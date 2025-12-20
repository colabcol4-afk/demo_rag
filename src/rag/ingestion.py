"""
In-Memory RAG Ingestion Pipeline
Processes documents directly in memory without intermediate file storage.

Flow:
1. Parse PDFs from data folder (in memory)
2. Chunk parsed text (in memory)
3. Upload chunks directly to Qdrant vector store
4. Track processed documents in JSON file
"""

import argparse
import json
import sys
import os
from pathlib import Path
from typing import Dict, List, Set
from datetime import datetime
from tqdm import tqdm

# LLMWhisperer for parsing
from unstract.llmwhisperer import LLMWhispererClientV2
from unstract.llmwhisperer.client_v2 import LLMWhispererClientException

# LangChain for chunking and vector store
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
import uuid

# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths
SRC_DIR = Path(__file__).parent.parent
DATA_FOLDER = SRC_DIR / "data"
TRACKING_FILE = SRC_DIR.parent / "processed_documents.json"

# API Keys from environment variables
LLMWHISPERER_API_KEY = os.environ.get("LLMWHISPERER_API_KEY")
NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY")
QDRANT_URL = os.environ.get("QDRANT_URL")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")
COLLECTION_NAME = "document_chunks"

# Chunking Configuration
MAX_CHUNK_CHARS = 1600
CHUNK_OVERLAP = 200
MAX_SECTION_CHARS = 8000  # Pre-split large docs
EMBEDDING_DIMENSION = 1024

# ============================================================================
# TRACKING MANAGEMENT
# ============================================================================

def load_tracking_data() -> Dict:
    """Load tracking data from JSON file."""
    if TRACKING_FILE.exists():
        try:
            with open(TRACKING_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return {"last_updated": None, "processed_documents": {}}
    return {"last_updated": None, "processed_documents": {}}

def save_tracking_data(tracking_data: Dict):
    """Save tracking data to JSON file."""
    tracking_data["last_updated"] = datetime.now().isoformat()
    with open(TRACKING_FILE, 'w', encoding='utf-8') as f:
        json.dump(tracking_data, f, indent=2, ensure_ascii=False)

def get_processed_files(tracking_data: Dict) -> Set[str]:
    """Get set of already processed file names."""
    return set(tracking_data.get("processed_documents", {}).keys())

# ============================================================================
# STEP 1: PARSE DOCUMENTS (In-Memory)
# ============================================================================

def parse_document_to_text(file_path: Path, client: LLMWhispererClientV2) -> str:
    """
    Parse a single document and return text.

    Args:
        file_path: Path to document
        client: LLMWhisperer client

    Returns:
        Extracted text
    """
    print(f"  📄 Parsing: {file_path.name}")

    result = client.whisper(
        file_path=str(file_path),
        wait_for_completion=True,
        wait_timeout=200
    )

    if result["status_code"] == 200 and result["status"] == "processed":
        text = result["extraction"]["result_text"]
        print(f"     ✓ Extracted {len(text):,} characters")
        return text
    else:
        raise Exception(f"Parsing failed: {result.get('status', 'Unknown error')}")

def parse_new_documents(data_folder: Path, processed_files: Set[str]) -> Dict[str, str]:
    """
    Parse all new documents in data folder.

    Returns:
        Dict of {filename: text}
    """
    # Find new files
    all_files = list(data_folder.glob("*.pdf")) + \
                list(data_folder.glob("*.docx")) + \
                list(data_folder.glob("*.doc"))

    new_files = [f for f in all_files if f.stem not in processed_files]

    if not new_files:
        print("\n✅ All documents already processed!")
        return {}

    print(f"\n📂 Found {len(new_files)} new document(s) to process:")
    for f in new_files:
        print(f"   • {f.name}")

    # Initialize parser
    print("\n🔧 Initializing LLMWhisperer parser...")
    client = LLMWhispererClientV2(api_key=LLMWHISPERER_API_KEY)

    # Parse each file
    parsed_docs = {}
    print(f"\n📝 Parsing {len(new_files)} document(s)...\n")

    for file_path in new_files:
        try:
            text = parse_document_to_text(file_path, client)
            parsed_docs[file_path.stem] = text
        except Exception as e:
            print(f"     ❌ Error: {str(e)}")
            continue

    print(f"\n✅ Successfully parsed {len(parsed_docs)}/{len(new_files)} document(s)")
    return parsed_docs

# ============================================================================
# STEP 2: CHUNK DOCUMENTS (In-Memory)
# ============================================================================

def pre_split_large_document(text: str, max_chars: int) -> List[str]:
    """Pre-split very large documents into sections."""
    if len(text) <= max_chars:
        return [text]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chars,
        chunk_overlap=400,
        length_function=len,
        separators=["\n\n\n", "\n\n", "\n", ". ", " "]
    )
    return splitter.split_text(text)

def split_oversized_chunk(chunk: str, max_chars: int, overlap: int) -> List[str]:
    """Split an oversized chunk."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chars,
        chunk_overlap=overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]
    )
    return splitter.split_text(chunk)

def validate_chunks(chunks: List[str], max_chars: int, overlap: int) -> List[str]:
    """Validate and fix oversized chunks."""
    validated = []
    for chunk in chunks:
        if len(chunk) <= max_chars:
            validated.append(chunk)
        else:
            validated.extend(split_oversized_chunk(chunk, max_chars, overlap))
    return validated

def chunk_text(text: str, doc_name: str, embedder) -> List[str]:
    """
    Chunk a single document's text.

    Args:
        text: Document text
        doc_name: Document name for logging
        embedder: NVIDIA embedder

    Returns:
        List of text chunks
    """
    print(f"  📄 Chunking: {doc_name}")
    print(f"     → Document size: {len(text):,} characters")

    # Pre-split large documents
    if len(text) > MAX_SECTION_CHARS:
        sections = pre_split_large_document(text, MAX_SECTION_CHARS)
        print(f"     → Pre-split into {len(sections)} sections")
    else:
        sections = [text]

    # Semantic chunking
    chunker = SemanticChunker(
        embeddings=embedder,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=90,
    )

    all_chunks = []
    for section in sections:
        try:
            docs = chunker.create_documents([section])
            chunks = [d.page_content for d in docs]
            all_chunks.extend(chunks)
        except Exception as e:
            # Fallback to simple splitting
            print(f"     ⚠️  Semantic chunking failed, using fallback")
            fallback = RecursiveCharacterTextSplitter(
                chunk_size=MAX_CHUNK_CHARS,
                chunk_overlap=CHUNK_OVERLAP,
                length_function=len
            )
            chunks = fallback.split_text(section)
            all_chunks.extend(chunks)

    # Validate chunk sizes
    validated_chunks = validate_chunks(all_chunks, MAX_CHUNK_CHARS, CHUNK_OVERLAP)

    avg_size = sum(len(c) for c in validated_chunks) // len(validated_chunks)
    print(f"     ✓ Created {len(validated_chunks)} chunks (avg: {avg_size} chars)")

    return validated_chunks

def chunk_all_documents(parsed_docs: Dict[str, str]) -> Dict[str, List[str]]:
    """
    Chunk all parsed documents.

    Returns:
        Dict of {doc_name: [chunks]}
    """
    if not parsed_docs:
        return {}

    print(f"\n🔧 Initializing semantic chunker...")
    embedder = NVIDIAEmbeddings(
        model="nvidia/nv-embedqa-e5-v5",
        api_key=NVIDIA_API_KEY,
        truncate="END"
    )
    print(f"   ✓ Using NVIDIA embeddings: nvidia/nv-embedqa-e5-v5")

    print(f"\n✂️  Chunking {len(parsed_docs)} document(s)...\n")

    chunked_docs = {}
    for doc_name, text in parsed_docs.items():
        try:
            chunks = chunk_text(text, doc_name, embedder)
            chunked_docs[doc_name] = chunks
        except Exception as e:
            print(f"  ❌ Error chunking {doc_name}: {str(e)}")
            continue

    total_chunks = sum(len(chunks) for chunks in chunked_docs.values())
    print(f"\n✅ Created {total_chunks} total chunks from {len(chunked_docs)} document(s)")

    return chunked_docs, embedder

# ============================================================================
# STEP 3: UPLOAD TO VECTOR STORE
# ============================================================================

def create_documents_from_chunks(chunked_docs: Dict[str, List[str]]) -> List[Document]:
    """Convert chunks to LangChain Documents."""
    documents = []

    for doc_name, chunks in chunked_docs.items():
        for i, chunk in enumerate(chunks, 1):
            doc = Document(
                page_content=chunk,
                metadata={
                    "source": doc_name,
                    "chunk_number": i,
                    "total_chunks": len(chunks),
                    "chunk_chars": len(chunk),
                    "processed_date": datetime.now().isoformat()
                }
            )
            documents.append(doc)

    return documents

def upload_to_vector_store(chunked_docs: Dict[str, List[str]], embedder):
    """Upload chunks directly to Qdrant."""
    if not chunked_docs:
        print("\n✅ No new chunks to upload")
        return

    print(f"\n🔗 Connecting to Qdrant Cloud...")

    # Create client
    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        prefer_grpc=False,
    )

    # Get or create collection
    collections = client.get_collections().collections
    collection_exists = any(col.name == COLLECTION_NAME for col in collections)

    if not collection_exists:
        print(f"   → Creating collection: {COLLECTION_NAME}")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=EMBEDDING_DIMENSION,
                distance=Distance.COSINE
            )
        )
    else:
        print(f"   ✓ Using existing collection: {COLLECTION_NAME}")

    # Create vector store
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embedder,
    )

    # Convert to Documents
    documents = create_documents_from_chunks(chunked_docs)
    document_ids = [str(uuid.uuid4()) for _ in documents]

    # Upload in batches
    print(f"\n⬆️  Uploading {len(documents)} chunks to Qdrant...\n")

    batch_size = 100
    for i in tqdm(range(0, len(documents), batch_size), desc="Uploading"):
        batch_docs = documents[i:i + batch_size]
        batch_ids = document_ids[i:i + batch_size]
        vector_store.add_documents(documents=batch_docs, ids=batch_ids)

    # Verify
    collection_info = client.get_collection(collection_name=COLLECTION_NAME)
    print(f"\n✅ Upload complete!")
    print(f"   • Total vectors in collection: {collection_info.points_count}")

    return len(documents)

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_pipeline(auto_confirm=False):
    """Execute the complete in-memory RAG pipeline."""

    print("=" * 70)
    print("🚀 IN-MEMORY RAG INGESTION PIPELINE")
    print("=" * 70)

    # Check data folder
    if not DATA_FOLDER.exists():
        print(f"\n❌ Data folder not found: {DATA_FOLDER}")
        return

    print(f"\n📂 Data folder: {DATA_FOLDER}")

    try:
        # Load tracking
        print("\n📖 Loading processing history...")
        tracking_data = load_tracking_data()
        processed_files = get_processed_files(tracking_data)

        if processed_files:
            print(f"   ✓ Found {len(processed_files)} previously processed document(s)")
        else:
            print(f"   ℹ️  No documents processed yet")

        # Step 1: Parse documents
        print("\n" + "=" * 70)
        print("STEP 1: DOCUMENT PARSING")
        print("=" * 70)

        parsed_docs = parse_new_documents(DATA_FOLDER, processed_files)

        if not parsed_docs:
            print("\n✅ All documents up to date! Nothing to process.")
            return

        # Step 2: Chunk documents
        print("\n" + "=" * 70)
        print("STEP 2: SEMANTIC CHUNKING")
        print("=" * 70)

        chunked_docs, embedder = chunk_all_documents(parsed_docs)

        if not chunked_docs:
            print("\n❌ No chunks created")
            return

        # Step 3: Upload to vector store
        print("\n" + "=" * 70)
        print("STEP 3: VECTOR STORE UPLOAD")
        print("=" * 70)

        chunks_uploaded = upload_to_vector_store(chunked_docs, embedder)

        # Update tracking
        print("\n💾 Updating processing history...")
        for doc_name, chunks in chunked_docs.items():
            tracking_data["processed_documents"][doc_name] = {
                "chunks_count": len(chunks),
                "processed_date": datetime.now().isoformat()
            }
        save_tracking_data(tracking_data)
        print("   ✓ Tracking updated")

        # Final summary
        print("\n" + "=" * 70)
        print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 70)

        total_processed = len(tracking_data["processed_documents"])
        total_chunks = sum(d["chunks_count"] for d in tracking_data["processed_documents"].values())

        print(f"\n📊 Summary:")
        print(f"   • Documents processed this run: {len(chunked_docs)}")
        print(f"   • Chunks uploaded this run: {chunks_uploaded}")
        print(f"   • Total documents in vector store: {total_processed}")
        print(f"   • Total chunks in vector store: {total_chunks}")
        print()

    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Pipeline failed: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

# ============================================================================
# ENTRY POINT
# ============================================================================

def main(auto_confirm=False):
    """Main entry point."""
    run_pipeline(auto_confirm=auto_confirm)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="In-Memory RAG Ingestion Pipeline")
    parser.add_argument("--yes", "-y", action="store_true", help="Auto-confirm")
    args = parser.parse_args()

    main(auto_confirm=args.yes)
