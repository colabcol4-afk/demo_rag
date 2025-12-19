"""
Qdrant Vector Store Population Script with Incremental Updates

This script tracks processed documents in a JSON file and only uploads new chunks
on subsequent runs, making it efficient for ongoing document additions.

Features:
- Tracks processed folders in processed_documents.json
- Only processes NEW folders not in the tracking file
- Updates tracking file after successful upload
- Shows statistics about new vs existing documents
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Set
import uuid
from datetime import datetime
from tqdm import tqdm

from langchain_core.documents import Document
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams


# ============================================================================
# CONFIGURATION - UPDATE THESE VALUES
# ============================================================================

# Qdrant Cloud Configuration
QDRANT_URL = os.environ.get("QDRANT_URL")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")
COLLECTION_NAME = "document_chunks"  # Name for your collection

# NVIDIA Embeddings Configuration
NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY")
NVIDIA_MODEL = "nvidia/nv-embedqa-e5-v5"  # Embedding model

# Path Configuration (absolute paths relative to src directory)
SRC_DIR = Path(__file__).parent.parent  # Points to 'src/' directory
CHUNKS_DIR = SRC_DIR / "chunks"
TRACKING_FILE = SRC_DIR.parent / "processed_documents.json"  # In project root

# Vector Configuration
EMBEDDING_DIMENSION = 1024  # Dimension for nv-embedqa-e5-v5 model
DISTANCE_METRIC = Distance.COSINE  # Can be COSINE, EUCLID, DOT, or MANHATTAN


# ============================================================================
# TRACKING FILE MANAGEMENT
# ============================================================================

def load_tracking_data() -> Dict:
    """
    Load the tracking data from JSON file.
    
    Returns:
        Dictionary with tracking data or empty structure if file doesn't exist
    """
    if TRACKING_FILE.exists():
        try:
            with open(TRACKING_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"✓ Loaded tracking data from {TRACKING_FILE}")
            return data
        except json.JSONDecodeError as e:
            print(f"⚠️  Warning: Could not parse {TRACKING_FILE}. Starting fresh.")
            return create_empty_tracking_structure()
    else:
        print(f"ℹ️  No tracking file found. This appears to be the first run.")
        return create_empty_tracking_structure()


def create_empty_tracking_structure() -> Dict:
    """Create an empty tracking data structure."""
    return {
        "last_updated": None,
        "processed_documents": {}
    }


def save_tracking_data(tracking_data: Dict):
    """
    Save tracking data to JSON file.
    
    Args:
        tracking_data: Dictionary with tracking information
    """
    tracking_data["last_updated"] = datetime.now().isoformat()
    
    with open(TRACKING_FILE, 'w', encoding='utf-8') as f:
        json.dump(tracking_data, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Updated tracking file: {TRACKING_FILE}")


def get_processed_folders(tracking_data: Dict) -> Set[str]:
    """
    Get set of already processed folder names.
    
    Args:
        tracking_data: Tracking data dictionary
        
    Returns:
        Set of processed folder names
    """
    return set(tracking_data.get("processed_documents", {}).keys())


def identify_new_folders(chunks_dir: Path, processed_folders: Set[str]) -> List[Path]:
    """
    Identify folders that haven't been processed yet.
    
    Args:
        chunks_dir: Path to chunks directory
        processed_folders: Set of already processed folder names
        
    Returns:
        List of new folder paths
    """
    if not chunks_dir.exists():
        raise FileNotFoundError(f"Chunks directory not found: {chunks_dir}")
    
    all_folders = [f for f in chunks_dir.iterdir() if f.is_dir()]
    new_folders = [f for f in all_folders if f.name not in processed_folders]
    
    return new_folders


def add_document_to_tracking(
    tracking_data: Dict,
    folder_name: str,
    chunk_count: int,
    chunk_files: List[str]
):
    """
    Add a processed document to tracking data.
    
    Args:
        tracking_data: Tracking data dictionary
        folder_name: Name of the source folder
        chunk_count: Number of chunks processed
        chunk_files: List of chunk filenames
    """
    tracking_data["processed_documents"][folder_name] = {
        "chunks_count": chunk_count,
        "processed_date": datetime.now().isoformat(),
        "chunk_files": sorted(chunk_files)
    }


# ============================================================================
# DOCUMENT LOADING WITH TRACKING
# ============================================================================

def load_chunks_from_folders(
    folders: List[Path],
    tracking_data: Dict
) -> List[Document]:
    """
    Load all text chunks from specified folders and create Document objects.
    
    Args:
        folders: List of folder paths to process
        tracking_data: Tracking data to update
        
    Returns:
        List of Document objects with metadata
    """
    if not folders:
        print("\n✅ All documents are already processed. No new chunks to add.")
        return []
    
    documents = []
    
    print(f"\n🔍 Found {len(folders)} NEW documents to process")
    print(f"📂 Processing chunks from: {folders[0].parent.absolute()}\n")
    
    for source_folder in tqdm(folders, desc="Loading new documents"):
        source_name = source_folder.name
        
        # Get all .txt files in the folder, sorted numerically
        txt_files = sorted(
            source_folder.glob("*.txt"),
            key=lambda x: int(x.stem) if x.stem.isdigit() else x.stem
        )
        
        if not txt_files:
            print(f"⚠️  Warning: No .txt files found in {source_folder.name}")
            continue
        
        chunk_files_list = []
        folder_documents = []
        
        # Process each chunk file
        for chunk_file in txt_files:
            try:
                # Read chunk content
                with open(chunk_file, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                
                # Skip empty chunks
                if not content:
                    print(f"⚠️  Warning: Empty chunk in {source_folder.name}/{chunk_file.name}")
                    continue
                
                # Create Document with metadata
                doc = Document(
                    page_content=content,
                    metadata={
                        "source": source_name,  # Source document name
                        "chunk_file": chunk_file.name,  # Original chunk filename
                        "chunk_number": int(chunk_file.stem) if chunk_file.stem.isdigit() else chunk_file.stem,
                        "total_chars": len(content),
                        "processed_date": datetime.now().isoformat()
                    }
                )
                
                folder_documents.append(doc)
                chunk_files_list.append(chunk_file.name)
                
            except Exception as e:
                print(f"❌ Error reading {source_folder.name}/{chunk_file.name}: {str(e)}")
                continue
        
        # Add to main documents list
        documents.extend(folder_documents)
        
        # Add to tracking data
        add_document_to_tracking(
            tracking_data,
            source_name,
            len(folder_documents),
            chunk_files_list
        )
    
    # Print summary statistics
    print(f"\n✅ Loaded {len(documents)} NEW chunks from {len(folders)} documents")
    
    # Show breakdown by source
    if documents:
        print("\n📊 New chunks per source document:")
        source_counts = {}
        for doc in documents:
            source = doc.metadata['source']
            source_counts[source] = source_counts.get(source, 0) + 1
        
        for source, count in sorted(source_counts.items()):
            print(f"   • {source}: {count} chunks")
    
    return documents


def show_tracking_summary(tracking_data: Dict):
    """
    Display summary of processed documents from tracking data.
    
    Args:
        tracking_data: Tracking data dictionary
    """
    processed_docs = tracking_data.get("processed_documents", {})
    
    if not processed_docs:
        print("\n📊 No documents have been processed yet.")
        return
    
    total_chunks = sum(doc["chunks_count"] for doc in processed_docs.values())
    
    print(f"\n📊 Previously Processed Documents: {len(processed_docs)}")
    print(f"   Total chunks already in vector store: {total_chunks}")
    
    if len(processed_docs) <= 10:
        print("\n   Documents:")
        for name, info in sorted(processed_docs.items()):
            date = info["processed_date"].split("T")[0] if "T" in info["processed_date"] else info["processed_date"]
            print(f"   • {name}: {info['chunks_count']} chunks (added {date})")
    else:
        print(f"\n   (Run with --show-all to see full list)")


# ============================================================================
# EMBEDDER AND VECTOR STORE FUNCTIONS (Same as before)
# ============================================================================

def create_embedder():
    """Create and return NVIDIA embeddings instance."""
    print("\n🔧 Initializing NVIDIA Embeddings...")
    
    embedder = NVIDIAEmbeddings(
        model=NVIDIA_MODEL,
        api_key=NVIDIA_API_KEY,
        truncate="END",
    )
    
    print(f"   ✓ Using model: {NVIDIA_MODEL}")
    
    return embedder


def get_or_create_collection(client: QdrantClient, collection_name: str, vector_size: int):
    """
    Get existing collection or create if it doesn't exist.
    For incremental updates, we never recreate the collection.
    
    Args:
        client: QdrantClient instance
        collection_name: Name of the collection
        vector_size: Dimension of the embeddings
    """
    # Check if collection exists
    collections = client.get_collections().collections
    collection_exists = any(col.name == collection_name for col in collections)
    
    if collection_exists:
        print(f"✓ Using existing collection '{collection_name}'")
    else:
        print(f"\n🏗️  Creating new collection '{collection_name}'...")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=DISTANCE_METRIC
            )
        )
        print(f"   ✓ Collection created successfully")


def populate_vector_store(
    documents: List[Document],
    embedder,
    qdrant_url: str,
    qdrant_api_key: str,
    collection_name: str
):
    """
    Add new documents to Qdrant vector store.
    
    Args:
        documents: List of Document objects to upload
        embedder: Embeddings instance
        qdrant_url: Qdrant cloud URL
        qdrant_api_key: Qdrant API key
        collection_name: Name of the collection
    """
    if not documents:
        print("\n✅ No new documents to upload.")
        return
    
    print("\n🔗 Connecting to Qdrant Cloud...")
    print(f"   URL: {qdrant_url}")
    
    # Create Qdrant client
    client = QdrantClient(
        url=qdrant_url,
        api_key=qdrant_api_key,
        prefer_grpc=True,  # Use gRPC for better performance
    )
    
    print("   ✓ Connected successfully")
    
    # Get or create collection
    get_or_create_collection(client, collection_name, EMBEDDING_DIMENSION)
    
    # Create vector store instance
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embedder,
    )
    
    # Generate unique IDs for each document
    print(f"\n📝 Generating unique IDs for {len(documents)} new documents...")
    document_ids = [str(uuid.uuid4()) for _ in documents]
    
    # Upload documents in batches
    print(f"\n⬆️  Uploading {len(documents)} new documents to Qdrant...")
    print("   This may take a while depending on the number of documents...\n")
    
    try:
        # Add documents with IDs
        batch_size = 100  # Process in batches of 100
        
        for i in tqdm(range(0, len(documents), batch_size), desc="Uploading batches"):
            batch_docs = documents[i:i + batch_size]
            batch_ids = document_ids[i:i + batch_size]
            
            vector_store.add_documents(
                documents=batch_docs,
                ids=batch_ids
            )
        
        print(f"\n✅ Successfully uploaded all {len(documents)} new documents!")
        
        # Verify upload
        collection_info = client.get_collection(collection_name=collection_name)
        print(f"\n📊 Collection Statistics:")
        print(f"   • Total vectors in collection: {collection_info.points_count}")
        print(f"   • Collection name: {collection_name}")
        print(f"   • Vector dimension: {collection_info.config.params.vectors.size}")
        print(f"   • Distance metric: {collection_info.config.params.vectors.distance}")
        
    except Exception as e:
        print(f"\n❌ Error during upload: {str(e)}")
        raise


def test_similarity_search(
    qdrant_url: str,
    qdrant_api_key: str,
    collection_name: str,
    embedder,
    query: str = "What is this document about?",
    k: int = 3
):
    """Test the vector store with a similarity search."""
    print(f"\n🔍 Testing similarity search with query: '{query}'")
    
    # Create vector store instance
    client = QdrantClient(
        url=qdrant_url,
        api_key=qdrant_api_key,
        prefer_grpc=True,
    )
    
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embedder,
    )
    
    # Perform similarity search
    results = vector_store.similarity_search(query, k=k)
    
    print(f"\n📋 Top {k} Results:")
    for i, doc in enumerate(results, 1):
        print(f"\n--- Result {i} ---")
        print(f"Source: {doc.metadata.get('source', 'N/A')}")
        print(f"Chunk: {doc.metadata.get('chunk_file', 'N/A')}")
        print(f"Content preview: {doc.page_content[:200]}...")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main(auto_confirm=False):
    """Main execution function with incremental update support.

    Args:
        auto_confirm: If True, skip all confirmation prompts
    """

    print("=" * 70)
    print("🚀 QDRANT VECTOR STORE - INCREMENTAL UPDATE")
    print("=" * 70)
    
    try:
        # Step 1: Load tracking data
        print("\n📖 Loading tracking data...")
        tracking_data = load_tracking_data()
        
        # Show summary of what's already processed
        show_tracking_summary(tracking_data)
        
        # Step 2: Identify new folders
        print("\n🔍 Scanning for new documents...")
        processed_folders = get_processed_folders(tracking_data)
        new_folders = identify_new_folders(CHUNKS_DIR, processed_folders)
        
        if not new_folders:
            print("\n" + "=" * 70)
            print("✅ ALL DOCUMENTS UP TO DATE!")
            print("=" * 70)
            print("\nNo new documents found in chunks folder.")
            print(f"All {len(processed_folders)} documents are already in the vector store.")
            return
        
        print(f"\n🆕 Found {len(new_folders)} new document(s) to process:")
        for folder in new_folders:
            print(f"   • {folder.name}")

        # Confirm processing (skip if auto_confirm)
        if not auto_confirm:
            print("\n" + "-" * 70)
            proceed = input("\nProceed with uploading these new documents? [y/n]: ")
            if proceed.lower() != 'y':
                print("\n❌ Upload cancelled by user.")
                return
        else:
            print("\n✓ Auto-confirmed, proceeding with upload...")
        
        # Step 3: Load chunks from new folders only
        documents = load_chunks_from_folders(new_folders, tracking_data)
        
        if not documents:
            print("\n⚠️  No valid chunks found in new documents.")
            return
        
        # Step 4: Create embedder
        embedder = create_embedder()
        
        # Step 5: Upload only new documents
        populate_vector_store(
            documents=documents,
            embedder=embedder,
            qdrant_url=QDRANT_URL,
            qdrant_api_key=QDRANT_API_KEY,
            collection_name=COLLECTION_NAME
        )
        
        # Step 6: Save tracking data
        print("\n💾 Saving tracking data...")
        save_tracking_data(tracking_data)
        
        # Step 7: Show final summary
        print("\n" + "=" * 70)
        print("📊 FINAL SUMMARY")
        print("=" * 70)
        
        total_processed = len(tracking_data["processed_documents"])
        total_chunks = sum(
            doc["chunks_count"] 
            for doc in tracking_data["processed_documents"].values()
        )
        
        print(f"\n   Total documents in vector store: {total_processed}")
        print(f"   Total chunks in vector store: {total_chunks}")
        print(f"   New documents added this run: {len(new_folders)}")
        print(f"   New chunks added this run: {len(documents)}")

        # Step 8: Test the vector store (optional) - skip if auto_confirm
        if not auto_confirm:
            print("\n" + "=" * 70)
            test_search = input("\nWould you like to test the vector store with a search? [y/n]: ")

            if test_search.lower() == 'y':
                test_query = input("Enter your search query: ")
                test_similarity_search(
                    qdrant_url=QDRANT_URL,
                    qdrant_api_key=QDRANT_API_KEY,
                    collection_name=COLLECTION_NAME,
                    embedder=embedder,
                    query=test_query,
                    k=3
                )
        
        print("\n" + "=" * 70)
        print("✅ SCRIPT COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ Fatal Error: {str(e)}")
        raise


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Populate Qdrant Vector Store")
    parser.add_argument(
        "--yes", "-y",
        action="store_true",
        help="Auto-confirm and skip all prompts"
    )
    args = parser.parse_args()

    main(auto_confirm=args.yes)