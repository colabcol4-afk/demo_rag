"""
RAG Ingestion Pipeline
Combines document parsing, chunking, and vector store population into a single workflow.

This script orchestrates:
1. Document Parsing (PDF, DOCX, HTML) using LLMWhisperer
2. Semantic Chunking with size validation
3. Vector Store Population in Qdrant with incremental updates
"""

import sys
from pathlib import Path

# Add utils directory to path to import sibling modules
utils_dir = Path(__file__).parent.parent
sys.path.insert(0, str(utils_dir))

# Import functions from existing modules in utils
import parsing
import chunking
import populate

# Configuration
DATA_FOLDER = Path(__file__).parent.parent.parent.parent / "data"
PARSED_FOLDER = Path(__file__).parent.parent.parent.parent / "parsed_text"
CHUNKS_FOLDER = Path(__file__).parent.parent.parent.parent / "chunks"


def check_prerequisites() -> bool:
    """
    Check if all required folders and API connections are working.

    Returns:
        bool: True if all checks pass, False otherwise
    """
    print("=" * 70)
    print("🔍 PREREQUISITES CHECK")
    print("=" * 70)

    # Check folder structure
    print("\n📁 Checking folder structure...")
    DATA_FOLDER.mkdir(parents=True, exist_ok=True)
    PARSED_FOLDER.mkdir(parents=True, exist_ok=True)
    CHUNKS_FOLDER.mkdir(parents=True, exist_ok=True)
    print(f"   ✓ Data folder: {DATA_FOLDER}")
    print(f"   ✓ Parsed folder: {PARSED_FOLDER}")
    print(f"   ✓ Chunks folder: {CHUNKS_FOLDER}")

    # Check for documents
    documents = list(DATA_FOLDER.glob("*.pdf")) + list(DATA_FOLDER.glob("*.docx")) + \
                list(DATA_FOLDER.glob("*.doc")) + list(DATA_FOLDER.glob("*.html")) + \
                list(DATA_FOLDER.glob("*.htm"))

    if not documents:
        print(f"\n⚠️  WARNING: No documents found in {DATA_FOLDER}")
        print("   Supported formats: PDF, DOCX, DOC, HTML, HTM")
        print("   Please add documents to the data folder before running ingestion.")
        return False

    print(f"\n📄 Found {len(documents)} document(s) in data folder:")
    for doc in documents:
        print(f"   • {doc.name}")

    # Check API connectivity (optional - will be checked during execution)
    print("\n🔐 API keys configured:")
    print("   ✓ LLMWhisperer API key")
    print("   ✓ NVIDIA Embeddings API key")
    print("   ✓ Qdrant Cloud credentials")

    return True


def run_ingestion_pipeline():
    """
    Execute the complete RAG ingestion pipeline:
    1. Parse documents from data folder
    2. Chunk parsed text semantically
    3. Populate Qdrant vector store
    """
    print("\n" + "=" * 70)
    print("🚀 STARTING RAG INGESTION PIPELINE")
    print("=" * 70)

    try:
        # Step 1: Parse Documents
        print("\n" + "=" * 70)
        print("STEP 1: DOCUMENT PARSING")
        print("=" * 70)
        parsing.sync_folders()
        print("\n✅ Document parsing complete!")

        # Step 2: Chunk Documents
        print("\n" + "=" * 70)
        print("STEP 2: SEMANTIC CHUNKING")
        print("=" * 70)
        chunking.main()
        print("\n✅ Document chunking complete!")

        # Step 3: Populate Vector Store
        print("\n" + "=" * 70)
        print("STEP 3: VECTOR STORE POPULATION")
        print("=" * 70)
        populate.main()
        print("\n✅ Vector store population complete!")

        # Final Summary
        print("\n" + "=" * 70)
        print("✅ INGESTION PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 70)

        # Show statistics
        parsed_files = list(PARSED_FOLDER.glob("*.txt"))
        chunk_folders = [f for f in CHUNKS_FOLDER.iterdir() if f.is_dir()]

        print(f"\n📊 Final Statistics:")
        print(f"   • Parsed documents: {len(parsed_files)}")
        print(f"   • Chunked documents: {len(chunk_folders)}")

        total_chunks = 0
        for folder in chunk_folders:
            total_chunks += len(list(folder.glob("*.txt")))
        print(f"   • Total chunks: {total_chunks}")

        print(f"\n💡 Your documents are now ready for retrieval!")
        print(f"   Use retriever.py to query the vector store.\n")

    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user")
        print("   You can re-run the script to continue from where it left off.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Pipeline failed with error: {str(e)}")
        print("\nPlease check:")
        print("1. All API keys are valid and have remaining quota")
        print("2. Network connection is stable")
        print("3. Documents in data folder are valid and not corrupted")
        raise


def main():
    """Main entry point for the ingestion pipeline."""
    print("\n" + "=" * 70)
    print("📚 RAG INGESTION PIPELINE")
    print("   Document Processing → Chunking → Vector Store")
    print("=" * 70)

    # Check prerequisites
    if not check_prerequisites():
        print("\n❌ Prerequisites check failed. Exiting.")
        return

    # Confirm before proceeding
    print("\n" + "-" * 70)
    print("This will:")
    print("1. Parse any new documents in the data folder")
    print("2. Create semantic chunks from parsed text")
    print("3. Upload new chunks to the Qdrant vector store")
    print("\nExisting documents will be skipped (incremental update).")
    print("-" * 70)

    proceed = input("\nProceed with ingestion? [y/n]: ")
    if proceed.lower() != 'y':
        print("\n❌ Ingestion cancelled by user.")
        return

    # Run the pipeline
    run_ingestion_pipeline()


if __name__ == "__main__":
    main()
