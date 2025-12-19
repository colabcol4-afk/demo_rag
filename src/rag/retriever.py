"""
RAG Retriever Module
Provides a clean interface for retrieving relevant documents from the Qdrant vector store.

Features:
- Dense semantic search using NVIDIA embeddings
- LLM-based re-ranking with Groq
- Returns top 5 most relevant results
- Clear formatting and demarkation of results
- Production-ready error handling
"""

import sys
import os
from pathlib import Path
from typing import List, Sequence
from langchain_core.documents import Document
from langchain_core.callbacks import Callbacks
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from langchain_groq import ChatGroq
from qdrant_client import QdrantClient

# Note: In LangChain v1.x, the import path has moved to a sub-package
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors.base import BaseDocumentCompressor

# --- CONFIGURATION ---
QDRANT_URL = os.environ.get("QDRANT_URL")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")
COLLECTION_NAME = "document_chunks"

# NVIDIA Embeddings Configuration
NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY")
NVIDIA_MODEL = "nvidia/nv-embedqa-e5-v5"

# Groq Configuration for Re-ranking
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODEL = "deepseek-r1-distill-llama-70b"

# Retrieval Configuration
INITIAL_RETRIEVAL_COUNT = 15  # Get top 15 from vector store
FINAL_RESULT_COUNT = 5  # Re-rank and return top 5


# --- ENVIRONMENT VALIDATION ---
def validate_environment():
    """
    Validate all required environment variables are set.
    
    Raises:
        EnvironmentError: If any required environment variable is missing
    """
    required_vars = {
        "QDRANT_URL": QDRANT_URL,
        "QDRANT_API_KEY": QDRANT_API_KEY,
        "NVIDIA_API_KEY": NVIDIA_API_KEY,
        "GROQ_API_KEY": GROQ_API_KEY,
    }
    
    missing = [name for name, value in required_vars.items() if not value]
    
    if missing:
        error_msg = f"""
❌ Missing required environment variables: {', '.join(missing)}

Please set these in your Streamlit Cloud secrets:
1. Go to your app dashboard
2. Click 'Settings' → 'Secrets'
3. Add the missing variables:
   
   QDRANT_URL = "your-qdrant-url"
   QDRANT_API_KEY = "your-api-key"
   NVIDIA_API_KEY = "your-nvidia-key"
   GROQ_API_KEY = "your-groq-key"
        """
        raise EnvironmentError(error_msg)
    
    print("✅ All environment variables loaded successfully")


# Validate immediately on import
validate_environment()


# --- INITIALIZE MODELS ---

# Dense Embeddings (NVIDIA)
try:
    dense_embeddings = NVIDIAEmbeddings(
        api_key=NVIDIA_API_KEY,
        model=NVIDIA_MODEL,
        truncate="END"
    )
    print(f"✅ NVIDIA Embeddings initialized: {NVIDIA_MODEL}")
except Exception as e:
    raise ConnectionError(f"❌ Failed to initialize NVIDIA embeddings: {e}")

# Re-ranker LLM (Groq)
try:
    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model=GROQ_MODEL,
        temperature=0,
    )
    print(f"✅ Groq LLM initialized: {GROQ_MODEL}")
except Exception as e:
    raise ConnectionError(f"❌ Failed to initialize Groq LLM: {e}")


# --- CUSTOM GROQ RE-RANKER ---
class GroqReranker(BaseDocumentCompressor):
    """Custom re-ranker using Groq LLM for intelligent ranking."""

    llm: ChatGroq
    top_n: int = FINAL_RESULT_COUNT

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Callbacks = None
    ) -> Sequence[Document]:
        """
        Re-rank documents based on relevance to query.

        Args:
            documents: List of documents to re-rank
            query: Search query
            callbacks: Optional callbacks

        Returns:
            Re-ranked and filtered list of top documents
        """
        if not documents:
            return []

        prompt = ChatPromptTemplate.from_template(
            """Rank these documents for the query: {query}
            Return ONLY JSON: {{"relevant_indices": [idx1, idx2, ...]}}
            Documents:
            {context}"""
        )

        context_str = "\n".join([f"ID {i}: {doc.page_content[:400]}" for i, doc in enumerate(documents)])
        chain = prompt | self.llm | JsonOutputParser()

        try:
            result = chain.invoke({"query": query, "context": context_str})
            indices = result.get("relevant_indices", [])
            return [documents[i] for i in indices if 0 <= i < len(documents)][:self.top_n]
        except Exception as e:
            print(f"⚠️ Re-ranking failed: {e}. Using fallback order.")
            # Fallback: return first N documents if re-ranking fails
            return documents[:self.top_n]


# --- VECTOR STORE INITIALIZATION (with error handling) ---
def initialize_vector_store():
    """
    Initialize vector store with proper error handling and validation.
    
    Returns:
        QdrantVectorStore: Initialized vector store
        
    Raises:
        ConnectionError: If connection to Qdrant fails
        ValueError: If collection doesn't exist
    """
    try:
        # Test connection first
        print(f"🔗 Connecting to Qdrant at: {QDRANT_URL[:50]}...")
        test_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        
        # Check if collection exists
        collections = test_client.get_collections()
        collection_names = [col.name for col in collections.collections]
        
        if COLLECTION_NAME not in collection_names:
            error_msg = f"""
❌ Collection '{COLLECTION_NAME}' not found in Qdrant!

Available collections: {collection_names if collection_names else 'None'}

🔧 To fix this:
1. Run your data ingestion script to create the collection
2. Or create it manually in Qdrant dashboard
3. Make sure the collection name matches: '{COLLECTION_NAME}'
            """
            raise ValueError(error_msg)
        
        print(f"✅ Collection '{COLLECTION_NAME}' found")
        
        # Collection exists, create vector store
        vector_store = QdrantVectorStore.from_existing_collection(
            embedding=dense_embeddings,
            collection_name=COLLECTION_NAME,
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            retrieval_mode=RetrievalMode.DENSE,
        )
        
        print(f"✅ Successfully connected to Qdrant vector store")
        return vector_store
        
    except ValueError:
        # Re-raise ValueError with our custom message
        raise
    except Exception as e:
        error_msg = f"""
❌ Failed to connect to Qdrant vector store!

Error: {str(e)}

🔧 Troubleshooting:
1. Check QDRANT_URL is correct (currently: {QDRANT_URL[:50]}...)
2. Verify QDRANT_API_KEY is set correctly
3. Ensure collection '{COLLECTION_NAME}' exists in Qdrant
4. Check network connectivity to Qdrant
5. Verify Qdrant service is running
        """
        raise ConnectionError(error_msg) from e


# Initialize vector store
vector_store = initialize_vector_store()


# --- SETUP RETRIEVAL PIPELINE ---
base_retriever = vector_store.as_retriever(search_kwargs={"k": INITIAL_RETRIEVAL_COUNT})
reranker = GroqReranker(llm=llm, top_n=FINAL_RESULT_COUNT)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=reranker,
    base_retriever=base_retriever
)

print("✅ Retrieval pipeline initialized successfully")
print(f"📊 Configuration: Initial retrieval={INITIAL_RETRIEVAL_COUNT}, Final results={FINAL_RESULT_COUNT}")


# --- RETRIEVAL FUNCTION ---

def retrieve(query: str, top_k: int = 5, verbose: bool = True) -> List[Document]:
    """
    Retrieve the most relevant documents for a given query.

    Args:
        query: Search query string
        top_k: Number of results to return (default: 5)
        verbose: Whether to print formatted results (default: True)

    Returns:
        List of Document objects with content and metadata
    """
    if not query or not query.strip():
        print("⚠️ Empty query provided")
        return []
    
    # Update the reranker's top_n if different from default
    if top_k != FINAL_RESULT_COUNT:
        reranker.top_n = top_k

    try:
        # Perform retrieval
        results = compression_retriever.invoke(query)

        if verbose:
            print_results(query, results)

        return results
    
    except Exception as e:
        print(f"❌ Error during retrieval: {e}")
        raise


def print_results(query: str, results: List[Document]):
    """
    Print formatted retrieval results with clear demarkation.

    Args:
        query: The search query
        results: List of retrieved documents
    """
    print("\n" + "=" * 80)
    print(f"🔍 QUERY: {query}")
    print("=" * 80)

    if not results:
        print("\n⚠️  No results found for this query.")
        print("=" * 80)
        return

    print(f"\n📊 Found {len(results)} relevant result(s):\n")

    for i, doc in enumerate(results, 1):
        print("─" * 80)
        print(f"📄 RESULT #{i}")
        print("─" * 80)

        # Print metadata
        metadata = doc.metadata
        print(f"📌 Source Document: {metadata.get('source', 'N/A')}")
        print(f"📑 Chunk: {metadata.get('chunk_file', 'N/A')}")
        print(f"🔢 Chunk Number: {metadata.get('chunk_number', 'N/A')}")
        print(f"📏 Length: {metadata.get('total_chars', len(doc.page_content))} characters")

        # Print content
        print(f"\n📝 CONTENT:")
        print("─" * 80)
        print(doc.page_content)
        print("─" * 80)
        print()

    print("=" * 80)
    print(f"✅ Retrieval complete! Returned {len(results)} result(s).")
    print("=" * 80)
    print()


def get_concatenated_results(query: str, top_k: int = 5, separator: str = "\n\n---\n\n") -> str:
    """
    Retrieve results and return them as a single concatenated string.

    Args:
        query: Search query string
        top_k: Number of results to return
        separator: String to separate different results (default: newlines with ---)

    Returns:
        Concatenated string of all results
    """
    try:
        results = retrieve(query, top_k=top_k, verbose=False)

        if not results:
            return "No results found."

        concatenated = []
        for i, doc in enumerate(results, 1):
            result_text = f"[RESULT {i} - Source: {doc.metadata.get('source', 'N/A')}]\n"
            result_text += doc.page_content
            concatenated.append(result_text)

        return separator.join(concatenated)
    
    except Exception as e:
        return f"Error retrieving results: {str(e)}"


# --- MAIN EXECUTION (for testing) ---

def main():
    """Main function for testing the retriever."""
    print("\n" + "=" * 80)
    print("🤖 RAG RETRIEVER - Interactive Mode")
    print("=" * 80)
    print("\nType your query below (or 'quit' to exit):\n")

    while True:
        try:
            query = input("🔍 Query: ").strip()

            if query.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break

            if not query:
                print("⚠️  Please enter a valid query.\n")
                continue

            # Retrieve and display results
            retrieve(query, top_k=5, verbose=True)
            
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")


if __name__ == "__main__":
    main()