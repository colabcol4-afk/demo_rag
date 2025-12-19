"""
RAG Retriever Module
Provides a clean interface for retrieving relevant documents from the Qdrant vector store.

Features:
- Dense semantic search using NVIDIA embeddings
- LLM-based re-ranking with Groq
- Returns top 5 most relevant results
- Clear formatting and demarkation of results
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


# --- INITIALIZE MODELS ---

# Dense Embeddings (NVIDIA)
dense_embeddings = NVIDIAEmbeddings(
    api_key=NVIDIA_API_KEY,
    model=NVIDIA_MODEL,
    truncate="END"
)

# Re-ranker LLM (Groq)
llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model=GROQ_MODEL,
    temperature=0,
)


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
        except:
            # Fallback: return first N documents if re-ranking fails
            return documents[:self.top_n]


# --- SETUP VECTOR STORE RETRIEVER ---
vector_store = QdrantVectorStore.from_existing_collection(
    embedding=dense_embeddings,
    collection_name=COLLECTION_NAME,
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    retrieval_mode=RetrievalMode.DENSE,
)

# --- FINAL RETRIEVAL PIPELINE ---
base_retriever = vector_store.as_retriever(search_kwargs={"k": INITIAL_RETRIEVAL_COUNT})
reranker = GroqReranker(llm=llm, top_n=FINAL_RESULT_COUNT)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=reranker,
    base_retriever=base_retriever
)


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
    # Update the reranker's top_n if different from default
    if top_k != FINAL_RESULT_COUNT:
        reranker.top_n = top_k

    # Perform retrieval
    results = compression_retriever.invoke(query)

    if verbose:
        print_results(query, results)

    return results


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
    results = retrieve(query, top_k=top_k, verbose=False)

    if not results:
        return "No results found."

    concatenated = []
    for i, doc in enumerate(results, 1):
        result_text = f"[RESULT {i} - Source: {doc.metadata.get('source', 'N/A')}]\n"
        result_text += doc.page_content
        concatenated.append(result_text)

    return separator.join(concatenated)


# --- MAIN EXECUTION (for testing) ---

def main():
    """Main function for testing the retriever."""
    print("\n" + "=" * 80)
    print("🤖 RAG RETRIEVER - Interactive Mode")
    print("=" * 80)
    print("\nType your query below (or 'quit' to exit):\n")

    while True:
        query = input("🔍 Query: ").strip()

        if query.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Goodbye!")
            break

        if not query:
            print("⚠️  Please enter a valid query.\n")
            continue

        # Retrieve and display results
        retrieve(query, top_k=5, verbose=True)


if __name__ == "__main__":
    main()
