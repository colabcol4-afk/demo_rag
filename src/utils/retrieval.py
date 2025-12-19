import os
from typing import Sequence
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

# --- 1. Configuration (Set your keys here) ---
QDRANT_URL = os.environ.get("QDRANT_URL")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")
COLLECTION_NAME = "document_chunks"  # Name for your collection

# NVIDIA Embeddings Configuration
NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY")
NVIDIA_MODEL = "nvidia/nv-embedqa-e5-v5"  # Embedding model

GROQ_API_KEY = os.environ.get("GROQ_API_KEY") 

# --- 2. Initialize Models ---

# Simple Dense Embeddings (NVIDIA)
dense_embeddings = NVIDIAEmbeddings(
    api_key=NVIDIA_API_KEY,
    model="nvidia/nv-embedqa-e5-v5",
    truncate="END"
)

# Re-ranker LLM (Groq)
llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model="deepseek-r1-distill-llama-70b",
    temperature=0,
)

# --- 3. Custom Groq Re-ranker ---
class GroqReranker(BaseDocumentCompressor):
    llm: ChatGroq
    top_n: int = 6

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Callbacks = None
    ) -> Sequence[Document]:
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
            return documents[:self.top_n]

# --- 4. Setup Simple Vector Store Retriever ---
vector_store = QdrantVectorStore.from_existing_collection(
    embedding=dense_embeddings,
    collection_name=COLLECTION_NAME,
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    retrieval_mode=RetrievalMode.DENSE, # Pure semantic search
)

# --- 5. Final Retrieval Pipeline ---
base_retriever = vector_store.as_retriever(search_kwargs={"k": 10})
reranker = GroqReranker(llm=llm, top_n=3)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=reranker,
    base_retriever=base_retriever
)

# --- 6. Execution ---
query = "What are world models and how are they different from normal models ?"
results = compression_retriever.invoke(query)

for i, doc in enumerate(results):
    print(f"Rank {i+1}: {doc.page_content[:200]}...")