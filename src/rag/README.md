# RAG Module

This module provides a complete RAG (Retrieval-Augmented Generation) pipeline with ingestion and retrieval capabilities.

## Files

### 1. `ingestion.py`
Combined pipeline that orchestrates document processing, chunking, and vector store population.

**Usage:**
```bash
cd src/rag
python ingestion.py
```

**What it does:**
1. Parses documents from `data/` folder (PDF, DOCX, HTML) using LLMWhisperer
2. Creates semantic chunks with size validation using NVIDIA embeddings
3. Uploads new chunks to Qdrant vector store (incremental updates)

**Features:**
- Prerequisite checks (folders, documents, API keys)
- Interactive confirmation before processing
- Incremental updates (only processes new documents)
- Progress tracking and detailed statistics

### 2. `retriever.py`
Clean retrieval interface for querying the vector store.

**Usage as a script:**
```bash
cd src/rag
python retriever.py
```

**Usage as a module:**
```python
from rag import retrieve, get_concatenated_results

# Get top 5 results with formatted output
results = retrieve("What are world models?", top_k=5)

# Get concatenated string of results
text = get_concatenated_results("What are world models?", top_k=5)
```

**Features:**
- Dense semantic search using NVIDIA embeddings
- LLM-based re-ranking with Groq (DeepSeek model)
- Returns top 5 most relevant results by default
- Clear formatting with source metadata
- Interactive mode for testing

## Configuration

All configurations (API keys, URLs, models) are extracted from your existing implementation:

- **LLMWhisperer**: Document parsing
- **NVIDIA Embeddings**: `nvidia/nv-embedqa-e5-v5`
- **Qdrant Cloud**: Vector storage
- **Groq**: Re-ranking with `deepseek-r1-distill-llama-70b`

## Folder Structure

```
agent_demo/
├── data/              # Input documents (PDF, DOCX, HTML)
├── parsed_text/       # Parsed text files
├── chunks/            # Semantic chunks organized by document
└── src/
    ├── rag/           # This module
    │   ├── __init__.py
    │   ├── ingestion.py
    │   ├── retriever.py
    │   └── README.md
    └── utils/
        ├── parsing.py
        ├── chunking.py
        ├── populate.py
        └── retrieval.py
```

## Workflow

### First Time Setup
1. Add documents to `data/` folder
2. Run `ingestion.py` to process all documents
3. Use `retriever.py` to query the vector store

### Adding New Documents
1. Add new documents to `data/` folder
2. Run `ingestion.py` again (only processes new documents)
3. New documents are now searchable

### Querying
- Interactive mode: `python retriever.py`
- In your code: Import and use the `retrieve()` function

## API Keys Used

The following API keys from your implementation are used:
- LLMWhisperer API Key
- NVIDIA API Key
- Qdrant Cloud URL & API Key
- Groq API Key

All keys are hardcoded from your existing implementation for convenience.
