"""
RAG Module
Provides ingestion and retrieval functionality for the RAG system.
"""

from .retriever import retrieve, get_concatenated_results

__all__ = ['retrieve', 'get_concatenated_results']
