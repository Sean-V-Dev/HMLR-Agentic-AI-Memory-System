"""
Vector Embeddings Module for RAG Retrieval

This module handles:
- Vector embedding generation
- Similarity search
- Hybrid retrieval (vector + keyword)
"""

from .embedding_manager import EmbeddingManager

__all__ = ['EmbeddingManager']
