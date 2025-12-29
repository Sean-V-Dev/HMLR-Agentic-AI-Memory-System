"""
Embedding Manager - Handles vector embedding generation and search

Uses sentence-transformers for fast, high-quality embeddings.
Model: all-MiniLM-L6-v2 (384 dimensions, fast, good quality)
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import pickle
from hmlr.core.model_config import model_config

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("sentence-transformers not installed. Install with: pip install sentence-transformers")


# Global model cache to avoid reloading models
_MODEL_CACHE = {}


class EmbeddingManager:
    """
    Manages vector embeddings for conversation turns.
    """
    
    def __init__(self, model_name: str = None):
        """
        Initialize embedding manager.
        
        Args:
            model_name: SentenceTransformer model name (defaults to centralized config)
        """
        self.model_name = model_name or model_config.EMBEDDING_MODEL_NAME
        self.dimension = model_config.EMBEDDING_DIMENSION
        
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "sentence-transformers required. Install with: pip install sentence-transformers"
            )
        
        # Check if model already loaded in cache
        if self.model_name in _MODEL_CACHE:
            cached = _MODEL_CACHE[self.model_name]
            self.model = cached['model']
            self.device = cached['device']
            return
        
        # Load model for first time
        import torch
        device = 'cpu'
        gpu_info = "CPU only"
        
        try:
            if torch.cuda.is_available():
                device = 'cuda'
                gpu_name = torch.cuda.get_device_name(0)
                gpu_info = f"GPU ({gpu_name})"
                print(f" GPU detected: {gpu_name}", flush=True)
            else:
                print(f"⚠️  No GPU detected - using CPU", flush=True)
                print(f"   To enable GPU: pip install torch --index-url https://download.pytorch.org/whl/cu121", flush=True)
        except Exception as e:
            print(f"⚠️  GPU check failed: {e} - using CPU", flush=True)
        
        print(f"Loading embedding model: {self.model_name} on {gpu_info}...", flush=True)
        self.model = SentenceTransformer(self.model_name, device=device, local_files_only=True)
        self.device = device
        print(f"✓ Model loaded: {self.model_name} ({self.dimension}D) on {device.upper()}", flush=True)
        
        # Cache for future instances
        _MODEL_CACHE[self.model_name] = {
            'model': self.model,
            'device': self.device
        }
    
    def encode(self, text: str) -> np.ndarray:
        """
        Generate embedding for text.
        
        Args:
            text: Text to embed
            
        Returns:
            Numpy array of shape (384,)
        """
        return self.model.encode(text, show_progress_bar=False)
    
    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for multiple texts (faster).
        
        Args:
            texts: List of texts to embed
            
        Returns:
            Numpy array of shape (N, 384)
        """
        return self.model.encode(texts, show_progress_bar=True, batch_size=32)
    
    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Similarity score (0-1)
        """
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_product / (norm1 * norm2))
    
    def serialize_embedding(self, embedding: np.ndarray) -> bytes:
        """
        Serialize embedding to bytes for database storage.
        
        Args:
            embedding: Numpy array
            
        Returns:
            Pickled bytes
        """
        return pickle.dumps(embedding)
    
    def deserialize_embedding(self, data: bytes) -> np.ndarray:
        """
        Deserialize embedding from database bytes.
        
        Args:
            data: Pickled bytes
            
        Returns:
            Numpy array
        """
        return pickle.loads(data)
    
    def find_similar(self, query_embedding: np.ndarray, 
                    stored_embeddings: List[Tuple[str, np.ndarray, str]], 
                    top_k: int = 10,
                    min_similarity: float = 0.0) -> List[Dict]:
        """
        Find most similar embeddings to query.
        
        Args:
            query_embedding: Query vector
            stored_embeddings: List of (embedding_id, vector, text) tuples
            top_k: Number of results to return
            min_similarity: Minimum similarity threshold (0-1)
            
        Returns:
            List of dicts with embedding_id, similarity, text
        """
        results = []
        
        for embedding_id, stored_vec, text in stored_embeddings:
            # Skip embeddings with mismatched dimensions (from old models)
            if stored_vec.shape[0] != query_embedding.shape[0]:
                continue
                
            similarity = self.cosine_similarity(query_embedding, stored_vec)
            
            if similarity >= min_similarity:
                results.append({
                    'embedding_id': embedding_id,
                    'similarity': similarity,
                    'text': text,
                    'vector': stored_vec  # Include vector for visualization
                })
        
        # Sort by similarity descending
        results.sort(key=lambda x: x['similarity'], reverse=True)
        
        return results[:top_k]


class EmbeddingStorage:
    """
    Helper class to store and retrieve embeddings from database.
    Works with Storage class from storage.py
    """
    
    def __init__(self, storage):
        """
        Initialize with storage instance.
        
        Args:
            storage: Storage instance from storage.py
        """
        self.storage = storage
        self.embedding_manager = EmbeddingManager()
    
    def save_turn_embeddings(self, turn_id: str, chunks: List[str]) -> int:
        """
        Generate and save embeddings for turn chunks.
        
        Args:
            turn_id: Turn identifier
            chunks: List of text chunks
            
        Returns:
            Number of embeddings saved
        """
        cursor = self.storage.conn.cursor()
        
        for idx, chunk_text in enumerate(chunks):
            # Generate embedding
            embedding = self.embedding_manager.encode(chunk_text)
            
            # Serialize for storage
            embedding_bytes = self.embedding_manager.serialize_embedding(embedding)
            
            # Generate embedding ID
            embedding_id = f"{turn_id}_chunk_{idx}"
            
            # Store in database
            cursor.execute("""
                INSERT OR REPLACE INTO embeddings 
                (embedding_id, turn_id, chunk_index, embedding, text_content, dimension, model_name)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                embedding_id,
                turn_id,
                idx,
                embedding_bytes,
                chunk_text,
                self.embedding_manager.dimension,
                self.embedding_manager.model_name
            ))
        
        self.storage.conn.commit()
        return len(chunks)
    
    def get_all_embeddings(self) -> List[Tuple[str, np.ndarray, str, str]]:
        """
        Load all embeddings from database.
        
        Returns:
            List of (embedding_id, vector, text, turn_id) tuples
        """
        cursor = self.storage.conn.cursor()
        
        rows = cursor.execute("""
            SELECT embedding_id, embedding, text_content, turn_id
            FROM embeddings
        """).fetchall()
        
        results = []
        for row in rows:
            embedding_id = row[0]
            embedding_bytes = row[1]
            text = row[2]
            turn_id = row[3]
            
            # Deserialize embedding
            vector = self.embedding_manager.deserialize_embedding(embedding_bytes)
            
            results.append((embedding_id, vector, text, turn_id))
        
        return results
    
    def search_similar(self, query: str, top_k: int = 10, 
                      min_similarity: float = 0.55) -> List[Dict]:
        """
        Search for similar conversations using vector similarity.
        ONLY searches gardened_memory (long-term storage), not embeddings table (query vectors).
        
        Args:
            query: Search query
            top_k: Number of results
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of results with turn_id (chunk_id), similarity, text
        """
        # Encode query
        query_embedding = self.embedding_manager.encode(query)
        
        # ONLY load embeddings from gardened_memory (long-term searchable content)
        # The embeddings table contains query vectors, not searchable content
        gardened_embeddings = self._get_gardened_embeddings()
        
        if not gardened_embeddings:
            return []
        
        # Find similar
        similar = self.embedding_manager.find_similar(
            query_embedding,
            [(e[0], e[1], e[2]) for e in gardened_embeddings],
            top_k=top_k,
            min_similarity=min_similarity
        )
        
        # Add chunk_id to results (stored as turn_id for backward compatibility)
        embedding_to_chunk = {e[0]: e[3] for e in gardened_embeddings}
        for result in similar:
            result['turn_id'] = embedding_to_chunk[result['embedding_id']]  # This is actually chunk_id
            result['query_vector'] = query_embedding
        
        return similar
    
    def _get_gardened_embeddings(self) -> List[Tuple[str, np.ndarray, str, str]]:
        """
        Get embeddings from gardened_memory chunks.
        
        Returns:
            List of (embedding_id, vector, text, chunk_id) tuples
        """
        cursor = self.storage.conn.cursor()
        
        # Check if gardened_memory table exists
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='gardened_memory'
        """)
        
        if not cursor.fetchone():
            return []  # Table doesn't exist yet
        
        # Get all gardened chunks that have embeddings
        cursor.execute("""
            SELECT g.chunk_id, e.embedding, g.text_content
            FROM gardened_memory g
            JOIN embeddings e ON e.turn_id = g.chunk_id
        """)
        
        results = []
        for row in cursor.fetchall():
            chunk_id = row[0]
            embedding_bytes = row[1]
            text = row[2]
            
            # Deserialize embedding
            vector = self.embedding_manager.deserialize_embedding(embedding_bytes)
            
            # Format: (embedding_id, vector, text, chunk_id)
            results.append((chunk_id, vector, text, chunk_id))
        
        return results


if __name__ == "__main__":
    print(" Testing EmbeddingManager...\n")
    
    # Test encoding
    manager = EmbeddingManager()
    
    text = "Python is a programming language"
    embedding = manager.encode(text)
    print(f" Encoded text: {text}")
    print(f"   Embedding shape: {embedding.shape}")
    print(f"   First 5 values: {embedding[:5]}\n")
    
    # Test similarity
    text1 = "I love programming in Python"
    text2 = "Python is my favorite coding language"
    text3 = "I enjoy eating pizza"
    
    vec1 = manager.encode(text1)
    vec2 = manager.encode(text2)
    vec3 = manager.encode(text3)
    
    sim_12 = manager.cosine_similarity(vec1, vec2)
    sim_13 = manager.cosine_similarity(vec1, vec3)
    
    print(f"Similarity tests:")
    print(f"  '{text1}' vs '{text2}': {sim_12:.3f} (should be HIGH)")
    print(f"  '{text1}' vs '{text3}': {sim_13:.3f} (should be LOW)")
