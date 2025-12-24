"""
Dossier Embedding Storage - Fact-level vector search for dossier retrieval.

This module provides embedding storage and search specifically for dossier facts.
Each fact is embedded individually to enable granular semantic search, while
maintaining the association with its parent dossier.


"""

import sqlite3
import json
import numpy as np
import logging
from sentence_transformers import SentenceTransformer
from typing import List, Tuple, Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class DossierEmbeddingStorage:
    """
    Manages fact-level embeddings for dossier retrieval.
    
    Each fact within a dossier is embedded individually, enabling granular
    semantic search. When a query matches any fact in a dossier, the entire
    dossier can be retrieved and provided as context.
    
    Architecture:
    - Embedding Model: snowflake-arctic-embed-l (1024D, best accuracy)
    - Storage: SQLite table dossier_fact_embeddings
    - Search: Cosine similarity with configurable threshold (default 0.4)
    """
    
    def __init__(self, 
                 db_path: str, 
                 model_name: str = 'Snowflake/snowflake-arctic-embed-l'):
        """
        Initialize dossier embedding storage.
        
        Args:
            db_path: Path to SQLite database (same as main storage)
            model_name: Model for embedding and search (same model for both)
        """
        self.db_path = db_path
        self.model_name = model_name
        
        # Load model (used for both embedding and search)
        self.model = SentenceTransformer(model_name)
        
        self._initialize_table()
        logger.info(f"DossierEmbeddingStorage initialized with model: {model_name}")
    
    def _initialize_table(self):
        """Create embedding tables if they don't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Fact-level embeddings (existing)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS dossier_fact_embeddings (
                fact_id TEXT PRIMARY KEY,
                dossier_id TEXT NOT NULL,
                embedding BLOB NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (fact_id) REFERENCES dossier_facts(fact_id) ON DELETE CASCADE,
                FOREIGN KEY (dossier_id) REFERENCES dossiers(dossier_id) ON DELETE CASCADE
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_dfe_dossier ON dossier_fact_embeddings(dossier_id)")
        
        # Dossier-level search embeddings
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS dossier_search_embeddings (
                dossier_id TEXT PRIMARY KEY,
                embedding BLOB NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (dossier_id) REFERENCES dossiers(dossier_id) ON DELETE CASCADE
            )
        """)
        
        conn.commit()
        conn.close()
        logger.debug("Dossier embedding tables initialized")
    
    def save_fact_embedding(self, fact_id: str, dossier_id: str, fact_text: str) -> bool:
        """
        Embed and store a single fact.
        
        Args:
            fact_id: Unique fact ID (format: fact_YYYYMMDD_HHMMSS_XXX)
            dossier_id: Parent dossier ID
            fact_text: The actual fact text to embed
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Generate embedding
            embedding = self.model.encode(fact_text)
            embedding_blob = embedding.astype(np.float32).tobytes()
            
            # Store in database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO dossier_fact_embeddings 
                (fact_id, dossier_id, embedding, created_at)
                VALUES (?, ?, ?, ?)
            """, (fact_id, dossier_id, embedding_blob, datetime.now().isoformat()))
            conn.commit()
            conn.close()
            
            logger.debug(f"Embedded fact {fact_id} for dossier {dossier_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save fact embedding for {fact_id}: {e}")
            return False
    
    def save_dossier_search_embedding(self, dossier_id: str, search_summary: str) -> bool:
        """
        Embed and store dossier-level search summary for broad topic matching.
        
        This enables broad retrieval like "which car for family trip" to match
        dossiers with search_summary containing general concepts about vehicles,
        transportation, family use, etc.
        
        Args:
            dossier_id: Dossier ID
            search_summary: Search-optimized summary text
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Generate embedding
            embedding = self.model.encode(search_summary)
            embedding_blob = embedding.astype(np.float32).tobytes()
            
            # Store in database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO dossier_search_embeddings 
                (dossier_id, embedding, created_at)
                VALUES (?, ?, ?)
            """, (dossier_id, embedding_blob, datetime.now().isoformat()))
            conn.commit()
            conn.close()
            
            logger.debug(f"Embedded search summary for dossier {dossier_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save dossier search embedding for {dossier_id}: {e}")
            return False
    
    def search_similar_facts(
        self,
        query: str,
        top_k: int = 10,
        threshold: float = 0.4
    ) -> List[Tuple[str, str, float]]:
        """
        Search for facts similar to the query.
        
        This is the core of the Multi-Vector Voting system. Each incoming fact
        packet will search for similar facts, and dossiers with the most matching
        facts will "bubble up" as candidates.
        
        Args:
            query: Query text to search for
            top_k: Maximum number of results to return
            threshold: Minimum similarity score (0-1, default 0.4)
        
        Returns:
            List of tuples: (fact_id, dossier_id, similarity_score)
            Ordered by similarity score descending
        """
        try:
            # Embed query
            query_embedding = self.model.encode(query).astype(np.float32)
            
            # Retrieve all embeddings
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT fact_id, dossier_id, embedding FROM dossier_fact_embeddings")
            
            results = []
            for fact_id, dossier_id, embedding_blob in cursor.fetchall():
                # Deserialize embedding
                fact_embedding = np.frombuffer(embedding_blob, dtype=np.float32)
                
                # Check for dimension mismatch (happens when switching embedding models)
                if len(fact_embedding) != len(query_embedding):
                    logger.warning(f"Skipping fact {fact_id}: embedding dimension mismatch "
                                 f"({len(fact_embedding)} vs {len(query_embedding)}). "
                                 f"Consider regenerating embeddings with current model.")
                    continue
                
                # Compute cosine similarity
                similarity = float(np.dot(query_embedding, fact_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(fact_embedding)
                ))
                
                # Filter by threshold
                if similarity >= threshold:
                    results.append((fact_id, dossier_id, similarity))
            
            conn.close()
            
            # Sort by similarity descending and limit to top_k
            results.sort(key=lambda x: x[2], reverse=True)
            results = results[:top_k]
            
            logger.debug(f"Found {len(results)} facts above threshold {threshold} for query: '{query[:50]}...'")
            return results
            
        except Exception as e:
            logger.error(f"Failed to search similar facts: {e}")
            return []
    
    def search_similar_dossiers(
        self,
        query: str,
        top_k: int = 10,
        threshold: float = 0.3
    ) -> List[Tuple[str, float]]:
        """
        Search for dossiers using search_summary embeddings (broad topic matching).
               
        Args:
            query: Query text to search for
            top_k: Maximum number of results to return
            threshold: Minimum similarity score (0-1, default 0.3 - lower for broad matching)
        
        Returns:
            List of tuples: (dossier_id, similarity_score)
            Ordered by similarity score descending
        """
        try:
            # Encode query
            query_embedding = self.model.encode(query)
            
            # Get all dossier search embeddings
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT dossier_id, embedding FROM dossier_search_embeddings")
            rows = cursor.fetchall()
            conn.close()
            
            if not rows:
                logger.debug("No dossier search embeddings found")
                return []
            
            # Compute similarities
            results = []
            for dossier_id, embedding_blob in rows:
                dossier_embedding = np.frombuffer(embedding_blob, dtype=np.float32)
                
                # Check for dimension mismatch
                if len(dossier_embedding) != len(query_embedding):
                    logger.warning(f"Skipping dossier {dossier_id}: embedding dimension mismatch "
                                 f"({len(dossier_embedding)} vs {len(query_embedding)}). "
                                 f"Consider regenerating embeddings with current model.")
                    continue
                
                # Cosine similarity
                similarity = np.dot(query_embedding, dossier_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(dossier_embedding)
                )
                
                if similarity >= threshold:
                    results.append((dossier_id, float(similarity)))
            
            # Sort by similarity descending and limit to top_k
            results.sort(key=lambda x: x[1], reverse=True)
            results = results[:top_k]
            
            logger.debug(f"Found {len(results)} dossiers above threshold {threshold} for query: '{query[:50]}...'")
            return results
            
        except Exception as e:
            logger.error(f"Failed to search similar dossiers: {e}")
            return []
    
    def get_dossier_by_fact_id(self, fact_id: str) -> Optional[str]:
        """
        Get dossier ID for a given fact ID.
        
        Args:
            fact_id: Fact ID to look up
        
        Returns:
            Dossier ID, or None if not found
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT dossier_id FROM dossier_fact_embeddings WHERE fact_id = ?", (fact_id,))
            result = cursor.fetchone()
            conn.close()
            return result[0] if result else None
        except Exception as e:
            logger.error(f"Failed to get dossier for fact {fact_id}: {e}")
            return None
    
    def get_fact_count(self, dossier_id: str = None) -> int:
        """
        Get count of embedded facts.
        
        Args:
            dossier_id: Optional filter by dossier
        
        Returns:
            Number of embedded facts
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            if dossier_id:
                cursor.execute("SELECT COUNT(*) FROM dossier_fact_embeddings WHERE dossier_id = ?", (dossier_id,))
            else:
                cursor.execute("SELECT COUNT(*) FROM dossier_fact_embeddings")
            
            count = cursor.fetchone()[0]
            conn.close()
            return count
            
        except Exception as e:
            logger.error(f"Failed to get fact count: {e}")
            return 0
    
    def delete_dossier_embeddings(self, dossier_id: str) -> bool:
        """
        Delete all fact embeddings for a dossier.
        
        This is called when a dossier is deleted or archived.
        
        Args:
            dossier_id: Dossier whose fact embeddings should be deleted
        
        Returns:
            True if successful, False otherwise
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("DELETE FROM dossier_fact_embeddings WHERE dossier_id = ?", (dossier_id,))
            deleted_count = cursor.rowcount
            conn.commit()
            conn.close()
            
            logger.info(f"Deleted {deleted_count} fact embeddings for dossier {dossier_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete embeddings for dossier {dossier_id}: {e}")
            return False


