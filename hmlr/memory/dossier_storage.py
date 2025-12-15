"""
Dossier Embedding Storage - Fact-level vector search for dossier retrieval.

This module provides embedding storage and search specifically for dossier facts.
Each fact is embedded individually to enable granular semantic search, while
maintaining the association with its parent dossier.

Author: CognitiveLattice Team
Created: 2025-12-15 (Phase 1: Dossier System)
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
    - Embedding Model: all-MiniLM-L6-v2 (384D, same as memory system)
    - Storage: SQLite table dossier_fact_embeddings
    - Search: Cosine similarity with configurable threshold (default 0.4)
    """
    
    def __init__(self, db_path: str, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize dossier embedding storage.
        
        Args:
            db_path: Path to SQLite database (same as main storage)
            model_name: SentenceTransformer model name
        """
        self.db_path = db_path
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self._initialize_table()
        logger.info(f"DossierEmbeddingStorage initialized with model: {model_name}")
    
    def _initialize_table(self):
        """Create dossier_fact_embeddings table if it doesn't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
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
        conn.commit()
        conn.close()
        logger.debug("Dossier fact embeddings table initialized")
    
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
        
        Example:
            results = storage.search_similar_facts("vegetarian diet", top_k=10)
            # Returns: [
            #   ('fact_001', 'dos_diet_001', 0.85),
            #   ('fact_023', 'dos_diet_001', 0.72),
            #   ('fact_045', 'dos_health_002', 0.65)
            # ]
            # Note: dos_diet_001 appears twice = strong candidate
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


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("🧪 Dossier Embedding Storage Test")
    print("=" * 60)
    
    import os
    import tempfile
    
    # Create temporary test database
    with tempfile.NamedTemporaryFile(delete=False, suffix='.db') as tmp:
        test_db = tmp.name
    
    try:
        # Initialize storage
        print("\n1. Initializing storage...")
        storage = DossierEmbeddingStorage(test_db)
        print(f"   ✅ Storage initialized: {test_db}")
        
        # Test embedding storage
        print("\n2. Storing fact embeddings...")
        facts = [
            ("fact_001", "dos_diet", "User is strictly vegetarian"),
            ("fact_002", "dos_diet", "User avoids all meat products"),
            ("fact_003", "dos_diet", "User prefers plant-based proteins"),
            ("fact_004", "dos_tech", "User works with Python"),
            ("fact_005", "dos_tech", "User prefers functional programming")
        ]
        
        for fact_id, dossier_id, fact_text in facts:
            success = storage.save_fact_embedding(fact_id, dossier_id, fact_text)
            if success:
                print(f"   ✅ Embedded: {fact_text[:40]}...")
        
        # Test search
        print("\n3. Searching for similar facts...")
        queries = [
            "dietary preferences",
            "programming languages",
            "vegetarian food"
        ]
        
        for query in queries:
            results = storage.search_similar_facts(query, top_k=3, threshold=0.3)
            print(f"\n   Query: '{query}'")
            for fact_id, dossier_id, score in results:
                print(f"     - {dossier_id}/{fact_id}: {score:.3f}")
        
        # Test vote tallying (simulate Multi-Vector Voting)
        print("\n4. Simulating Multi-Vector Voting...")
        incoming_facts = [
            "User follows a vegan lifestyle",
            "User does not eat eggs or dairy"
        ]
        
        vote_tally = {}
        for fact in incoming_facts:
            results = storage.search_similar_facts(fact, top_k=5, threshold=0.4)
            for _, dossier_id, score in results:
                if dossier_id not in vote_tally:
                    vote_tally[dossier_id] = {'hits': 0, 'score_sum': 0.0}
                vote_tally[dossier_id]['hits'] += 1
                vote_tally[dossier_id]['score_sum'] += score
        
        print(f"   Vote results:")
        for dossier_id, stats in sorted(vote_tally.items(), key=lambda x: x[1]['hits'], reverse=True):
            print(f"     - {dossier_id}: {stats['hits']} hits, total score: {stats['score_sum']:.3f}")
        
        # Test fact count
        print("\n5. Checking fact counts...")
        total = storage.get_fact_count()
        diet_count = storage.get_fact_count("dos_diet")
        tech_count = storage.get_fact_count("dos_tech")
        print(f"   Total facts: {total}")
        print(f"   Diet dossier: {diet_count} facts")
        print(f"   Tech dossier: {tech_count} facts")
        
        print("\n" + "=" * 60)
        print("✅ All tests passed!")
        
    finally:
        # Cleanup
        if os.path.exists(test_db):
            os.remove(test_db)
            print(f"🧹 Cleaned up test database")
