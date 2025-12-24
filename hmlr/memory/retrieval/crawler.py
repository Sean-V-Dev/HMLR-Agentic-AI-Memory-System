"""
Lattice Crawler - Retrieves relevant context from stored memory.

Uses semantic similarity via vector embeddings to find relevant chunks:
- Embeds the query into a 1024D vector (bge-large-en-v1.5)
- Performs cosine similarity search against stored chunk embeddings
- Returns top-k most similar chunks for LLM filtering (The Governor)
"""

from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
import sys
import os
import logging
import json
import re


logger = logging.getLogger(__name__)

from hmlr.core.model_config import model_config
try:
    from hmlr.memory.models import Intent, RetrievedContext, DayNode, TaskState
    from hmlr.memory.sliding_window import SlidingWindow
    from hmlr.memory.storage import Storage
    from hmlr.memory.embeddings.embedding_manager import EmbeddingStorage
    from hmlr.memory import UserPlan
    from hmlr.core.exceptions import RetrievalError, VectorDatabaseError
except ImportError:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    from hmlr.memory.models import Intent, RetrievedContext, DayNode, TaskState
    from hmlr.memory.sliding_window import SlidingWindow
    from hmlr.memory.storage import Storage
    from hmlr.memory.embeddings.embedding_manager import EmbeddingStorage
    from hmlr.memory import UserPlan
    from hmlr.core.exceptions import RetrievalError, VectorDatabaseError


class LatticeCrawler:
    """Retrieves relevant memory chunks using semantic similarity."""

    def __init__(
        self, 
        storage: Storage, 
        max_days_back: int = None, 
        recency_weight: float = 0.5, 
        use_vector_search: bool = True
    ):
        """Initialize crawler with embedding-based semantic search."""
        self.storage = storage
        self.max_days_back = max_days_back
        self.recency_weight = recency_weight
        self.use_vector_search = use_vector_search
        
        # Initialize embedding storage if vector search enabled
        if self.use_vector_search:
            try:
                self.embedding_storage = EmbeddingStorage(storage)
                logger.info("Vector search enabled (embeddings initialized)")
            except Exception as e:
                logger.warning(f"Vector search init failed: {e}. Falling back to keyword search.", exc_info=True)
                # We don't raise here because we can fallback to non-vector search (fail-open)
                # But we log it clearly as a system degradation.
                self.use_vector_search = False
                self.embedding_storage = None
        else:
            self.embedding_storage = None
    
    def _search_gardened_memory(
        self,
        query: str,
        top_k: int = 20,
        min_similarity: float = 0.55
    ) -> List[Dict]:
        """
       
        Args:
            query: Query text to embed and search
            top_k: Number of chunks to retrieve
            min_similarity: Minimum similarity threshold (0.0-1.0)
            
        Returns:
            List of dicts with chunk_id, chunk_type, text_content, parent_id, 
            global_tags, similarity, block_id, topic_label
        """
        if not self.use_vector_search or not self.embedding_storage:
            return []
        
        print(f"    Searching gardened memory: '{query[:60]}...'")
        
        try:
            # Search using embedding manager (now searches gardened_memory too)
            results = self.embedding_storage.search_similar(
                query=query,
                top_k=top_k,
                min_similarity=min_similarity or model_config.MIN_SIMILARITY_THRESHOLD
            )
        except Exception as e:
            logger.error(f"Vector database search failed: {e}", exc_info=True)
            raise RetrievalError(f"Critical failure in vector search: {e}") from e
            
        # Filter to ONLY gardened chunks (chunk_id format: sent_YYYYMMDD_HHMMSS_hash or para_YYYYMMDD_HHMMSS_hash)
        gardened_results = []
        
        for result in results:
            chunk_id = result.get('turn_id')  # EmbeddingStorage uses 'turn_id' field
            
            # Only process if it's a gardened chunk (starts with 'sent_' or 'para_')
            if not chunk_id or not (chunk_id.startswith('sent_') or chunk_id.startswith('para_')):
                continue
            
            try:
                # Get full chunk data from gardened_memory table
                cursor = self.storage.conn.cursor()
                cursor.execute("""
                    SELECT chunk_id, chunk_type, text_content, parent_id, global_tags, block_id
                    FROM gardened_memory
                    WHERE chunk_id = ?
                """, (chunk_id,))
                
                chunk_row = cursor.fetchone()
                if not chunk_row:
                    continue
                
                # Get topic label from original bridge block
                cursor.execute("""
                    SELECT content_json
                    FROM daily_ledger
                    WHERE block_id = ?
                """, (chunk_row[5],))  # block_id
                
                ledger_row = cursor.fetchone()
                topic_label = "Unknown Topic"
                if ledger_row:
                    try:
                        content_json = json.loads(ledger_row[0])
                        topic_label = content_json.get('topic_label', 'Unknown Topic')
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse content_json for block {chunk_row[5]}: {e}")
                        topic_label = "Corrupt Topic Data"
                
                # Parse global tags (stored as JSON string)
                global_tags = []
                if chunk_row[4]:  # global_tags column
                    try:
                        global_tags = json.loads(chunk_row[4])
                    except json.JSONDecodeError as e:
                        logger.warning(f"Data Corruption Alert: Failed to parse global_tags for chunk {chunk_row[0]}: {e}")
                        global_tags = []
                
                gardened_results.append({
                    'chunk_id': chunk_row[0],
                    'chunk_type': chunk_row[1],
                    'text_content': chunk_row[2],
                    'parent_id': chunk_row[3],
                    'global_tags': global_tags,
                    'block_id': chunk_row[5],
                    'topic_label': topic_label,
                    'similarity': result.get('similarity', 0.0)
                })
            except Exception as e:
                # Individual chunk retrieval errors shouldn't crash the whole search, but should be logged
                logger.warning(f"Failed to retrieve details for chunk {chunk_id}: {e}")
                continue
        
        # print(f"      Found {len(gardened_results)} gardened chunks (similarity ≥ {min_similarity})")
        
        return gardened_results
    
    def _search_with_vectors(
        self,
        query: str,
        top_k: int = 20,
        min_similarity: float = 0.55
    ) -> List[Dict]:
       
        # Delegate to new gardened memory search
        return self._search_gardened_memory(query, top_k, min_similarity)
    
 
    def retrieve_context(
        self,
        intent: Intent,
        current_day_id: str,
        max_results: int = 10,
        window: Optional[SlidingWindow] = None
    ) -> RetrievedContext:
        
        #Main retrieval method - searches memory and returns structured context.
        
        
        
        # print(f" Crawler: Searching gardened memory with intent:")
        # print(f"   Query Type: {intent.query_type.value}")
        # primary_topics = getattr(intent, 'primary_topics', [])
        # if primary_topics:
        #    print(f"   Primary Topics: {primary_topics}")
        
        # Track what we retrieve for lineage
        retrieved_chunk_ids = []
        relevant_tasks = []
        
        # Vector search across gardened memory chunks
        gardened_results = []
        if self.use_vector_search:
            # print(f"\n GARDENED MEMORY SEARCH (Long-term HMLR storage):")
            
            # Use raw query for semantic search
            search_query = intent.raw_query
            
            gardened_results = self._search_with_vectors(
                query=search_query,
                top_k=max_results * 2,  # Get more candidates for filtering
                min_similarity=model_config.MIN_SIMILARITY_THRESHOLD
            )
            # print(f"    Search query: '{search_query}'")
        
        # print(f"\n SEARCH RESULTS SUMMARY:")
        # print(f"   Gardened chunks found: {len(gardened_results)}")
        
        # 3. Deduplicate chunks if window provided
        # Note: Chunks are deduplicated by chunk_id, not turn_id
        if window and gardened_results:
            before_dedup = len(gardened_results)
            filtered_results = []
            
            for result in gardened_results:
                chunk_id = result.get('chunk_id')
                if not chunk_id:
                    filtered_results.append(result)
                    continue
                
                # Check if chunk already in window
                if not window.is_in_window(chunk_id):
                    filtered_results.append(result)
                else:
                    # High similarity - include even if seen before
                    similarity = result.get('similarity', 0)
                    if similarity >= 0.6:
                        filtered_results.append(result)
                        # print(f"    Keeping high-similarity chunk {chunk_id[:30]}... (similarity={similarity:.3f})")
            
            gardened_results = filtered_results
            after_dedup = len(gardened_results)
            if before_dedup > after_dedup:
                # print(f"    Deduplicated chunks: {before_dedup} → {after_dedup} "
                #       f"({before_dedup - after_dedup} skipped)")
                pass
        
        # 4. Sort by similarity (already scored by embedding search)
        gardened_results.sort(key=lambda x: x.get('similarity', 0), reverse=True)
        
        # 5. Take top N results
        top_results = gardened_results[:max_results]
        
        # Mark retrieved chunks as loaded in window
        if window:
            for result in top_results:
                chunk_id = result.get('chunk_id')
                if chunk_id:
                    window.mark_loaded(chunk_id)
                    retrieved_chunk_ids.append(chunk_id)
            
            for task in relevant_tasks:
                window.mark_loaded(task.task_id)
        
        # 6. Build RetrievedContext
        # Extract day_ids by querying daily_ledger.date field (not parsing block_id)
        sources = set()
        for result in top_results:
            block_id = result.get('block_id', '')
            if block_id:
                # Query daily_ledger for the date field
                cursor = self.storage.conn.cursor()
                cursor.execute("SELECT date FROM daily_ledger WHERE block_id = ?", (block_id,))
                row = cursor.fetchone()
                if row and row[0]:
                    sources.add(row[0])  # date field is already in YYYY-MM-DD format
                else:
                    logger.warning(f"Could not find date for block_id: {block_id}")
        
        context = RetrievedContext(
            contexts=top_results,  # Chunk-based contexts with global_tags
            active_tasks=relevant_tasks,
            sources=list(sources),
            retrieved_turn_ids=retrieved_chunk_ids  # NOTE: Now contains chunk_ids, not turn_ids
        )
        
        # Log chunk details
        for i, chunk in enumerate(top_results[:3], 1):
            chunk_type = chunk.get('chunk_type', 'unknown')
            topic = chunk.get('topic_label', 'Unknown')[:30]
            tags = chunk.get('global_tags', [])
            sim = chunk.get('similarity', 0)
            
        
        return context
    
    def _get_search_range(self, current_day_id: str) -> List[str]:
        """
        Generate list of day IDs to search (today backward N days).
        Only used if max_days_back is set. Otherwise searches all days.
        
        Args:
            current_day_id: Today's day ID (YYYY-MM-DD)
            
        Returns:
            List of day IDs in descending order (newest first)
        """
        if not self.max_days_back:
            return None  # No limit
        
        current_date = datetime.strptime(current_day_id, "%Y-%m-%d")
        day_ids = []
        
        for i in range(self.max_days_back):
            day = current_date - timedelta(days=i)
            day_ids.append(day.strftime("%Y-%m-%d"))
        
        return day_ids
    
    def _parse_time_range(self, time_range: Tuple[str, str], current_day_id: str) -> List[str]:

        current_date = datetime.strptime(current_day_id, "%Y-%m-%d")
        day_ids = []
        
        # Parse start time
        start_str = time_range[0].lower() if len(time_range) > 0 else "today"
        
        if start_str == "today":
            start_date = current_date
        elif start_str == "yesterday":
            start_date = current_date - timedelta(days=1)
        elif "last week" in start_str or "past week" in start_str:
            start_date = current_date - timedelta(days=7)
        elif "last month" in start_str or "past month" in start_str:
            start_date = current_date - timedelta(days=30)
        else:
            # Try to parse as date
            try:
                start_date = datetime.strptime(start_str, "%Y-%m-%d")
            except:
                start_date = current_date  # Fallback to today
        
        # Generate day range from start_date to current_date
        days = (current_date - start_date).days + 1
        for i in range(days):
            day = current_date - timedelta(days=i)
            day_ids.append(day.strftime("%Y-%m-%d"))
        
        return day_ids


