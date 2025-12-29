"""
Chunk storage integration - Saves chunks to database with FTS5 indexing.

This module extends the existing Storage class with chunk-specific operations.
"""
import sqlite3
import json
from typing import List, Optional
from ..chunking.chunk_engine import Chunk
import logging

logger = logging.getLogger(__name__)


class ChunkStorage:
    """
    Storage operations for hierarchical chunks.
    
    Integrates with existing Storage class to add chunk persistence.
    """
    
    def __init__(self, storage):
        """
        Initialize chunk storage.
        
        Args:
            storage: Instance of memory.storage.Storage
        """
        self.storage = storage
        self.conn = storage.conn
    
    def save_chunks(self, chunks: List[Chunk]) -> None:
        """
        Save a batch of chunks to the database.
        
        Args:
            chunks: List of Chunk objects to persist
        """
        if not chunks:
            return
        
        cursor = self.conn.cursor()
        
        try:
            for chunk in chunks:
                cursor.execute("""
                    INSERT INTO chunks (
                        chunk_id, parent_chunk_id, chunk_type,
                        text_verbatim, lexical_filters,
                        span_id, turn_id, block_id,
                        created_at, token_count, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    chunk.chunk_id,
                    chunk.parent_chunk_id,
                    chunk.chunk_type,
                    chunk.text_verbatim,
                    json.dumps(chunk.lexical_filters),  # Store as JSON array
                    chunk.span_id,
                    chunk.turn_id,
                    None,  
                    chunk.metadata.get('created_at', 'now'),
                    chunk.token_count,
                    json.dumps(chunk.metadata)
                ))
            
            self.conn.commit()
            logger.info(f"Saved {len(chunks)} chunks to database")
        
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Failed to save chunks: {e}")
            raise
    
    def get_chunks_by_turn(self, turn_id: str) -> List[Chunk]:
        """
        Retrieve all chunks for a specific turn.
        
        Args:
            turn_id: Turn identifier
            
        Returns:
            List of Chunk objects
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT chunk_id, parent_chunk_id, chunk_type,
                   text_verbatim, lexical_filters,
                   span_id, turn_id, block_id,
                   created_at, token_count, metadata
            FROM chunks
            WHERE turn_id = ?
            ORDER BY CASE chunk_type 
                WHEN 'sentence' THEN 1 
                WHEN 'paragraph' THEN 2 
                ELSE 3 END
        """, (turn_id,))
        
        chunks = []
        for row in cursor.fetchall():
            chunks.append(self._row_to_chunk(row))
        
        return chunks
    
    def get_chunks_by_span(self, span_id: str, chunk_type: Optional[str] = None) -> List[Chunk]:
        """
        Retrieve chunks for a conversation span.
        
        Args:
            span_id: Span identifier
            chunk_type: Optional filter by type ('sentence', 'paragraph', 'bridge_block')
            
        Returns:
            List of Chunk objects
        """
        cursor = self.conn.cursor()
        
        if chunk_type:
            cursor.execute("""
                SELECT chunk_id, parent_chunk_id, chunk_type,
                       text_verbatim, lexical_filters,
                       span_id, turn_id, block_id,
                       created_at, token_count, metadata
                FROM chunks
                WHERE span_id = ? AND chunk_type = ?
                ORDER BY created_at
            """, (span_id, chunk_type))
        else:
            cursor.execute("""
                SELECT chunk_id, parent_chunk_id, chunk_type,
                       text_verbatim, lexical_filters,
                       span_id, turn_id, block_id,
                       created_at, token_count, metadata
                FROM chunks
                WHERE span_id = ?
                ORDER BY created_at
            """, (span_id,))
        
        chunks = []
        for row in cursor.fetchall():
            chunks.append(self._row_to_chunk(row))
        
        return chunks
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[Chunk]:
        """
        Retrieve a specific chunk by ID.
        
        Args:
            chunk_id: Chunk identifier
            
        Returns:
            Chunk object or None if not found
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT chunk_id, parent_chunk_id, chunk_type,
                   text_verbatim, lexical_filters,
                   span_id, turn_id, block_id,
                   created_at, token_count, metadata
            FROM chunks
            WHERE chunk_id = ?
        """, (chunk_id,))
        
        row = cursor.fetchone()
        return self._row_to_chunk(row) if row else None
    
    def search_chunks_lexical(self, keywords: List[str], limit: int = 50) -> List[Chunk]:
        """
        Search chunks using FTS5 lexical (keyword) matching.
        
        Args:
            keywords: List of keywords to search for
            limit: Maximum number of results
            
        Returns:
            List of matching Chunk objects
        """
        # Build FTS5 query (OR logic for multiple keywords)
        fts_query = ' OR '.join(keywords)
        
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT c.chunk_id, c.parent_chunk_id, c.chunk_type,
                   c.text_verbatim, c.lexical_filters,
                   c.span_id, c.turn_id, c.block_id,
                   c.created_at, c.token_count, c.metadata,
                   fts.rank
            FROM chunks c
            JOIN chunks_fts fts ON c.rowid = fts.rowid
            WHERE chunks_fts MATCH ?
            ORDER BY fts.rank
            LIMIT ?
        """, (fts_query, limit))
        
        chunks = []
        for row in cursor.fetchall():
            chunk = self._row_to_chunk(row[:-1])  # Exclude rank from row
            chunk.metadata['lexical_rank'] = row[-1]  # Store rank in metadata
            chunks.append(chunk)
        
        return chunks
    
    def update_chunk_block_id(self, chunk_id: str, block_id: str) -> None:
        """
        Link a chunk to a bridge block 
        
        Args:
            chunk_id: Chunk identifier
            block_id: Bridge block identifier
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            UPDATE chunks
            SET block_id = ?
            WHERE chunk_id = ?
        """, (block_id, chunk_id))
        
        self.conn.commit()
    
    def get_child_chunks(self, parent_chunk_id: str) -> List[Chunk]:
        """
        Get all child chunks of a parent (e.g., sentences in a paragraph).
        
        Args:
            parent_chunk_id: Parent chunk identifier
            
        Returns:
            List of child Chunk objects
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT chunk_id, parent_chunk_id, chunk_type,
                   text_verbatim, lexical_filters,
                   span_id, turn_id, block_id,
                   created_at, token_count, metadata
            FROM chunks
            WHERE parent_chunk_id = ?
            ORDER BY created_at
        """, (parent_chunk_id,))
        
        chunks = []
        for row in cursor.fetchall():
            chunks.append(self._row_to_chunk(row))
        
        return chunks
    
    def _row_to_chunk(self, row) -> Chunk:
        """Convert database row to Chunk object."""
        return Chunk(
            chunk_id=row[0],
            parent_chunk_id=row[1],
            chunk_type=row[2],
            text_verbatim=row[3],
            lexical_filters=json.loads(row[4]) if row[4] else [],
            span_id=row[5],
            turn_id=row[6],
            token_count=row[9],
            metadata=json.loads(row[10]) if row[10] else {}
        )
    
    def get_chunk_count(self, chunk_type: Optional[str] = None) -> int:
        """
        Get total count of chunks.
        
        Args:
            chunk_type: Optional filter by type
            
        Returns:
            Total number of chunks
        """
        cursor = self.conn.cursor()
        
        if chunk_type:
            cursor.execute("SELECT COUNT(*) FROM chunks WHERE chunk_type = ?", (chunk_type,))
        else:
            cursor.execute("SELECT COUNT(*) FROM chunks")
        
        return cursor.fetchone()[0]
