"""
Long-Horizon Memory System - Storage Layer

This module provides persistent storage for the memory system using SQLite.
It handles:
- Day nodes with temporal linking
- Task state persistence
- Keywords with time ranges
- Summaries and affect tracking
- Metadata staging for synthesis

"""

import sqlite3
import json
import logging
import time
from datetime import datetime
from typing import List, Dict, Optional, Any, Tuple, Callable
from pathlib import Path
from hmlr.core.model_config import model_config

logger = logging.getLogger(__name__)

# Import plan models
from .models import UserPlan, PlanItem, PlanFeedback, PlanModification
import os

from .models import (
    DayNode,
    TaskState,
    Keyword,
    Summary,
    Affect,
    DaySynthesis,
    ConversationTurn,
    TaskStatus,
    TaskType,
    create_day_id,
    Span
)
from .persistence.schema import initialize_database
from .persistence.dossier_store import DossierStore
from .persistence.ledger_store import LedgerStore
from .id_generator import (
    generate_turn_id,
    generate_session_id,
    generate_keyword_id,
    generate_summary_id,
    generate_affect_id,
    generate_task_id,
    parse_id,
    get_id_type
)


class Storage:
    """
    SQLite-based storage layer for the memory system.
    Provides CRUD operations for all memory components.
    """
    
    def __init__(self, db_path: str = None):
        """
        Initialize storage with SQLite database.
        
        Args:
            db_path: Path to SQLite database file. 
                     If None, checks HMLR_DB_PATH env var, 
                     else defaults to ~/.hmlr/cognitive_lattice_memory.db
        """
        if db_path is None:
            # 1. Check environment variable
            env_path = os.getenv("HMLR_DB_PATH")
            if env_path:
                db_path = env_path
            else:
                # 2. Default to user home directory (XDG style)
                # This ensures persistence survives package updates/reinstalls
                home_dir = Path.home() / ".hmlr"
                home_dir.mkdir(parents=True, exist_ok=True)
                db_path = str(home_dir / "cognitive_lattice_memory.db")
        
        self.db_path = db_path
        
        # Ensure directory exists if explicit path provided
        db_dir = os.path.dirname(self.db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)

        self.conn = None
        self._initialize_database()
    
    def _initialize_database(self):
        """Create database and tables if they don't exist with WAL and busy timeout."""
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=5.0)
        self.conn.row_factory = sqlite3.Row  # Enable dict-like access

        # Enable WAL for better concurrent access and set a busy timeout
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA busy_timeout=5000;")  # 5s base busy timeout

        # Delegate schema creation to persistence/schema.py
        initialize_database(self.conn)
        logger.info(f"Storage initialized: {self.db_path} (WAL enabled, busy_timeout=5000ms)")

    def _with_retry(self, fn: Callable[[], Any], retries: int = 3, base_delay: float = 0.1) -> Any:
        """Retry helper for transient SQLite busy/lock errors."""
        attempt = 0
        while True:
            try:
                return fn()
            except sqlite3.OperationalError as e:
                msg = str(e).lower()
                if "locked" not in msg and "busy" not in msg:
                    raise
                attempt += 1
                if attempt > retries:
                    raise
                sleep_for = base_delay * (2 ** (attempt - 1))
                time.sleep(sleep_for)
            except Exception:
                raise

    # =========================================================================
    # DAY NODE OPERATIONS
    # =========================================================================
    
    def create_day(self, day_id: str = None) -> DayNode:
        """Create a new day node"""
        if day_id is None:
            day_id = create_day_id()
        
        cursor = self.conn.cursor()

        # Check if day already exists
        cursor.execute("SELECT day_id FROM days WHERE day_id = ?", (day_id,))
        if cursor.fetchone():
            return self.get_day(day_id)

        # Create new day
        created_at = datetime.now()

        def _insert_day():
            cursor.execute(
                """
                INSERT INTO days (day_id, created_at)
                VALUES (?, ?)
                """,
                (day_id, created_at),
            )

        self._with_retry(_insert_day)

        # Link to previous day if it exists
        prev_day_id = self._get_previous_day_id(day_id)
        if prev_day_id:
            def _link_prev():
                cursor.execute(
                    """
                    UPDATE days SET prev_day = ? WHERE day_id = ?
                    """,
                    (prev_day_id, day_id),
                )
                cursor.execute(
                    """
                    UPDATE days SET next_day = ? WHERE day_id = ?
                    """,
                    (day_id, prev_day_id),
                )

            self._with_retry(_link_prev)

        self.conn.commit()
        
        return DayNode(
            day_id=day_id,
            created_at=created_at,
            prev_day=prev_day_id
        )
    
    def get_day(self, day_id: str) -> Optional[DayNode]:
        """Get a day node by ID"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT day_id, created_at, prev_day, next_day
            FROM days WHERE day_id = ?
        """, (day_id,))
        
        row = cursor.fetchone()
        if not row:
            return None
        
        # Get session IDs
        cursor.execute("""
            SELECT session_id FROM day_sessions WHERE day_id = ?
        """, (day_id,))
        session_ids = [r['session_id'] for r in cursor.fetchall()]
        
        return DayNode(
            day_id=row['day_id'],
            created_at=datetime.fromisoformat(row['created_at']),
            prev_day=row['prev_day'],
            next_day=row['next_day'],
            session_ids=session_ids,
            keywords=[],
            summaries=[],
            affect_patterns=[],
            synthesis=None
        )
    
    def add_session_to_day(self, day_id: str, session_id: str) -> None:
        """Associate a session with a day"""
        cursor = self.conn.cursor()

        # Ensure day exists
        cursor.execute("SELECT day_id FROM days WHERE day_id = ?", (day_id,))
        if not cursor.fetchone():
            self.create_day(day_id)

        # Add session (ignore if duplicate)
        def _insert_session():
            cursor.execute(
                """
                INSERT INTO day_sessions (day_id, session_id)
                VALUES (?, ?)
                """,
                (day_id, session_id),
            )

        try:
            self._with_retry(_insert_session)
            self.conn.commit()
        except sqlite3.IntegrityError as e:
            if "UNIQUE constraint" in str(e) or "unique constraint" in str(e):
                return
            logger.error(f"Database error adding session {session_id} to day {day_id}: {e}")
            raise
    
    def _get_previous_day_id(self, day_id: str) -> Optional[str]:
        """Find the most recent day before this one"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT day_id FROM days
            WHERE day_id < ?
            ORDER BY day_id DESC
            LIMIT 1
        """, (day_id,))
        
        row = cursor.fetchone()
        return row['day_id'] if row else None
    
    # =========================================================================
    # METADATA STAGING OPERATIONS (Pre-Synthesis)
    # =========================================================================
    
    def stage_turn_metadata(self, turn: ConversationTurn) -> None:
        """
        Stage turn metadata for later synthesis.

        """
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO metadata_staging
            (turn_id, turn_sequence, session_id, day_id, timestamp, 
             user_message, assistant_response, keywords, user_summary, 
             assistant_summary, detected_affect, active_topics, retrieval_sources,
             summary_id, keyword_ids, affect_ids, task_created_id, 
             task_updated_ids, loaded_turn_ids, span_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            turn.turn_id,                           # NEW: String ID like t_...
            turn.turn_sequence,                     # NEW: Sequential number
            turn.session_id,                        # String ID like sess_...
            turn.day_id,
            turn.timestamp,
            turn.user_message,
            turn.assistant_response,
            json.dumps(turn.keywords),
            turn.user_summary,
            turn.assistant_summary,
            json.dumps(turn.detected_affect),
            json.dumps(turn.active_topics),
            json.dumps(turn.retrieval_sources),
            turn.summary_id,                        # NEW: s_t_...
            json.dumps(turn.keyword_ids),           # NEW: [k1_..., k2_...]
            json.dumps(turn.affect_ids),            # NEW: [a_t_...]
            turn.task_created_id,                   # NEW: tsk_...
            json.dumps(turn.task_updated_ids),      # NEW: [tsk_...]
            json.dumps(turn.loaded_turn_ids),        # NEW: [t_..., t_...]
            turn.span_id if hasattr(turn, 'span_id') else None  # HMLR v1: span link
        ))
        self.conn.commit()
    
    def get_staged_turns(self, day_id: str) -> List[ConversationTurn]:
        """
        Get all staged turns for a day.
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM metadata_staging
            WHERE day_id = ?
            ORDER BY turn_sequence
        """, (day_id,))
        
        return self._rows_to_turns(cursor.fetchall())

    def get_session_history(self, session_id: str, limit: int = 25, offset: int = 0) -> List[ConversationTurn]:
        """
        Retrieve turns for a specific session with pagination.
        Used by VirtualSlidingWindow for stateless operation.
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM metadata_staging
            WHERE session_id = ?
            ORDER BY turn_sequence DESC
            LIMIT ? OFFSET ?
        """, (session_id, limit, offset))
        
        # Reverse to get chronological order (since we queried DESC)
        turns = self._rows_to_turns(cursor.fetchall())
        turns.reverse()
        return turns

    def get_recent_turns(self, day_id: Optional[str] = None, limit: int = 20) -> List[ConversationTurn]:
        """
        Get most recent turns across all sessions or for a specific day.
        """
        cursor = self.conn.cursor()
        if day_id:
            cursor.execute("""
                SELECT * FROM metadata_staging
                WHERE day_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (day_id, limit))
        else:
            cursor.execute("""
                SELECT * FROM metadata_staging
                ORDER BY timestamp DESC
                LIMIT ?
            """, (limit,))
            
        return self._rows_to_turns(cursor.fetchall())

    def _rows_to_turns(self, rows: List[sqlite3.Row]) -> List[ConversationTurn]:
        """Helper to convert DB rows to ConversationTurn objects"""
        turns = []
        for row in rows:
            turns.append(ConversationTurn(
                turn_id=row['turn_id'],
                turn_sequence=row['turn_sequence'],
                session_id=row['session_id'],
                day_id=row['day_id'],
                timestamp=datetime.fromisoformat(row['timestamp']),
                user_message=row['user_message'],
                assistant_response=row['assistant_response'],
                keywords=json.loads(row['keywords']) if row['keywords'] else [],
                detected_affect=json.loads(row['detected_affect']) if row['detected_affect'] else [],
                user_summary=row['user_summary'],
                assistant_summary=row['assistant_summary'],
                active_topics=json.loads(row['active_topics']) if row['active_topics'] else [],
                retrieval_sources=json.loads(row['retrieval_sources']) if row['retrieval_sources'] else [],
                summary_id=row['summary_id'],
                keyword_ids=json.loads(row['keyword_ids']) if row['keyword_ids'] else [],
                affect_ids=json.loads(row['affect_ids']) if row['affect_ids'] else [],
                task_created_id=row['task_created_id'],
                task_updated_ids=json.loads(row['task_updated_ids']) if row['task_updated_ids'] else [],
                loaded_turn_ids=json.loads(row['loaded_turn_ids']) if row['loaded_turn_ids'] else [],
                span_id=row['span_id']
            ))
        return turns
    
    def get_turn_by_id(self, turn_id: str) -> Optional[ConversationTurn]:
        """
        Fetch a single conversation turn by its ID.
        
        Args:
            turn_id: The unique turn identifier (t_...)
            
        Returns:
            ConversationTurn object or None if not found
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM metadata_staging WHERE turn_id = ?", (turn_id,))
        row = cursor.fetchone()
        
        if not row:
            return None
            
        return ConversationTurn(
            turn_id=row['turn_id'],
            turn_sequence=row['turn_sequence'],
            session_id=row['session_id'],
            day_id=row['day_id'],
            timestamp=datetime.fromisoformat(row['timestamp']) if isinstance(row['timestamp'], str) else row['timestamp'],
            user_message=row['user_message'],
            assistant_response=row['assistant_response'],
            keywords=json.loads(row['keywords']) if row['keywords'] else [],
            detected_affect=json.loads(row['detected_affect']) if row['detected_affect'] else [],
            user_summary=row['user_summary'],
            assistant_summary=row['assistant_summary'],
            active_topics=json.loads(row['active_topics']) if row['active_topics'] else [],
            retrieval_sources=json.loads(row['retrieval_sources']) if row['retrieval_sources'] else [],
            summary_id=row['summary_id'],
            keyword_ids=json.loads(row['keyword_ids']) if row['keyword_ids'] else [],
            affect_ids=json.loads(row['affect_ids']) if row['affect_ids'] else [],
            task_created_id=row['task_created_id'],
            task_updated_ids=json.loads(row['task_updated_ids']) if row['task_updated_ids'] else [],
            loaded_turn_ids=json.loads(row['loaded_turn_ids']) if row['loaded_turn_ids'] else [],
            span_id=row['span_id'] if 'span_id' in row.keys() else None
        )

    def get_facts_by_turn_id(self, turn_id: str) -> List[Dict[str, Any]]:
        """
        Fetch all facts originating from a specific turn.
        
        Args:
            turn_id: The turn identifier (t_...)
            
        Returns:
            List of fact dictionaries
        """
        cursor = self.conn.cursor()

        # 1. Get from fact_store
        cursor.execute("SELECT * FROM fact_store WHERE source_turn_id = ?", (turn_id,))
        facts = [dict(row) for row in cursor.fetchall()]

        # 2. Get from dossier_facts (if not already found)
        cursor.execute("SELECT * FROM dossier_facts WHERE source_turn_id = ?", (turn_id,))
        dossier_facts = [dict(row) for row in cursor.fetchall()]

        all_facts = facts + dossier_facts
        return all_facts
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def get_stats(self) -> Dict[str, Any]:
        cursor = self.conn.cursor()
        
        stats = {}
        
        cursor.execute("SELECT COUNT(*) as count FROM days")
        stats['total_days'] = cursor.fetchone()['count']
        
        cursor.execute("""
            SELECT status, COUNT(*) as count
            FROM tasks
            GROUP BY status
        """)
        stats['tasks_by_status'] = {row['status']: row['count'] for row in cursor.fetchall()}
        
        cursor.execute("SELECT COUNT(*) as count FROM metadata_staging")
        stats['staged_turns'] = cursor.fetchone()['count']
        
        cursor.execute("SELECT COUNT(*) as count FROM embeddings")
        stats['total_embeddings'] = cursor.fetchone()['count']
        
        return stats
    
    # ============================================================================
    # VECTOR EMBEDDINGS METHODS
    # ============================================================================
    
    def save_embedding(self, embedding_id: str, turn_id: str, chunk_index: int,
                      embedding_bytes: bytes, text_content: str,
                      dimension: int = None, model_name: str = None):
        """
        Save a vector embedding to the database.
        
        Args:
            embedding_id: Unique embedding identifier
            turn_id: Associated turn ID
            chunk_index: Chunk number (0 for first chunk)
            embedding_bytes: Serialized numpy array
            text_content: The text that was embedded
            dimension: Embedding dimension (defaults to centralized config)
            model_name: Model used to generate embedding (defaults to centralized config)
        """
        # Use centralized config for defaults
        dimension = dimension or model_config.EMBEDDING_DIMENSION
        model_name = model_name or model_config.EMBEDDING_MODEL_NAME
        cursor = self.conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO embeddings
            (embedding_id, turn_id, chunk_index, embedding, text_content, dimension, model_name)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (embedding_id, turn_id, chunk_index, embedding_bytes, text_content, dimension, model_name))
        
        self.conn.commit()
    
    def get_all_embeddings(self) -> List[tuple]:
        """
        Retrieve all embeddings from database.
        
        Returns:
            List of (embedding_id, embedding_bytes, text_content, turn_id) tuples
        """
        cursor = self.conn.cursor()
        
        rows = cursor.execute("""
            SELECT embedding_id, embedding, text_content, turn_id
            FROM embeddings
        """).fetchall()
        
        return [(row[0], row[1], row[2], row[3]) for row in rows]
    
    def get_turn_embeddings(self, turn_id: str) -> List[tuple]:
        """
        Get all embedding chunks for a specific turn.
        
        Args:
            turn_id: Turn identifier
            
        Returns:
            List of (embedding_id, chunk_index, embedding_bytes, text_content) tuples
        """
        cursor = self.conn.cursor()
        
        rows = cursor.execute("""
            SELECT embedding_id, chunk_index, embedding, text_content
            FROM embeddings
            WHERE turn_id = ?
            ORDER BY chunk_index
        """, (turn_id,)).fetchall()
        
        return [(row[0], row[1], row[2], row[3]) for row in rows]
    
    def delete_turn_embeddings(self, turn_id: str):
        """
        Delete all embeddings for a turn.
        
        Args:
            turn_id: Turn identifier
        """
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM embeddings WHERE turn_id = ?", (turn_id,))
        self.conn.commit()
    
    def get_embedding_count(self) -> int:
        """
        Get total number of embeddings stored.
        
        Returns:
            Count of embeddings
        """
        cursor = self.conn.cursor()
        result = cursor.execute("SELECT COUNT(*) FROM embeddings").fetchone()
        return result[0] if result else 0
    

    # ============================================================================
    # Fact Store & Daily Ledger Query Methods
    # ========================================================================
    
    def query_fact_store(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Query fact_store for exact keyword match 
        
        Args:
            key: The exact key to search for (e.g., "HMLR", "API_KEY")
        
        Returns:
            Fact dictionary if found, None otherwise
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT fact_id, key, value, category, 
                   source_span_id, source_chunk_id, source_paragraph_id,
                   source_block_id, source_turn_id, evidence_snippet, created_at
            FROM fact_store
            WHERE key = ?
            ORDER BY created_at DESC
            LIMIT 1
        """, (key,))
        
        row = cursor.fetchone()
        if not row:
            return None
        
        return {
            'fact_id': row[0],
            'key': row[1],
            'value': row[2],
            'category': row[3],
            'source_span_id': row[4],
            'source_chunk_id': row[5],
            'source_paragraph_id': row[6],
            'source_block_id': row[7],
            'source_turn_id': row[8],
            'evidence_snippet': row[9],
            'created_at': row[10]
        }
    
    def get_facts_for_block(self, block_id: str) -> List[Dict[str, Any]]:
        """
        Get ALL facts associated with a specific Bridge Block.
        
        Returns facts ordered by most recent first. This allows the LLM
        to see all facts/secrets from this topic in its prompt.
        
        Args:
            block_id: The Bridge Block ID
        
        Returns:
            List of fact dictionaries (most recent first)
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT fact_id, key, value, category, 
                   source_span_id, source_chunk_id, source_paragraph_id,
                   source_block_id, source_turn_id, evidence_snippet, created_at
            FROM fact_store
            WHERE source_block_id = ?
            ORDER BY created_at DESC
        """, (block_id,))
        
        facts = []
        for row in cursor.fetchall():
            facts.append({
                'fact_id': row[0],
                'key': row[1],
                'value': row[2],
                'category': row[3],
                'source_span_id': row[4],
                'source_chunk_id': row[5],
                'source_paragraph_id': row[6],
                'source_block_id': row[7],
                'source_turn_id': row[8],
                'evidence_snippet': row[9],
                'created_at': row[10]
            })
        
        return facts
    
    def update_facts_block_id(self, turn_id: str, block_id: str) -> int:
        return LedgerStore.link_facts_to_block(self.conn, turn_id, block_id)
    
    def get_active_bridge_blocks(self) -> List[Dict[str, Any]]:
        return LedgerStore.get_active_bridge_blocks(self.conn)
    
    # ========================================================================
    # BRIDGE BLOCK STORAGE METHODS
    # ========================================================================

    def get_daily_ledger_metadata(self, day_id: str) -> List[Dict[str, Any]]:
        return LedgerStore.get_daily_ledger_metadata(self.conn, day_id)

    def get_bridge_block_full(self, block_id: str) -> Optional[Dict[str, Any]]:
        return LedgerStore.get_bridge_block_full(self.conn, block_id)

    def save_to_gardened_memory(self, chunks: List[Dict[str, Any]], block_id: str, global_tags: List[str]) -> int:
        return LedgerStore.save_to_gardened_memory(self.conn, chunks, block_id, global_tags)
    
    def append_turn_to_block(self, block_id: str, turn: Dict[str, Any], **kwargs) -> bool:
        return LedgerStore.append_turn_to_block(self.conn, block_id, turn, **kwargs)

    def update_bridge_block_status(self, block_id: str, new_status: str, exit_reason: Optional[str] = None) -> bool:
        return LedgerStore.update_bridge_block_status(self.conn, block_id, new_status, exit_reason)

    def update_last_active_flag(self, block_id: str) -> bool:
        return LedgerStore.update_last_active_flag(self.conn, block_id)

    def generate_block_summary(self, block_id: str) -> Optional[str]:
        return LedgerStore.generate_block_summary(self.conn, block_id)

    def create_new_bridge_block(
        self,
        day_id: str,
        topic_label: str,
        keywords: List[str],
        span_id: Optional[str] = None,
        **kwargs
    ) -> Optional[str]:
        return LedgerStore.create_new_bridge_block(self.conn, day_id, topic_label, keywords, span_id, **kwargs)

    def update_bridge_block_metadata(self, block_id: str, metadata: Dict[str, Any]) -> bool:
        return LedgerStore.update_bridge_block_metadata(self.conn, block_id, metadata)

    # =========================================================================
    # DOSSIER OPERATIONS
    # =========================================================================
    
    def create_dossier(self, dossier_id: str, title: str, summary: str = "", search_summary: str = "") -> bool:
        return DossierStore.create_dossier(self.conn, dossier_id, title, summary, search_summary)
    
    def get_dossier(self, dossier_id: str) -> Optional[Dict[str, Any]]:
        return DossierStore.get_dossier(self.conn, dossier_id)
    
    def get_all_dossiers(self, status: str = 'active') -> List[Dict[str, Any]]:
        return DossierStore.get_all_dossiers(self.conn, status)
    
    def add_fact_to_dossier(
        self,
        dossier_id: str,
        fact_id: str,
        fact_text: str,
        source_block_id: str,
        source_turn_id: str = None,
        fact_type: str = None,
        confidence: float = 1.0
    ) -> bool:
        return DossierStore.add_fact_to_dossier(
            self.conn, dossier_id, fact_id, fact_text, source_block_id,
            source_turn_id, fact_type, confidence
        )
    
    def get_dossier_facts(self, dossier_id: str) -> List[Dict[str, Any]]:
        return DossierStore.get_dossier_facts(self.conn, dossier_id)
    
    def update_dossier_summary(self, dossier_id: str, new_summary: str) -> bool:
        return DossierStore.update_dossier_summary(self.conn, dossier_id, new_summary)
    
    def add_provenance_entry(
        self,
        dossier_id: str,
        operation: str,
        provenance_id: str,
        source_block_id: str = None,
        source_turn_id: str = None,
        details: str = None
    ) -> bool:
        return DossierStore.add_provenance_entry(
            self.conn, dossier_id, operation, provenance_id,
            source_block_id, source_turn_id, details
        )
    
    def get_dossier_history(self, dossier_id: str) -> List[Dict[str, Any]]:
        return DossierStore.get_dossier_history(self.conn, dossier_id)
    
    # =========================================================================
    # BLOCK METADATA OPERATIONS 
    # =========================================================================
    
    def save_block_metadata(self, block_id: str, global_tags: List[str], 
                           section_rules: List[Dict]) -> None:
        return LedgerStore.save_block_metadata(self.conn, block_id, global_tags, section_rules)
    
    def get_block_metadata(self, block_id: str) -> Dict[str, Any]:
        return LedgerStore.get_block_metadata(self.conn, block_id)

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            logger.info("Storage connection closed")
    
    def __enter__(self):
        """Context manager support"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager support"""
        self.close()
