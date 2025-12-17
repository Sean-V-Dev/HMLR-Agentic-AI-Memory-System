"""
Long-Horizon Memory System - Storage Layer

This module provides persistent storage for the memory system using SQLite.
It handles:
- Day nodes with temporal linking
- Task state persistence
- Keywords with time ranges
- Summaries and affect tracking
- Metadata staging for synthesis

Author: CognitiveLattice Team
Created: 2025-10-10
"""

import sqlite3
import json
import logging
from datetime import datetime
from typing import List, Dict, Optional, Any, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

# Import plan models
from .models import UserPlan, PlanItem, PlanFeedback, PlanModification
import os

try:
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
        Span,
        HierarchicalSummary
    )
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
except ImportError:
    # For standalone testing
    from models import (
        DayNode,
        TaskState,
        Keyword,
        Summary,
        Affect,
        DaySynthesis,
        ConversationTurn,
        TaskStatus,
        TaskType,
        create_day_id
    )
    from id_generator import (
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
            db_path: Path to SQLite database file. If None, uses default in memory/ folder
        """
        if db_path is None:
            # Default to memory folder in project root
            memory_dir = Path(__file__).parent
            db_path = str(memory_dir / "cognitive_lattice_memory.db")
        
        self.db_path = db_path
        self.conn = None
        self._initialize_database()
    
    def _initialize_database(self):
        """Create database and tables if they don't exist"""
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row  # Enable dict-like access
        
        cursor = self.conn.cursor()
        
        # === DAYS TABLE ===
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS days (
                day_id TEXT PRIMARY KEY,
                created_at TIMESTAMP NOT NULL,
                prev_day TEXT,
                next_day TEXT
            )
        """)
        
        # === DAY SESSIONS TABLE ===
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS day_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                day_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                FOREIGN KEY (day_id) REFERENCES days(day_id),
                UNIQUE(day_id, session_id)
            )
        """)
        
        # === DAY KEYWORDS TABLE ===
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS day_keywords (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                day_id TEXT NOT NULL,
                keyword TEXT NOT NULL,
                first_mentioned TIMESTAMP NOT NULL,
                last_mentioned TIMESTAMP NOT NULL,
                frequency INTEGER DEFAULT 1,
                turn_ids TEXT,
                FOREIGN KEY (day_id) REFERENCES days(day_id)
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_keywords ON day_keywords(keyword)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_day_keywords ON day_keywords(day_id, keyword)")
        
        # === DAY SUMMARIES TABLE ===
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS day_summaries (
                turn_id INTEGER NOT NULL,
                day_id TEXT NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                user_query_summary TEXT,
                assistant_response_summary TEXT,
                keywords_this_turn TEXT,
                PRIMARY KEY (day_id, turn_id),
                FOREIGN KEY (day_id) REFERENCES days(day_id)
            )
        """)
        
        # === DAY AFFECT TABLE ===
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS day_affect (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                day_id TEXT NOT NULL,
                affect_label TEXT NOT NULL,
                first_detected TIMESTAMP NOT NULL,
                last_detected TIMESTAMP NOT NULL,
                intensity REAL DEFAULT 0.5,
                associated_topics TEXT,
                turn_ids TEXT,
                FOREIGN KEY (day_id) REFERENCES days(day_id)
            )
        """)
        
        # === TASKS TABLE ===
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                task_id TEXT PRIMARY KEY,
                task_type TEXT NOT NULL,
                status TEXT NOT NULL,
                created_date TEXT NOT NULL,
                created_at TIMESTAMP NOT NULL,
                last_updated TIMESTAMP NOT NULL,
                completed_at TIMESTAMP,
                task_title TEXT,
                original_query TEXT,
                total_steps INTEGER DEFAULT 0,
                completed_steps INTEGER DEFAULT 0,
                skipped_steps INTEGER DEFAULT 0,
                tags TEXT,
                notes TEXT,
                state_json TEXT
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_task_status ON tasks(status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_task_date ON tasks(created_date)")
        
        # === TASK DAYS TABLE (task-day associations) ===
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS task_days (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT NOT NULL,
                day_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                FOREIGN KEY (task_id) REFERENCES tasks(task_id),
                FOREIGN KEY (day_id) REFERENCES days(day_id)
            )
        """)
        
        # === METADATA STAGING TABLE (pre-synthesis) ===
        # Updated for Phase B: String IDs with lineage tracking
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS metadata_staging (
                turn_id TEXT NOT NULL,
                turn_sequence INTEGER NOT NULL,
                session_id TEXT NOT NULL,
                day_id TEXT NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                user_message TEXT,
                assistant_response TEXT,
                keywords TEXT,
                user_summary TEXT,
                assistant_summary TEXT,
                detected_affect TEXT,
                active_topics TEXT,
                retrieval_sources TEXT,
                summary_id TEXT,
                keyword_ids TEXT,
                affect_ids TEXT,
                task_created_id TEXT,
                task_updated_ids TEXT,
                loaded_turn_ids TEXT,
                span_id TEXT,
                PRIMARY KEY (turn_id),
                UNIQUE (session_id, turn_sequence)
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_turn_day ON metadata_staging(day_id, timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_turn_session ON metadata_staging(session_id, turn_sequence)")
        
        # === NEW: SUMMARIES TABLE (with lineage) ===
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS summaries (
                summary_id TEXT PRIMARY KEY,
                source_turn_id TEXT NOT NULL,
                day_id TEXT NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                user_query_summary TEXT NOT NULL,
                assistant_response_summary TEXT NOT NULL,
                keywords_this_turn TEXT,
                derived_from TEXT NOT NULL,
                derived_by TEXT NOT NULL,
                extraction_method TEXT NOT NULL,
                FOREIGN KEY (source_turn_id) REFERENCES metadata_staging(turn_id)
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_summary_turn ON summaries(source_turn_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_summary_day ON summaries(day_id, timestamp)")
        
        # === NEW: KEYWORDS TABLE (with lineage) ===
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS keywords (
                keyword_id TEXT PRIMARY KEY,
                keyword TEXT NOT NULL,
                source_turn_id TEXT NOT NULL,
                day_id TEXT NOT NULL,
                first_mentioned TIMESTAMP NOT NULL,
                last_mentioned TIMESTAMP NOT NULL,
                frequency INTEGER DEFAULT 1,
                derived_from TEXT NOT NULL,
                derived_by TEXT NOT NULL,
                confidence REAL DEFAULT 1.0,
                FOREIGN KEY (source_turn_id) REFERENCES metadata_staging(turn_id)
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_keyword_word ON keywords(keyword)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_keyword_day ON keywords(day_id, keyword)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_keyword_turn ON keywords(source_turn_id)")
        
        # === NEW: AFFECT TABLE (with lineage) ===
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS affect (
                affect_id TEXT PRIMARY KEY,
                affect_label TEXT NOT NULL,
                source_turn_id TEXT NOT NULL,
                day_id TEXT NOT NULL,
                first_detected TIMESTAMP NOT NULL,
                last_detected TIMESTAMP NOT NULL,
                intensity REAL DEFAULT 0.5,
                confidence REAL DEFAULT 0.8,
                associated_topics TEXT,
                derived_from TEXT NOT NULL,
                derived_by TEXT NOT NULL,
                detection_method TEXT NOT NULL,
                trigger_context TEXT,
                FOREIGN KEY (source_turn_id) REFERENCES metadata_staging(turn_id)
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_affect_label ON affect(affect_label, day_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_affect_turn ON affect(source_turn_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_affect_day ON affect(day_id, first_detected)")
        
        # === HMLR v1: SPANS TABLE ===
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS spans (
                span_id TEXT PRIMARY KEY,
                day_id TEXT NOT NULL,
                created_at TIMESTAMP NOT NULL,
                last_active_at TIMESTAMP NOT NULL,
                topic_label TEXT,
                is_active BOOLEAN DEFAULT 1,
                summary_id TEXT,
                parent_span_id TEXT,
                turn_ids_json TEXT
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_spans_day ON spans(day_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_spans_active ON spans(is_active)")

        # === HMLR v1: HIERARCHICAL SUMMARIES TABLE ===
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS hierarchical_summaries (
                summary_id TEXT PRIMARY KEY,
                created_at TIMESTAMP NOT NULL,
                content TEXT NOT NULL,
                embedding_json TEXT,
                level INTEGER DEFAULT 0,
                topics_json TEXT,
                span_ids_json TEXT,
                child_summary_ids_json TEXT
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_hsum_level ON hierarchical_summaries(level)")

        # === PHASE 11.1: DAILY LEDGER TABLE (Bridge Blocks) ===
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS daily_ledger (
                block_id TEXT PRIMARY KEY,
                prev_block_id TEXT,
                span_id TEXT,
                content_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT,
                status TEXT DEFAULT 'PAUSED',
                exit_reason TEXT,
                embedding_status TEXT DEFAULT 'PENDING',
                embedded_at TEXT
            )
        """)
        
        # Migration: Add updated_at column if it doesn't exist
        try:
            cursor.execute("SELECT updated_at FROM daily_ledger LIMIT 1")
        except:
            logger.info("Migrating daily_ledger: Adding updated_at column")
            cursor.execute("ALTER TABLE daily_ledger ADD COLUMN updated_at TEXT")
        
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_ledger_status ON daily_ledger(status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_ledger_date ON daily_ledger(created_at)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_ledger_span ON daily_ledger(span_id)")

        # === PHASE 11.1: FACT STORE TABLE ===
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS fact_store (
                fact_id INTEGER PRIMARY KEY AUTOINCREMENT,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                category TEXT,
                source_span_id TEXT,
                source_chunk_id TEXT,
                source_paragraph_id TEXT,
                source_block_id TEXT,
                evidence_snippet TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (source_span_id) REFERENCES spans(span_id)
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_fact_key ON fact_store(key)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_fact_span ON fact_store(source_span_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_fact_block ON fact_store(source_block_id)")

        # === DAY SYNTHESIS TABLE ===
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS day_synthesis (
                day_id TEXT PRIMARY KEY,
                created_at TIMESTAMP NOT NULL,
                emotional_arc TEXT,
                key_patterns TEXT,
                topic_affect_mapping TEXT,
                behavioral_notes TEXT,
                narrative_summary TEXT,
                notable_moments TEXT,
                FOREIGN KEY (day_id) REFERENCES days(day_id)
            )
        """)
        
        # === PLANNING SYSTEM TABLES ===
        # User plans metadata
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_plans (
                plan_id TEXT PRIMARY KEY,
                topic TEXT NOT NULL,
                title TEXT NOT NULL,
                created_date TEXT NOT NULL,
                duration_weeks INTEGER DEFAULT 4,
                status TEXT DEFAULT 'active',
                progress_percentage REAL DEFAULT 0.0,
                last_updated TEXT,
                notes TEXT
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_plan_status ON user_plans(status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_plan_topic ON user_plans(topic)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_plan_date ON user_plans(created_date)")
        
        # Individual plan items
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS plan_items (
                plan_id TEXT NOT NULL,
                date TEXT NOT NULL,
                task TEXT NOT NULL,
                duration_minutes INTEGER NOT NULL,
                completed BOOLEAN DEFAULT FALSE,
                notes TEXT,
                actual_duration INTEGER,
                completion_time TEXT,
                FOREIGN KEY (plan_id) REFERENCES user_plans(plan_id),
                PRIMARY KEY (plan_id, date, task)
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_plan_item_date ON plan_items(date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_plan_item_completed ON plan_items(completed)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_plan_item_plan ON plan_items(plan_id)")
        
        # Plan feedback and modifications (for future phases)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS plan_feedback (
                feedback_id TEXT PRIMARY KEY,
                plan_id TEXT NOT NULL,
                date TEXT NOT NULL,
                feedback_type TEXT NOT NULL,
                user_feedback TEXT NOT NULL,
                llm_response TEXT,
                emotional_context TEXT,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (plan_id) REFERENCES user_plans(plan_id)
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_feedback_plan ON plan_feedback(plan_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_feedback_date ON plan_feedback(date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_feedback_type ON plan_feedback(feedback_type)")
        
        # === EMBEDDINGS TABLE (Vector storage for RAG) ===
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                embedding_id TEXT PRIMARY KEY,
                turn_id TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                embedding BLOB NOT NULL,
                text_content TEXT NOT NULL,
                dimension INTEGER DEFAULT 384,
                model_name TEXT DEFAULT 'all-MiniLM-L6-v2',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (turn_id) REFERENCES metadata_staging(turn_id)
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_embedding_turn ON embeddings(turn_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_embedding_chunk ON embeddings(turn_id, chunk_index)")
        
        # === DOSSIER SYSTEM TABLES (Phase 1) ===
        # Dossiers: Meta-layer for aggregating semantically related facts
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS dossiers (
                dossier_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                summary TEXT,
                search_summary TEXT,
                created_at TEXT NOT NULL,
                last_updated TEXT NOT NULL,
                permissions TEXT DEFAULT '{"access": "full"}',
                status TEXT DEFAULT 'active'
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_dossiers_updated ON dossiers(last_updated)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_dossiers_status ON dossiers(status)")
        
        # Dossier Facts: Individual facts within dossiers
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS dossier_facts (
                fact_id TEXT PRIMARY KEY,
                dossier_id TEXT NOT NULL,
                fact_text TEXT NOT NULL,
                fact_type TEXT,
                added_at TEXT NOT NULL,
                source_block_id TEXT,
                source_turn_id TEXT,
                confidence REAL DEFAULT 1.0,
                FOREIGN KEY (dossier_id) REFERENCES dossiers(dossier_id) ON DELETE CASCADE
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_dossier_facts_dossier ON dossier_facts(dossier_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_dossier_facts_added ON dossier_facts(added_at)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_dossier_facts_source_block ON dossier_facts(source_block_id)")
        
        # Dossier Provenance: History tracking for dossier evolution
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS dossier_provenance (
                provenance_id TEXT PRIMARY KEY,
                dossier_id TEXT NOT NULL,
                operation TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                source_block_id TEXT,
                source_turn_id TEXT,
                details TEXT,
                FOREIGN KEY (dossier_id) REFERENCES dossiers(dossier_id) ON DELETE CASCADE
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_provenance_dossier ON dossier_provenance(dossier_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_provenance_timestamp ON dossier_provenance(timestamp)")
        
        # === BLOCK METADATA TABLE (Phase 2) ===
        # Stores sticky meta tags at bridge block level (not per chunk)
        # Tags stored ONCE per block, referenced by chunks via block_id
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS block_metadata (
                block_id TEXT PRIMARY KEY,
                global_tags TEXT,  -- JSON: ["env: python-3.9", "os: windows"]
                section_rules TEXT,  -- JSON: [{"start_turn": 10, "end_turn": 15, "rule": "no-eval"}]
                created_at TEXT NOT NULL,
                FOREIGN KEY (block_id) REFERENCES daily_ledger(block_id) ON DELETE CASCADE
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_block_metadata_block ON block_metadata(block_id)")
        
        # === MIGRATIONS ===
        # Add search_summary column to dossiers if it doesn't exist
        cursor.execute("PRAGMA table_info(dossiers)")
        columns = [row[1] for row in cursor.fetchall()]
        if 'search_summary' not in columns:
            cursor.execute("ALTER TABLE dossiers ADD COLUMN search_summary TEXT")
            print("  🔄 Migration: Added search_summary column to dossiers")
        
        self.conn.commit()
        print(f"Storage initialized: {self.db_path}")
    
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
        cursor.execute("""
            INSERT INTO days (day_id, created_at)
            VALUES (?, ?)
        """, (day_id, created_at))
        
        # Link to previous day if it exists
        prev_day_id = self._get_previous_day_id(day_id)
        if prev_day_id:
            # Update this day's prev_day
            cursor.execute("""
                UPDATE days SET prev_day = ? WHERE day_id = ?
            """, (prev_day_id, day_id))
            
            # Update previous day's next_day
            cursor.execute("""
                UPDATE days SET next_day = ? WHERE day_id = ?
            """, (day_id, prev_day_id))
        
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
        
        # Get keywords
        keywords = self.get_day_keywords(day_id)
        
        # Get summaries
        summaries = self.get_day_summaries(day_id)
        
        # Get affect patterns
        affect_patterns = self.get_day_affect(day_id)
        
        # Get synthesis if exists
        synthesis = self.get_day_synthesis(day_id)
        
        return DayNode(
            day_id=row['day_id'],
            created_at=datetime.fromisoformat(row['created_at']),
            prev_day=row['prev_day'],
            next_day=row['next_day'],
            session_ids=session_ids,
            keywords=keywords,
            summaries=summaries,
            affect_patterns=affect_patterns,
            synthesis=synthesis
        )
    
    def add_session_to_day(self, day_id: str, session_id: str) -> None:
        """Associate a session with a day"""
        cursor = self.conn.cursor()
        
        # Ensure day exists
        cursor.execute("SELECT day_id FROM days WHERE day_id = ?", (day_id,))
        if not cursor.fetchone():
            self.create_day(day_id)
        
        # Add session (ignore if duplicate)
        try:
            cursor.execute("""
                INSERT INTO day_sessions (day_id, session_id)
                VALUES (?, ?)
            """, (day_id, session_id))
            self.conn.commit()
        except sqlite3.IntegrityError:
            # Session already associated with this day
            pass
    
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
    # KEYWORD OPERATIONS
    # =========================================================================
    
    def add_keyword(self, day_id: str, keyword: Keyword) -> None:
        """Add or update a keyword for a day"""
        cursor = self.conn.cursor()
        
        # Check if keyword exists for this day
        cursor.execute("""
            SELECT id, frequency, turn_ids FROM day_keywords
            WHERE day_id = ? AND keyword = ?
        """, (day_id, keyword.keyword))
        
        existing = cursor.fetchone()
        
        if existing:
            # Update existing keyword
            old_turn_ids = json.loads(existing['turn_ids']) if existing['turn_ids'] else []
            new_turn_ids = list(set(old_turn_ids + keyword.turn_ids))
            
            cursor.execute("""
                UPDATE day_keywords
                SET last_mentioned = ?,
                    frequency = frequency + ?,
                    turn_ids = ?
                WHERE id = ?
            """, (
                keyword.last_mentioned,
                keyword.frequency,
                json.dumps(new_turn_ids),
                existing['id']
            ))
        else:
            # Insert new keyword
            cursor.execute("""
                INSERT INTO day_keywords 
                (day_id, keyword, first_mentioned, last_mentioned, frequency, turn_ids)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                day_id,
                keyword.keyword,
                keyword.first_mentioned,
                keyword.last_mentioned,
                keyword.frequency,
                json.dumps(keyword.turn_ids)
            ))
        
        self.conn.commit()
    
    def get_day_keywords(self, day_id: str) -> List[Keyword]:
        """Get all keywords for a day"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT keyword, first_mentioned, last_mentioned, frequency, turn_ids
            FROM day_keywords
            WHERE day_id = ?
            ORDER BY frequency DESC
        """, (day_id,))
        
        keywords = []
        for row in cursor.fetchall():
            turn_ids_list = json.loads(row['turn_ids']) if row['turn_ids'] else []
            # Use first turn as source if available, otherwise generate ID
            source_turn_id = turn_ids_list[0] if turn_ids_list else f"t_{day_id}_unknown"
            keyword_id = f"k1_{row['keyword']}_{day_id}"
            
            keywords.append(Keyword(
                keyword_id=keyword_id,
                keyword=row['keyword'],
                source_turn_id=source_turn_id,
                day_id=day_id,
                first_mentioned=datetime.fromisoformat(row['first_mentioned']),
                last_mentioned=datetime.fromisoformat(row['last_mentioned']),
                frequency=row['frequency'],
                turn_ids=turn_ids_list
            ))
        
        return keywords
    
    def search_keywords(self, keywords: List[str], date_range: Tuple[str, str] = None) -> List[Dict[str, Any]]:
        """
        Search for keywords across days (Phase 3 schema).
        Returns matching day_id, keyword, turn content, and metadata.
        
        Now searches the 'keywords' table (Phase 3) with full context from metadata_staging.
        """
        cursor = self.conn.cursor()
        
        # Build query - search new keywords table with context from metadata_staging
        placeholders = ','.join('?' * len(keywords))
        query = f"""
            SELECT 
                k.day_id, 
                k.keyword, 
                k.source_turn_id,
                k.first_mentioned, 
                k.last_mentioned, 
                k.frequency,
                m.user_message,
                m.assistant_response,
                m.timestamp as turn_timestamp
            FROM keywords k
            LEFT JOIN metadata_staging m ON k.source_turn_id = m.turn_id
            WHERE k.keyword IN ({placeholders})
        """
        params = list(keywords)
        
        if date_range:
            query += " AND k.day_id BETWEEN ? AND ?"
            params.extend(date_range)
        
        query += " ORDER BY k.day_id DESC, k.frequency DESC"
        
        cursor.execute(query, params)
        
        results = []
        for row in cursor.fetchall():
            # Build context from turn content
            context = f"User: {row['user_message']}\nAssistant: {row['assistant_response']}"
            
            results.append({
                'day_id': row['day_id'],
                'keyword': row['keyword'],
                'first_mentioned': row['first_mentioned'],
                'last_mentioned': row['last_mentioned'],
                'frequency': row['frequency'],
                'turn_ids': [row['source_turn_id']],  # Single turn per keyword entry
                'context': context,  # NEW: Full conversation context
                'timestamp': row['turn_timestamp'],  # NEW: Turn timestamp
                'session_id': None  # Compatibility with old schema
            })
        
        return results
    
    # =========================================================================
    # SUMMARY OPERATIONS
    # =========================================================================
    
    def add_summary(self, day_id: str, summary: Summary) -> None:
        """Add a turn summary"""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO day_summaries
            (turn_id, day_id, timestamp, user_query_summary, 
             assistant_response_summary, keywords_this_turn)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            summary.turn_id,
            day_id,
            summary.timestamp,
            summary.user_query_summary,
            summary.assistant_response_summary,
            json.dumps(summary.keywords_this_turn)
        ))
        self.conn.commit()
    
    def get_day_summaries(self, day_id: str) -> List[Summary]:
        """Get all summaries for a day"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT turn_id, timestamp, user_query_summary,
                   assistant_response_summary, keywords_this_turn
            FROM day_summaries
            WHERE day_id = ?
            ORDER BY turn_id
        """, (day_id,))
        
        summaries = []
        for row in cursor.fetchall():
            summaries.append(Summary(
                turn_id=row['turn_id'],
                timestamp=datetime.fromisoformat(row['timestamp']),
                user_query_summary=row['user_query_summary'],
                assistant_response_summary=row['assistant_response_summary'],
                keywords_this_turn=json.loads(row['keywords_this_turn']) if row['keywords_this_turn'] else []
            ))
        
        return summaries
    
    # =========================================================================
    # AFFECT OPERATIONS
    # =========================================================================
    
    def add_affect(self, day_id: str, affect: Affect) -> None:
        """Add or update an affect pattern"""
        cursor = self.conn.cursor()
        
        # Check if affect exists for this day
        cursor.execute("""
            SELECT id FROM day_affect
            WHERE day_id = ? AND affect_label = ?
        """, (day_id, affect.affect_label))
        
        existing = cursor.fetchone()
        
        if existing:
            # Update existing
            cursor.execute("""
                UPDATE day_affect
                SET last_detected = ?,
                    intensity = ?,
                    associated_topics = ?,
                    turn_ids = ?
                WHERE id = ?
            """, (
                affect.last_detected,
                affect.intensity,
                json.dumps(affect.associated_topics),
                json.dumps(affect.turn_ids),
                existing['id']
            ))
        else:
            # Insert new
            cursor.execute("""
                INSERT INTO day_affect
                (day_id, affect_label, first_detected, last_detected,
                 intensity, associated_topics, turn_ids)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                day_id,
                affect.affect_label,
                affect.first_detected,
                affect.last_detected,
                affect.intensity,
                json.dumps(affect.associated_topics),
                json.dumps(affect.turn_ids)
            ))
        
        self.conn.commit()
    
    def get_day_affect(self, day_id: str) -> List[Affect]:
        """Get all affect patterns for a day"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT affect_label, first_detected, last_detected,
                   intensity, associated_topics, turn_ids
            FROM day_affect
            WHERE day_id = ?
        """, (day_id,))
        
        affects = []
        for row in cursor.fetchall():
            turn_ids_list = json.loads(row['turn_ids']) if row['turn_ids'] else []
            # Use first turn as source if available
            source_turn_id = turn_ids_list[0] if turn_ids_list else f"t_{day_id}_unknown"
            affect_id = f"a1_{row['affect_label']}_{day_id}"
            
            affects.append(Affect(
                affect_id=affect_id,
                affect_label=row['affect_label'],
                source_turn_id=source_turn_id,
                day_id=day_id,
                first_detected=datetime.fromisoformat(row['first_detected']),
                last_detected=datetime.fromisoformat(row['last_detected']),
                intensity=row['intensity'],
                associated_topics=json.loads(row['associated_topics']) if row['associated_topics'] else [],
                turn_ids=turn_ids_list
            ))
        
        return affects
    
    # =========================================================================
    # TASK OPERATIONS
    # =========================================================================
    
    def save_task(self, task: TaskState) -> None:
        """Save or update a task"""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO tasks
            (task_id, task_type, status, created_date, created_at, last_updated,
             completed_at, task_title, original_query, total_steps, completed_steps,
             skipped_steps, tags, notes, state_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            task.task_id,
            task.task_type.value,
            task.status.value,
            task.created_date,
            task.created_at,
            task.last_updated,
            task.completed_at,
            task.task_title,
            task.original_query,
            task.total_steps,
            task.completed_steps,
            task.skipped_steps,
            json.dumps(task.tags),
            task.notes,
            json.dumps(task.state_json)
        ))
        self.conn.commit()
    
    def get_task(self, task_id: str) -> Optional[TaskState]:
        """Get a task by ID"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM tasks WHERE task_id = ?
        """, (task_id,))
        
        row = cursor.fetchone()
        if not row:
            return None
        
        return TaskState(
            task_id=row['task_id'],
            task_type=TaskType(row['task_type']),
            status=TaskStatus(row['status']),
            created_date=row['created_date'],
            created_at=datetime.fromisoformat(row['created_at']),
            last_updated=datetime.fromisoformat(row['last_updated']),
            completed_at=datetime.fromisoformat(row['completed_at']) if row['completed_at'] else None,
            task_title=row['task_title'] or "",
            original_query=row['original_query'] or "",
            total_steps=row['total_steps'],
            completed_steps=row['completed_steps'],
            skipped_steps=row['skipped_steps'],
            tags=json.loads(row['tags']) if row['tags'] else [],
            notes=row['notes'] or "",
            state_json=json.loads(row['state_json']) if row['state_json'] else {}
        )
    
    def get_active_tasks(self) -> List[TaskState]:
        """Get all active tasks"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM tasks
            WHERE status = ?
            ORDER BY last_updated DESC
        """, (TaskStatus.ACTIVE.value,))
        
        tasks = []
        for row in cursor.fetchall():
            tasks.append(TaskState(
                task_id=row['task_id'],
                task_type=TaskType(row['task_type']),
                status=TaskStatus(row['status']),
                created_date=row['created_date'],
                created_at=datetime.fromisoformat(row['created_at']),
                last_updated=datetime.fromisoformat(row['last_updated']),
                completed_at=datetime.fromisoformat(row['completed_at']) if row['completed_at'] else None,
                task_title=row['task_title'] or "",
                original_query=row['original_query'] or "",
                total_steps=row['total_steps'],
                completed_steps=row['completed_steps'],
                skipped_steps=row['skipped_steps'],
                tags=json.loads(row['tags']) if row['tags'] else [],
                notes=row['notes'] or "",
                state_json=json.loads(row['state_json']) if row['state_json'] else {}
            ))
        
        return tasks
    
    def link_task_to_day(self, task_id: str, day_id: str, event_type: str) -> None:
        """Link a task event to a day"""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO task_days (task_id, day_id, event_type, timestamp)
            VALUES (?, ?, ?, ?)
        """, (task_id, day_id, event_type, datetime.now()))
        self.conn.commit()
    
    # =========================================================================
    # METADATA STAGING OPERATIONS (Pre-Synthesis)
    # =========================================================================
    
    def stage_turn_metadata(self, turn: ConversationTurn) -> None:
        """
        Stage turn metadata for later synthesis.
        
        Updated for Phase B: Supports new string IDs and lineage references.
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
        
        Updated for Phase B: Returns turns with new ID structure.
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM metadata_staging
            WHERE day_id = ?
            ORDER BY turn_sequence
        """, (day_id,))
        
        turns = []
        for row in cursor.fetchall():
            turns.append(ConversationTurn(
                turn_id=row['turn_id'],
                turn_sequence=row['turn_sequence'],        # NEW
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
                summary_id=row['summary_id'],              # NEW
                keyword_ids=json.loads(row['keyword_ids']) if row['keyword_ids'] else [],  # NEW
                affect_ids=json.loads(row['affect_ids']) if row['affect_ids'] else [],      # NEW
                task_created_id=row['task_created_id'],    # NEW
                task_updated_ids=json.loads(row['task_updated_ids']) if row['task_updated_ids'] else [],  # NEW
                loaded_turn_ids=json.loads(row['loaded_turn_ids']) if row['loaded_turn_ids'] else []      # NEW
            ))
        
        return turns
    
    def clear_staged_turns(self, day_id: str) -> None:
        """Clear staged turns after synthesis"""
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM metadata_staging WHERE day_id = ?", (day_id,))
        self.conn.commit()
    
    # =========================================================================
    # DAY SYNTHESIS OPERATIONS
    # =========================================================================
    
    def save_day_synthesis(self, synthesis: DaySynthesis) -> None:
        """Save day synthesis results"""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO day_synthesis
            (day_id, created_at, emotional_arc, key_patterns, topic_affect_mapping,
             behavioral_notes, narrative_summary, notable_moments)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            synthesis.day_id,
            synthesis.created_at,
            synthesis.emotional_arc,
            json.dumps(synthesis.key_patterns),
            json.dumps(synthesis.topic_affect_mapping),
            synthesis.behavioral_notes,
            synthesis.narrative_summary,
            json.dumps(synthesis.notable_moments)
        ))
        self.conn.commit()
    
    def get_day_synthesis(self, day_id: str) -> Optional[DaySynthesis]:
        """Get day synthesis if it exists"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM day_synthesis WHERE day_id = ?
        """, (day_id,))
        
        row = cursor.fetchone()
        if not row:
            return None
        
        return DaySynthesis(
            day_id=row['day_id'],
            created_at=datetime.fromisoformat(row['created_at']),
            emotional_arc=row['emotional_arc'],
            key_patterns=json.loads(row['key_patterns']) if row['key_patterns'] else [],
            topic_affect_mapping=json.loads(row['topic_affect_mapping']) if row['topic_affect_mapping'] else {},
            behavioral_notes=row['behavioral_notes'] or "",
            narrative_summary=row['narrative_summary'],
            notable_moments=json.loads(row['notable_moments']) if row['notable_moments'] else []
        )
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def get_day_range(self, start_date: str, end_date: str) -> List[DayNode]:
        """Get all days in a date range"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT day_id FROM days
            WHERE day_id BETWEEN ? AND ?
            ORDER BY day_id
        """, (start_date, end_date))
        
        return [self.get_day(row['day_id']) for row in cursor.fetchall()]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics"""
        cursor = self.conn.cursor()
        
        stats = {}
        
        # Count days
        cursor.execute("SELECT COUNT(*) as count FROM days")
        stats['total_days'] = cursor.fetchone()['count']
        
        # Count tasks by status
        cursor.execute("""
            SELECT status, COUNT(*) as count
            FROM tasks
            GROUP BY status
        """)
        stats['tasks_by_status'] = {row['status']: row['count'] for row in cursor.fetchall()}
        
        # Count keywords
        cursor.execute("SELECT COUNT(DISTINCT keyword) as count FROM day_keywords")
        stats['unique_keywords'] = cursor.fetchone()['count']
        
        # Count staged turns
        cursor.execute("SELECT COUNT(*) as count FROM metadata_staging")
        stats['staged_turns'] = cursor.fetchone()['count']
        
        # Count embeddings
        cursor.execute("SELECT COUNT(*) as count FROM embeddings")
        stats['total_embeddings'] = cursor.fetchone()['count']
        
        return stats
    
    # ============================================================================
    # VECTOR EMBEDDINGS METHODS
    # ============================================================================
    
    def save_embedding(self, embedding_id: str, turn_id: str, chunk_index: int,
                      embedding_bytes: bytes, text_content: str,
                      dimension: int = 384, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Save a vector embedding to the database.
        
        Args:
            embedding_id: Unique embedding identifier
            turn_id: Associated turn ID
            chunk_index: Chunk number (0 for first chunk)
            embedding_bytes: Serialized numpy array
            text_content: The text that was embedded
            dimension: Embedding dimension
            model_name: Model used to generate embedding
        """
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
    # PLANNING SYSTEM METHODS
    # ============================================================================
    
    def save_user_plan(self, plan: 'UserPlan'):
        """
        Save or update a user plan.
        
        Args:
            plan: UserPlan object to save
        """
        cursor = self.conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO user_plans
            (plan_id, topic, title, created_date, duration_weeks, status, 
             progress_percentage, last_updated, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            plan.plan_id,
            plan.topic,
            plan.title,
            plan.created_date,
            plan.duration_weeks,
            plan.status,
            plan.progress_percentage,
            plan.last_updated.isoformat() if plan.last_updated else None,
            plan.notes
        ))
        
        # Save plan items
        for item in plan.items:
            cursor.execute("""
                INSERT OR REPLACE INTO plan_items
                (plan_id, date, task, duration_minutes, completed, notes, 
                 actual_duration, completion_time)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                plan.plan_id,
                item.date,
                item.task,
                item.duration_minutes,
                item.completed,
                item.notes,
                item.actual_duration,
                item.completion_time.isoformat() if item.completion_time else None
            ))
        
        self.conn.commit()
    
    def get_user_plan(self, plan_id: str) -> Optional['UserPlan']:
        """
        Retrieve a complete user plan with all items.
        
        Args:
            plan_id: Plan identifier
            
        Returns:
            UserPlan object or None if not found
        """
        cursor = self.conn.cursor()
        
        # Get plan metadata
        plan_row = cursor.execute("""
            SELECT * FROM user_plans WHERE plan_id = ?
        """, (plan_id,)).fetchone()
        
        if not plan_row:
            return None
        
        # Get plan items
        item_rows = cursor.execute("""
            SELECT * FROM plan_items WHERE plan_id = ?
            ORDER BY date, task
        """, (plan_id,)).fetchall()
        
        # Convert to PlanItem objects
        items = []
        for row in item_rows:
            items.append(PlanItem(
                plan_id=row['plan_id'],
                date=row['date'],
                task=row['task'],
                duration_minutes=row['duration_minutes'],
                completed=bool(row['completed']),
                notes=row['notes'] or "",
                actual_duration=row['actual_duration'],
                completion_time=datetime.fromisoformat(row['completion_time']) if row['completion_time'] else None
            ))
        
        return UserPlan(
            plan_id=plan_row['plan_id'],
            topic=plan_row['topic'],
            title=plan_row['title'],
            created_date=plan_row['created_date'],
            duration_weeks=plan_row['duration_weeks'],
            items=items,
            status=plan_row['status'],
            progress_percentage=plan_row['progress_percentage'],
            last_updated=datetime.fromisoformat(plan_row['last_updated']) if plan_row['last_updated'] else None,
            notes=plan_row['notes'] or ""
        )
    
    def get_plans_for_date(self, date: str) -> List['PlanItem']:
        """
        Get all plan items for a specific date.
        
        Args:
            date: Date string in YYYY-MM-DD format
            
        Returns:
            List of PlanItem objects for that date
        """
        cursor = self.conn.cursor()
        
        rows = cursor.execute("""
            SELECT * FROM plan_items 
            WHERE date = ? AND completed = FALSE
            ORDER BY task
        """, (date,)).fetchall()
        
        items = []
        for row in rows:
            items.append(PlanItem(
                plan_id=row['plan_id'],
                date=row['date'],
                task=row['task'],
                duration_minutes=row['duration_minutes'],
                completed=bool(row['completed']),
                notes=row['notes'] or "",
                actual_duration=row['actual_duration'],
                completion_time=datetime.fromisoformat(row['completion_time']) if row['completion_time'] else None
            ))
        
        return items
    
    def update_plan_item_completion(self, plan_id: str, date: str, task: str,
                                   completed: bool, actual_duration: Optional[int] = None):
        """
        Update completion status of a plan item.
        
        Args:
            plan_id: Plan identifier
            date: Date string
            task: Task description
            completed: Completion status
            actual_duration: Actual time spent (optional)
        """
        cursor = self.conn.cursor()
        
        update_data = {
            'completed': completed,
            'completion_time': datetime.now().isoformat() if completed else None
        }
        
        if actual_duration is not None:
            update_data['actual_duration'] = actual_duration
        
        # Build dynamic update query
        set_clause = ', '.join(f"{key} = ?" for key in update_data.keys())
        values = list(update_data.values()) + [plan_id, date, task]
        
        cursor.execute(f"""
            UPDATE plan_items 
            SET {set_clause}
            WHERE plan_id = ? AND date = ? AND task = ?
        """, values)
        
        # Update plan progress
        self._update_plan_progress(plan_id)
        
        self.conn.commit()
    
    def get_active_plans(self) -> List['UserPlan']:
        """
        Get all active (non-completed) user plans.
        
        Returns:
            List of active UserPlan objects
        """
        cursor = self.conn.cursor()
        
        plan_rows = cursor.execute("""
            SELECT plan_id FROM user_plans 
            WHERE status IN ('active', 'paused')
            ORDER BY created_date DESC
        """).fetchall()
        
        plans = []
        for row in plan_rows:
            plan = self.get_user_plan(row['plan_id'])
            if plan:
                plans.append(plan)
        
        return plans
    
    def _update_plan_progress(self, plan_id: str):
        """
        Update the progress percentage for a plan.
        
        Args:
            plan_id: Plan identifier
        """
        cursor = self.conn.cursor()
        
        # Calculate progress
        result = cursor.execute("""
            SELECT 
                COUNT(*) as total_items,
                COUNT(CASE WHEN completed = 1 THEN 1 END) as completed_items
            FROM plan_items 
            WHERE plan_id = ?
        """, (plan_id,)).fetchone()
        
        if result and result['total_items'] > 0:
            progress = (result['completed_items'] / result['total_items']) * 100.0
            
            cursor.execute("""
                UPDATE user_plans 
                SET progress_percentage = ?, last_updated = ?
                WHERE plan_id = ?
            """, (progress, datetime.now().isoformat(), plan_id))
    
    # ============================================================================
    # DEPRECATED: HMLR v1 SPAN MANAGEMENT (Moved to memory/deprecated/span_methods.py)

    # Phase 11.5: Fact Store & Daily Ledger Query Methods
    # ========================================================================
    
    def query_fact_store(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Query fact_store for exact keyword match (Phase 11.5).
        
        Args:
            key: The exact key to search for (e.g., "HMLR", "API_KEY")
        
        Returns:
            Fact dictionary if found, None otherwise
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT fact_id, key, value, category, 
                   source_span_id, source_chunk_id, source_paragraph_id,
                   source_block_id, evidence_snippet, created_at
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
            'evidence_snippet': row[8],
            'created_at': row[9]
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
                   source_block_id, evidence_snippet, created_at
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
                'evidence_snippet': row[8],
                'created_at': row[9]
            })
        
        return facts
    
    def update_facts_block_id(self, turn_id: str, block_id: str) -> int:
        """
        Update all facts from a specific turn with the final block_id.
        
        Facts are initially created with block_id=None during extraction.
        After the Governor assigns a block_id, this method updates all
        facts from that turn to link them to the correct Bridge Block.
        
        Strategy:
        - turn_id format: "turn_YYYYMMDD_HHMMSS"
        - chunk_id format: "sent_YYYYMMDD_HHMMSS_randomhex" or "para_YYYYMMDD_HHMMSS_randomhex"
        - Match on timestamp portion (chars after underscore in turn_id)
        
        Args:
            turn_id: The turn identifier (e.g., "turn_20250115_143022")
            block_id: The final Bridge Block ID assigned by the Governor
        
        Returns:
            Number of facts updated
        """
        cursor = self.conn.cursor()
        
        # Extract timestamp from turn_id (e.g., "turn_20250115_143022" -> "20250115_143022")
        timestamp = turn_id.replace("turn_", "")
        
        # Update facts where source_chunk_id contains the timestamp
        # This matches both sent_TIMESTAMP_* and para_TIMESTAMP_*
        cursor.execute("""
            UPDATE fact_store
            SET source_block_id = ?
            WHERE source_chunk_id LIKE ?
              AND (source_block_id IS NULL OR source_block_id = '')
        """, (block_id, f"%{timestamp}%"))
        
        updated_count = cursor.rowcount
        self.conn.commit()
        
        return updated_count
    
    def get_active_bridge_blocks(self) -> List[Dict[str, Any]]:
        """
        Retrieve all Bridge Blocks from today (hot path - Phase 11.5).
        
        Returns:
            List of Bridge Block content_json from today
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT block_id, content_json, created_at, status, exit_reason
            FROM daily_ledger
            WHERE DATE(created_at) = DATE('now')
            AND status IN ('ACTIVE', 'PAUSED')
            ORDER BY created_at DESC
        """)
        
        results = []
        for row in cursor.fetchall():
            try:
                content = json.loads(row[1])
                results.append({
                    'block_id': row[0],
                    'content': content,
                    'created_at': row[2],
                    'status': row[3],
                    'exit_reason': row[4]
                })
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse bridge block {row[0]}: {e}")
                continue
        
        return results
    
    # ========================================================================
    # PHASE 11.9.B: BRIDGE BLOCK STORAGE METHODS (Governor Integration)
    # ========================================================================

    def get_daily_ledger_metadata(self, day_id: str) -> List[Dict[str, Any]]:
        """
        Get metadata summaries of all Bridge Blocks for a specific day (CRITICAL for LLM routing).
        Excludes heavy turns[] array to keep payload lightweight.
        
        Args:
            day_id: The day ID in format YYYY-MM-DD
        
        Returns:
            List of lightweight block metadata dictionaries with:
            - block_id: Unique identifier
            - topic_label: Human-readable topic name
            - summary: Brief description of conversation
            - keywords: List of topic keywords
            - open_loops: Unresolved questions/tasks
            - decisions: Key decisions made
            - turn_count: Number of turns in block
            - last_updated: Timestamp of last activity
            - is_last_active: Boolean flag (only one block should be True)
            - status: ACTIVE or PAUSED
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT block_id, content_json, updated_at, status
            FROM daily_ledger
            WHERE DATE(created_at) = DATE(?)
            AND status IN ('ACTIVE', 'PAUSED')
            ORDER BY updated_at DESC
        """, (day_id,))
        
        metadata_list = []
        for row in cursor.fetchall():
            try:
                content = json.loads(row[1])
                
                # Extract metadata only (exclude heavy turns[] array)
                metadata = {
                    'block_id': row[0],
                    'topic_label': content.get('topic_label', 'Unknown Topic'),
                    'summary': content.get('summary', ''),
                    'keywords': content.get('keywords', []),
                    'open_loops': content.get('open_loops', []),
                    'decisions_made': content.get('decisions_made', []),
                    'turn_count': len(content.get('turns', [])),
                    'last_updated': row[2],
                    'is_last_active': (row[3] == 'ACTIVE'),  # Only active block is "last active"
                    'status': row[3]
                }
                metadata_list.append(metadata)
                
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Failed to extract metadata from block {row[0]}: {e}")
                continue
        
        return metadata_list

    def get_bridge_block_full(self, block_id: str) -> Optional[Dict[str, Any]]:
        """
        Load complete Bridge Block including turns[] array (called AFTER routing decision).
        
        Args:
            block_id: The unique block identifier
        
        Returns:
            Full Bridge Block JSON dictionary including:
            - All metadata fields
            - turns[]: Complete conversation history (verbatim user/AI messages)
            - chunk_ids: References to pre-chunked content
            
            Returns None if block not found.
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT content_json, status, created_at, updated_at
            FROM daily_ledger
            WHERE block_id = ?
        """, (block_id,))
        
        row = cursor.fetchone()
        if not row:
            logger.warning(f"Bridge block {block_id} not found")
            return None
        
        try:
            content = json.loads(row[0])
            content['_db_status'] = row[1]  # Include current DB status
            content['_db_created_at'] = row[2]
            content['_db_updated_at'] = row[3]
            return content
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse bridge block {block_id}: {e}")
            return None

    def append_turn_to_block(self, block_id: str, turn: Dict[str, Any]) -> bool:
        """
        Append a new conversation turn to Bridge Block's turns[] array.
        
        Args:
            block_id: The block to update
            turn: Dictionary containing:
                - turn_id: Unique turn identifier
                - timestamp: ISO-8601 timestamp
                - user_message: User's query (verbatim)
                - ai_response: AI's response (verbatim)
                - chunks: List of chunk IDs (for gardener)
        
        Returns:
            True if successful, False otherwise
        """
        cursor = self.conn.cursor()
        
        # Get current block content
        cursor.execute("SELECT content_json FROM daily_ledger WHERE block_id = ?", (block_id,))
        row = cursor.fetchone()
        if not row:
            logger.error(f"Cannot append turn: block {block_id} not found")
            return False
        
        try:
            content = json.loads(row[0])
            
            # Append turn to turns[] array
            if 'turns' not in content:
                content['turns'] = []
            content['turns'].append(turn)
            
            # Update timestamp
            content['last_updated'] = datetime.now().isoformat()
            
            # Save back to DB
            cursor.execute("""
                UPDATE daily_ledger
                SET content_json = ?, updated_at = ?
                WHERE block_id = ?
            """, (json.dumps(content), datetime.now().isoformat(), block_id))
            
            self.conn.commit()
            logger.info(f"Appended turn {turn.get('turn_id')} to block {block_id}")
            return True
            
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to append turn to block {block_id}: {e}")
            return False

    def update_bridge_block_status(self, block_id: str, new_status: str, exit_reason: Optional[str] = None) -> bool:
        """
        Change Bridge Block status between ACTIVE ↔ PAUSED.
        
        Args:
            block_id: The block to update
            new_status: 'ACTIVE' or 'PAUSED'
            exit_reason: Optional reason for pause (e.g., 'topic_shift', 'day_rollover')
        
        Returns:
            True if successful, False otherwise
        """
        if new_status not in ['ACTIVE', 'PAUSED', 'ARCHIVED']:
            logger.error(f"Invalid status: {new_status}. Must be ACTIVE, PAUSED, or ARCHIVED")
            return False
        
        cursor = self.conn.cursor()
        
        # Also update status in content_json for consistency
        cursor.execute("SELECT content_json FROM daily_ledger WHERE block_id = ?", (block_id,))
        row = cursor.fetchone()
        if not row:
            logger.warning(f"Block {block_id} not found for status update")
            return False
        
        try:
            content = json.loads(row[0])
            content['status'] = new_status
            if exit_reason:
                content['exit_reason'] = exit_reason
            
            cursor.execute("""
                UPDATE daily_ledger
                SET status = ?, exit_reason = ?, updated_at = ?, content_json = ?
                WHERE block_id = ?
            """, (new_status, exit_reason, datetime.now().isoformat(), json.dumps(content), block_id))
            
            self.conn.commit()
            logger.info(f"Updated block {block_id} status: {new_status} (reason: {exit_reason})")
            return True
            
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to update status for block {block_id}: {e}")
            self.conn.rollback()
            return False

    def update_last_active_flag(self, block_id: str) -> bool:
        """
        Set is_last_active flag for specified block and clear it for all others in the same day.
        This ensures only ONE block is marked as last active per day.
        
        Args:
            block_id: The block to mark as last active
        
        Returns:
            True if successful, False otherwise
        """
        cursor = self.conn.cursor()
        
        # Get the day from block's created_at timestamp
        cursor.execute("SELECT created_at FROM daily_ledger WHERE block_id = ?", (block_id,))
        row = cursor.fetchone()
        if not row:
            logger.error(f"Block {block_id} not found")
            return False
        
        day_id = row[0][:10]  # Extract YYYY-MM-DD from ISO timestamp
        
        try:
            # Clear all last_active flags for this day
            cursor.execute("""
                UPDATE daily_ledger
                SET status = 'PAUSED'
                WHERE DATE(created_at) = DATE(?)
                AND status = 'ACTIVE'
                AND block_id != ?
            """, (day_id, block_id))
            
            # Set the specified block as ACTIVE (which implies last_active)
            cursor.execute("""
                UPDATE daily_ledger
                SET status = 'ACTIVE', updated_at = ?
                WHERE block_id = ?
            """, (datetime.now().isoformat(), block_id))
            
            self.conn.commit()
            logger.info(f"Block {block_id} marked as last active for day {day_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update last_active flag: {e}")
            self.conn.rollback()
            return False

    def generate_block_summary(self, block_id: str) -> Optional[str]:
        """
        Generate summary of Bridge Block when pausing (called before status change to PAUSED).
        Uses LLM to create concise summary of conversation.
        
        Args:
            block_id: The block to summarize
        
        Returns:
            Generated summary string, or None if failed
        
        Note:
            This is a placeholder for Phase 11.9. Full LLM integration happens in Governor.
            For now, generates a simple summary from turns count and topic.
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT content_json FROM daily_ledger WHERE block_id = ?", (block_id,))
        row = cursor.fetchone()
        if not row:
            logger.warning(f"Block {block_id} not found for summarization")
            return None
        
        try:
            content = json.loads(row[0])
            topic = content.get('topic_label', 'Unknown Topic')
            turn_count = len(content.get('turns', []))
            keywords = content.get('keywords', [])
            
            # Simple summary (TODO: Replace with LLM call in Governor)
            summary = f"{turn_count}-turn discussion about {topic}"
            if keywords:
                summary += f" (keywords: {', '.join(keywords[:3])})"
            
            # Update the block with generated summary
            content['summary'] = summary
            cursor.execute("""
                UPDATE daily_ledger
                SET content_json = ?, updated_at = ?
                WHERE block_id = ?
            """, (json.dumps(content), datetime.now().isoformat(), block_id))
            
            self.conn.commit()
            logger.info(f"Generated summary for block {block_id}: {summary}")
            return summary
            
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to generate summary for block {block_id}: {e}")
            return None

    def create_new_bridge_block(
        self,
        day_id: str,
        topic_label: str,
        keywords: List[str],
        span_id: Optional[str] = None
    ) -> Optional[str]:
        """
        Create a new Bridge Block with LLM-suggested metadata.
        
        Args:
            day_id: The day this block belongs to (YYYY-MM-DD)
            topic_label: LLM-suggested topic name (e.g., "AWS Lambda Pricing")
            keywords: Extracted keywords from query
            span_id: Optional span reference (for lineage tracking)
        
        Returns:
            block_id of created block, or None if failed
        """
        from uuid import uuid4
        
        block_id = f"bb_{day_id.replace('-', '')}_{uuid4().hex[:8]}"
        timestamp = datetime.now().isoformat()
        
        # Create Bridge Block structure
        content = {
            'block_id': block_id,
            'prev_block_id': None,  # TODO: Link to previous block if related
            'span_id': span_id,
            'timestamp': timestamp,
            'status': 'ACTIVE',
            'exit_reason': None,
            'topic_label': topic_label,
            'summary': '',  # Generated when paused
            'user_affect': '',  # TODO: Extract from first turn
            'bot_persona': '',  # TODO: Infer from context
            'open_loops': [],
            'decisions_made': [],
            'active_variables': {},
            'keywords': keywords,
            'turns': []  # Empty initially, populated as conversation progresses
        }
        
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO daily_ledger (
                    block_id, prev_block_id, span_id, content_json,
                    created_at, updated_at, status, exit_reason, embedding_status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                block_id,
                None,  # prev_block_id
                span_id,
                json.dumps(content),
                timestamp,
                timestamp,
                'ACTIVE',
                None,
                'PENDING'
            ))
            
            self.conn.commit()
            logger.info(f"Created new bridge block: {block_id} (topic: {topic_label})")
            return block_id
            
        except Exception as e:
            logger.error(f"Failed to create bridge block: {e}")
            self.conn.rollback()
            return None

    def update_bridge_block_metadata(self, block_id: str, metadata: Dict[str, Any]) -> bool:
        """
        Update Bridge Block header metadata from LLM-generated JSON.
        
        This is called after the main LLM generates/updates block metadata.
        Updates fields like topic_label, keywords, summary, open_loops, decisions_made.
        
        Args:
            block_id: The block to update
            metadata: Dictionary with metadata fields (from LLM JSON response)
        
        Returns:
            True if successful, False otherwise
        
        Example metadata:
            {
                "topic_label": "AWS Lambda Pricing",
                "keywords": ["aws", "lambda", "serverless", "pricing"],
                "summary": "Discussion about Lambda pricing tiers and cost optimization",
                "open_loops": ["Research reserved capacity pricing"],
                "decisions_made": ["Use on-demand for initial deployment"]
            }
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT content_json FROM daily_ledger WHERE block_id = ?", (block_id,))
        row = cursor.fetchone()
        
        if not row:
            logger.error(f"Block {block_id} not found for metadata update")
            return False
        
        try:
            content = json.loads(row[0])
            
            # Update fields present in metadata
            if 'topic_label' in metadata:
                content['topic_label'] = metadata['topic_label']
            if 'keywords' in metadata:
                content['keywords'] = metadata['keywords']
            if 'summary' in metadata:
                content['summary'] = metadata['summary']
            if 'open_loops' in metadata:
                content['open_loops'] = metadata['open_loops']
            if 'decisions_made' in metadata:
                content['decisions_made'] = metadata['decisions_made']
            if 'user_affect' in metadata:
                content['user_affect'] = metadata['user_affect']
            if 'bot_persona' in metadata:
                content['bot_persona'] = metadata['bot_persona']
            
            # Write back to DB
            cursor.execute("""
                UPDATE daily_ledger
                SET content_json = ?, updated_at = ?
                WHERE block_id = ?
            """, (json.dumps(content), datetime.now().isoformat(), block_id))
            
            self.conn.commit()
            
            updated_fields = list(metadata.keys())
            logger.info(f"Updated metadata for block {block_id}: {updated_fields}")
            return True
            
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to update metadata for block {block_id}: {e}")
            self.conn.rollback()
            return False

    # =========================================================================
    # DOSSIER OPERATIONS (Phase 1)
    # =========================================================================
    
    def create_dossier(self, dossier_id: str, title: str, summary: str = "", search_summary: str = "") -> bool:
        """
        Create a new dossier.
        
        Args:
            dossier_id: Unique ID (format: dos_YYYYMMDD_HHMMSS)
            title: Dossier title (from cluster_label)
            summary: Initial summary (optional)
            search_summary: Search-optimized summary for broad retrieval (optional)
        
        Returns:
            True if successful, False otherwise
        """
        try:
            cursor = self.conn.cursor()
            now = datetime.now().isoformat()
            cursor.execute("""
                INSERT INTO dossiers (dossier_id, title, summary, search_summary, created_at, last_updated)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (dossier_id, title, summary, search_summary, now, now))
            self.conn.commit()
            logger.info(f"Created dossier: {dossier_id} - {title}")
            return True
        except Exception as e:
            logger.error(f"Failed to create dossier {dossier_id}: {e}")
            self.conn.rollback()
            return False
    
    def get_dossier(self, dossier_id: str) -> Optional[Dict[str, Any]]:
        """
        Get dossier by ID.
        
        Returns:
            Dictionary with dossier fields, or None if not found
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM dossiers WHERE dossier_id = ?", (dossier_id,))
        row = cursor.fetchone()
        return dict(row) if row else None
    
    def get_all_dossiers(self, status: str = 'active') -> List[Dict[str, Any]]:
        """
        Get all dossiers with given status.
        
        Args:
            status: Filter by status ('active', 'archived', etc.)
        
        Returns:
            List of dossier dictionaries
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM dossiers WHERE status = ? ORDER BY last_updated DESC", (status,))
        return [dict(row) for row in cursor.fetchall()]
    
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
        """
        Add a fact to a dossier.
        
        Args:
            dossier_id: Target dossier ID
            fact_id: Unique fact ID (format: fact_YYYYMMDD_HHMMSS_XXX)
            fact_text: The actual fact content
            source_block_id: Bridge block that contributed this fact
            source_turn_id: Optional turn ID within block
            fact_type: Optional categorization (observation, preference, etc.)
            confidence: Confidence score 0-1
        
        Returns:
            True if successful, False otherwise
        """
        try:
            cursor = self.conn.cursor()
            now = datetime.now().isoformat()
            cursor.execute("""
                INSERT INTO dossier_facts 
                (fact_id, dossier_id, fact_text, fact_type, added_at, 
                 source_block_id, source_turn_id, confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (fact_id, dossier_id, fact_text, fact_type, now, 
                  source_block_id, source_turn_id, confidence))
            
            # Update dossier's last_updated timestamp
            cursor.execute("""
                UPDATE dossiers SET last_updated = ? WHERE dossier_id = ?
            """, (now, dossier_id))
            
            self.conn.commit()
            logger.info(f"Added fact {fact_id} to dossier {dossier_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to add fact to dossier: {e}")
            self.conn.rollback()
            return False
    
    def get_dossier_facts(self, dossier_id: str) -> List[Dict[str, Any]]:
        """
        Get all facts for a dossier.
        
        Args:
            dossier_id: Target dossier ID
        
        Returns:
            List of fact dictionaries, ordered by added_at
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM dossier_facts 
            WHERE dossier_id = ? 
            ORDER BY added_at ASC
        """, (dossier_id,))
        return [dict(row) for row in cursor.fetchall()]
    
    def update_dossier_summary(self, dossier_id: str, new_summary: str) -> bool:
        """
        Update dossier summary (incremental updates).
        
        Args:
            dossier_id: Target dossier ID
            new_summary: Updated summary text
        
        Returns:
            True if successful, False otherwise
        """
        try:
            cursor = self.conn.cursor()
            now = datetime.now().isoformat()
            cursor.execute("""
                UPDATE dossiers 
                SET summary = ?, last_updated = ? 
                WHERE dossier_id = ?
            """, (new_summary, now, dossier_id))
            self.conn.commit()
            logger.info(f"Updated summary for dossier {dossier_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to update dossier summary: {e}")
            self.conn.rollback()
            return False
    
    def add_provenance_entry(
        self,
        dossier_id: str,
        operation: str,
        provenance_id: str,
        source_block_id: str = None,
        source_turn_id: str = None,
        details: str = None
    ) -> bool:
        """
        Add provenance tracking entry for dossier operations.
        
        Args:
            dossier_id: Target dossier ID
            operation: Operation type (created, fact_added, summary_updated, etc.)
            provenance_id: Unique provenance ID (format: prov_YYYYMMDD_HHMMSS_XXX)
            source_block_id: Optional bridge block that triggered operation
            source_turn_id: Optional turn that triggered operation
            details: Optional JSON with operation-specific details
        
        Returns:
            True if successful, False otherwise
        """
        try:
            cursor = self.conn.cursor()
            now = datetime.now().isoformat()
            cursor.execute("""
                INSERT INTO dossier_provenance 
                (provenance_id, dossier_id, operation, timestamp, 
                 source_block_id, source_turn_id, details)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (provenance_id, dossier_id, operation, now, 
                  source_block_id, source_turn_id, details))
            self.conn.commit()
            logger.info(f"Added provenance: {operation} for dossier {dossier_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to add provenance entry: {e}")
            self.conn.rollback()
            return False
    
    def get_dossier_history(self, dossier_id: str) -> List[Dict[str, Any]]:
        """
        Get provenance history for a dossier.
        
        Args:
            dossier_id: Target dossier ID
        
        Returns:
            List of provenance entries, ordered chronologically
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM dossier_provenance 
            WHERE dossier_id = ? 
            ORDER BY timestamp ASC
        """, (dossier_id,))
        return [dict(row) for row in cursor.fetchall()]
    
    # =========================================================================
    # BLOCK METADATA OPERATIONS (Phase 2 - Sticky Meta Tags)
    # =========================================================================
    
    def save_block_metadata(self, block_id: str, global_tags: List[str], 
                           section_rules: List[Dict]) -> None:
        """
        Save sticky meta tags for a bridge block.
        Tags are stored ONCE per block, not per chunk (pointer model).
        
        Args:
            block_id: Bridge block ID
            global_tags: List of global environment/constraint tags
            section_rules: List of section-specific rules with turn ranges
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO block_metadata 
            (block_id, global_tags, section_rules, created_at)
            VALUES (?, ?, ?, ?)
        """, (
            block_id,
            json.dumps(global_tags),
            json.dumps(section_rules),
            datetime.now().isoformat()
        ))
        self.conn.commit()
        logger.info(f"Saved block metadata for {block_id}: {len(global_tags)} global tags, "
                   f"{len(section_rules)} section rules")
    
    def get_block_metadata(self, block_id: str) -> Dict[str, Any]:
        """
        Retrieve sticky meta tags for a bridge block.
        Used during read-side chunk hydration (group-by-block pattern).
        
        Args:
            block_id: Bridge block ID
            
        Returns:
            Dictionary with global_tags and section_rules
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT global_tags, section_rules 
            FROM block_metadata 
            WHERE block_id = ?
        """, (block_id,))
        
        row = cursor.fetchone()
        if not row:
            return {'global_tags': [], 'section_rules': []}
        
        return {
            'global_tags': json.loads(row[0]) if row[0] else [],
            'section_rules': json.loads(row[1]) if row[1] else []
        }

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            print("✅ Storage connection closed")
    
    def __enter__(self):
        """Context manager support"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager support"""
        self.close()


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("🧪 Storage Layer Test")
    print("=" * 60)
    
    # Create test database
    test_db = "test_memory.db"
    if os.path.exists(test_db):
        os.remove(test_db)
    
    with Storage(test_db) as storage:
        # Test 1: Create days
        print("\n1. Creating day nodes...")
        today = create_day_id()
        day = storage.create_day(today)
        print(f"   ✅ Created day: {day.day_id}")
        
        # Test 2: Add session
        print("\n2. Adding session to day...")
        storage.add_session_to_day(today, "session_test_001")
        print(f"   ✅ Session added")
        
        # Test 3: Add keywords
        print("\n3. Adding keywords...")
        keyword = Keyword(
            keyword="rowing",
            first_mentioned=datetime.now(),
            last_mentioned=datetime.now(),
            frequency=5,
            turn_ids=[1, 3, 5, 7, 9]
        )
        storage.add_keyword(today, keyword)
        print(f"   ✅ Keyword 'rowing' added (frequency: {keyword.frequency})")
        
        # Test 4: Save task
        print("\n4. Creating task...")
        try:
            from hmlr.memory.models import create_task_id
        except ImportError:
            from models import create_task_id
        
        task = TaskState(
            task_id=create_task_id(TaskType.RECURRING_PLAN),
            task_type=TaskType.RECURRING_PLAN,
            status=TaskStatus.ACTIVE,
            created_date=today,
            created_at=datetime.now(),
            last_updated=datetime.now(),
            task_title="30-day rowing challenge",
            original_query="I want to row every day for 30 days",
            total_steps=30,
            completed_steps=7
        )
        storage.save_task(task)
        storage.link_task_to_day(task.task_id, today, "created")
        print(f"   ✅ Task saved: {task.task_title}")
        print(f"   Progress: {task.progress_percentage():.1f}%")
        
        # Test 5: Search keywords
        print("\n5. Searching keywords...")
        results = storage.search_keywords(["rowing"])
        print(f"   ✅ Found {len(results)} matches for 'rowing'")
        
        # Test 6: Get active tasks
        print("\n6. Getting active tasks...")
        active_tasks = storage.get_active_tasks()
        print(f"   ✅ Found {len(active_tasks)} active tasks")
        
        # Test 7: Retrieve day with all data
        print("\n7. Retrieving full day node...")
        retrieved_day = storage.get_day(today)
        print(f"   ✅ Day: {retrieved_day.day_id}")
        print(f"   Sessions: {len(retrieved_day.session_ids)}")
        print(f"   Keywords: {len(retrieved_day.keywords)}")
        
        # Test 8: Stats
        print("\n8. Storage statistics...")
        stats = storage.get_stats()
        print(f"   ✅ Total days: {stats['total_days']}")
        print(f"   ✅ Active tasks: {stats.get('tasks_by_status', {}).get('active', 0)}")
        print(f"   ✅ Unique keywords: {stats['unique_keywords']}")
    
    print("\n" + "=" * 60)
    print("🎉 All storage tests passed!")
    print(f"Test database created: {test_db}")
    print("(You can delete this file after inspection)")

