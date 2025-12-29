import sqlite3
import logging

logger = logging.getLogger(__name__)


def initialize_database(conn: sqlite3.Connection):
    """
    Apply DDL and migrations to the provided SQLite connection.
    Delegates to specialized functions for DDL and migrations.
    """
    _create_tables(conn)
    _run_migrations(conn)

def _create_tables(conn: sqlite3.Connection):
    """
    Execute strict DDL to create tables if they don't exist.
    """
    cursor = conn.cursor()
    
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
    
    # === METADATA STAGING TABLE (pre-synthesis) ===
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
    
    # === SUMMARIES TABLE ===
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
    
    # === KEYWORDS TABLE ===
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
    
    # === AFFECT TABLE ===
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
    
    # === SPANS TABLE ===
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

    # === DAILY LEDGER TABLE (Bridge Blocks) ===
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
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_ledger_status ON daily_ledger(status)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_ledger_date ON daily_ledger(created_at)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_ledger_span ON daily_ledger(span_id)")
    
    # === LEDGER TURNS TABLE (Normalized Bridge Block Turns) ===
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS ledger_turns (
            turn_id TEXT PRIMARY KEY,
            block_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            user_message TEXT NOT NULL,
            assistant_response TEXT NOT NULL,
            metadata_json TEXT,
            FOREIGN KEY (block_id) REFERENCES daily_ledger(block_id) ON DELETE CASCADE
        )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_ledger_turns_block ON ledger_turns(block_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_ledger_turns_time ON ledger_turns(timestamp)")

    # === FACT STORE TABLE ===
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
            source_turn_id TEXT,
            evidence_snippet TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (source_span_id) REFERENCES spans(span_id)
        )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_fact_key ON fact_store(key)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_fact_span ON fact_store(source_span_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_fact_block ON fact_store(source_block_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_fact_turn ON fact_store(source_turn_id)")

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
    
    # === EMBEDDINGS TABLE ===
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
    
    # === DOSSIER SYSTEM TABLES ===
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
    
    # === BLOCK METADATA TABLE ===
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS block_metadata (
            block_id TEXT PRIMARY KEY,
            global_tags TEXT,
            section_rules TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (block_id) REFERENCES daily_ledger(block_id) ON DELETE CASCADE
        )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_block_metadata_block ON block_metadata(block_id)")
    
    # === GARDENED MEMORY TABLE ===
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS gardened_memory (
            chunk_id TEXT PRIMARY KEY,
            block_id TEXT NOT NULL,
            turn_id TEXT NOT NULL,
            chunk_type TEXT NOT NULL,
            text_content TEXT NOT NULL,
            parent_id TEXT,
            global_tags TEXT,
            token_count INTEGER,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (block_id) REFERENCES daily_ledger(block_id)
        )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_gardened_block ON gardened_memory(block_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_gardened_turn ON gardened_memory(turn_id)")
    
    conn.commit()

def _run_migrations(conn: sqlite3.Connection):
    """
    Execute necessary migrations for existing databases.
    """
    cursor = conn.cursor()
    
    # Migration: Add updated_at if missing (daily_ledger)
    try:
        cursor.execute("SELECT updated_at FROM daily_ledger LIMIT 1")
    except sqlite3.OperationalError:
        logger.info("Migrating daily_ledger: Adding updated_at column")
        cursor.execute("ALTER TABLE daily_ledger ADD COLUMN updated_at TEXT")

    # Migration: Add date if missing (daily_ledger)
    try:
        cursor.execute("SELECT date FROM daily_ledger LIMIT 1")
    except sqlite3.OperationalError:
        logger.info("Migrating daily_ledger: Adding date column")
        cursor.execute("ALTER TABLE daily_ledger ADD COLUMN date TEXT")
        # Populate existing rows with created_at date (extract YYYY-MM-DD from timestamp)
        cursor.execute("""
            UPDATE daily_ledger 
            SET date = substr(created_at, 1, 10) 
            WHERE date IS NULL
        """)

    # Migration: Add source_turn_id if missing (fact_store)
    try:
        cursor.execute("SELECT source_turn_id FROM fact_store LIMIT 1")
    except sqlite3.OperationalError:
        logger.info("Migrating fact_store: Adding source_turn_id column")
        cursor.execute("ALTER TABLE fact_store ADD COLUMN source_turn_id TEXT")
    
    # Migration: Add search_summary to dossiers
    try:
        cursor.execute("SELECT search_summary FROM dossiers LIMIT 1")
    except sqlite3.OperationalError:
        logger.info("Migrating dossiers: Adding search_summary column")
        cursor.execute("ALTER TABLE dossiers ADD COLUMN search_summary TEXT")

    # Migration: Add turn_id to gardened_memory
    try:
        cursor.execute("SELECT turn_id FROM gardened_memory LIMIT 1")
    except sqlite3.OperationalError:
        logger.info("Migrating gardened_memory: Adding turn_id column")
        cursor.execute("ALTER TABLE gardened_memory ADD COLUMN turn_id TEXT")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_gardened_turn ON gardened_memory(turn_id)")
    
    conn.commit()
