import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
import sqlite3

logger = logging.getLogger(__name__)

class DossierStore:
    """
    Logic for managing dossiers, dossier facts, and provenance.
    Extracted from Storage to reduce file size.
    """
    
    @staticmethod
    def create_dossier(conn: sqlite3.Connection, dossier_id: str, title: str, summary: str = "", search_summary: str = "") -> bool:
        try:
            cursor = conn.cursor()
            now = datetime.now().isoformat()
            cursor.execute("""
                INSERT INTO dossiers (dossier_id, title, summary, search_summary, created_at, last_updated)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (dossier_id, title, summary, search_summary, now, now))
            conn.commit()
            logger.info(f"Created dossier: {dossier_id} - {title}")
            return True
        except Exception as e:
            logger.error(f"Failed to create dossier {dossier_id}: {e}")
            conn.rollback()
            return False

    @staticmethod
    def get_dossier(conn: sqlite3.Connection, dossier_id: str) -> Optional[Dict[str, Any]]:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM dossiers WHERE dossier_id = ?", (dossier_id,))
        row = cursor.fetchone()
        return dict(row) if row else None

    @staticmethod
    def get_all_dossiers(conn: sqlite3.Connection, status: str = 'active') -> List[Dict[str, Any]]:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM dossiers WHERE status = ? ORDER BY last_updated DESC", (status,))
        return [dict(row) for row in cursor.fetchall()]

    @staticmethod
    def add_fact_to_dossier(
        conn: sqlite3.Connection,
        dossier_id: str,
        fact_id: str,
        fact_text: str,
        source_block_id: str,
        source_turn_id: str = None,
        fact_type: str = None,
        confidence: float = 1.0
    ) -> bool:
        try:
            cursor = conn.cursor()
            now = datetime.now().isoformat()
            cursor.execute("""
                INSERT INTO dossier_facts 
                (fact_id, dossier_id, fact_text, fact_type, added_at, 
                 source_block_id, source_turn_id, confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (fact_id, dossier_id, fact_text, fact_type, now, 
                  source_block_id, source_turn_id, confidence))
            
            cursor.execute("""
                UPDATE dossiers SET last_updated = ? WHERE dossier_id = ?
            """, (now, dossier_id))
            
            conn.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to add fact to dossier: {e}")
            conn.rollback()
            return False

    @staticmethod
    def get_dossier_facts(conn: sqlite3.Connection, dossier_id: str) -> List[Dict[str, Any]]:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM dossier_facts WHERE dossier_id = ? ORDER BY added_at ASC", (dossier_id,))
        return [dict(row) for row in cursor.fetchall()]

    @staticmethod
    def update_dossier_summary(conn: sqlite3.Connection, dossier_id: str, new_summary: str) -> bool:
        try:
            cursor = conn.cursor()
            now = datetime.now().isoformat()
            cursor.execute("UPDATE dossiers SET summary = ?, last_updated = ? WHERE dossier_id = ?", (new_summary, now, dossier_id))
            conn.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to update dossier summary: {e}")
            conn.rollback()
            return False

    @staticmethod
    def add_provenance_entry(
        conn: sqlite3.Connection,
        dossier_id: str,
        operation: str,
        provenance_id: str,
        source_block_id: str = None,
        source_turn_id: str = None,
        details: str = None
    ) -> bool:
        try:
            cursor = conn.cursor()
            now = datetime.now().isoformat()
            cursor.execute("""
                INSERT INTO dossier_provenance 
                (provenance_id, dossier_id, operation, timestamp, 
                 source_block_id, source_turn_id, details)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (provenance_id, dossier_id, operation, now, 
                  source_block_id, source_turn_id, details))
            conn.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to add provenance entry: {e}")
            conn.rollback()
            return False

    @staticmethod
    def get_dossier_history(conn: sqlite3.Connection, dossier_id: str) -> List[Dict[str, Any]]:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM dossier_provenance WHERE dossier_id = ? ORDER BY timestamp ASC", (dossier_id,))
        return [dict(row) for row in cursor.fetchall()]
