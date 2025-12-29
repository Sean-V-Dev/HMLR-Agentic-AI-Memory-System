import sqlite3
import json
import logging
from datetime import datetime
from typing import List, Dict, Optional, Any, Tuple

logger = logging.getLogger(__name__)

class LedgerStore:
    """
    Handles Bridge Block and Daily Ledger persistence operations.
    """

    @staticmethod
    def get_active_bridge_blocks(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
        """
        Retrieve all active/paused Bridge Blocks.
        Note: Removed date filter - status='ACTIVE' is sufficient filter for current session blocks.
        """
        cursor = conn.cursor()
        cursor.execute("""
            SELECT block_id, content_json, created_at, status, exit_reason
            FROM daily_ledger
            WHERE status IN ('ACTIVE', 'PAUSED')
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

    @staticmethod
    def get_daily_ledger_metadata(conn: sqlite3.Connection, day_id: str) -> List[Dict[str, Any]]:
        """
        Get metadata summaries of all Bridge Blocks for a specific day.
        Uses a join to ledger_turns for accurate turn counts.
        """
        cursor = conn.cursor()
        cursor.execute("""
            SELECT l.block_id, l.content_json, l.updated_at, l.status, COUNT(t.turn_id) as turn_count
            FROM daily_ledger l
            LEFT JOIN ledger_turns t ON l.block_id = t.block_id
            WHERE DATE(l.created_at) = DATE(?)
            AND l.status IN ('ACTIVE', 'PAUSED')
            GROUP BY l.block_id
            ORDER BY l.updated_at DESC
        """, (day_id,))
        
        metadata_list = []
        for row in cursor.fetchall():
            try:
                content = json.loads(row[1])
                metadata = {
                    'block_id': row[0],
                    'topic_label': content.get('topic_label', 'Unknown Topic'),
                    'summary': content.get('summary', ''),
                    'keywords': content.get('keywords', []),
                    'open_loops': content.get('open_loops', []),
                    'decisions_made': content.get('decisions_made', []),
                    'turn_count': row[4],  # From the COUNT() join
                    'last_updated': row[2],
                    'is_last_active': (row[3] == 'ACTIVE'),
                    'status': row[3]
                }
                metadata_list.append(metadata)
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Failed to extract metadata from block {row[0]}: {e}")
                continue
        
        return metadata_list

    @staticmethod
    def get_bridge_block_full(conn: sqlite3.Connection, block_id: str) -> Optional[Dict[str, Any]]:
        """
        Load complete Bridge Block including turns fetched from normalized table.
        """
        logger.debug(f"get_bridge_block_full() called for block_id={block_id}")
        cursor = conn.cursor()
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
            content['_db_status'] = row[1]
            content['_db_created_at'] = row[2]
            content['_db_updated_at'] = row[3]
            
            # Fetch turns from ledger_turns table
            cursor.execute("""
                SELECT turn_id, timestamp, user_message, assistant_response, metadata_json
                FROM ledger_turns
                WHERE block_id = ?
                ORDER BY timestamp ASC
            """, (block_id,))
            
            logger.debug(f"Querying ledger_turns for block_id={block_id}")
            rows = cursor.fetchall()
            logger.debug(f"Query returned {len(rows)} rows")
            
            turns = []
            for t_row in rows:
                turns.append({
                    'turn_id': t_row[0],
                    'timestamp': t_row[1],
                    'user_message': t_row[2],
                    'ai_response': t_row[3],
                    'metadata': json.loads(t_row[4]) if t_row[4] else {}
                })
            
            content['turns'] = turns
            logger.debug(f"Retrieved block {block_id}: {len(turns)} turns loaded from ledger_turns table")
            return content
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse bridge block {block_id}: {e}")
            return None

    @staticmethod
    def create_new_bridge_block(
        conn: sqlite3.Connection,
        day_id: str,
        topic_label: str,
        keywords: List[str],
        span_id: Optional[str] = None,
        **kwargs
    ) -> Optional[str]:
        """
        Create a new Bridge Block.
        Turns are stored in a separate table, so turns[] is removed from content_json.
        Any additional kwargs are merged into the content JSON for extensibility.
        """
        from uuid import uuid4
        block_id = f"bb_{day_id.replace('-', '')}_{uuid4().hex[:8]}"
        timestamp = datetime.now().isoformat()
        
        content = {
            'block_id': block_id,
            'prev_block_id': None,
            'span_id': span_id,
            'timestamp': timestamp,
            'status': 'ACTIVE',
            'exit_reason': None,
            'topic_label': topic_label,
            'summary': '',
            'user_affect': '',
            'bot_persona': '',
            'open_loops': [],
            'decisions_made': [],
            'active_variables': {},
            'keywords': keywords
        }
        
        # Merge any additional kwargs into content for extensibility
        if kwargs:
            content.update(kwargs)
        
        try:
            cursor = conn.cursor()
            # Extract date from timestamp (YYYY-MM-DD format)
            date_str = timestamp[:10]  # timestamp is "YYYY-MM-DD HH:MM:SS"
            cursor.execute("""
                INSERT INTO daily_ledger (
                    block_id, prev_block_id, span_id, content_json,
                    created_at, updated_at, status, exit_reason, embedding_status, date
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                block_id, None, span_id, json.dumps(content),
                timestamp, timestamp, 'ACTIVE', None, 'PENDING', date_str
            ))
            conn.commit()
            logger.info(f"Created new bridge block: {block_id} (topic: {topic_label}, date: {date_str})")
            return block_id
        except Exception as e:
            logger.error(f"Failed to create bridge block: {e}", exc_info=True)
            conn.rollback()
            return None

    @staticmethod
    def append_turn_to_block(conn: sqlite3.Connection, block_id: str, turn: Dict[str, Any], **kwargs) -> bool:
        """
        Append a new conversation turn to Bridge Block via the ledger_turns table.
        This provides O(1) appends and prevents race conditions on large JSON blobs.
        """
        cursor = conn.cursor()
        
        try:
            # Build metadata from turn dict, preserving chunks and other non-standard fields
            metadata = turn.get('metadata', {})
            
            # Add chunks to metadata if present at top level
            if 'chunks' in turn:
                metadata['chunks'] = turn['chunks']
            
            # Merge kwargs into metadata
            if kwargs:
                metadata.update(kwargs)

            # 1. Insert into normalized ledger_turns table
            cursor.execute("""
                INSERT INTO ledger_turns (
                    turn_id, block_id, timestamp, user_message, 
                    assistant_response, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                turn.get('turn_id'),
                block_id,
                turn.get('timestamp', datetime.now().isoformat()),
                turn.get('user_message', ''),
                turn.get('ai_response') or turn.get('assistant_response', ''),
                json.dumps(metadata)
            ))
            
            # 2. Update updated_at in daily_ledger to signal activity
            cursor.execute("""
                UPDATE daily_ledger
                SET updated_at = ?
                WHERE block_id = ?
            """, (datetime.now().isoformat(), block_id))
            
            conn.commit()
            
            # Verify the write succeeded and force WAL checkpoint
            cursor.execute("PRAGMA wal_checkpoint(PASSIVE);")
            cursor.execute("SELECT COUNT(*) FROM ledger_turns WHERE block_id = ?", (block_id,))
            turn_count = cursor.fetchone()[0]
            logger.debug(f"Appended turn {turn.get('turn_id')} to block {block_id} (verified {turn_count} turns in DB)")
            return True
        except Exception as e:
            logger.error(f"Failed to append turn to block {block_id}: {e}", exc_info=True)
            conn.rollback()
            return False

    @staticmethod
    def update_bridge_block_status(conn: sqlite3.Connection, block_id: str, new_status: str, exit_reason: Optional[str] = None) -> bool:
        """
        Change Bridge Block status between ACTIVE ↔ PAUSED.
        """
        if new_status not in ['ACTIVE', 'PAUSED', 'ARCHIVED']:
            logger.error(f"Invalid status: {new_status}")
            return False
            
        cursor = conn.cursor()
        cursor.execute("SELECT content_json FROM daily_ledger WHERE block_id = ?", (block_id,))
        row = cursor.fetchone()
        if not row:
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
            conn.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to update status for block {block_id}: {e}")
            conn.rollback()
            return False

    @staticmethod
    def update_last_active_flag(conn: sqlite3.Connection, block_id: str) -> bool:
        """
        Set is_last_active flag for specified block.
        """
        cursor = conn.cursor()
        cursor.execute("SELECT created_at FROM daily_ledger WHERE block_id = ?", (block_id,))
        row = cursor.fetchone()
        if not row:
            return False
            
        day_id = row[0][:10]
        try:
            cursor.execute("""
                UPDATE daily_ledger
                SET status = 'PAUSED'
                WHERE DATE(created_at) = DATE(?)
                AND status = 'ACTIVE'
                AND block_id != ?
            """, (day_id, block_id))
            
            cursor.execute("""
                UPDATE daily_ledger
                SET status = 'ACTIVE', updated_at = ?
                WHERE block_id = ?
            """, (datetime.now().isoformat(), block_id))
            conn.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to update last_active flag: {e}")
            conn.rollback()
            return False

    @staticmethod
    def generate_block_summary(conn: sqlite3.Connection, block_id: str) -> Optional[str]:
        """
        Generate summary of Bridge Block.
        Now counts turns from the ledger_turns table.
        """
        cursor = conn.cursor()
        
        # Get metadata
        cursor.execute("SELECT content_json FROM daily_ledger WHERE block_id = ?", (block_id,))
        row = cursor.fetchone()
        if not row:
            return None
            
        # Get turn count from normalized table
        cursor.execute("SELECT COUNT(*) FROM ledger_turns WHERE block_id = ?", (block_id,))
        turn_count = cursor.fetchone()[0]
            
        try:
            content = json.loads(row[0])
            topic = content.get('topic_label', 'Unknown Topic')
            summary = f"{turn_count}-turn discussion about {topic}"
            
            content['summary'] = summary
            cursor.execute("""
                UPDATE daily_ledger
                SET content_json = ?, updated_at = ?
                WHERE block_id = ?
            """, (json.dumps(content), datetime.now().isoformat(), block_id))
            conn.commit()
            return summary
        except Exception as e:
            logger.error(f"Failed to generate summary for block {block_id}: {e}")
            return None

    @staticmethod
    def update_bridge_block_metadata(conn: sqlite3.Connection, block_id: str, metadata: Dict[str, Any]) -> bool:
        """
        Update Bridge Block header metadata.
        """
        cursor = conn.cursor()
        cursor.execute("SELECT content_json FROM daily_ledger WHERE block_id = ?", (block_id,))
        row = cursor.fetchone()
        if not row:
            return False
            
        try:
            content = json.loads(row[0])
            for field in ['topic_label', 'keywords', 'summary', 'open_loops', 'decisions_made', 'user_affect', 'bot_persona']:
                if field in metadata:
                    content[field] = metadata[field]
                    
            cursor.execute("""
                UPDATE daily_ledger
                SET content_json = ?, updated_at = ?
                WHERE block_id = ?
            """, (json.dumps(content), datetime.now().isoformat(), block_id))
            conn.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to update metadata for block {block_id}: {e}")
            conn.rollback()
            return False

    @staticmethod
    def save_block_metadata(conn: sqlite3.Connection, block_id: str, global_tags: List[str], section_rules: List[Dict]) -> None:
        """
        Save sticky meta tags for a bridge block.
        """
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO block_metadata 
            (block_id, global_tags, section_rules, created_at)
            VALUES (?, ?, ?, ?)
        """, (block_id, json.dumps(global_tags), json.dumps(section_rules), datetime.now().isoformat()))
        conn.commit()

    @staticmethod
    def get_block_metadata(conn: sqlite3.Connection, block_id: str) -> Dict[str, Any]:
        """
        Retrieve sticky meta tags for a bridge block.
        """
        cursor = conn.cursor()
        cursor.execute("SELECT global_tags, section_rules FROM block_metadata WHERE block_id = ?", (block_id,))
        row = cursor.fetchone()
        if not row:
            return {'global_tags': [], 'section_rules': []}
        return {
            'global_tags': json.loads(row[0]) if row[0] else [],
            'section_rules': json.loads(row[1]) if row[1] else []
        }

    @staticmethod
    def link_facts_to_block(conn: sqlite3.Connection, turn_id: str, block_id: str) -> int:
        """
        Update facts to link them to the correct Bridge Block.
        """
        cursor = conn.cursor()
        timestamp = turn_id.replace("turn_", "")
        cursor.execute("""
            UPDATE fact_store
            SET source_block_id = ?
            WHERE source_chunk_id LIKE ?
              AND (source_block_id IS NULL OR source_block_id = '')
        """, (block_id, f"%{timestamp}%"))
        count = cursor.rowcount
        conn.commit()
        return count

    @staticmethod
    def save_to_gardened_memory(conn: sqlite3.Connection, chunks: List[Dict[str, Any]], block_id: str, global_tags: List[str]) -> int:
        """
        Save tagged chunks to gardened_memory for long-term retrieval.
        """
        cursor = conn.cursor()
        saved_count = 0
        try:
            for chunk in chunks:
                cursor.execute("""
                    INSERT OR REPLACE INTO gardened_memory (
                        chunk_id, block_id, turn_id, chunk_type,
                        text_content, parent_id, global_tags, token_count
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    chunk.get('chunk_id'),
                    block_id,
                    chunk.get('turn_id', ''),
                    chunk.get('chunk_type', 'turn'),
                    chunk.get('text_verbatim', chunk.get('text_content', '')),
                    chunk.get('parent_chunk_id'),
                    json.dumps(global_tags),
                    chunk.get('token_count', 0)
                ))
                saved_count += 1
            conn.commit()
            return saved_count
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to save chunks to gardened_memory: {e}")
            raise
