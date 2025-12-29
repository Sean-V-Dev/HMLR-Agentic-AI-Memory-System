"""
Stateless Sliding Window for HMLR.

This module provides a query-based view of the conversation history,
ensuring consistency across multiple workers and LangGraph nodes.
"""

import logging
from typing import List, Dict, Optional, Any, Set
from datetime import datetime
from hmlr.memory.models import ConversationTurn

logger = logging.getLogger(__name__)

class SlidingWindow:
    """
    Query-based Sliding Window.
    Stateless: fetches data from Storage on demand.
    """
    
    def __init__(self, storage=None, session_id: str = None, max_turns: int = 25):
        self.storage = storage
        self.session_id = session_id
        self.max_turns = max_turns
        
        # Transient sets for the current processing turn (deduplication)
        # These are request-scoped and cleared between requests via clear_transient_state()
        self._loaded_turn_ids: Set[str] = set()
        self._loaded_keyword_ids: Set[str] = set()
        self._loaded_task_ids: Set[str] = set()
        self._loaded_summary_ids: Set[str] = set()

    def set_session(self, session_id: str):
        """Set the active session for the window."""
        self.session_id = session_id
        # Reset transient state for new session
        self.clear_transient_state()

    def clear_transient_state(self):
        """Clear session-level tracking (deduplication sets)."""
        self._loaded_turn_ids.clear()
        self._loaded_keyword_ids.clear()
        self._loaded_task_ids.clear()
        self._loaded_summary_ids.clear()

    @property
    def turns(self) -> List[ConversationTurn]:
        """Fetch the most recent turns for the active session.
        
        Always queries the database to ensure consistency across
        multiple workers and LangGraph nodes. No caching.
        """
        if not self.storage or not self.session_id:
            return []
        
        return self.storage.get_session_history(self.session_id, limit=self.max_turns)

    def add_turn(self, turn: ConversationTurn) -> None:
        """
        Placeholder for compatibility. 
        In the stateless model, turns are added via the database (ConversationManager.log_turn).
        """
        # We don't need to append to an in-memory list anymore.
        # But we might want to update transient tracking if needed.
        self.mark_loaded(turn.turn_id)
        if turn.summary_id:
            self.mark_loaded(turn.summary_id)
        for kid in turn.keyword_ids:
            self.mark_loaded(kid)
        for tid in turn.task_updated_ids:
            self.mark_loaded(tid)

    def get_turn(self, turn_id: str) -> Optional[ConversationTurn]:
        """Fetch a specific turn from the database."""
        if not self.storage:
            return None
        return self.storage.get_turn_by_id(turn_id)

    def is_in_window(self, item_id: str) -> bool:
        """
        Check if item is CURRENTLY in the window or was loaded this turn.
        """
        if item_id in self._loaded_turn_ids or \
           item_id in self._loaded_keyword_ids or \
           item_id in self._loaded_task_ids or \
           item_id in self._loaded_summary_ids:
            return True
            
        # Check if it's in the actual turns currently in the window
        turns = self.turns
        for turn in turns:
            if turn.turn_id == item_id or \
               turn.summary_id == item_id or \
               item_id in turn.keyword_ids or \
               item_id in turn.task_updated_ids or \
               turn.task_created_id == item_id:
                return True
        return False

    def is_recently_seen(self, item_id: str) -> bool:
        """
        Check if item was in the previous page of history (recently pruned).
        """
        if not self.storage or not self.session_id:
            return False
            
        # Query turns 26-50
        recent = self.storage.get_session_history(self.session_id, offset=self.max_turns, limit=self.max_turns)
        for turn in recent:
            if turn.turn_id == item_id or \
               turn.summary_id == item_id or \
               item_id in turn.keyword_ids or \
               item_id in turn.task_updated_ids:
                return True
        return False

    def mark_loaded(self, item_id: str) -> None:
        """Mark an item as 'loaded' (blocks retrieval in current turn)."""
        # We use prefix checking for simplicity if id_generator isn't available
        if item_id.startswith('t_'):
            self._loaded_turn_ids.add(item_id)
        elif item_id.startswith('k'):
            self._loaded_keyword_ids.add(item_id)
        elif item_id.startswith('tsk_'):
            self._loaded_task_ids.add(item_id)
        elif item_id.startswith('s_'):
            self._loaded_summary_ids.add(item_id)

    def get_loaded_topics(self) -> List[str]:
        topics = set()
        for turn in self.turns:
            topics.update(turn.active_topics)
        return list(topics)

    def is_topic_active(self, topic: str) -> bool:
        return topic in self.get_loaded_topics()

    def clear(self) -> None:
        """Clear transient state. Does NOT delete turns from DB."""
        self.clear_transient_state()
        logger.info(f"Sliding window transient state cleared for session {self.session_id}")

    @staticmethod
    def load_from_file(filepath: str = None) -> 'SlidingWindow':
        """Compatibility method - now returns a fresh stateless window."""
        return SlidingWindow()

    def save_to_file(self, filepath: str = None) -> None:
        """No-op for stateless window."""
        pass
