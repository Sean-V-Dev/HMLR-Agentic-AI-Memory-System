"""
HMLR Client - Main public API for HMLR memory system.

This provides a clean, user-friendly wrapper around the internal
component factory and conversation engine.
"""

import os
import warnings
from typing import Optional, Dict, Any
from .core.component_factory import ComponentFactory


class HMLRClient:
    """
    Main client for HMLR memory system.
    
    Example:
        ```python
        from hmlr import HMLRClient
        
        client = HMLRClient(
            api_key="your-openai-key",
            db_path="memory.db"
        )
        
        response = await client.chat("Tell me about Python")
        print(response["content"])
        ```
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        db_path: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize HMLR client.
        
        Args:
            api_key: Optional API key. If not provided, system uses environment variables.
            db_path: Optional path to SQLite database. If not provided, uses default.
            **kwargs: Additional configuration options (unused)
        """
        from hmlr.core.model_config import model_config
        model_name = model_config.get_main_model()
        
        # Initialize components using centralized config AND injected secrets
        # print(f"🏗️  Initializing HMLR with {model_name}...")
        self.components = ComponentFactory.create_all_components(
            api_key=api_key,
            db_path=db_path
        )
        
        self.db_path = self.components.storage.db_path
        self.model = model_name
        
        # Create conversation engine
        self.engine = ComponentFactory.create_conversation_engine(
            self.components
        )
        
        # print(f"✅ HMLR initialized successfully")
    
    async def chat(
        self,
        message: str,
        session_id: str = "default_session",
        force_intent: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Send a message and get a response with memory.
        
        Args:
            message: User's message
            session_id: Unique identifier for the conversation session (NEW)
            force_intent: Override intent detection (optional)
            **kwargs: Additional parameters for conversation engine
        
        Returns:
            Response dictionary
        """
        response = await self.engine.process_user_message(
            message,
            session_id=session_id,
            force_intent=force_intent,
            **kwargs
        )
        
        return {
            "content": response.content,
            "status": response.status.value,
            "metadata": response.metadata if hasattr(response, 'metadata') else {}
        }
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the memory system.
        
        Returns:
            Dictionary with memory statistics:
            - total_turns: Total conversation turns stored
            - sliding_window_size: Current sliding window size
            - db_path: Path to database file
            - model: Model being used
        """
        stats = self.engine.get_memory_stats()
        stats["db_path"] = self.db_path
        return stats
    
    def get_recent_conversations(self, limit: int = 10) -> list:
        """
        Get recent conversation turns.
        
        Args:
            limit: Maximum number of turns to retrieve
        
        Returns:
            List of recent conversation turns
        """
        return self.engine.get_recent_turns(limit=limit)
    
    def clear_sliding_window(self):
        """
        Clear the sliding window transient state.
        
        In the stateless model, this only clears deduplication sets and caches
        for the current session.
        """
        self.engine.clear_session_state()
        # print("🧹 Sliding window transient state cleared")
    
    def close(self):
        """
        Close the client and cleanup resources.
        
        Always call this when done to ensure proper cleanup.
        """
        # Close storage connection
        if hasattr(self.components.storage, 'close'):
            self.components.storage.close()
        
        # print("👋 HMLR client closed")
    
    def __enter__(self):
        """Support context manager protocol."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Support context manager protocol."""
        self.close()
        return False
