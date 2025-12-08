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
            db_path="memory.db",
            model="gpt-4.1-mini"  # ONLY tested model!
        )
        
        response = await client.chat("Tell me about Python")
        print(response["content"])
        ```
    
    WARNING: Only tested with gpt-4.1-mini. Other models may fail.
    """
    
    def __init__(
        self,
        api_key: str,
        db_path: str = "hmlr_memory.db",
        model: str = "gpt-4.1-mini",
        use_llm_intent_mode: bool = False,
        context_budget_tokens: int = 12000,
        max_sliding_window_turns: int = 15,
        crawler_recency_weight: float = 0.3,
        **kwargs
    ):
        """
        Initialize HMLR client.
        
        Args:
            api_key: OpenAI API key
            db_path: Path to SQLite database for memory storage
            model: OpenAI model to use (default: gpt-4.1-mini)
                   WARNING: Only gpt-4.1-mini has been tested!
            use_llm_intent_mode: Use LLM for intent detection (default: False)
            context_budget_tokens: Max tokens for context (default: 12000)
            max_sliding_window_turns: Max turns in sliding window (default: 15)
            crawler_recency_weight: Weight for recent turns in retrieval (default: 0.3)
            **kwargs: Additional configuration options
        """
        # Model compatibility warning
        if model != "gpt-4.1-mini":
            warnings.warn(
                f"⚠️  Model '{model}' has NOT been tested!\n"
                f"HMLR is only validated with 'gpt-4.1-mini'.\n"
                f"Other models may produce incorrect results or fail completely.\n"
                f"Use at your own risk!",
                UserWarning,
                stacklevel=2
            )
        
        # Set API key in environment
        os.environ["OPENAI_API_KEY"] = api_key
        
        # Store configuration
        self.db_path = db_path
        self.model = model
        
        # Initialize components
        print(f"🏗️  Initializing HMLR with {model}...")
        self.components = ComponentFactory.create_all_components(
            db_path=db_path,
            use_llm_intent_mode=use_llm_intent_mode,
            context_budget_tokens=context_budget_tokens,
            max_sliding_window_turns=max_sliding_window_turns,
            crawler_recency_weight=crawler_recency_weight,
            **kwargs
        )
        
        # Create conversation engine
        self.engine = ComponentFactory.create_conversation_engine(
            self.components
        )
        
        print(f"✅ HMLR initialized successfully")
    
    async def chat(
        self,
        message: str,
        force_intent: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Send a message and get a response with memory.
        
        Args:
            message: User's message
            force_intent: Override intent detection (optional)
            **kwargs: Additional parameters for conversation engine
        
        Returns:
            Response dictionary with:
            - content: The AI's response text
            - status: Response status (success/error)
            - metadata: Additional information about the response
        """
        response = await self.engine.process_user_message(
            message,
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
        storage = self.components.storage
        
        # Get total turns (try to get all, fall back to estimate)
        try:
            all_turns = storage.get_recent_turns(day_id=None, limit=100000)
            total_turns = len(all_turns)
        except Exception:
            total_turns = "unknown"
        
        return {
            "total_turns": total_turns,
            "sliding_window_size": len(self.components.sliding_window.turns),
            "db_path": self.db_path,
            "model": self.model
        }
    
    def get_recent_conversations(self, limit: int = 10) -> list:
        """
        Get recent conversation turns.
        
        Args:
            limit: Maximum number of turns to retrieve
        
        Returns:
            List of recent conversation turns
        """
        storage = self.components.storage
        return storage.get_recent_turns(day_id=None, limit=limit)
    
    def clear_sliding_window(self):
        """
        Clear the sliding window (keeps database intact).
        
        This only clears the in-memory sliding window, not the
        permanent storage. Use this to start fresh while keeping history.
        """
        self.components.sliding_window.turns.clear()
        print("🧹 Sliding window cleared")
    
    def close(self):
        """
        Close the client and cleanup resources.
        
        Always call this when done to ensure proper cleanup.
        """
        # Close storage connection
        if hasattr(self.components.storage, 'close'):
            self.components.storage.close()
        
        print("👋 HMLR client closed")
    
    def __enter__(self):
        """Support context manager protocol."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Support context manager protocol."""
        self.close()
        return False
