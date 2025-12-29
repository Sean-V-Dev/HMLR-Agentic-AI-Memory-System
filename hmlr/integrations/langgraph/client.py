"""
HMLR Client Manager for LangGraph

Manages HMLR client instances to avoid re-initialization on every call.
"""

import logging
import threading
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class HMLRClientManager:
    """
    Manages HMLR client lifecycle for LangGraph nodes.
    
    This ensures we don't re-initialize HMLR components on every node call,
    which would be expensive (embedding models, DB connections, etc.)
    
    Usage:
        manager = HMLRClientManager()
        engine = manager.get_engine(config)
        
    Thread Safety:
        This class is thread-safe. Multiple LangGraph workers can share
        the same manager instance.
    """
    
    _instance: Optional['HMLRClientManager'] = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Singleton pattern - only one manager per process."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self._engines: Dict[str, Any] = {}  # key -> ConversationEngine
        self._components: Dict[str, Any] = {}  # key -> ComponentBundle
        self._engine_lock = threading.Lock()
        self._initialized = True
        logger.info("HMLRClientManager initialized")
    
    def _make_key(self, config: dict) -> str:
        """Create a cache key from config."""
        db_path = config.get("hmlr_db_path", "default")
        profile_path = config.get("hmlr_profile_path", "default")
        return f"{db_path}::{profile_path}"
    
    def get_engine(self, config: dict = None, raise_on_error: bool = True):
        """
        Get or create a ConversationEngine for the given config.
        
        Args:
            config: LangGraph config dict with optional keys:
                - hmlr_db_path: Path to HMLR database
                - hmlr_profile_path: Path to user profile
                - openai_api_key: API key (or uses env)
            raise_on_error: If True, propagate exceptions instead of
                          returning error responses
        
        Returns:
            ConversationEngine instance
        """
        config = config or {}
        key = self._make_key(config)
        
        with self._engine_lock:
            if key not in self._engines:
                logger.info(f"Creating new HMLR engine for config: {key}")
                self._engines[key] = self._create_engine(config, raise_on_error)
            return self._engines[key]
    
    def get_components(self, config: dict = None):
        """
        Get or create ComponentBundle for the given config.
        
        Useful for health checking before running the graph.
        """
        config = config or {}
        key = self._make_key(config)
        
        with self._engine_lock:
            if key not in self._components:
                self._create_engine(config, raise_on_error=False)
            return self._components.get(key)
    
    def _create_engine(self, config: dict, raise_on_error: bool):
        """Create HMLR components and engine."""
        import os
        from hmlr.core.component_factory import ComponentFactory
        
        # Apply config to environment if provided
        if config.get("hmlr_db_path"):
            os.environ["COGNITIVE_LATTICE_DB"] = config["hmlr_db_path"]
        if config.get("hmlr_profile_path"):
            os.environ["USER_PROFILE_PATH"] = config["hmlr_profile_path"]
        
        # Create components
        factory = ComponentFactory()
        api_key = config.get("openai_api_key")
        components = factory.create_all_components(api_key=api_key)
        
        # Cache components for health checking
        key = self._make_key(config)
        self._components[key] = components
        
        # Log health status
        if not components.is_fully_operational():
            degraded = components.get_degraded_components()
            logger.warning(f"HMLR components degraded: {degraded}")
        else:
            logger.info("HMLR components healthy")
        
        # Create engine
        engine = ComponentFactory.create_conversation_engine(
            components, 
            raise_on_error=raise_on_error
        )
        
        return engine
    
    def clear_cache(self):
        """Clear cached engines (useful for testing)."""
        with self._engine_lock:
            self._engines.clear()
            self._components.clear()
            logger.info("HMLR engine cache cleared")
    
    def is_healthy(self, config: dict = None) -> bool:
        """Check if HMLR is fully operational for given config."""
        components = self.get_components(config)
        if components is None:
            return False
        return components.is_fully_operational()
    
    def get_degraded_components(self, config: dict = None) -> list:
        """Get list of degraded component names."""
        components = self.get_components(config)
        if components is None:
            return ["all"]
        return components.get_degraded_components()


# Global singleton instance
_manager: Optional[HMLRClientManager] = None


def get_client_manager() -> HMLRClientManager:
    """Get the global HMLRClientManager instance."""
    global _manager
    if _manager is None:
        _manager = HMLRClientManager()
    return _manager
