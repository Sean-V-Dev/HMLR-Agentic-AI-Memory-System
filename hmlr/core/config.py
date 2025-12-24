"""
HMLR Configuration - Environment and API settings

Note: Model configuration (model names, token budgets, temperature) has been
moved to model_config.py for better organization. Import from there:
    from hmlr.core.model_config import model_config
"""

import os
from typing import Optional


class HMLRConfig:
    """
    Centralized configuration for HMLR system.
    Loads from environment variables with sensible defaults.
    
    Scope: API providers, file paths, database settings
    For model config: See hmlr.core.model_config
    """
    # API Providers
    API_PROVIDER = os.getenv("API_PROVIDER", "openai")

    # File Paths
    # DEBUG_LLM_FLOW_PATH removed - debug file writes eliminated (use logger.debug instead)
    DB_PATH = os.getenv("COGNITIVE_LATTICE_DB")


# Global singleton instance
config = HMLRConfig()


