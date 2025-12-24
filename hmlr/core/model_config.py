"""
Model Configuration for CognitiveLattice
Centralized model names, token limits, temperature, and API parameters.

Design Philosophy:
- Hierarchical configuration: Global defaults + operation-specific overrides
- Token budgets are explicit (no fallbacks, each operation has unique needs)
- Models and temperature use fallback chains for flexibility
- Environment variables override all defaults
- Multi-provider support: Works with OpenAI ✅, Google Gemini ✅, xAI Grok ✅, Anthropic Claude ✅
- Coming Soon: DeepSeek ⚠️, Local Ollama ⚙️

Usage:
    from hmlr.core.model_config import model_config
    
    # Get models (with automatic fallback)
    model = model_config.get_main_model()
    
    # Get temperature (with automatic fallback)
    temp = model_config.get_main_temperature()
    
    # Get token budgets (explicit values)
    max_tokens = model_config.MAX_RESPONSE_TOKENS

Multi-Provider Examples:
    # OpenAI (✅ fully supported)
    export API_PROVIDER="openai"
    export OPENAI_API_KEY="sk-..."
    export HMLR_DEFAULT_MODEL="gpt-4.1-mini"
    
    # Google Gemini (✅ fully supported)
    export API_PROVIDER="gemini"
    export GEMINI_API_KEY="..."
    export HMLR_DEFAULT_MODEL="gemini-2.0-flash-thinking-exp"
    
    # xAI Grok (✅ fully supported)
    export API_PROVIDER="grok"
    export XAI_API_KEY="..."
    export HMLR_DEFAULT_MODEL="grok-2-latest"
    
    # Anthropic Claude (✅ fully supported)
    export API_PROVIDER="anthropic"
    export ANTHROPIC_API_KEY="sk-ant-..."
    export HMLR_DEFAULT_MODEL="claude-3-5-sonnet-20241022"
    
    # Mix providers (requires setting API_PROVIDER per operation, not currently supported)
    # Future: Could add MAIN_PROVIDER, WORKER_PROVIDER for multi-provider mixing
"""

import os
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class ModelConfig:
    """
    Centralized configuration for all LLM and embedding models.
    
    Design principle: Hierarchical configuration with global default + operation-specific overrides.
    - Change DEFAULT_MODEL to affect the entire stack
    - Override specific operations (LATTICE_MODEL, SYNTHESIS_MODEL) for fine-grained control
    - Environment variables can set either global or specific overrides
    
    Example use cases:
    - Fast/cheap everywhere: Set DEFAULT_MODEL="gpt-4.1-mini"
    - Mix fast workers + thinking endpoint: DEFAULT_MODEL="gpt-4.1-mini", MAIN_MODEL="gpt-4.1-pro"
    - Different providers: Mix OpenAI, Google, Anthropic models per operation
    """
    
    # ===== GLOBAL DEFAULT MODEL =====
    # Change this to affect ALL operations that don't have explicit overrides
    # 
    # Provider-specific model names (✅ = fully supported, ⚠️ = needs implementation):
    # - OpenAI ✅: "gpt-4.1-mini", "gpt-4.1-pro", "gpt-4o", "gpt-4o-mini", "o1", "o1-mini"
    # - Google ✅: "gemini-2.0-flash-thinking-exp", "gemini-1.5-pro", "gemini-1.5-flash"
    # - xAI ✅: "grok-2-latest", "grok-beta"
    # - Anthropic ⚠️: "claude-3-5-sonnet-20241022", "claude-3-7-sonnet-20250219" (not yet in external_api_client.py)
    # - DeepSeek ⚠️: "deepseek-chat", "deepseek-reasoner" (not yet in external_api_client.py)
    # - Local ⚙️: "ollama/llama3.2:3b", "ollama/mistral:7b" (via OpenAI-compatible API)
    #
    # Note: Set API_PROVIDER in config.py to match your model choice
    # To add new providers: Update external_api_client.py with provider-specific API calls
    DEFAULT_MODEL: str = os.getenv("HMLR_DEFAULT_MODEL", "gpt-4.1-mini")
    
    # ===== OPERATION-SPECIFIC MODELS (None = use DEFAULT_MODEL) =====
    # Main conversation model (user-facing responses)
    MAIN_MODEL: Optional[str] = os.getenv("HMLR_MAIN_MODEL")
    
    # Lightweight model for metadata extraction
    NANO_MODEL: Optional[str] = os.getenv("HMLR_NANO_MODEL")
    
    # Model for topic classification and filtering (Lattice operations)
    LATTICE_MODEL: Optional[str] = os.getenv("HMLR_LATTICE_MODEL")
    
    # Model for dossier synthesis and fact processing
    SYNTHESIS_MODEL: Optional[str] = os.getenv("HMLR_SYNTHESIS_MODEL")
    
    # ===== TOKEN BUDGETS =====
    # Note: These are operation-specific and don't have a "default" fallback
    # because each operation has unique requirements. Change individually as needed.
    
    # Global context budget (total tokens available for context assembly)
    CONTEXT_BUDGET_TOKENS: int = int(os.getenv("CONTEXT_BUDGET_TOKENS", "6000"))
    
    # Maximum tokens for LLM response
    MAX_RESPONSE_TOKENS: int = int(os.getenv("MAX_RESPONSE_TOKENS", "2000"))
    
    # Fact extraction response limit - 2000 tokens sufficient with auto-chunking at 10k
    FACT_EXTRACTION_MAX_TOKENS: int = int(os.getenv("FACT_EXTRACTION_MAX_TOKENS", "2000"))
    
    # User profile context token budget (compact summary)
    USER_PROFILE_MAX_TOKENS: int = int(os.getenv("USER_PROFILE_MAX_TOKENS", "300"))
    
    # Context hydrator budget (large for comprehensive retrieval)
    HYDRATOR_MAX_TOKENS: int = int(os.getenv("HYDRATOR_MAX_TOKENS", "50000"))
    
    # ===== EMBEDDING MODELS =====
    # Sentence transformer model for generating embeddings
    # Decision: Standardized on bge-large-en-v1.5 for better quality
    EMBEDDING_MODEL_NAME: str = os.getenv(
        "HMLR_EMBEDDING_MODEL", 
        "BAAI/bge-large-en-v1.5"  # 1024D, high quality
    )
    
    # Embedding dimension (must match model)
    EMBEDDING_DIMENSION: int = int(os.getenv("HMLR_EMBEDDING_DIM", "1024"))
    
    # ===== TEMPERATURE SETTINGS (Hierarchical like models) =====
    # Default temperature for all operations (if not overridden)
    DEFAULT_TEMPERATURE: float = float(os.getenv("HMLR_DEFAULT_TEMPERATURE", "0.1"))
    
    # Operation-specific overrides (None = use DEFAULT_TEMPERATURE)
    # Main conversation temperature (user-facing, can be higher for better UX)
    MAIN_TEMPERATURE: Optional[float] = (
        float(os.getenv("HMLR_MAIN_TEMPERATURE")) 
        if os.getenv("HMLR_MAIN_TEMPERATURE") else None
    )
    
    # Worker operation temperatures (fact scrubber, synthesis, lattice)
    # Generally should stay low (0.1) for deterministic results
    WORKER_TEMPERATURE: Optional[float] = (
        float(os.getenv("HMLR_WORKER_TEMPERATURE"))
        if os.getenv("HMLR_WORKER_TEMPERATURE") else None
    )
    
    # ===== ADVANCED MODEL PARAMETERS (Optional, for thinking models) =====
    # These parameters are only used by specific models and are optional
    
    # Thinking/Reasoning effort (for models like o1, o3, Gemini thinking)
    # - OpenAI o1/o3: "low", "medium", "high" 
    # - Gemini thinking: Integer 1-10 (thinking budget)
    # If None, model uses its default behavior
    DEFAULT_REASONING_EFFORT: Optional[str] = os.getenv("HMLR_DEFAULT_REASONING_EFFORT")
    
    # Override reasoning effort for main conversation only
    MAIN_REASONING_EFFORT: Optional[str] = os.getenv("HMLR_MAIN_REASONING_EFFORT")
    
    # Top P (nucleus sampling) - typically 0.0-1.0
    # Controls diversity of responses (lower = more focused)
    DEFAULT_TOP_P: Optional[float] = (
        float(os.getenv("HMLR_DEFAULT_TOP_P"))
        if os.getenv("HMLR_DEFAULT_TOP_P") else None
    )
    
    # Top K (for providers that support it, like Gemini)
    # Limits sampling to top K tokens
    DEFAULT_TOP_K: Optional[int] = (
        int(os.getenv("HMLR_DEFAULT_TOP_K"))
        if os.getenv("HMLR_DEFAULT_TOP_K") else None
    )
    
    # Frequency penalty (reduces repetition, -2.0 to 2.0)
    DEFAULT_FREQUENCY_PENALTY: Optional[float] = (
        float(os.getenv("HMLR_DEFAULT_FREQUENCY_PENALTY"))
        if os.getenv("HMLR_DEFAULT_FREQUENCY_PENALTY") else None
    )
    
    DEFAULT_PRESENCE_PENALTY: Optional[float] = (
        float(os.getenv("HMLR_DEFAULT_PRESENCE_PENALTY"))
        if os.getenv("HMLR_DEFAULT_PRESENCE_PENALTY") else None
    )
    
    # ===== INTENT & RETRIEVAL =====
    # Use LLM for intent detection (True) or keyword-based heuristics (False)
    USE_LLM_INTENT_MODE: bool = os.getenv("HMLR_USE_LLM_INTENT_MODE", "False").lower() == "true"
    
    # Weight for recency in crawler retrieval (0.0 to 1.0)
    CRAWLER_RECENCY_WEIGHT: float = float(os.getenv("HMLR_CRAWLER_RECENCY_WEIGHT", "0.5"))
    
    # Minimum similarity threshold for semantic retrieval
    MIN_SIMILARITY_THRESHOLD: float = float(os.getenv("HMLR_MIN_SIMILARITY", "0.4"))
    
    # Default score for retrieved candidates
    DEFAULT_CANDIDATE_SCORE: float = float(os.getenv("HMLR_DEFAULT_SCORE", "1.0"))
    
    # ===== SLIDING WINDOW =====
    SLIDING_WINDOW_SIZE: int = int(os.getenv("SLIDING_WINDOW_SIZE", "20"))
    
    # ===== MODEL RESOLUTION METHODS =====
    @classmethod
    def get_main_model(cls) -> str:
        """Get model for main conversation (falls back to DEFAULT_MODEL)"""
        return cls.MAIN_MODEL or cls.DEFAULT_MODEL
    
    @classmethod
    def get_nano_model(cls) -> str:
        """Get model for metadata extraction (falls back to DEFAULT_MODEL)"""
        return cls.NANO_MODEL or cls.DEFAULT_MODEL
    
    @classmethod
    def get_lattice_model(cls) -> str:
        """Get model for lattice operations (falls back to DEFAULT_MODEL)"""
        return cls.LATTICE_MODEL or cls.DEFAULT_MODEL
    
    @classmethod
    def get_synthesis_model(cls) -> str:
        """Get model for synthesis/dossier operations (falls back to DEFAULT_MODEL)"""
        return cls.SYNTHESIS_MODEL or cls.DEFAULT_MODEL
    
    @classmethod
    def get_main_temperature(cls) -> float:
        """Get temperature for main conversation (falls back to DEFAULT_TEMPERATURE)"""
        return cls.MAIN_TEMPERATURE if cls.MAIN_TEMPERATURE is not None else cls.DEFAULT_TEMPERATURE
    
    @classmethod
    def get_worker_temperature(cls) -> float:
        """Get temperature for worker operations (falls back to DEFAULT_TEMPERATURE)"""
        return cls.WORKER_TEMPERATURE if cls.WORKER_TEMPERATURE is not None else cls.DEFAULT_TEMPERATURE
    
    @classmethod
    def get_reasoning_effort(cls, operation: str = "default") -> Optional[str]:
        """
        Get reasoning effort for thinking models.
        
        Args:
            operation: "main" for user-facing, "default" for workers
            
        Returns:
            Reasoning effort string or None (model uses default)
        """
        if operation == "main" and cls.MAIN_REASONING_EFFORT is not None:
            return cls.MAIN_REASONING_EFFORT
        return cls.DEFAULT_REASONING_EFFORT
    
    @classmethod
    def get_advanced_params(cls) -> Dict[str, Any]:
        """
        Get all advanced model parameters as a dictionary.
        Only includes non-None values.
        
        Returns:
            Dict of advanced parameters (empty if none set)
        """
        params = {}
        
        if cls.DEFAULT_TOP_P is not None:
            params['top_p'] = cls.DEFAULT_TOP_P
        
        if cls.DEFAULT_TOP_K is not None:
            params['top_k'] = cls.DEFAULT_TOP_K
        
        if cls.DEFAULT_FREQUENCY_PENALTY is not None:
            params['frequency_penalty'] = cls.DEFAULT_FREQUENCY_PENALTY
        
        if cls.DEFAULT_PRESENCE_PENALTY is not None:
            params['presence_penalty'] = cls.DEFAULT_PRESENCE_PENALTY
        
        return params
    
    @classmethod
    def to_dict(cls) -> Dict[str, Any]:
        """Export all config values as a dictionary (useful for debugging)."""
        return {
            key: getattr(cls, key)
            for key in dir(cls)
            if key.isupper() and not key.startswith('_')
        }
    
    @classmethod
    def validate(cls) -> None:
        """
        Validate configuration consistency.
        Raises ValueError if critical mismatches found.
        
        Rationale: The embedding dimension MUST match the model being used.
        Mismatches cause silent failures or dimension errors during vector search.
        """
        # Check embedding dimension matches model
        known_dimensions = {
            "all-MiniLM-L6-v2": 384,
            "BAAI/bge-large-en-v1.5": 1024,
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
        }
        
        expected_dim = known_dimensions.get(cls.EMBEDDING_MODEL_NAME)
        if expected_dim and expected_dim != cls.EMBEDDING_DIMENSION:
            raise ValueError(
                f"Embedding dimension mismatch: {cls.EMBEDDING_MODEL_NAME} "
                f"expects {expected_dim}D, but config has {cls.EMBEDDING_DIMENSION}D. "
                f"Set HMLR_EMBEDDING_DIM={expected_dim}"
            )
        
        # Validate temperature ranges
        if not (0.0 <= cls.DEFAULT_TEMPERATURE <= 2.0):
            raise ValueError(
                f"DEFAULT_TEMPERATURE must be between 0.0 and 2.0, got {cls.DEFAULT_TEMPERATURE}"
            )
        
        if cls.MAIN_TEMPERATURE is not None and not (0.0 <= cls.MAIN_TEMPERATURE <= 2.0):
            raise ValueError(
                f"MAIN_TEMPERATURE must be between 0.0 and 2.0, got {cls.MAIN_TEMPERATURE}"
            )
        
        if cls.WORKER_TEMPERATURE is not None and not (0.0 <= cls.WORKER_TEMPERATURE <= 2.0):
            raise ValueError(
                f"WORKER_TEMPERATURE must be between 0.0 and 2.0, got {cls.WORKER_TEMPERATURE}"
            )
        
        # Validate retrieval parameters
        if not (0.0 <= cls.MIN_SIMILARITY_THRESHOLD <= 1.0):
            raise ValueError(
                f"MIN_SIMILARITY_THRESHOLD must be between 0.0 and 1.0, got {cls.MIN_SIMILARITY_THRESHOLD}"
            )
        
        if cls.DEFAULT_CANDIDATE_SCORE < 0.0:
            raise ValueError(
                f"DEFAULT_CANDIDATE_SCORE must be non-negative, got {cls.DEFAULT_CANDIDATE_SCORE}"
            )
        
        # Validate token budgets are positive
        token_fields = [
            'CONTEXT_BUDGET_TOKENS', 'MAX_RESPONSE_TOKENS', 
            'FACT_EXTRACTION_MAX_TOKENS', 'USER_PROFILE_MAX_TOKENS',
            'HYDRATOR_MAX_TOKENS'
        ]
        
        for field in token_fields:
            value = getattr(cls, field)
            if value <= 0:
                raise ValueError(f"{field} must be positive, got {value}")


# Global singleton instance
model_config = ModelConfig()


# Run validation on import (fail fast if misconfigured)
try:
    model_config.validate()
except ValueError as e:
    logger.warning(f"Model configuration validation failed: {e}")
    logger.warning("Fix your environment variables or model_config.py defaults")
    # Don't raise - let the system start, but warn loudly
