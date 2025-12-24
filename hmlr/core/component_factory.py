"""
Component Factory for CognitiveLattice

This module provides centralized component initialization with dependency injection.
All CognitiveLattice interfaces (CLI, Flask API, Discord bot, etc.) can use this
factory to get a consistent, properly-wired set of components.

"""

import os
import logging
from dataclasses import dataclass
from typing import Optional
from hmlr.core.config import config

logger = logging.getLogger(__name__)
from hmlr.memory import Storage
from hmlr.memory.conversation_manager import ConversationManager
from hmlr.memory.sliding_window import SlidingWindow
from hmlr.memory.embeddings.embedding_manager import EmbeddingManager, EmbeddingStorage
from hmlr.memory.retrieval.crawler import LatticeCrawler
from hmlr.memory.retrieval.context_hydrator import ContextHydrator
from hmlr.memory.retrieval.lattice import LatticeRetrieval, TheGovernor
from hmlr.memory.retrieval.hmlr_hydrator import Hydrator
from hmlr.memory.retrieval.dossier_retriever import DossierRetriever
from hmlr.memory.synthesis.user_profile_manager import UserProfileManager
from hmlr.memory.synthesis.scribe import Scribe
from hmlr.memory.synthesis.dossier_governor import DossierGovernor
from hmlr.memory.dossier_storage import DossierEmbeddingStorage
from hmlr.memory.id_generator import IDGenerator
from hmlr.memory.chunking.chunk_engine import ChunkEngine
from hmlr.memory.fact_scrubber import FactScrubber



@dataclass
class ComponentBundle:
    """
    Container for all initialized CognitiveLattice components.
    
    This bundle provides all components needed by ConversationEngine
    and other parts of the system, properly initialized and wired together.
    """
    # Core storage and state
    storage: Storage
    conversation_mgr: ConversationManager
    sliding_window: SlidingWindow
    
    
    # Retrieval components
    crawler: LatticeCrawler
    context_hydrator: ContextHydrator
    
    # HMLR Components
    lattice_retrieval: LatticeRetrieval
    governor: TheGovernor
    hydrator: Hydrator
    
    # Dossier System 
    dossier_retriever: Optional[DossierRetriever]
    dossier_governor: Optional[DossierGovernor]
    dossier_storage: Optional[DossierEmbeddingStorage]
    
    # User Profile & Scribe
    user_profile_manager: UserProfileManager
    scribe: Scribe
    
    # Chunking and Fact Extraction
    chunk_engine: ChunkEngine
    fact_scrubber: Optional[FactScrubber]
    
    # Utilities
    embedding_storage: EmbeddingStorage
    
    # Session state
    previous_day: str


class ComponentFactory:
    """
    Factory for creating and wiring all CognitiveLattice components.
    
    This factory handles all the complex initialization logic, ensuring
    components are properly configured and dependencies are correctly wired.
    
    Usage:
        # Simple initialization
        components = ComponentFactory.create_all_components()
        
        # Use with ConversationEngine
        engine = ConversationEngine(
            storage=components.storage,
            sliding_window=components.sliding_window,
            ...
        )
    """
    
    @staticmethod
    def create_all_components(
        api_key: Optional[str] = None,
        db_path: Optional[str] = None
    ) -> ComponentBundle:
        """
        Create and wire all CognitiveLattice components.
        
        This method initializes all components in the correct order,
        handling dependencies and configuration.
        
        Args:
            api_key: Optional API key to use (overrides environment)
            db_path: Optional DB path (overrides environment and defaults)
            
        Returns:
            ComponentBundle with all initialized components
        """
        from hmlr.core.model_config import model_config
        logger.info("Initializing CognitiveLattice components...")
        
        
        # Database and Storage initialization (DI)
        db_path = db_path or os.environ.get('COGNITIVE_LATTICE_DB')
        storage = Storage(db_path=db_path)
        
        # Initialize stateless sliding window first so it can be injected
        logger.info(f"Initializing stateless sliding window...")
        sliding_window = SlidingWindow(storage=storage)
        
        conversation_mgr = ConversationManager(storage, sliding_window=sliding_window)
        previous_day = conversation_mgr.current_day
        
        # === Utilities === #
        logger.info("Initializing utilities...")
        logger.info(f"Persistent memory enabled (day: {conversation_mgr.current_day})")
        logger.info(f"Metadata extraction enabled")
        
        # Create unified embedding storage (handles encoding AND database)
        embedding_storage = EmbeddingStorage(storage)
        logger.info(f"Embedding storage initialized")
        
        # === Retrieval System === #
        logger.info("Initializing retrieval system...")
        crawler = LatticeCrawler(storage, recency_weight=model_config.CRAWLER_RECENCY_WEIGHT)
        context_hydrator = ContextHydrator(storage=storage, max_tokens=model_config.CONTEXT_BUDGET_TOKENS)
        
        # === HMLR Components === #
        logger.info("Initializing HMLR components...")
        lattice_retrieval = LatticeRetrieval(crawler)
        hydrator = Hydrator(storage, token_limit=model_config.CONTEXT_BUDGET_TOKENS)
        
        # We no longer load from file as it's database-backed.
        # Session ID will be set by the ConversationEngine.
        
        logger.info(f"Retrieval system enabled")
        mode_desc = "LLM mode" if model_config.USE_LLM_INTENT_MODE else "Heuristic mode"
        logger.info(f"Intent Analyzer: {mode_desc}")
        logger.info(f"Context Hydrator: {model_config.CONTEXT_BUDGET_TOKENS} token budget")
        
        # === User Profile & Scribe === #
        logger.info("Initializing user profile and scribe...")
        user_profile_manager = UserProfileManager()
        logger.info(f"User profile and scribe ready")
        
        # === External Services === #
        logger.info("Initializing external services...")
        
        try:
            from hmlr.core.external_api_client import ExternalAPIClient
            from hmlr.core.config import config
            external_api = ExternalAPIClient(
                api_provider=config.API_PROVIDER,
                api_key=api_key
            )
            logger.info(f"External API client ({config.API_PROVIDER}) initialized")
        except Exception as e:
            logger.error(f"Could not initialize External API Client: {e}", exc_info=True)
            external_api = None
        
        # === Dossier System === #
        logger.info("Initializing dossier system...")
        dossier_storage = None
        dossier_retriever = None
        dossier_governor = None
        
        try:
            # Initialize dossier embedding storage
            dossier_storage = DossierEmbeddingStorage(storage.db_path)
            logger.info(f"Dossier storage initialized")
            
            # Initialize dossier retriever
            dossier_retriever = DossierRetriever(storage, dossier_storage)
            logger.info(f"Dossier retriever initialized")
            
            # Initialize dossier governor (write-side)
            if external_api:
                id_generator = IDGenerator()
                dossier_governor = DossierGovernor(
                    storage=storage,
                    dossier_storage=dossier_storage,
                    llm_client=external_api,
                    id_generator=id_generator
                )
                logger.info(f"Dossier governor initialized")
            else:
                logger.warning(f"Dossier governor offline (no API)")
        except Exception as e:
            logger.error(f"Could not initialize dossier system: {e}", exc_info=True)
            dossier_storage = None
            dossier_retriever = None
            dossier_governor = None
        
        # Initialize Governor now that we have API client and dossier_retriever
        governor = TheGovernor(
            external_api, storage, crawler, dossier_retriever=dossier_retriever
        ) if external_api else None
        if governor:
            logger.info(f"The Governor is online")
        else:
            logger.warning(f"The Governor is offline (no API)")
            
        # Initialize Scribe now that we have API client
        scribe = Scribe(external_api, user_profile_manager) if external_api else None
        if scribe:
            logger.info(f"The Scribe is online")
        else:
            logger.warning(f"The Scribe is offline (no API)")
        
        # === Chunking and Fact Extraction === #
        logger.info("Initializing chunking and fact extraction...")
        chunk_engine = ChunkEngine()
        fact_scrubber = FactScrubber(storage, external_api) if external_api else None
        if fact_scrubber:
            logger.info(f"FactScrubber is online")
        else:
            logger.warning(f"FactScrubber is offline (no API)")
        
        logger.info("All components initialized successfully")
        
        return ComponentBundle(
            storage=storage,
            conversation_mgr=conversation_mgr,
            sliding_window=sliding_window,
            crawler=crawler,
            context_hydrator=context_hydrator,
            lattice_retrieval=lattice_retrieval,
            governor=governor,
            hydrator=hydrator,
            dossier_retriever=dossier_retriever,
            dossier_governor=dossier_governor,
            dossier_storage=dossier_storage,
            user_profile_manager=user_profile_manager,
            scribe=scribe,
            chunk_engine=chunk_engine,
            fact_scrubber=fact_scrubber,
            embedding_storage=embedding_storage,
            previous_day=previous_day
        )
    
    @staticmethod
    def create_conversation_engine(components: ComponentBundle):
        """
        Create a ConversationEngine from a ComponentBundle.
        
        This is a convenience method that wires the components into
        a ConversationEngine with the correct parameters.
        
        Args:
            components: ComponentBundle from create_all_components()
        
        Returns:
            Initialized ConversationEngine
        """
        from hmlr.core.conversation_engine import ConversationEngine
        
        logger.info("Creating ConversationEngine...")
        
        engine = ConversationEngine(
            storage=components.storage,
            sliding_window=components.sliding_window,
            conversation_mgr=components.conversation_mgr,
            crawler=components.crawler,
            lattice_retrieval=components.lattice_retrieval,
            governor=components.governor,
            hydrator=components.hydrator,
            context_hydrator=components.context_hydrator,
            user_profile_manager=components.user_profile_manager,
            scribe=components.scribe,
            chunk_engine=components.chunk_engine,
            fact_scrubber=components.fact_scrubber,
            embedding_storage=components.embedding_storage,
            previous_day=components.previous_day
        )
        
        logger.info("ConversationEngine initialized")
        
        return engine
