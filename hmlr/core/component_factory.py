"""
Component Factory for CognitiveLattice

This module provides centralized component initialization with dependency injection.
All CognitiveLattice interfaces (CLI, Flask API, Discord bot, etc.) can use this
factory to get a consistent, properly-wired set of components.

Extracted from main.py as part of Phase 3 refactor.
"""

import os
from dataclasses import dataclass
from typing import Optional
from hmlr.memory import Storage
from hmlr.memory.conversation_manager import ConversationManager
from hmlr.memory.models import SlidingWindow
from hmlr.memory.metadata_extractor import LLMMetadataExtractor
from hmlr.memory.embeddings.embedding_manager import EmbeddingManager, EmbeddingStorage
from hmlr.memory.retrieval.intent_analyzer import IntentAnalyzer
from hmlr.memory.retrieval.crawler import LatticeCrawler
from hmlr.memory.retrieval.context_hydrator import ContextHydrator
from hmlr.memory.retrieval.lattice import LatticeRetrieval, TheGovernor
from hmlr.memory.retrieval.hmlr_hydrator import Hydrator
from hmlr.memory.retrieval.dossier_retriever import DossierRetriever
from hmlr.memory.synthesis import SynthesisManager
from hmlr.memory.synthesis.user_profile_manager import UserProfileManager
from hmlr.memory.synthesis.scribe import Scribe
from hmlr.memory.synthesis.dossier_governor import DossierGovernor
from hmlr.memory.dossier_storage import DossierEmbeddingStorage
from hmlr.memory.id_generator import IDGenerator
from hmlr.memory.chunking.chunk_engine import ChunkEngine
from hmlr.memory.fact_scrubber import FactScrubber
from hmlr.core.cognitive_lattice import SessionManager


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
    session_manager: SessionManager
    
    # Retrieval components
    crawler: LatticeCrawler
    intent_analyzer: IntentAnalyzer
    context_hydrator: ContextHydrator
    
    # HMLR Components
    lattice_retrieval: LatticeRetrieval
    governor: TheGovernor
    hydrator: Hydrator
    
    # Dossier System (Phase 4)
    dossier_retriever: Optional[DossierRetriever]
    dossier_governor: Optional[DossierGovernor]
    dossier_storage: Optional[DossierEmbeddingStorage]
    
    # Synthesis
    synthesis_manager: SynthesisManager
    user_profile_manager: UserProfileManager
    scribe: Scribe
    
    # Chunking and Fact Extraction
    chunk_engine: ChunkEngine
    fact_scrubber: Optional[FactScrubber]
    
    # Utilities
    metadata_extractor: LLMMetadataExtractor
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
        use_llm_intent_mode: bool = False,
        max_sliding_window_turns: int = 20,
        context_budget_tokens: int = 6000,
        crawler_recency_weight: float = 0.5
    ) -> ComponentBundle:
        """
        Create and wire all CognitiveLattice components.
        
        This method initializes all components in the correct order,
        handling dependencies and configuration.
        
        Args:
            use_llm_intent_mode: Whether to use LLM for intent analysis (default: False)
            max_sliding_window_turns: Max turns in sliding window (default: 20)
            context_budget_tokens: Token budget for context hydration (default: 6000)
            crawler_recency_weight: Weight for recent results (default: 0.5)
        
        Returns:
            ComponentBundle with all initialized components
        """
        print("🏗️  Initializing CognitiveLattice components...")
        
        # === Core Storage and State === #
        print("   📦 Initializing storage and session...")
        session_manager = SessionManager()
        print(f"   🧠 Cognitive Lattice initialized for session: {session_manager.lattice.session_id}")
        
        # Check for test database path override
        db_path = os.environ.get('COGNITIVE_LATTICE_DB')
        storage = Storage(db_path=db_path) if db_path else Storage()
        conversation_mgr = ConversationManager(storage)
        previous_day = conversation_mgr.current_day
        
        # === Utilities === #
        print("   🛠️  Initializing utilities...")
        metadata_extractor = LLMMetadataExtractor()
        print(f"   💾 Persistent memory enabled (day: {conversation_mgr.current_day})")
        print(f"   📊 Metadata extraction enabled")
        
        # Create unified embedding storage (handles encoding AND database)
        embedding_storage = EmbeddingStorage(storage)
        print(f"   🔍 Embedding storage initialized")
        
        # === Phase 3: Retrieval System === #
        print("   🔍 Initializing retrieval system...")
        intent_analyzer = IntentAnalyzer(use_llm_mode=use_llm_intent_mode)
        crawler = LatticeCrawler(storage, recency_weight=crawler_recency_weight)
        context_hydrator = ContextHydrator(storage=storage, max_tokens=context_budget_tokens)
        
        # === HMLR Components === #
        print("   🏛️  Initializing HMLR components...")
        lattice_retrieval = LatticeRetrieval(crawler)
        hydrator = Hydrator(storage, token_limit=context_budget_tokens)
        
        # Load or create sliding window
        print(f"   📂 Loading sliding window...")
        sliding_window = SlidingWindow.load_from_file()
        
        if len(sliding_window.turns) == 0:
            # No saved state - load from database
            print(f"      ℹ️  No saved state - loading from database...")
            try:
                recent_turns = storage.get_recent_turns(day_id=None, limit=max_sliding_window_turns)
                for turn in recent_turns:
                    sliding_window.add_turn(turn)
                
                if recent_turns:
                    print(f"      📝 Loaded {len(recent_turns)} turns from database")
            except Exception as e:
                print(f"      ⚠️  Could not load from database: {e}")
        else:
            print(f"      📂 Loaded sliding window state: {len(sliding_window.turns)} turns")
        
        print(f"   🔍 Retrieval system enabled")
        mode_desc = "LLM mode" if use_llm_intent_mode else "Heuristic mode"
        print(f"      - Intent Analyzer: {mode_desc}")
        print(f"      - Context Hydrator: {context_budget_tokens} token budget")
        
        # === Synthesis System === #
        print("   🧠 Initializing synthesis system...")
        synthesis_manager = SynthesisManager(storage)
        user_profile_manager = UserProfileManager()
        print(f"   🧠 Synthesis system enabled")
        
        # === External Services === #
        print("   🌐 Initializing external services...")
        
        try:
            from hmlr.core.external_api_client import ExternalAPIClient
            external_api = ExternalAPIClient()
            print(f"   🌐 External API client initialized")
        except Exception as e:
            print(f"   ⚠️  Could not initialize External API Client: {e}")
            external_api = None
        
        # === Dossier System (Phase 4) === #
        print("   📂 Initializing dossier system...")
        dossier_storage = None
        dossier_retriever = None
        dossier_governor = None
        
        try:
            # Initialize dossier embedding storage
            dossier_storage = DossierEmbeddingStorage(storage.db_path)
            print(f"   📂 Dossier storage initialized")
            
            # Initialize dossier retriever
            dossier_retriever = DossierRetriever(storage, dossier_storage)
            print(f"   🔍 Dossier retriever initialized")
            
            # Initialize dossier governor (write-side)
            if external_api:
                id_generator = IDGenerator()
                dossier_governor = DossierGovernor(
                    storage=storage,
                    dossier_storage=dossier_storage,
                    llm_client=external_api,
                    id_generator=id_generator
                )
                print(f"   📝 Dossier governor initialized")
            else:
                print(f"   ⚠️  Dossier governor offline (no API)")
        except Exception as e:
            print(f"   ⚠️  Could not initialize dossier system: {e}")
            dossier_storage = None
            dossier_retriever = None
            dossier_governor = None
        
        # Initialize Governor now that we have API client and dossier_retriever
        governor = TheGovernor(
            external_api, storage, crawler, dossier_retriever=dossier_retriever
        ) if external_api else None
        if governor:
            print(f"   🏛️  The Governor is online")
        else:
            print(f"   ⚠️  The Governor is offline (no API)")
            
        # Initialize Scribe now that we have API client
        scribe = Scribe(external_api, user_profile_manager) if external_api else None
        if scribe:
            print(f"   ✍️  The Scribe is online")
        else:
            print(f"   ⚠️  The Scribe is offline (no API)")
        
        # === Chunking and Fact Extraction === #
        print("   🔧 Initializing chunking and fact extraction...")
        chunk_engine = ChunkEngine()
        fact_scrubber = FactScrubber(storage, external_api) if external_api else None
        if fact_scrubber:
            print(f"   🧹 FactScrubber is online")
        else:
            print(f"   ⚠️  FactScrubber is offline (no API)")
        
        print("   ✅ All components initialized successfully")
        
        return ComponentBundle(
            storage=storage,
            conversation_mgr=conversation_mgr,
            sliding_window=sliding_window,
            session_manager=session_manager,
            crawler=crawler,
            intent_analyzer=intent_analyzer,
            context_hydrator=context_hydrator,
            lattice_retrieval=lattice_retrieval,
            governor=governor,
            hydrator=hydrator,
            dossier_retriever=dossier_retriever,
            dossier_governor=dossier_governor,
            dossier_storage=dossier_storage,
            synthesis_manager=synthesis_manager,
            user_profile_manager=user_profile_manager,
            scribe=scribe,
            chunk_engine=chunk_engine,
            fact_scrubber=fact_scrubber,
            metadata_extractor=metadata_extractor,
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
        
        print("🚀 Creating ConversationEngine...")
        
        engine = ConversationEngine(
            storage=components.storage,
            sliding_window=components.sliding_window,
            session_manager=components.session_manager,
            conversation_mgr=components.conversation_mgr,
            crawler=components.crawler,
            intent_analyzer=components.intent_analyzer,
            lattice_retrieval=components.lattice_retrieval,
            governor=components.governor,
            hydrator=components.hydrator,
            context_hydrator=components.context_hydrator,
            synthesis_manager=components.synthesis_manager,
            user_profile_manager=components.user_profile_manager,
            scribe=components.scribe,
            chunk_engine=components.chunk_engine,
            fact_scrubber=components.fact_scrubber,
            embedding_storage=components.embedding_storage,
            previous_day=components.previous_day
        )
        
        print("🚀 ConversationEngine initialized")
        
        return engine
