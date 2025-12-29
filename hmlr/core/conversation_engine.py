"""
ConversationEngine - Unified conversation processing for CognitiveLattice.

This module provides the core conversation processing logic that can be
used by multiple interfaces (CLI, Flask API, Discord bot, etc.).
"""

import re
import traceback
import asyncio
import logging
from typing import Tuple, Optional, List, Dict, Any
from datetime import datetime

from hmlr.core.models import ConversationResponse, ResponseStatus
from .exceptions import ApiConnectionError, ConfigurationError, RetrievalError, StorageWriteError
from .config import config
from .model_config import model_config
from . import prompts
from hmlr.memory.models import Intent, QueryType
from hmlr.memory.retrieval.lattice import LatticeRetrieval, TheGovernor
from hmlr.memory.retrieval.hmlr_hydrator import Hydrator

logger = logging.getLogger(__name__)


class ConversationEngine:
    """
    Unified conversation processing engine for CognitiveLattice.
    
    Handles intent detection, context retrieval, LLM interaction,
    and response generation across all conversation types.
    
    This engine maintains session state and can be used by multiple
    interfaces without code duplication.
    """
    
    def __init__(
        self,
        storage,
        sliding_window,
        conversation_mgr,
        crawler,
        lattice_retrieval,
        governor,
        hydrator,
        context_hydrator,
        user_profile_manager,
        scribe,
        chunk_engine,
        fact_scrubber,
        embedding_storage,
        previous_day=None,
        raise_on_error: bool = False
    ):
        """
        Initialize ConversationEngine with all required components.
        
        Args:
            storage: DailyStorage instance
            sliding_window: SlidingWindow instance
            : SessionManager instance
            conversation_mgr: ConversationManager instance
            crawler: LatticeCrawler instance
            lattice_retrieval: LatticeRetrieval instance
            governor: TheGovernor instance
            hydrator: Hydrator instance
            context_hydrator: ContextHydrator instance
            user_profile_manager: UserProfileManager instance
            scribe: Scribe instance
            chunk_engine: ChunkEngine instance
            fact_scrubber: FactScrubber instance
            embedding_storage: EmbeddingStorage instance
            previous_day: Optional[str] ID of the previous day
            raise_on_error: If True, exceptions propagate instead of returning
                          ConversationResponse with ERROR status. Set True for
                          LangGraph integration so graph can handle errors.
        """
        self.storage = storage
        self.sliding_window = sliding_window
        self.conversation_mgr = conversation_mgr
        self.crawler = crawler
        self.lattice_retrieval = lattice_retrieval
        self.governor = governor
        self.hydrator = hydrator
        self.context_hydrator = context_hydrator
        self.user_profile_manager = user_profile_manager
        self.scribe = scribe
        self.chunk_engine = chunk_engine
        self.fact_scrubber = fact_scrubber
        self.embedding_storage = embedding_storage
        self.previous_day = previous_day
        self.raise_on_error = raise_on_error
        
        self.logger = logging.getLogger(__name__)
        
        self.main_model = model_config.get_main_model()
        self.nano_model = model_config.get_nano_model()
    
    async def process_user_message(
        self,
        user_query: str,
        session_id: str = "default_session",
        force_intent: Optional[str] = None,
        await_background_tasks: bool = False,
        **kwargs
    ) -> ConversationResponse:
        """
        Main entry point for processing user messages.
        
        Args:
            user_query: User's input message
            session_id: Unique session identifier
            force_intent: Optional intent override (used for task lock or session override)
            await_background_tasks: If True, wait for Scribe/background tasks to complete
                                   before returning. Set True for LangGraph integration.
            **kwargs: Additional parameters passed to internals
        
        Returns:
            ConversationResponse object with response text, metadata, and status
        """
        # Set session on stateless sliding window
        if hasattr(self.sliding_window, 'set_session'):
            self.sliding_window.set_session(session_id)
            
        start_time = datetime.now()
        
        try:
            # Default to chat mode (planning/task features removed)
            logger.info("Processing in chat mode")

            # 3. Trigger Scribe (Background User Profile Update)
            if self.scribe:
                logger.info("Triggering Scribe in background")
                # Use BackgroundTaskManager for safety
                if not hasattr(self, 'background_manager'):
                    # Lazy init if not present (though better in __init__)
                    from hmlr.core.background_tasks import BackgroundTaskManager
                    self.background_manager = BackgroundTaskManager()
                
                self.background_manager.add_task(
                    self.scribe.run_scribe_agent(user_query),
                    name=f"scribe_agent_{session_id}"
                )
            
            # 4. Route to chat handler
            response = await self._handle_chat(user_query, session_id=session_id, **kwargs)
            
            # 4. Wait for background tasks if requested (LangGraph mode)
            if await_background_tasks and hasattr(self, 'background_manager'):
                await self.background_manager.shutdown(timeout=10.0)
            
            # 5. Calculate processing time
            end_time = datetime.now()
            response.processing_time_ms = int((end_time - start_time).total_seconds() * 1000)
            
            return response
            
        except Exception as e:
            # Handle unexpected errors
            error_trace = traceback.format_exc()
            logger.error(f"Error in ConversationEngine: {e}", exc_info=True)
            
            # For LangGraph integration: propagate exceptions so graph can handle them
            if self.raise_on_error:
                raise
            
            end_time = datetime.now()
            processing_time = int((end_time - start_time).total_seconds() * 1000)
            
            return ConversationResponse(
                response_text="I encountered an error processing your request.",
                status=ResponseStatus.ERROR,
                detected_intent="error",
                detected_action="error",
                error_message=str(e),
                error_traceback=error_trace,
                processing_time_ms=processing_time
            )
    
    async def _handle_chat(self, user_query: str, session_id: str = "default_session", **kwargs) -> ConversationResponse:
        """
        Processes a chat message using the HMLR architecture.
        """
        logger.info("[Bridge Block Chat]")
        
        if not self.governor or not self.governor.api_client:
            return ConversationResponse(
                response_text="I'm here to chat! (External API not available)",
                status=ResponseStatus.PARTIAL,
                detected_intent="chat",
                detected_action="chat"
            )
        
        try:
            # 1. Chunking & Fact Extraction (Parallel)
            turn_id = f"turn_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            chunks = self._chunk_user_query(user_query, turn_id)
            
            # 2. Routing & Retrieval (Parallel)
            day_id = self.conversation_mgr.current_day
            routing_decision, filtered_memories, facts, dossiers, fact_task = await self._orchestrate_retrieval(
                user_query, day_id, turn_id, chunks
            )
            
            # 3. Routing Execution
            block_id, is_new_topic = await self._execute_routing_strategy(routing_decision, day_id)
            
            # 4. Link Facts to Block (Await parallel extraction)
            await self._finalize_fact_extraction(fact_task, turn_id, block_id)
            
            # 5. Hydration & LLM Generation
            response_text, metadata_json = await self._generate_llm_response(
                block_id, filtered_memories, user_query, is_new_topic, dossiers
            )
            
            # 6. Persistence & State Update
            await self._persist_chat_turn(block_id, turn_id, user_query, response_text, chunks, metadata_json, session_id)
            
            return ConversationResponse(
                response_text=response_text,
                status=ResponseStatus.SUCCESS,
                detected_intent="chat",
                detected_action="chat",
                contexts_retrieved=len(filtered_memories)
            )
            
        except ApiConnectionError as e:
            logger.error(f"Chat API connection failed: {e}", exc_info=True)
            return ConversationResponse(
                response_text="I apologize, but I'm having trouble connecting to my brain right now.",
                status=ResponseStatus.ERROR,
                detected_intent="chat",
                detected_action="chat"
            )
        except Exception as e:
            logger.error(f"Unexpected error in chat: {e}", exc_info=True)
            return ConversationResponse(
                response_text="I'm here to chat, but I'm having trouble connecting to my chat system right now.",
                status=ResponseStatus.ERROR,
                error_message=str(e)
            )

    def _chunk_user_query(self, user_query: str, turn_id: str) -> List[Any]:
        if not self.chunk_engine:
            logger.warning("ChunkEngine not available, skipping chunking")
            return []
        chunks = self.chunk_engine.chunk_turn(text=user_query, turn_id=turn_id, span_id=None)
        logger.debug(f"Created {len(chunks)} chunks")
        return chunks

    async def _orchestrate_retrieval(self, user_query: str, day_id: str, turn_id: str, chunks: List[Any]):
        # Start fact extraction in parallel
        fact_task = None
        if self.fact_scrubber and chunks:
            logger.debug("FactScrubber: Starting extraction in parallel")
            fact_task = asyncio.create_task(
                self.fact_scrubber.extract_and_save(
                    turn_id=turn_id, message_text=user_query, chunks=chunks, span_id=None, block_id=None
                )
            )
            
        try:
            routing_decision, memories, facts, dossiers = await self.governor.govern(user_query, day_id)
        except RetrievalError as e:
            logger.error(f"Critical retrieval failure: {e}. Proceeding with memory disabled.")
            routing_decision = {'matched_block_id': None, 'is_new_topic': True, 'suggested_label': 'General Discussion'}
            memories, facts, dossiers = [], [], []

        return routing_decision, memories, facts, dossiers, fact_task

    async def _execute_routing_strategy(self, routing_decision: Dict, day_id: str) -> Tuple[str, bool]:
        matched_block_id = routing_decision.get('matched_block_id')
        is_new = routing_decision.get('is_new_topic', False)
        suggested_label = routing_decision.get('suggested_label', 'General Discussion')
        
        active_blocks = self.storage.get_active_bridge_blocks()
        last_active_block = next((b for b in active_blocks if b.get('status') == 'ACTIVE'), None)
        
        if matched_block_id and last_active_block and matched_block_id == last_active_block['block_id']:
            logger.info(f"Routing Strategy: Topic Continuation ({matched_block_id})")
            return matched_block_id, False
            
        if matched_block_id and not is_new:
            logger.info(f"Routing Strategy: Topic Resumption ({matched_block_id})")
            if last_active_block:
                self.storage.update_bridge_block_status(last_active_block['block_id'], 'PAUSED')
                self.storage.generate_block_summary(last_active_block['block_id'])
            self.storage.update_bridge_block_status(matched_block_id, 'ACTIVE')
            return matched_block_id, False
            
        # New Topic
        logger.info(f"Routing Strategy: New Topic ('{suggested_label}')")
        if last_active_block:
            self.storage.update_bridge_block_status(last_active_block['block_id'], 'PAUSED')
            self.storage.generate_block_summary(last_active_block['block_id'])
            
        metadata = getattr(self, '_current_metadata', {})
        keywords = metadata.get('keywords', [])
        block_id = self.storage.create_new_bridge_block(day_id=day_id, topic_label=suggested_label, keywords=keywords)
        return block_id, True

    async def _finalize_fact_extraction(self, fact_task, turn_id: str, block_id: str):
        if fact_task:
            extracted_facts = await fact_task
            if extracted_facts and block_id:
                self.storage.update_facts_block_id(turn_id, block_id)

    async def _generate_llm_response(self, block_id: str, memories: List, user_query: str, is_new_topic: bool, dossiers: List):
        block_facts = self.storage.get_facts_for_block(block_id)
        full_prompt = self.context_hydrator.hydrate_bridge_block(
            block_id=block_id, memories=memories, facts=block_facts, system_prompt=prompts.CHAT_SYSTEM_PROMPT,
            user_message=user_query, is_new_topic=is_new_topic, dossiers=dossiers
        )
        
        chat_response = await self.governor.api_client.query_external_api_async(full_prompt)
        
        # Parse metadata JSON
        metadata_json = None
        response_text = chat_response
        json_pattern = r'```json\s*(\{[^`]+\})\s*```'
        json_match = re.search(json_pattern, chat_response, re.DOTALL)
        
        if json_match:
            try:
                import json
                metadata_json = json.loads(json_match.group(1))
                response_text = re.sub(json_pattern, '', chat_response, flags=re.DOTALL).strip()
            except Exception as e:
                logger.warning(f"Failed to parse metadata JSON: {e}")
                
        return response_text, metadata_json

    async def _persist_chat_turn(self, block_id, turn_id, user_query, response_text, chunks, metadata_json, session_id):
        if metadata_json:
            self.storage.update_bridge_block_metadata(block_id, metadata_json)
        
        turn_data = {
            "turn_id": turn_id,
            "timestamp": datetime.now().isoformat(),
            "user_message": user_query,
            "ai_response": response_text,
            "chunks": [self._format_chunk(c) for c in chunks] if chunks else []
        }
        
        if not self.storage.append_turn_to_block(block_id, turn_data):
            logger.error(f"Failed to append turn {turn_id} to block {block_id}")
            
        self.log_conversation_turn(user_query, response_text, session_id=session_id)

    def _format_chunk(self, chunk: Any) -> Dict:
        return {
            "chunk_id": chunk.chunk_id,
            "chunk_type": chunk.chunk_type,
            "text_verbatim": chunk.text_verbatim,
            "parent_chunk_id": chunk.parent_chunk_id,
            "token_count": chunk.token_count
        }


    
    def log_conversation_turn(self, user_msg: str, assistant_msg: str, session_id: str = "default_session",
                             keywords: List[str] = None, topics: List[str] = None, affect: str = None):
        """
        Log turn to storage, embeddings, and sliding window.
        
        This function handles:
        1. Metadata extraction from messages (or uses provided metadata)
        2. Turn creation and storage
        3. Embedding generation
        4. Sliding window updates (stateless)
        5. Day synthesis trigger
        
        Args:
            user_msg: User's message
            assistant_msg: Assistant's response
            session_id: Current session ID
            
        """
        try:
            logger.debug(f"Logging turn to storage (session={session_id})...")
            turn = self.conversation_mgr.log_turn(
                session_id=session_id,
                user_message=user_msg,
                assistant_response=assistant_msg,
                keywords=keywords or [],
                active_topics=topics or [],
                affect=affect or "neutral"
            )

            logger.debug("Updating sliding window...")
            self.sliding_window.add_turn(turn)

            logger.debug(f"Turn logged: {turn.turn_id}")

            # Generate embeddings from turn text chunks (for vector search)
            # IMPORTANT: Only embed USER queries, not assistant responses
            # Rationale: Sources of truth come from user input or external sources referenced by user.
            # Governor searches user queries to find relevant turns, then hydrates full turn (including assistant response).
            try:
                # Only embed the user query
                turn_text = user_msg
                
                # Chunk the user query
                text_chunks = []
                if self.chunk_engine:
                    chunks = self.chunk_engine.chunk_turn(
                        text=turn_text,
                        turn_id=turn.turn_id,
                        span_id=None
                    )
                    # Extract text_verbatim from sentence-level chunks only
                    text_chunks = [chunk.text_verbatim for chunk in chunks 
                                   if hasattr(chunk, 'text_verbatim') and chunk.chunk_type == 'sentence']
                    logger.debug(f"Created {len(text_chunks)} sentence chunks from user query")
                else:
                    # Fallback: embed full user query as single chunk
                    text_chunks = [turn_text]
                    logger.debug("ChunkEngine unavailable, embedding full user query")
                
                if text_chunks:
                    self.embedding_storage.save_turn_embeddings(turn.turn_id, text_chunks)
                    logger.debug(f"Generated embeddings for {len(text_chunks)} user query chunks")
                else:
                    logger.warning("No text chunks generated for embedding")
                    
            except Exception as embed_err:
                logger.error(f"Embedding generation failed: {embed_err}", exc_info=True)

            current_day = self.conversation_mgr.current_day
            if current_day != self.previous_day:
                logger.info(f"Day changed from {self.previous_day} to {current_day}")
                self.previous_day = current_day

        except Exception as e:
            logger.error(
                f"Failed to log turn to storage (session={session_id}): {e}",
                exc_info=True
            )
            raise StorageWriteError(f"Turn persistence failed for session {session_id}") from e

    # =========================================================================
    # ENCAPSULATION FACADE METHODS
    # =========================================================================

    def clear_session_state(self, session_id: str = "default_session"):
        """
        Clear transient state for the given session.
        Delegates to SlidingWindow.
        """
        if hasattr(self.sliding_window, 'set_session'):
            self.sliding_window.set_session(session_id)
        if hasattr(self.sliding_window, 'clear'):
            self.sliding_window.clear()
            
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the memory system.
        """
        total_turns = "unknown"
        if self.storage:
            try:
                # Basic count query (naive implementation for compatibility)
                # Ideally storage should have a count_turns() method
                all_turns = self.storage.get_recent_turns(limit=100000)
                total_turns = len(all_turns)
            except Exception:
                logger.warning("Failed to calculate total turns for memory stats", exc_info=True)
        
        window_size = 0
        if hasattr(self.sliding_window, 'turns'):
            window_size = len(self.sliding_window.turns)
            
        return {
            "total_turns": total_turns,
            "sliding_window_size": window_size,
            "model": self.main_model
        }

    def get_recent_turns(self, limit: int = 10) -> List[Any]:
        """
        Get recent conversation turns from storage.
        """
        if self.storage:
            return self.storage.get_recent_turns(limit=limit)
        return []
