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
        previous_day=None
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
        
        self.logger = logging.getLogger(__name__)
        
        self.main_model = model_config.get_main_model()
        self.nano_model = model_config.get_nano_model()
    
    async def process_user_message(
        self,
        user_query: str,
        session_id: str = "default_session",
        force_intent: Optional[str] = None,
        **kwargs
    ) -> ConversationResponse:
        """
        Main entry point for processing user messages.
        
        Args:
            user_query: User's input message
            session_id: Unique session identifier
            force_intent: Optional intent override (used for task lock or session override)
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
            
            # 4. Calculate processing time
            end_time = datetime.now()
            response.processing_time_ms = int((end_time - start_time).total_seconds() * 1000)
            
            return response
            
        except Exception as e:
            # Handle unexpected errors
            error_trace = traceback.format_exc()
            logger.error(f"Error in ConversationEngine: {e}", exc_info=True)
            
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
        
        
        This implements the CORRECTED HMLR architecture:
        1. Governor: 3 parallel tasks (routing, memory retrieval, fact lookup)
        2. Execute 1 of 4 routing scenarios
        3. Hydrator: Load Bridge Block + format context + metadata instructions
        4. Main LLM: Generate response + optional metadata JSON
        5. Parse response, update block headers, append turn
        
        Args:
            user_query: User's chat message
        
        Returns:
            ConversationResponse with chat response and metadata
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
            # Start debug logging for this turn
            self.logger.info(f"Starting turn for query: {user_query[:50]}...")
            
            # --- Chunking & Fact Extraction --- #
            # Generate turn_id immediately (needed for chunking)
            turn_id = f"turn_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            logger.debug(f"ChunkEngine: Chunking query (turn_id={turn_id})...")
            
            # Chunk the query into hierarchical structure (turn → paragraph → sentence)
            chunks = []
            if self.chunk_engine:
                chunks = self.chunk_engine.chunk_turn(
                    text=user_query,
                    turn_id=turn_id,
                    span_id=None  # Daily conversations don't have span_id yet
                )
                logger.debug(f"Created {len(chunks)} chunks")
            else:
                logger.warning(f"ChunkEngine not available, skipping chunking")
            
            # === USE GPT-4.1-NANO METADATA (extracted during intent detection) === #
            metadata = getattr(self, '_current_metadata', {})
            gpt_nano_keywords = metadata.get('keywords', [])
            gpt_nano_topics = metadata.get('topics', [])
            
            if gpt_nano_keywords:
                logger.debug(f"Using GPT-4.1-nano keywords: {gpt_nano_keywords[:5]}")
            
            # --- Governor: Routing & Context Retrieval --- #
            day_id = self.conversation_mgr.current_day
            logger.info(f"Governor: Running parallel tasks for day {day_id}")
            
            # Start fact extraction in parallel with Governor (if chunks available)
            fact_extraction_task = None
            if self.fact_scrubber and chunks:
                logger.debug("FactScrubber: Starting extraction in parallel")
                print(f"    FactScrubber: Starting extraction for {len(chunks)} chunks...")
                fact_extraction_task = asyncio.create_task(
                    self.fact_scrubber.extract_and_save(
                        turn_id=turn_id,
                        message_text=user_query,
                        chunks=chunks,
                        span_id=None,
                        block_id=None  # Will update after Governor assigns block_id
                    )
                )
            else:
                if not self.fact_scrubber:
                    print(f"     FactScrubber not available")
                if not chunks:
                    print(f"     No chunks available for fact extraction")
            
            # Call async govern() method
            try:
                routing_decision, filtered_memories, facts, dossiers = await self.governor.govern(user_query, day_id)
            except RetrievalError as e:
                logger.error(f"Critical retrieval failure: {e}. Proceeding with memory disabled.")
                # Fallback: No memory, default routing
                routing_decision = {'matched_block_id': None, 'is_new_topic': True, 'suggested_label': 'General Discussion'}
                filtered_memories = []
                facts = []
                dossiers = []

            logger.info(
                f"Governor results: routing={routing_decision.get('matched_block_id')}, "
                f"memories={len(filtered_memories)}, facts={len(facts)}, dossiers={len(dossiers)}"
            )
            
            # --- Routing Execution --- #
            block_id = None
            is_new_topic = False
            
            matched_block_id = routing_decision.get('matched_block_id')
            is_new = routing_decision.get('is_new_topic', False)
            suggested_label = routing_decision.get('suggested_label', 'General Discussion')
            
            # Get last active block (should be only one with status='ACTIVE')
            active_blocks = self.storage.get_active_bridge_blocks()
            last_active_block = None
            for block in active_blocks:
                if block.get('status') == 'ACTIVE':
                    last_active_block = block
                    break
            
            # Determine which scenario to execute
            if matched_block_id and last_active_block and matched_block_id == last_active_block['block_id']:
                # Strategy: Topic Continuation
                logger.info(f"Routing Strategy: Topic Continuation (block {matched_block_id})")
                block_id = matched_block_id
                is_new_topic = False
                # No status changes needed
                
            elif matched_block_id and not is_new:
                # Strategy: Topic Resumption
                logger.info(f"Routing Strategy: Topic Resumption (reactivate block {matched_block_id})")
                
                # Pause current active block if exists
                if last_active_block:
                    old_active_id = last_active_block['block_id']
                    logger.debug(f"Pausing old block: {old_active_id}")
                    self.storage.update_bridge_block_status(old_active_id, 'PAUSED')
                    self.storage.generate_block_summary(old_active_id)
                
                # Reactivate matched block
                self.storage.update_bridge_block_status(matched_block_id, 'ACTIVE')
                block_id = matched_block_id
                is_new_topic = False
                
            elif is_new and not last_active_block:
                # Strategy: New Topic Creation (no active blocks)
                logger.info("Routing Strategy: New Topic Creation (first topic today)")
                
                # Extract keywords from query
                keywords = gpt_nano_keywords or []
                
                # Create new Bridge Block
                block_id = self.storage.create_new_bridge_block(
                    day_id=day_id,
                    topic_label=suggested_label,
                    keywords=keywords
                )
                logger.debug(f"Created block: {block_id}")
                is_new_topic = True
                
                
            elif is_new and last_active_block:
                # Strategy: Topic Shift to New
                logger.info("Routing Strategy: Topic Shift to New")
                
                # Pause current active block
                old_active_id = last_active_block['block_id']
                logger.debug(f"Pausing old block: {old_active_id}")
                self.storage.update_bridge_block_status(old_active_id, 'PAUSED')
                self.storage.generate_block_summary(old_active_id)
                
                # Extract keywords from query
                keywords = gpt_nano_keywords or []
                
                # Create new Bridge Block
                block_id = self.storage.create_new_bridge_block(
                    day_id=day_id,
                    topic_label=suggested_label,
                    keywords=keywords
                )
                logger.debug(f"Created block: {block_id}")
                is_new_topic = True
            
            else:
                # Fallback: Shouldn't happen, but create new block if needed
                logger.warning("Routing Fallback: Creating new block (unexpected scenario)")
                keywords = gpt_nano_keywords or []
                block_id = self.storage.create_new_bridge_block(
                    day_id=day_id,
                    topic_label=suggested_label,
                    keywords=keywords
                )
                is_new_topic = True
            
            # --- Update Facts with Block ID --- #
            # Wait for fact extraction to complete (if running)
            if fact_extraction_task:
                logger.debug("Waiting for FactScrubber to complete")
                print(f"    Waiting for FactScrubber to complete...")
                extracted_facts = await fact_extraction_task
                logger.debug(f"Extracted {len(extracted_facts)} facts")
                print(f"    FactScrubber extracted {len(extracted_facts)} facts")
                
                # Update facts with final block_id
                if extracted_facts and block_id:
                    logger.debug(f"Linking {len(extracted_facts)} facts to block {block_id}")
                    print(f"    Linking {len(extracted_facts)} facts to block {block_id}...")
                    updated_count = self.storage.update_facts_block_id(turn_id, block_id)
                    logger.debug(f"Updated {updated_count} facts with block_id")
                    print(f"    Updated {updated_count} facts with block_id")
            else:
                print(f"    No fact extraction task to await")
            
            # === HYDRATOR: Format Context === #
            logger.debug(f"Hydrator: Building context for block {block_id}")
            
            # Get ALL facts for this specific block (not keyword-filtered facts from Governor)
            # This allows LLM to fuzzy-match vague queries like "what was that credential?"
            block_facts = self.storage.get_facts_for_block(block_id)
            logger.debug(f"Loaded {len(block_facts)} facts for this block")
            
            # Build system prompt from centralized template
            system_prompt = prompts.CHAT_SYSTEM_PROMPT
            
            # Call hydrator with is_new_topic flag
            full_prompt = self.context_hydrator.hydrate_bridge_block(
                block_id=block_id,
                memories=filtered_memories,
                facts=block_facts,  
                system_prompt=system_prompt,
                user_message=user_query,
                is_new_topic=is_new_topic,
                dossiers=dossiers  
            )
            
            logger.debug(f"Full prompt length: {len(full_prompt)} chars")
            
            # === MAIN LLM: Generate Response === #
            logger.info("Calling main LLM")
            chat_response = await self.governor.api_client.query_external_api_async(full_prompt)
            logger.debug("Response received from LLM")
            
            # === PARSE METADATA JSON === #
            logger.debug("Parsing metadata")
            metadata_json = None
            response_text = chat_response
            
            # Extract JSON code block if present
            import re
            json_pattern = r'```json\s*(\{[^`]+\})\s*```'
            json_match = re.search(json_pattern, chat_response, re.DOTALL)
            
            if json_match:
                import json
                try:
                    metadata_json = json.loads(json_match.group(1))
                    logger.debug(f"Metadata JSON extracted: {list(metadata_json.keys())}")
                    
                    # Strip JSON block from user-facing response
                    response_text = re.sub(json_pattern, '', chat_response, flags=re.DOTALL).strip()
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse metadata JSON: {e}")
            
            # === UPDATE BRIDGE BLOCK HEADER === #
            if metadata_json:
                logger.info("Updating Bridge Block header")
                try:
                    # Update metadata in storage
                    self.storage.update_bridge_block_metadata(block_id, metadata_json)
                    logger.debug("Header updated successfully")
                except Exception as e:
                    logger.warning(f"Failed to update header: {e}", exc_info=True)
            
            # === APPEND TURN TO BRIDGE BLOCK === #
            print(f"    DEBUG: Appending turn {turn_id} to block {block_id}...")
            turn_data = {
                "turn_id": turn_id,  # Reuse turn_id generated at start for chunking
                "timestamp": datetime.now().isoformat(),
                "user_message": user_query,
                "ai_response": response_text,
                "chunks": [{
                    "chunk_id": chunk.chunk_id,
                    "chunk_type": chunk.chunk_type,
                    "text_verbatim": chunk.text_verbatim,
                    "parent_chunk_id": chunk.parent_chunk_id,
                    "token_count": chunk.token_count
                } for chunk in chunks] if chunks else []
            }
            
            success = self.storage.append_turn_to_block(block_id, turn_data)
            print(f"    DEBUG: append_turn_to_block returned: {success}")
            if not success:
                print(f"    ERROR: Failed to append turn {turn_id} to block {block_id}")
            else:
                print(f"    Turn {turn_id} appended to block {block_id}")
            
            # === LOG AND RETURN === #
            logger.debug(f"Response: {response_text[:200]}...")
            
            # Log the response
            self.logger.info(f"LLM Response received ({len(response_text)} chars)")
            
            # Log to session
            self.log_conversation_turn(user_query, response_text, session_id=session_id)
            
            return ConversationResponse(
                response_text=response_text,
                status=ResponseStatus.SUCCESS,
                detected_intent="chat",
                detected_action="chat",
                contexts_retrieved=len(filtered_memories),
                sliding_window_turns=0,  # Bridge Blocks replace sliding window
                citations_found=0,
                context_efficiency=100.0  # Governor already filtered
            )
            
        except ApiConnectionError as e:
            logger.error(f"Chat API connection failed: {e}", exc_info=True)
            return ConversationResponse(
                response_text="I apologize, but I'm having trouble connecting to my brain right now. Please try again in a moment.",
                status=ResponseStatus.ERROR,
                detected_intent="chat",
                detected_action="chat"
            )
        except Exception as e:
            logger.error(f"Unexpected error in chat: {e}", exc_info=True)
            error_trace = traceback.format_exc()
            
            fallback_response = "I'm here to chat, but I'm having trouble connecting to my chat system right now."
            
            # Log to persistent storage
            self.log_conversation_turn(user_query, fallback_response)
            
            return ConversationResponse(
                response_text=fallback_response,
                status=ResponseStatus.ERROR,
                detected_intent="chat",
                detected_action="chat",
                error_message=str(e),
                error_traceback=error_trace
            )
    
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
                pass
        
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
