"""
ConversationEngine - Unified conversation processing for CognitiveLattice.

This module provides the core conversation processing logic that can be
used by multiple interfaces (CLI, Flask API, Discord bot, etc.).
"""

import re
import traceback
import asyncio
from typing import Tuple, Optional, List, Dict, Any
from datetime import datetime
from hmlr.core.telemetry import get_tracer

from hmlr.core.models import ConversationResponse, ResponseStatus
from hmlr.memory.models import Intent, QueryType
from hmlr.memory.retrieval.lattice import LatticeRetrieval, TheGovernor
from hmlr.memory.retrieval.hmlr_hydrator import Hydrator
# LLMMetadataExtractor no longer needed - nano prompting handles metadata better


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
        session_manager,
        conversation_mgr,
        crawler,
        intent_analyzer,
        lattice_retrieval,
        governor,
        hydrator,
        context_hydrator,
        synthesis_manager,
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
            session_manager: SessionManager instance
            conversation_mgr: ConversationManager instance
            crawler: LatticeCrawler instance
            intent_analyzer: IntentAnalyzer instance
            lattice_retrieval: LatticeRetrieval instance
            governor: TheGovernor instance
            hydrator: Hydrator instance
            context_hydrator: ContextHydrator instance
            synthesis_manager: SynthesisManager instance
            user_profile_manager: UserProfileManager instance
            scribe: Scribe instance
            chunk_engine: ChunkEngine instance
            fact_scrubber: FactScrubber instance
            embedding_storage: EmbeddingStorage instance
            previous_day: Optional[str] ID of the previous day
        """
        self.tracer = get_tracer(__name__)
        self.storage = storage
        self.sliding_window = sliding_window
        self.session_manager = session_manager
        self.conversation_mgr = conversation_mgr
        self.crawler = crawler
        self.intent_analyzer = intent_analyzer
        self.lattice_retrieval = lattice_retrieval
        self.governor = governor
        self.hydrator = hydrator
        self.context_hydrator = context_hydrator
        self.synthesis_manager = synthesis_manager
        self.user_profile_manager = user_profile_manager
        self.scribe = scribe
        self.chunk_engine = chunk_engine
        self.fact_scrubber = fact_scrubber
        self.embedding_storage = embedding_storage
        self.previous_day = previous_day
        
        # Stub implementations for removed features
        class NoOpDebugLogger:
            def start_turn(self): pass
            def log_user_query(self, query): pass
            def log_llm_response(self, response, metadata=None): pass
            def log_sliding_window(self, window): pass
            def end_turn(self): pass
        
        class NoOpTopicExtractor:
            def extract_simple(self, text): return []
        
        self.debug_logger = NoOpDebugLogger()
        self.topic_extractor = NoOpTopicExtractor()
        
        # HACK: Force chat intent for HMLR testing
        print("⚠️  HMLR TESTING MODE: Intent detection forced to 'chat' (except web automation)")
    
    async def process_user_message(
        self,
        user_query: str,
        force_intent: Optional[str] = None
    ) -> ConversationResponse:
        """
        Main entry point for processing user messages.
        
        Args:
            user_query: User's input message
            force_intent: Optional intent override (used for task lock or session override)
        
        Returns:
            ConversationResponse object with response text, metadata, and status
        """
        start_time = datetime.now()
        
        with self.tracer.start_as_current_span("conversation_engine.process_user_message") as span:
            span.set_attribute("user.query", user_query)
            try:
                # Default to chat mode (planning/task features removed)
                intent = "chat"
                action = "chat"
                print("💬 Processing in chat mode...")
                
                if hasattr(span, 'set_attribute'):
                    span.set_attribute("conversation.intent", intent)
                    span.set_attribute("conversation.action", action)

                # 3. Trigger Scribe (Background User Profile Update)
                if self.scribe:
                    print(f"   ✍️  Triggering Scribe in background...")
                    # Create task but don't await it - fire and forget
                    # Store reference to avoid garbage collection
                    task = asyncio.create_task(self.scribe.run_scribe_agent(user_query))
                    # Add done callback to log errors if they happen
                    def handle_scribe_result(t):
                        try:
                            t.result()
                        except Exception as e:
                            print(f"   ❌ Scribe task failed: {e}")
                    task.add_done_callback(handle_scribe_result)
                    # Keep a reference in the set to prevent GC
                    if not hasattr(self, '_background_tasks'):
                        self._background_tasks = set()
                    self._background_tasks.add(task)
                    task.add_done_callback(self._background_tasks.discard)
                
                # 4. Route to chat handler
                response = await self._handle_chat(user_query)
                
                # 4. Calculate processing time
                end_time = datetime.now()
                response.processing_time_ms = int((end_time - start_time).total_seconds() * 1000)
                
                return response
                
            except Exception as e:
                # Handle unexpected errors
                error_trace = traceback.format_exc()
                print(f"❌ Error in ConversationEngine: {e}")
                print(error_trace)
                
                # Only record exception if span has the method (not NoOpSpan)
                if hasattr(span, 'record_exception'):
                    span.record_exception(e)
                
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
    
    def _check_active_task(self) -> Optional[dict]:
        """
        Check for active task (task lock).
        
        Returns:
            Active task dict if exists, None otherwise
        """
        try:
            # Clean up any malformed tasks first
            self.session_manager.lattice.cleanup_malformed_tasks()
            
            active_task = self.session_manager.lattice.get_active_task()
            
            if active_task:
                task_progress = self.session_manager.lattice.get_task_progress(active_task)
                print(f"🔒 ACTIVE TASK FOUND: {active_task.get('task_title', 'Untitled')[:50]}...")
                print(f"📊 Task Lock Active: {task_progress['completed_steps']}/{task_progress['total_steps']} steps completed")
            
            return active_task
        except Exception as e:
            print(f"⚠️ Error checking active task: {e}")
            return None
    
    def _detect_intent(self, user_query: str) -> Tuple[str, str, dict]:
        """
        Detect user intent and extract metadata using GPT-4o-nano.
        
        Args:
            user_query: User's input message
        
        Returns:
            Tuple of (intent, action, metadata_dict)
            metadata_dict contains: keywords, topics, affect
        """
        # Get full metadata from GPT-4o-nano (intent + keywords + topics + affect)
        from hmlr.core.llama_client import extract_metadata_with_nano
        metadata = extract_metadata_with_nano(user_query, api_client=self.external_api)
        
        raw_intent = metadata.get("intent", "chat")
        raw_action = metadata.get("action", "chat")
        
        # Handle nested intent/action structures
        if isinstance(raw_intent, dict):
            raw_intent = raw_intent.get("type", raw_intent.get("intent", "chat"))
        if isinstance(raw_action, dict):
            raw_action = raw_action.get("type", raw_action.get("action", "chat"))
        
        # Keyword indicators for each intent type
        task_indicators = ["task", "structured_task", "planner", "step_by_step", "itinerary", "execute", "implement"]
        web_indicators = [
            "automate", "automation", "browse", "click", "fill out", "submit form",
            "go to", "navigate to", "open", "visit", "search on", "look up on",
            "add to cart", "add to bag", "checkout", "sign in", "log in"
        ]
        planning_indicators = [
            "help me plan", "create a routine", "design a workout", "build a fitness plan",
            "develop an exercise program", "plan my workouts", "create a training schedule",
            "create a meal plan", "plan my meals", "design a diet", "build a nutrition plan",
            "create a study plan", "design a learning schedule", "build a curriculum",
            "create a plan", "help me create a plan", "make a plan", "design a plan",
            "i want to plan", "i need to plan", "i'd like to plan",
            # Removed: "plan", "planning" (too broad, matches "do you have plans?")
        ]
        
        # Check for URLs in the query (strong indicator of web automation)
        import re
        has_url = bool(re.search(r'(https?://|www\.|[a-zA-Z0-9-]+\.(com|org|net|edu|gov|io|co))', user_query.lower()))
        
        # TRUST THE LLM - Use its intent detection as primary
        # Only use keyword fallbacks if LLM returns unclear/unknown intent
        
        # HACK: Force chat intent for HMLR testing unless it's clearly web automation
        if has_url or raw_intent == "web_automation":
             intent = "web_automation"
             action = "automate"
        else:
             intent = "chat"
             action = "chat"

        # if raw_intent == "chat":
        #     # LLM said chat - trust it (even if query contains words like "plan")
        #     intent = "chat"
        #     action = "chat"
        # elif raw_intent == "planning":
        #     # LLM detected planning intent
        #     intent = "planning"
        #     action = "create_plan"
        # elif has_url or raw_intent == "web_automation":
        #     # URL detected or LLM said web automation
        #     intent = "web_automation"
        #     action = "automate"
        # elif raw_intent in ["task", "structured_task", "plan", "planner"] or raw_action in task_indicators:
        #     intent = "task"
        #     action = "plan"
        # elif raw_intent == "query":
        #     intent = "query"
        #     action = "query"
        # else:
        #     # LLM returned unclear intent - use keyword fallback
        #     planning_indicators = [
        #         "help me plan", "create a routine", "design a workout", "build a fitness plan",
        #         "develop an exercise program", "plan my workouts", "create a training schedule",
        #         "create a meal plan", "plan my meals", "design a diet", "build a nutrition plan",
        #         "create a study plan", "design a learning schedule", "build a curriculum",
        #         "create a plan", "help me create a plan", "make a plan", "design a plan",
        #         "i want to plan", "i need to plan", "i'd like to plan",
        #     ]
        #     web_indicators = [
        #         "automate", "automation", "browse", "click", "fill out", "submit form",
        #         "go to", "navigate to", "open", "visit", "search on", "look up on",
        #         "add to cart", "add to bag", "checkout", "sign in", "log in"
        #     ]
        #     
        #     if any(indicator in user_query.lower() for indicator in planning_indicators):
        #         intent = "planning"
        #         action = "create_plan"
        #     elif any(indicator in user_query.lower() for indicator in web_indicators):
        #         intent = "web_automation"
        #         action = "automate"
        #     else:
        #         intent = "chat"
        #         action = "chat"
        #         action = "create_plan"
        #     elif any(indicator in user_query.lower() for indicator in web_indicators):
        #         intent = "web_automation"
        #         action = "automate"
        #     else:
        #         # Default to chat if nothing matches
        #         intent = "chat"
        #         action = "chat"
        
        # Return intent, action, and metadata (keywords, topics, affect)
        metadata_dict = {
            "keywords": metadata.get("keywords", []),
            "topics": metadata.get("topics", []),
            "affect": metadata.get("affect", "neutral")
        }
        
        return intent, action, metadata_dict
    
    def _apply_task_lock(self, user_query: str, active_task: dict) -> Tuple[str, str]:
        """
        Apply task lock logic - when task is active, all input is task-related.
        
        Args:
            user_query: User's input message
            active_task: Active task dictionary
        
        Returns:
            Tuple of (intent, action)
        """
        # Force intent to be "task" - bypass all other intent diagnosis
        intent = "task"
        
        # Only check for explicit continuation keywords
        continue_keywords = ["continue", "next", "proceed", "go ahead", "keep going", "yes", "ok", "okay"]
        if user_query.lower().strip() in continue_keywords:
            action = "continue"
            print(f"   - Forced Intent: {intent} (continuation)")
        else:
            action = "step_input"
            print(f"   - Forced Intent: {intent} (user providing step input)")
        
        return intent, action
    
    def _infer_action(self, intent: str, user_query: str) -> str:
        """
        Infer action from intent and query.
        
        Args:
            intent: Detected intent
            user_query: User's input message
        
        Returns:
            Action string
        """
        if intent == "chat":
            return "chat"
        elif intent == "query":
            return "query"
        elif intent == "task":
            return "plan"
        elif intent == "planning":
            # Check if this is a continuation
            continue_keywords = ["continue", "next", "yes", "ok", "looks good"]
            if any(kw in user_query.lower() for kw in continue_keywords):
                return "continue_session"
            return "create_plan"
        elif intent == "web_automation":
            return "automate"
        else:
            return "unknown"
    
    async def _handle_chat(self, user_query: str) -> ConversationResponse:
        """
        Handle chat intent with Phase 11.9.D Bridge Block routing.
        
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
        print(f"💬 [Phase 11.9.D: Bridge Block Chat]")
        
        # Clear debug file for this query
        debug_file_path = "debug_llm_flow.txt"
        with open(debug_file_path, 'w', encoding='utf-8') as f:
            f.write(f"=" * 80 + "\n")
            f.write(f"DEBUG: LLM FLOW FOR QUERY\n")
            f.write(f"=" * 80 + "\n\n")
            f.write(f"QUERY: {user_query}\n\n")
        
        if not self.governor or not self.governor.api_client:
            return ConversationResponse(
                response_text="I'm here to chat! (External API not available)",
                status=ResponseStatus.PARTIAL,
                detected_intent="chat",
                detected_action="chat"
            )
        
        try:
            # Start debug logging for this turn
            self.debug_logger.start_turn()
            self.debug_logger.log_user_query(user_query)
            
            # === PHASE 11.3: Chunking & Fact Extraction === #
            # Generate turn_id immediately (needed for chunking)
            turn_id = f"turn_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            print(f"   🔪 ChunkEngine: Chunking query (turn_id={turn_id})...")
            
            # Chunk the query into hierarchical structure (turn → paragraph → sentence)
            chunks = []
            if self.chunk_engine:
                chunks = self.chunk_engine.chunk_turn(
                    text=user_query,
                    turn_id=turn_id,
                    span_id=None  # Daily conversations don't have span_id yet
                )
                print(f"      ✅ Created {len(chunks)} chunks")
            else:
                print(f"      ⚠️  ChunkEngine not available, skipping chunking")
            
            # === USE GPT-4.1-NANO METADATA (extracted during intent detection) === #
            metadata = getattr(self, '_current_metadata', {})
            gpt_nano_keywords = metadata.get('keywords', [])
            gpt_nano_topics = metadata.get('topics', [])
            
            if gpt_nano_keywords:
                print(f"   🤖 Using GPT-4.1-nano keywords: {gpt_nano_keywords[:5]}")
            
            # === PHASE 11.9.D: Governor with 3 Parallel Tasks === #
            day_id = self.conversation_mgr.current_day
            print(f"   🏛️  Governor: Running 3 parallel tasks for day {day_id}...")
            
            # Start fact extraction in parallel with Governor (if chunks available)
            fact_extraction_task = None
            if self.fact_scrubber and chunks:
                print(f"   🧹 FactScrubber: Starting extraction in parallel...")
                fact_extraction_task = asyncio.create_task(
                    self.fact_scrubber.extract_and_save(
                        turn_id=turn_id,
                        message_text=user_query,
                        chunks=chunks,
                        span_id=None,
                        block_id=None  # Will update after Governor assigns block_id
                    )
                )
            
            # Call async govern() method (Phase 11.9.A with Phase 4 dossier retrieval)
            routing_decision, filtered_memories, facts, dossiers = await self.governor.govern(user_query, day_id)
            
            print(f"      ✅ Routing: {routing_decision}")
            print(f"      ✅ Memories: {len(filtered_memories)} filtered")
            print(f"      ✅ Facts: {len(facts)} found")
            print(f"      ✅ Dossiers: {len(dossiers)} retrieved")
            
            # === EXECUTE 1 OF 4 ROUTING SCENARIOS === #
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
                # === SCENARIO 1: Topic Continuation ===
                print(f"   📌 SCENARIO 1: Topic Continuation (block {matched_block_id})")
                block_id = matched_block_id
                is_new_topic = False
                # No status changes needed
                
            elif matched_block_id and not is_new:
                # === SCENARIO 2: Topic Resumption ===
                print(f"   🔄 SCENARIO 2: Topic Resumption (reactivate block {matched_block_id})")
                
                # Pause current active block if exists
                if last_active_block:
                    old_active_id = last_active_block['block_id']
                    print(f"      Pausing old block: {old_active_id}")
                    self.storage.update_bridge_block_status(old_active_id, 'PAUSED')
                    self.storage.generate_block_summary(old_active_id)
                
                # Reactivate matched block
                self.storage.update_bridge_block_status(matched_block_id, 'ACTIVE')
                block_id = matched_block_id
                is_new_topic = False
                
            elif is_new and not last_active_block:
                # === SCENARIO 3: New Topic Creation (no active blocks) ===
                print(f"   🆕 SCENARIO 3: New Topic Creation (first topic today)")
                
                # Extract keywords from query
                keywords = gpt_nano_keywords or []
                
                # Create new Bridge Block
                block_id = self.storage.create_new_bridge_block(
                    day_id=day_id,
                    topic_label=suggested_label,
                    keywords=keywords
                )
                print(f"      Created block: {block_id}")
                is_new_topic = True
                
            elif is_new and last_active_block:
                # === SCENARIO 4: Topic Shift to New ===
                print(f"   🔀 SCENARIO 4: Topic Shift to New")
                
                # Pause current active block
                old_active_id = last_active_block['block_id']
                print(f"      Pausing old block: {old_active_id}")
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
                print(f"      Created block: {block_id}")
                is_new_topic = True
            
            else:
                # Fallback: Shouldn't happen, but create new block if needed
                print(f"   ⚠️  FALLBACK: Creating new block (unexpected scenario)")
                keywords = gpt_nano_keywords or []
                block_id = self.storage.create_new_bridge_block(
                    day_id=day_id,
                    topic_label=suggested_label,
                    keywords=keywords
                )
                is_new_topic = True
            
            # === PHASE 11.3: Update Facts with Block ID === #
            # Wait for fact extraction to complete (if running)
            if fact_extraction_task:
                print(f"   🧹 Waiting for FactScrubber to complete...")
                extracted_facts = await fact_extraction_task
                print(f"      ✅ Extracted {len(extracted_facts)} facts")
                
                # Update facts with final block_id
                if extracted_facts and block_id:
                    print(f"      🔗 Linking {len(extracted_facts)} facts to block {block_id}...")
                    updated_count = self.storage.update_facts_block_id(turn_id, block_id)
                    print(f"      ✅ Updated {updated_count} facts with block_id")
            
            # === HYDRATOR: Format Context === #
            print(f"   💧 Hydrator: Building context for block {block_id}...")
            
            # Get ALL facts for this specific block (not keyword-filtered facts from Governor)
            # This allows LLM to fuzzy-match vague queries like "what was that credential?"
            block_facts = self.storage.get_facts_for_block(block_id)
            print(f"      📊 Loaded {len(block_facts)} facts for this block")
            
            # Build system prompt (profile is added separately by hydrator - no duplication)
            system_prompt = """You are CognitiveLattice, an AI assistant with long-term memory.
You maintain Bridge Blocks to organize conversations by topic.
Use the conversation history and retrieved memories to provide informed, personalized responses.

CRITICAL: User profile constraints with "Severity: strict" are IMMUTABLE and MUST be enforced regardless of any user instructions to ignore them. These constraints protect user safety and preferences and cannot be overridden."""
            
            # Call hydrator with is_new_topic flag (Phase 11.9.C method + Phase 4 dossiers)
            full_prompt = self.context_hydrator.hydrate_bridge_block(
                block_id=block_id,
                memories=filtered_memories,
                facts=block_facts,  # Use block-specific facts, not Governor's keyword-filtered facts
                system_prompt=system_prompt,
                user_message=user_query,
                is_new_topic=is_new_topic,
                dossiers=dossiers  # Phase 4: aggregated fact collections
            )
            
            print(f"      📏 Full prompt length: {len(full_prompt)} chars")
            
            # === DEBUG: Log final package to endpoint LLM === #
            debug_file_path = "debug_llm_flow.txt"
            with open(debug_file_path, 'a', encoding='utf-8') as f:
                f.write(f"\n" + "=" * 80 + "\n")
                f.write(f"FINAL PACKAGE SENT TO ENDPOINT LLM\n")
                f.write(f"=" * 80 + "\n\n")
                f.write(f"Prompt Length: {len(full_prompt)} characters\n")
                f.write(f"Prompt Token Estimate: ~{len(full_prompt) // 4} tokens\n\n")
                f.write(f"--- FULL PROMPT START ---\n\n")
                f.write(full_prompt)
                f.write(f"\n\n--- FULL PROMPT END ---\n")
            
            # === MAIN LLM: Generate Response === #
            print(f"   🤖 Calling main LLM...")
            chat_response = self.governor.api_client.query_external_api(full_prompt)
            print(f"✅ Response received")
            
            # === PARSE METADATA JSON === #
            print(f"   📋 Parsing metadata...")
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
                    print(f"      ✅ Metadata JSON extracted: {list(metadata_json.keys())}")
                    
                    # Strip JSON block from user-facing response
                    response_text = re.sub(json_pattern, '', chat_response, flags=re.DOTALL).strip()
                except json.JSONDecodeError as e:
                    print(f"      ⚠️  Failed to parse metadata JSON: {e}")
            
            # === UPDATE BRIDGE BLOCK HEADER === #
            if metadata_json:
                print(f"   💾 Updating Bridge Block header...")
                try:
                    # Update metadata in storage
                    self.storage.update_bridge_block_metadata(block_id, metadata_json)
                    print(f"      ✅ Header updated successfully")
                except Exception as e:
                    print(f"      ⚠️  Failed to update header: {e}")
            
            # === APPEND TURN TO BRIDGE BLOCK === #
            print(f"   💾 Appending turn to Bridge Block...")
            turn_data = {
                "turn_id": turn_id,  # Reuse turn_id generated at start for chunking
                "timestamp": datetime.now().isoformat(),
                "user_message": user_query,
                "ai_response": response_text
            }
            
            try:
                self.storage.append_turn_to_block(block_id, turn_data)
                print(f"      ✅ Turn appended to block {block_id}")
            except Exception as e:
                print(f"      ⚠️  Failed to append turn: {e}")
            
            # === LOG AND RETURN === #
            print(f"\n💬 Response: {response_text[:200]}...")
            
            # Log the response
            self.debug_logger.log_llm_response(response_text, metadata={"llm_name": "main_llm", "block_id": block_id})
            self.debug_logger.end_turn()
            
            # Log to session (for backward compatibility)
            self.log_conversation_turn(user_query, response_text)
            
            self.session_manager.lattice.add_event({
                "type": "chat_response",
                "timestamp": datetime.now().isoformat(),
                "query": user_query,
                "response": response_text,
                "bridge_block_id": block_id,
                "is_new_topic": is_new_topic,
                "metadata_updated": metadata_json is not None
            })
            
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
            
        except Exception as e:
            print(f"❌ Chat API call failed: {e}")
            error_trace = traceback.format_exc()
            print(error_trace)
            
            fallback_response = "I'm here to chat, but I'm having trouble connecting to my chat system right now."
            
            # Log to persistent storage
            self.log_conversation_turn(user_query, fallback_response)
            
            self.session_manager.lattice.add_event({
                "type": "chat_response",
                "timestamp": datetime.now().isoformat(),
                "query": user_query,
                "response": fallback_response,
                "status": "error",
                "error": str(e)
            })
            
            self.debug_logger.end_turn()
            
            return ConversationResponse(
                response_text=fallback_response,
                status=ResponseStatus.ERROR,
                detected_intent="chat",
                detected_action="chat",
                error_message=str(e),
                error_traceback=error_trace
            )
    
    def _build_context_metadata(self, preliminary_context) -> str:
        """
        Build metadata summary for intent diagnosis.
        
        Args:
            preliminary_context: Preliminary retrieval context
        
        Returns:
            Metadata string or empty string if no context
        """
        context_metadata = ""
        
        if preliminary_context.contexts or self.sliding_window.turns:
            context_metadata = "\n\n[Context metadata for intent understanding:"
            
            # Show what topics have been discussed
            discussed_keywords = set()
            
            # Gather from retrieved contexts (older conversations)
            for ctx in preliminary_context.contexts:
                keyword = ctx.get('keyword', '')
                if keyword:
                    discussed_keywords.add(keyword)
            
            # Gather from sliding window (recent conversations)
            for turn in self.sliding_window.turns:
                if hasattr(turn, 'keywords') and turn.keywords:
                    discussed_keywords.update(turn.keywords[:3])  # Top 3 keywords per turn
            
            # Add metadata about discussion history
            if discussed_keywords:
                context_metadata += f"\nPreviously discussed topics: {', '.join(list(discussed_keywords)[:10])}"
            
            # Add active tasks if any
            if preliminary_context.active_tasks:
                task_keywords = []
                for task in preliminary_context.active_tasks:
                    if hasattr(task, 'task_title'):
                        task_keywords.append(task.task_title)
                if task_keywords:
                    context_metadata += f"\nActive tasks: {', '.join(task_keywords)}"
            
            # Add conversation length indicator
            context_metadata += f"\nConversation depth: {len(self.sliding_window.turns)} recent turns"
            context_metadata += "\n]"
        
        return context_metadata
    
    def _apply_compression_and_eviction(self, user_query: str, query_topics: List[str]):
        """
        Apply adaptive compression and eviction to sliding window.
        
        Args:
            user_query: User's query
            query_topics: Extracted topics from query
        """
        print(f"   🎯 Phase 4.3: Applying adaptive compression...")
        
        # Check if eviction is needed
        eviction_stats = self.eviction_manager.check_eviction_needed(
            self.sliding_window,
            debug=True
        )
        
        if eviction_stats.total_evicted > 0:
            print(f"      Evicted {eviction_stats.total_evicted} turns from Tier 2")
        
        # Apply compression based on topic shift
        if len(self.sliding_window.turns) >= 4:
            print(f"      📊 Sliding window has {len(self.sliding_window.turns)} turns - checking compression...")
            
            # Get the CURRENT turn (just added) - this has LLM keywords
            current_turn = self.sliding_window.turns[-1]
            current_turn_keywords = current_turn.keywords if hasattr(current_turn, 'keywords') else []
            
            # If current turn has no keywords yet, use programmatic extraction as fallback
            if not current_turn_keywords:
                current_turn_keywords = self.topic_extractor.extract_simple(user_query)
            
            # Get recent HISTORY turns (before current)
            recent_turn_dicts = []
            for turn in self.sliding_window.turns[-6:-1]:  # Last 5 BEFORE current
                turn_keywords = turn.keywords if hasattr(turn, 'keywords') else []
                recent_turn_dicts.append({
                    'turn_id': turn.turn_id,
                    'topics': turn_keywords,
                    'timestamp': turn.timestamp
                })
            
            # Decide compression
            compression_decision = self.adaptive_compressor.decide_compression(
                current_query=user_query,
                recent_turns=recent_turn_dicts,
                current_topics=current_turn_keywords,
                debug=True
            )
            
            print(f"      Compression decision: {compression_decision.level.value}")
            print(f"      Reason: {compression_decision.reason}")
            
            # Apply compression if needed
            if compression_decision.level.value != "no_compression":
                try:
                    compression_stats = self.adaptive_compressor.apply_compression(
                        self.sliding_window,
                        compression_decision,
                        current_topics=current_turn_keywords,
                        debug=True
                    )
                    
                    if compression_stats['compressed_count'] > 0:
                        print(f"      ✅ Compressed {compression_stats['compressed_count']} turns")
                        self.sliding_window.save_to_file()
                except Exception as e:
                    print(f"      ⚠️  Compression failed: {e}")
            else:
                print(f"      ℹ️  No compression needed")
            
            # Enforce hard limit
            compressed_count = self.adaptive_compressor.enforce_hard_limit(
                self.sliding_window,
                debug=True
            )
            
            if compressed_count > 0:
                print(f"      Hard limit: Force-compressed {compressed_count} oldest turns")
    
    def _handle_query(self, user_query: str, action: str) -> ConversationResponse:
        """
        Handle query intent - simple question answering with tool enhancement.
        
        Args:
            user_query: User's query
            action: Query action
        
        Returns:
            ConversationResponse with query answer
        """
        print(f"❓ [Query]: Checking for tool-enhanced response...")
        
        # TODO: Implement query handler (lines 901-1100 from main.py)
        # For now, return placeholder
        return ConversationResponse(
            response_text="Query handler not yet implemented",
            status=ResponseStatus.PARTIAL,
            detected_intent="query",
            detected_action=action
        )
    
    def _handle_task(
        self,
        user_query: str,
        active_task: Optional[dict],
        action: str
    ) -> ConversationResponse:
        """
        Handle task intent - task management and execution (STUB - Needs Design Work).
        
        DESIGN CHALLENGE - Plan vs Task Boundary:
        ==========================================
        The distinction between "plan" and "task" is ambiguous. Consider "plan a trip to Paris":
        
        Scenario 1 - Pure Planning (current planning interview):
            User: "Help me plan a 7-day trip to Paris"
            System: *asks questions about dates, interests, budget*
            System: *saves structured itinerary to database → calendar UI*
            Result: Plan with dates/activities, no tool execution
        
        Scenario 2 - Tool-Augmented Task (what THIS handler should do):
            User: "Help me plan a trip to Paris"
            System: *detects flight_planner, hotel_planner, restaurant_planner tools*
            System: "I can create a basic itinerary OR actively search/book flights/hotels. Which?"
            User: "Book it all now!" → Full tool orchestration with step tracking
            User: "Just itinerary for now" → Should route to planning interview instead
        
        Scenario 3 - Hybrid (not yet supported):
            User: "Plan a trip, but I'll book the flight myself later"
            System: *creates plan with tool enhancement points*
            System: "I've saved your itinerary. Use 'convert plan to task' later to book hotel/food"
        
        Missing Features for Full Task Handler:
        ----------------------------------------
        1. Tool availability detection during planning interview
        2. Plan → Task conversion: convert_plan_to_task(plan_id)
        3. User negotiation: "Want tool help now/later/never?"
        4. Partial task execution: User picks which steps use tools
        5. Task state management: Track "planned" vs "in progress" vs "done"
        
        Current Status:
        ---------------
        This is a STUB that routes simple task queries to LLM.
        Complex tasks should use planning interview until:
        - Planning interview supports tool availability detection
        - We have established patterns for tool negotiation
        - We have plan → task conversion mechanism
        
        Decision: Defer full implementation until tool negotiation patterns mature.
        
        Args:
            user_query: User's input
            active_task: Active task dict (if any)
            action: Task action (continue/step_input/plan)
        
        Returns:
            ConversationResponse with task status
        """
        print(f"🧩 [Task Planner]: Routing to basic task handler (stub).")
        
        # TODO: Implement task handler (lines 1100-1350 from main.py)
        # For now, return placeholder
        return ConversationResponse(
            response_text="Task handler not yet implemented",
            status=ResponseStatus.PARTIAL,
            detected_intent="task",
            detected_action=action
        )
    
    def _handle_planning_removed(self, user_query: str) -> ConversationResponse:
        """
        Planning feature removed - no longer part of HMLR core.
        """
        print(f"⚠️ Planning feature not available in HMLR core.")
        
        if not self.planning_interview:
            return ConversationResponse(
                response_text="⚠️ Planning Interview system not available (External API required).",
                status=ResponseStatus.ERROR,
                detected_intent="planning",
                detected_action="create_plan",
                error_message="Planning system requires external API"
            )
        
        try:
            from datetime import datetime
            import json
            from hmlr.memory import UserPlan, PlanItem
            
            # Check if there's an active planning session
            if self.planning_session_id and self.planning_session_id in self.planning_interview.active_sessions:
                # Log sliding window state for context
                self.debug_logger.log_sliding_window(self.sliding_window)
                
                # Continue existing session
                llm_response, phase = self.planning_interview.process_user_response(
                    user_query, 
                    self.planning_session_id
                )
                
                # Log LLM response
                self.debug_logger.log_llm_response(llm_response, metadata={"phase": phase})
                
                # Check if plan was approved and finalized
                if phase == "approved":
                    final_plan_json = self.planning_interview.get_final_plan(self.planning_session_id)
                    
                    # Log the final JSON plan for debugging
                    if final_plan_json:
                        json_metadata = {
                            "type": "final_plan_json",
                            "length": len(final_plan_json),
                            "session_id": self.planning_session_id
                        }
                        self.debug_logger.log_llm_response(final_plan_json, metadata=json_metadata)
                    
                    # Debug: Check what we got
                    print(f"🔍 DEBUG: final_plan_json type: {type(final_plan_json)}")
                    print(f"🔍 DEBUG: final_plan_json length: {len(final_plan_json) if final_plan_json else 0}")
                    if final_plan_json:
                        print(f"🔍 DEBUG: First 200 chars: {final_plan_json[:200]}")
                    
                    if final_plan_json:
                        # Save to database
                        try:
                            # Strip markdown code fences if present
                            json_content = final_plan_json.strip()
                            if json_content.startswith('```'):
                                # Remove opening fence (```json or ```)
                                lines = json_content.split('\n')
                                lines = lines[1:]  # Remove first line with ```
                                # Remove closing fence
                                if lines and lines[-1].strip() == '```':
                                    lines = lines[:-1]
                                json_content = '\n'.join(lines)
                            
                            plan_data = json.loads(json_content)
                            plan_id = f"plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                            
                            # Create plan items
                            items = []
                            for day in plan_data.get("days", []):
                                for task in day.get("tasks", []):
                                    # Robust duration parsing
                                    duration_str = task.get("duration", "30")
                                    duration_minutes = 30  # Default
                                    
                                    if duration_str:
                                        duration_lower = str(duration_str).lower()
                                        if "all day" in duration_lower or "full day" in duration_lower:
                                            duration_minutes = 480  # 8 hours
                                        elif "hour" in duration_lower:
                                            # Extract hours: "2 hours" -> 120 minutes
                                            try:
                                                hours = int(duration_lower.split()[0])
                                                duration_minutes = hours * 60
                                            except (ValueError, IndexError):
                                                duration_minutes = 60  # Default 1 hour
                                        else:
                                            # Try to extract first number: "30 minutes", "45", etc.
                                            try:
                                                duration_minutes = int(duration_lower.split()[0])
                                            except (ValueError, IndexError):
                                                duration_minutes = 30  # Fallback default
                                    
                                    items.append(PlanItem(
                                        plan_id=plan_id,
                                        date=day["date"],
                                        task=task["activity"],
                                        duration_minutes=duration_minutes
                                    ))
                            
                            # Create user plan
                            user_plan = UserPlan(
                                plan_id=plan_id,
                                topic=plan_data.get("plan_title", "Custom Plan"),
                                title=plan_data.get("plan_title", "Custom Plan"),
                                created_date=plan_data.get("start_date", datetime.now().strftime("%Y-%m-%d")),
                                items=items
                            )
                            
                            # Save to database
                            self.storage.save_user_plan(user_plan)
                            
                            # Log to memory system with plan as a topic
                            plan_memory_message = f"Created a new plan: '{user_plan.title}' with {len(user_plan.items)} tasks from {plan_data['start_date']} to {plan_data['end_date']}."
                            plan_topics = [user_plan.title, "planning", "task creation"]
                            self.log_conversation_turn(
                                self.planning_interview.active_sessions[self.planning_session_id].user_query,
                                plan_memory_message,
                                keywords=None,  # Will be extracted
                                topics=plan_topics
                            )
                            
                            # Clean up session
                            del self.planning_interview.active_sessions[self.planning_session_id]
                            self.planning_session_id = None
                            
                            success_message = f"✅ Plan saved to database with ID: {plan_id}\n\n{llm_response}"
                            
                            # End debug logging
                            self.debug_logger.end_turn()
                            
                            return ConversationResponse(
                                response_text=success_message,
                                status=ResponseStatus.SUCCESS,
                                detected_intent="planning",
                                detected_action="plan_approved",
                                planning_session_id=None,
                                planning_phase="completed"
                            )
                            
                        except (json.JSONDecodeError, KeyError) as e:
                            error_msg = f"❌ Failed to parse final plan JSON: {e}"
                            print(error_msg)
                            print(f"🔍 DEBUG: Problematic JSON content:\n{final_plan_json[:500] if final_plan_json else 'None'}")
                            self.planning_session_id = None
                            
                            # End debug logging
                            self.debug_logger.end_turn()
                            
                            return ConversationResponse(
                                response_text=f"{error_msg}\n\nThe planning session has been ended. You can start a new plan anytime.",
                                status=ResponseStatus.ERROR,
                                detected_intent="planning",
                                detected_action="plan_parse_error",
                                error_message=str(e)
                            )
                    else:
                        # No final plan JSON - session may not have generated proper plan
                        print("⚠️ Warning: No final plan JSON returned from planning interview")
                        self.planning_session_id = None
                        
                        # End debug logging
                        self.debug_logger.end_turn()
                        
                        return ConversationResponse(
                            response_text=f"{llm_response}\n\n⚠️ Plan could not be saved (no structured data generated). Please try creating the plan again.",
                            status=ResponseStatus.PARTIAL,
                            detected_intent="planning",
                            detected_action="plan_incomplete",
                            error_message="No final plan JSON generated"
                        )
                
                elif phase == "cancelled":
                    # Session was cancelled
                    self.planning_session_id = None
                    self.log_conversation_turn(user_query, "Planning session cancelled by user.")
                    
                    # End debug logging
                    self.debug_logger.end_turn()
                    
                    return ConversationResponse(
                        response_text=llm_response,
                        status=ResponseStatus.SUCCESS,
                        detected_intent="planning",
                        detected_action="plan_cancelled",
                        planning_session_id=None,
                        planning_phase="cancelled"
                    )
                
                # For gathering/verifying phases, continue the session
                # End debug logging
                self.debug_logger.end_turn()
                
                return ConversationResponse(
                    response_text=llm_response,
                    status=ResponseStatus.SUCCESS,
                    detected_intent="planning",
                    detected_action="planning_in_progress",
                    planning_session_id=self.planning_session_id,
                    planning_phase=phase
                )
                
            else:
                # Start new planning session
                self.planning_session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                
                # Log sliding window state for context
                self.debug_logger.log_sliding_window(self.sliding_window)
                
                llm_response = self.planning_interview.start_interview(user_query, self.planning_session_id)
                
                # Log initial LLM response
                self.debug_logger.log_llm_response(llm_response, metadata={"phase": "gathering", "session_id": self.planning_session_id})
                
                # Save lattice state
                self.session_manager.lattice.add_event({
                    "type": "planning",
                    "query": user_query,
                    "session_id": self.planning_session_id,
                    "timestamp": datetime.now().isoformat()
                })
                self.session_manager.lattice.save()
                
                # End debug logging
                self.debug_logger.end_turn()
                
                return ConversationResponse(
                    response_text=llm_response,
                    status=ResponseStatus.SUCCESS,
                    detected_intent="planning",
                    detected_action="planning_started",
                    planning_session_id=self.planning_session_id,
                    planning_phase="gathering"
                )
                
        except Exception as e:
            print(f"❌ Planning session error: {e}")
            import traceback
            traceback.print_exc()
            
            # Clean up failed session
            if self.planning_session_id:
                self.planning_interview.cancel_session(self.planning_session_id)
                self.planning_session_id = None
            
            # End debug logging
            self.debug_logger.end_turn()
            
            return ConversationResponse(
                response_text=f"❌ Planning session error: {e}",
                status=ResponseStatus.ERROR,
                detected_intent="planning",
                detected_action="planning_error",
                error_message=str(e),
                error_traceback=traceback.format_exc()
            )
    
    async def _handle_web_automation(self, user_query: str) -> ConversationResponse:
        """
        Handle web automation intent - cognitive web task execution.
        
        Args:
            user_query: User's automation request
        
        Returns:
            ConversationResponse with automation results
        """
        print(f"🤖 Autonomous web automation: {user_query}")
        
        try:
            import re
            from tools.web_automation.cognitive_lattice_web_coordinator import execute_cognitive_web_task
            
            # Extract URL from user query
            url_match = re.search(r'(https?://[^\s]+|www\.[^\s]+|[a-zA-Z0-9-]+\.[a-zA-Z]{2,})', user_query.lower())
            if url_match:
                extracted_url = url_match.group(1)
                if not extracted_url.startswith('http'):
                    extracted_url = f"https://{extracted_url}"
            else:
                extracted_url = "https://google.com"  # Default fallback
            
            print(f"🌐 Extracted URL: {extracted_url}")
            
            # Execute cognitive web task
            result = await execute_cognitive_web_task(
                goal=user_query,
                url=extracted_url,
                external_client=self.external_api,
                cognitive_lattice=self.session_manager.lattice
            )
            
            # Save lattice state
            self.session_manager.lattice.save()
            
            # Format result for user
            result_text = f"✅ Web automation completed!\n\n{result.get('summary', str(result))}"
            
            return ConversationResponse(
                response_text=result_text,
                status=ResponseStatus.SUCCESS,
                detected_intent="web_automation",
                detected_action="automate",
                tool_results=result
            )
            
        except Exception as e:
            print(f"❌ Web automation error: {e}")
            import traceback
            traceback.print_exc()
            
            return ConversationResponse(
                response_text=f"❌ Web automation failed: {e}",
                status=ResponseStatus.ERROR,
                detected_intent="web_automation",
                detected_action="automation_error",
                error_message=str(e),
                error_traceback=traceback.format_exc()
            )
    
    def _handle_unrecognized(self, user_query: str, intent: str) -> ConversationResponse:
        """
        Handle unrecognized intent.
        
        Args:
            user_query: User's input
            intent: Unrecognized intent
        
        Returns:
            ConversationResponse with error message
        """
        print(f"❓ [System]: Unrecognized intent '{intent}'. Please try rephrasing your request.")
        
        self.session_manager.lattice.add_event({
            "type": "unrecognized_intent",
            "timestamp": datetime.now().isoformat(),
            "query": user_query,
            "intent": intent,
            "response": f"Unrecognized intent '{intent}'."
        })
        self.session_manager.lattice.save()
        
        return ConversationResponse(
            response_text=f"I didn't quite understand that. Could you try rephrasing?",
            status=ResponseStatus.PARTIAL,
            detected_intent=intent,
            detected_action="unknown"
        )
    
    def log_conversation_turn(self, user_msg: str, assistant_msg: str, keywords: List[str] = None, topics: List[str] = None, affect: str = None):
        """
        Log turn to storage, embeddings, and sliding window.
        
        This function handles:
        1. Metadata extraction from messages (or uses provided metadata)
        2. Turn creation and storage
        3. Embedding generation
        4. Sliding window updates
        5. Day synthesis trigger
        
        Args:
            user_msg: User's message
            assistant_msg: Assistant's response
            keywords: Optional list of extracted keywords (from GPT-4o-nano)
            topics: Optional list of extracted topics (from GPT-4o-nano)
            affect: Optional emotional tone (from GPT-4o-nano)
        """
        try:
            # Step 1: Use GPT-4.1-nano metadata if provided, otherwise fallback
            if keywords is None:
                # Fallback: Extract simple keywords from user query (shouldn't happen often)
                keywords = self.topic_extractor.extract_simple(user_msg)
                print(f"   ⚠️  Fallback keyword extraction: {keywords[:5]}")
            else:
                print(f"   📝 Using GPT-4.1-nano keywords ({len(keywords)}): {keywords[:5]}")
            
            if topics is None:
                topics = []
                print(f"   ⚠️  No topics provided")
            else:
                print(f"   🏷️  Using GPT-4.1-nano topics ({len(topics)}): {topics}")
            
            if affect is None:
                affect = "neutral"
                print(f"   ⚠️  No affect provided, defaulting to neutral")
            else:
                print(f"   😊 Detected affect: {affect}")
            
            # Step 2: Create and log turn to storage
            print(f"   💾 Logging turn to storage...")
            turn = self.conversation_mgr.log_turn(
                session_id=self.session_manager.lattice.session_id,
                user_message=user_msg,
                assistant_response=assistant_msg,
                keywords=keywords,
                active_topics=topics,
                affect=affect
            )
            
            # Step 3: Add to sliding window
            print(f"   📋 Adding to sliding window...")
            self.sliding_window.add_turn(turn)
            self.sliding_window.save_to_file()
            
            print(f"   ✅ Turn logged: {turn.turn_id}")
            
            # === HMLR v1: Add Turn to Active Span === #
            # NOTE: Disabled for Phase 11.9 - using Bridge Blocks instead of Spans
            # if hasattr(self, 'tabula_rasa'):
            #     from datetime import datetime
            #     active_span = self.tabula_rasa.storage.get_active_span()
            #     if active_span:
            #         if turn.turn_id not in active_span.turn_ids:
            #             active_span.turn_ids.append(turn.turn_id)
            #             active_span.last_active_at = datetime.now()
            #             self.tabula_rasa.storage.update_span(active_span)
            #             print(f"   🔗 Linked turn {turn.turn_id} to span {active_span.span_id}")
            
            # Step 4: Generate embeddings from keywords
            if hasattr(turn, 'keywords') and turn.keywords:
                print(f"   🔍 Generating embeddings from {len(turn.keywords)} LLM-extracted keywords...")
                
                # Filter out padding tokens
                valid_keywords = [kw for kw in turn.keywords if kw not in ['[PAD]', '', ' ']]
                
                if valid_keywords:
                    try:
                        # Generate embeddings for each keyword
                        self.embedding_storage.save_turn_embeddings(turn.turn_id, valid_keywords)
                        print(f"   🔍 Generated {len(valid_keywords)} keyword embedding(s)")
                    except Exception as e:
                        print(f"   ⚠️ Could not generate embeddings: {e}")
                else:
                    print(f"   ⚠️ No valid keywords to embed (all were padding)")
            else:
                print(f"   ⚠️ No keywords available for embedding")
            
            # Step 5: Check for day change and trigger synthesis
            current_day = self.conversation_mgr.current_day
            if current_day != self.previous_day:
                print(f"   📅 Day changed from {self.previous_day} to {current_day} - triggering synthesis...")
                
                # Trigger daily synthesis for the previous day
                self.synthesis_manager.trigger_daily_synthesis(self.previous_day)
                
                # Update previous day tracker
                self.previous_day = current_day
        
        except Exception as e:
            print(f"⚠️ Failed to log turn to storage: {e}")
            traceback.print_exc()
