"""
HMLR v1 - Lattice Retrieval & The Governor

This module implements the Read Path of the HMLR system.
1. LatticeRetrieval: Hybrid search to find candidates (wraps LatticeCrawler).
2. TheGovernor: LLM-based gating to filter candidates.


"""

import json
import logging
import re
import os
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
# from hmlr.core.telemetry import get_tracer  # Removed - telemetry module deleted

from hmlr.memory.storage import Storage
from hmlr.core.external_api_client import ExternalAPIClient
from hmlr.core.model_config import model_config
from hmlr.memory.retrieval.crawler import LatticeCrawler
from hmlr.memory.models import Intent, QueryType

logger = logging.getLogger(__name__)

@dataclass
class MemoryCandidate:
    memory_id: str
    content_preview: str
    score: float
    source_type: str # 'turn', 'summary', 'span'
    full_object: Any = None

class LatticeRetrieval:
    """
    Retrieves candidate memories using hybrid search (Vector + Keyword).
    Wraps the existing LatticeCrawler but formats for the Governor.
    """
    def __init__(self, crawler: LatticeCrawler):
        # self.tracer = get_tracer(__name__)  # Removed - telemetry deleted
        self.crawler = crawler

    def retrieve_candidates(self, query: str, intent: Intent, top_k: int = 20) -> List[MemoryCandidate]:
        """
        Get raw candidates from the memory lattice.
        """
        # Tracer span removed - telemetry deleted

        # Use the existing crawler to get contexts
        # We ask for more results than usual because the Governor will filter them down
        retrieved_context = self.crawler.retrieve_context(
            intent=intent,
            current_day_id="CURRENT", # Crawler handles this
            max_results=top_k,
            window=None # We don't want to filter by window here, we want raw candidates
        )

        candidates = []
        
        # Visualization data
        vis_query_vector = None
        vis_candidates = []

        for ctx in retrieved_context.contexts:
            # Crawler returns dicts usually
            mem_id = ctx.get('turn_id') or ctx.get('summary_id') or "unknown"
            text = ctx.get('user_message', '') + " | " + ctx.get('assistant_response', '')
            if not text.strip():
                text = ctx.get('content', str(ctx))
            
            # Capture vectors for Phoenix
            if 'vector' in ctx and ctx['vector'] is not None:
                vec = ctx['vector']
                if hasattr(vec, 'tolist'): vec = vec.tolist()
                vis_candidates.append((vec, text))
            
            if vis_query_vector is None and 'query_vector' in ctx and ctx['query_vector'] is not None:
                q_vec = ctx['query_vector']
                if hasattr(q_vec, 'tolist'): q_vec = q_vec.tolist()
                vis_query_vector = q_vec

            # Truncate preview
            preview = text[:300] + "..." if len(text) > 300 else text
            
            candidates.append(MemoryCandidate(
                memory_id=mem_id,
                content_preview=preview,
                score=ctx.get('similarity', 0.0),
                source_type='turn', # Assuming mostly turns for now
                full_object=ctx
            ))
        
        # Log embeddings to Phoenix
        if vis_query_vector:
            # Tracer span removed - telemetry deleted
            pass

        return candidates

class TheGovernor:
    """  
    Implements 3 parallel tasks + Bridge Block routing:
    - TASK 1: Bridge Block routing (LLM)
    - TASK 2: Memory retrieval + 2-key filtering (Vector + LLM)
    - TASK 3: Fact store lookup (SQLite)
    
    Then executes 1 of 4 routing scenarios based on LLM decision.
    """
    def __init__(
        self, 
        api_client: ExternalAPIClient, 
        storage: Storage,
        crawler: LatticeCrawler,
        profile_path: str = None,
        dossier_retriever = None  
    ):
        self.api_client = api_client
        self.storage = storage
        self.crawler = crawler
        self.dossier_retriever = dossier_retriever
        
        # Default to package-relative path
        if profile_path is None:
            from pathlib import Path
            profile_path = str(Path(__file__).parent.parent.parent / "config" / "user_profile_lite.json")
        
        self.profile = self._load_profile(profile_path)

    def _load_profile(self, path: str) -> Dict[str, str]:
        if not os.path.exists(path):
            logger.info(f"User profile not found at {path}, starting with empty profile.")
            return {}
            
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Malfored user profile JSON at {path}: {e}")
            return {}
        except Exception as e:
            logger.error(f"Unexpected error loading user profile from {path}: {e}", exc_info=True)
            return {}

    async def govern(
        self, 
        query: str, 
        day_id: str,
        candidates: Optional[List[MemoryCandidate]] = None
    ) -> Tuple[Dict[str, Any], List[MemoryCandidate], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """

        
        Executes 4 independent async tasks:
        1. Bridge Block routing (LLM)
        2. Memory retrieval + 2-key filtering (Vector + LLM)
        3. Fact store lookup (SQLite)
        4. Dossier retrieval (fact-level embeddings)
        
        Args:
            query: User query text
            day_id: Current day ID (e.g., "2025-01-15")
            candidates: Optional pre-fetched memory candidates (from retrieval layer)
        
        Returns:
            Tuple of (routing_decision, filtered_memories, facts, dossiers)
            - routing_decision: {matched_block_id, is_new_topic, reasoning, topic_label}
            - filtered_memories: List of MemoryCandidate objects (2-key filtered)
            - facts: List of fact dictionaries from fact_store
            - dossiers: List of dossier dictionaries
        """
        # Tracer span removed - telemetry deleted

        # ===================================================================
        # PARALLEL EXECUTION: 4 Independent Tasks 
        # ===================================================================
        # Note: _lookup_facts is synchronous, so we wrap it in run_in_executor
        loop = asyncio.get_event_loop()
        
        # Execute dossier retrieval if dossier_retriever is available
        if self.dossier_retriever:
            results = await asyncio.gather(
                self._route_to_bridge_block(query, day_id),
                self._retrieve_and_filter_memories(query, day_id, candidates),
                loop.run_in_executor(None, self._lookup_facts, query),
                loop.run_in_executor(None, self._retrieve_dossiers, query),
                return_exceptions=True
            )
            routing_decision, filtered_memories, facts, dossiers = results
        else:
            # Fallback for systems without dossier_retriever
            results = await asyncio.gather(
                self._route_to_bridge_block(query, day_id),
                self._retrieve_and_filter_memories(query, day_id, candidates),
                loop.run_in_executor(None, self._lookup_facts, query),
                return_exceptions=True
            )
            routing_decision, filtered_memories, facts = results
            dossiers = []
        
        # Handle exceptions from parallel tasks
        fail_count = 0
        if isinstance(routing_decision, Exception):
            logger.error(f"CRITICAL: Bridge routing failed: {routing_decision}", exc_info=True)
            routing_decision = {"matched_block_id": None, "is_new_topic": True, "reasoning": f"ERROR: routing_failed ({type(routing_decision).__name__})"}
            fail_count += 1
        
        if isinstance(filtered_memories, Exception):
            logger.error(f"CRITICAL: Memory retrieval failed: {filtered_memories}", exc_info=True)
            filtered_memories = []
            fail_count += 1
        
        if isinstance(facts, Exception):
            logger.error(f"CRITICAL: Fact lookup failed: {facts}", exc_info=True)
            facts = []
            fail_count += 1
        
        if isinstance(dossiers, Exception):
            logger.error(f"CRITICAL: Dossier retrieval failed: {dossiers}", exc_info=True)
            dossiers = []
            fail_count += 1
            
        if fail_count > 0:
            logger.warning(f"Governor: {fail_count} parallel tasks failed. Results will be degraded.")
        
        # ===================================================================
        #  1-HOP CAUSAL HYDRATION
        # ===================================================================
        routing_decision, filtered_memories, facts, dossiers = await self._causal_hydration(
            routing_decision, filtered_memories, facts, dossiers
        )
        
        # Log results
        logger.info(
            f"Governor results (hydrated): "
            f"Matched={routing_decision.get('matched_block_id')}, "
            f"NewTopic={routing_decision.get('is_new_topic')}, "
            f"Memories={len(filtered_memories)}, "
            f"Facts={len(facts)}, "
            f"Dossiers={len(dossiers)}"
        )
        
        return routing_decision, filtered_memories, facts, dossiers

    async def _causal_hydration(
        self, 
        routing_decision: Dict[str, Any], 
        memories: List[MemoryCandidate], 
        facts: List[Dict[str, Any]], 
        dossiers: List[Dict[str, Any]]
    ) -> Tuple[Dict[str, Any], List[MemoryCandidate], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Implements 1-hop causal linkage between facts and turns.
        Bidirectional: FACTS ↔ TURNS
        - If fact is chosen → load its source turn
        - If turn is chosen → load its extracted facts
        Strictly 1-hop: No recursive expansion.
        """
        logger.info("Governor: Performing 1-hop causal hydration")
        
        # Track what we already have to avoid duplicates
        seen_turn_ids = set()
        for m in memories:
            if m.source_type == 'turn':
                seen_turn_ids.add(m.memory_id)
            elif m.source_type == 'summary' and m.full_object:
                if hasattr(m.full_object, 'source_turn_id'):
                    seen_turn_ids.add(m.full_object.source_turn_id)
                elif isinstance(m.full_object, dict) and 'source_turn_id' in m.full_object:
                    seen_turn_ids.add(m.full_object['source_turn_id'])

        seen_fact_ids = {f.get('fact_id') for f in facts if f.get('fact_id')}
        
        new_memories = list(memories)
        new_facts = list(facts)
        
        hydrated_turn_count = 0
        hydrated_fact_count = 0
        
        # 1. DOSSIER → TURNS (Dossier facts bring in their originating turns)
        for dossier in dossiers:
            dossier_id = dossier.get('dossier_id')
            d_facts = self.storage.get_dossier_facts(dossier_id)
            for df in d_facts:
                turn_id = df.get('source_turn_id')
                if turn_id and turn_id not in seen_turn_ids:
                    turn_data = self.storage.get_turn_by_id(turn_id)
                    if turn_data:
                        # Handle both dict and object formats
                        if isinstance(turn_data, dict):
                            user_msg = turn_data.get('user_message', '')
                        else:
                            user_msg = getattr(turn_data, 'user_message', '')
                        
                        preview = f"[Hydrated from dossier] {user_msg[:100]}" if user_msg else "[Hydrated turn]"
                        
                        logger.debug(f"Hydrating turn {turn_id} from dossier {dossier_id}")
                        new_memories.append(MemoryCandidate(
                            memory_id=turn_id,
                            content_preview=preview,
                            score=model_config.DEFAULT_CANDIDATE_SCORE,
                            source_type='turn',
                            full_object=turn_data
                        ))
                        seen_turn_ids.add(turn_id)
                        hydrated_turn_count += 1

        # 2. FACTS → TURNS (Direct fact hits bring in their originating turns)
        for fact in facts:
            turn_id = fact.get('source_turn_id')
            if turn_id and turn_id not in seen_turn_ids:
                turn_data = self.storage.get_turn_by_id(turn_id)
                if turn_data:
                    # Handle both dict and object formats
                    if isinstance(turn_data, dict):
                        user_msg = turn_data.get('user_message', '')
                    else:
                        user_msg = getattr(turn_data, 'user_message', '')
                    
                    preview = f"[Hydrated from fact] {user_msg[:100]}" if user_msg else "[Hydrated turn]"
                    
                    logger.debug(f"Hydrating turn {turn_id} from fact {fact.get('fact_id')}")
                    new_memories.append(MemoryCandidate(
                        memory_id=turn_id,
                        content_preview=preview,
                        score=model_config.DEFAULT_CANDIDATE_SCORE,
                        source_type='turn',
                        full_object=turn_data
                    ))
                    seen_turn_ids.add(turn_id)
                    hydrated_turn_count += 1

        # 3. TURNS → FACTS (Memory hits bring in their extracted facts)
        for m in memories:
            turn_id = None
            if m.source_type == 'turn':
                turn_id = m.memory_id
            elif m.source_type == 'summary' and m.full_object:
                if hasattr(m.full_object, 'source_turn_id'):
                    turn_id = m.full_object.source_turn_id
                elif isinstance(m.full_object, dict) and 'source_turn_id' in m.full_object:
                    turn_id = m.full_object['source_turn_id']
            
            if turn_id:
                t_facts = self.storage.get_facts_by_turn_id(turn_id)
                for tf in t_facts:
                    fid = tf.get('fact_id')
                    if fid and fid not in seen_fact_ids:
                        logger.debug(f"Hydrating fact {fid} from turn {turn_id}")
                        new_facts.append(tf)
                        seen_fact_ids.add(fid)
                        hydrated_fact_count += 1
        
        if hydrated_turn_count > 0 or hydrated_fact_count > 0:
            logger.info(f"Causal hydration: +{hydrated_turn_count} turns, +{hydrated_fact_count} facts")

        return routing_decision, new_memories, new_facts, dossiers
    
    async def _route_to_bridge_block(self, query: str, day_id: str) -> Dict[str, Any]:
        """
        TASK 1: LLM-based Bridge Block routing (metadata only).
        
        Uses GPT-4.1 mini to determine if query matches existing topic or is new.
        
        Args:
            query: User query text
            day_id: Current day ID
        
        Returns:
            {
                "matched_block_id": str or None,
                "is_new_topic": bool,
                "reasoning": str,
                "topic_label": str (suggested label if new topic)
            }
        """
        # Tracer span removed - telemetry deleted
        
        # Get metadata for all active bridge blocks (excludes turns[])
        metadata_list = self.storage.get_daily_ledger_metadata(day_id)
        
        if not metadata_list:
            # No blocks exist today - this is the first query
            logger.info("Governor: No blocks exist, creating first topic")
            return {
                "matched_block_id": None,
                "is_new_topic": True,
                "reasoning": "first_query_of_day",
                "topic_label": "Initial Conversation"
            }
            
        # Build routing prompt with metadata AND FACTS
        blocks_text = ""
        for i, meta in enumerate(metadata_list):
            last_active_marker = " (LAST ACTIVE)" if meta.get('is_last_active') else ""
            status_marker = f" ({meta.get('status', 'UNKNOWN')})"
            
            blocks_text += f"{i+1}. [{meta.get('topic_label', 'Unknown')}]{last_active_marker}{status_marker}\n"
            blocks_text += f"   ID: {meta.get('block_id')}\n"
            blocks_text += f"   Summary: {meta.get('summary', 'No summary')[:150]}...\n"
            blocks_text += f"   Keywords: {', '.join(meta.get('keywords', [])[:5])}\n"
            
            if meta.get('open_loops'):
                blocks_text += f"   Open Loops: {', '.join(meta['open_loops'][:3])}\n"
            
            if meta.get('decisions_made'):
                blocks_text += f"   Decisions: {', '.join(meta['decisions_made'][:3])}\n"
            
            blocks_text += f"   Turn Count: {meta.get('turn_count', 0)}\n"
            blocks_text += f"   Last Updated: {meta.get('last_updated', 'Unknown')}\n"
            
            # CRITICAL: Add all facts extracted for this topic
            block_facts = self.storage.get_facts_for_block(meta.get('block_id'))
            if block_facts:
                blocks_text += f"   Facts Extracted ({len(block_facts)} total):\n"
                # Show up to 10 most recent facts
                for fact in block_facts[:10]:
                    fact_preview = fact.get('value', '')[:80]
                    if len(fact.get('value', '')) > 80:
                        fact_preview += '...'
                    blocks_text += f"      • {fact.get('key', 'unknown')}: {fact_preview}\n"
                if len(block_facts) > 10:
                    blocks_text += f"      ... and {len(block_facts) - 10} more facts\n"
            
            blocks_text += "\n"
        
        routing_prompt = f"""You are an intelligent topic routing assistant for a conversational memory system.

PREVIOUS TOPICS TODAY:
{blocks_text}

USER QUERY: "{query}"

YOUR TASK:
Analyze the user's query and determine which topic block it belongs to. Use your intelligence to understand the INTENT and SEMANTIC CONTEXT, not just surface-level keywords.

**CRITICAL: Use the Facts Extracted to understand topic depth and context**
Each topic shows "Facts Extracted" - these reveal what specific information has been learned during that conversation. 
- If the query relates to ANY of these facts, it belongs to that topic
- Facts show the ACTUAL content discussed, not just keywords
- A topic with many related facts indicates an ongoing deep discussion - don't abandon it prematurely

You have 3 possible decisions:
1. **Continue LAST ACTIVE topic** - Query relates to the ongoing conversation
2. **Resume PAUSED topic** - Query clearly relates to a previous topic
3. **Start NEW topic** - Query is genuinely about something new/different

DECISION PRINCIPLES (Guidelines, not strict rules):

**Semantic Context Over Keywords:**
- "Let's talk about Docker Compose" while discussing Docker → SAME TOPIC (Docker is the context)
- "Let's talk about hiking" while discussing Docker → NEW TOPIC (completely unrelated)
- Focus on whether the SUBJECT MATTER is the same, not just the exact phrasing

**Domain Continuity - CRITICAL:**
- If the query is about a SUBTOPIC or COMPONENT of the current domain, it's the SAME conversation
- Example: Docker Containerization → Docker Compose → Docker Volumes → Docker Networks (all Docker, ONE topic)
- Example: Python basics → async/await → threading → decorators (all Python, ONE topic)
- Creating new topics for every subtopic fragments conversations into dozens of blocks
- Only create new topic if it's a COMPLETELY DIFFERENT DOMAIN (Docker → cooking, Python → hiking)

**Natural Conversation Flow:**
- Subtopic exploration within a domain → CONTINUE (e.g., volumes → compose in Docker)
- Related questions, clarifications, deeper dives → CONTINUE
- "Also...", "What about...", "And..." typically signal continuation
- "Instead" doesn't mean abandon topic, it means shift WITHIN the topic

**True Topic Abandonment:**
- User explicitly says they want to stop discussing current topic AND move to unrelated domain
- Query is about a completely different domain (Docker → cooking, Python → travel)
- No semantic connection whatsoever to current context

**Vague Queries:**
- "Tell me more", "Why?", "Explain" → CONTINUE LAST ACTIVE (inherit context)
- "Go back to that thing about X" → Check if X matches a paused topic's keywords

**When in Doubt:**
- STRONGLY prefer CONTINUATION over creating new topics
- Consider the full context: keywords, summary, open loops, not just the query alone
- Ask yourself: "Is this a different DOMAIN or just a different PART of the same domain?"
- If it's the same domain (Docker, Python, cooking, etc.), CONTINUE the topic

**DEBUG MODE - EXPLAIN YOUR REASONING:**
In your "reasoning" field, you MUST explicitly answer these questions:
1. What is the DOMAIN of the current active topic? (e.g., "Docker", "Python", "cooking")
2. What is the DOMAIN of the user's query? (e.g., "Docker Compose", "async/await", "baking")
3. Are these the SAME domain or DIFFERENT domains?
4. If SAME domain: Why are you continuing vs resuming vs creating new?
5. If DIFFERENT domain: What makes them unrelated?

Return JSON:
{{
    "matched_block_id": "<block_id>" or null,
    "is_new_topic": true/false,
    "reasoning": "<DETAILED explanation answering the 5 debug questions above>",
    "topic_label": "<suggested label if new topic, otherwise empty>"
        }}
"""
        try:
            # Use fast, cheap model for routing
            response = await self.api_client.query_external_api_async(
                routing_prompt, 
                model=model_config.get_lattice_model()
            )
            
            # Parse JSON response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                decision = json.loads(json_match.group(0))
                
                # Validate structure
                if "matched_block_id" in decision and "is_new_topic" in decision:
                    # Log routing decision at debug level
                    logger.debug(
                        f"Routing decision: matched={decision.get('matched_block_id')}, "
                        f"new_topic={decision.get('is_new_topic')}, "
                        f"label={decision.get('topic_label', 'N/A')}"
                    )
                    logger.debug(f"Routing reasoning: {decision.get('reasoning', 'N/A')}")
                    
                    logger.info(
                        f"Routing decision: "
                        f"matched={decision.get('matched_block_id')}, "
                        f"new={decision.get('is_new_topic')}, "
                        f"reason={decision.get('reasoning')}"
                    )
                    return decision
            
            # Fallback: Default to last active block
            logger.warning("Failed to parse routing JSON, defaulting to last active")
            last_active = next((m for m in metadata_list if m.get('is_last_active')), metadata_list[0])
            return {
                "matched_block_id": last_active.get('block_id'),
                "is_new_topic": False,
                "reasoning": "routing_parse_failed_defaulted_to_last_active",
                "topic_label": ""
            }
            
        except Exception as e:
            logger.error(f"Bridge routing failed: {e}", exc_info=True)
            logger.warning("FALLBACK: Defaulting to last active block due to routing failure")
            # Fail safe: default to last active or create new
            if metadata_list:
                last_active = next((m for m in metadata_list if m.get('is_last_active')), metadata_list[0])
                return {
                    "matched_block_id": last_active.get('block_id'),
                    "is_new_topic": False,
                    "reasoning": "routing_exception_defaulted_to_last_active",
                    "topic_label": ""
                }
            else:
                return {
                    "matched_block_id": None,
                    "is_new_topic": True,
                    "reasoning": "routing_exception_no_blocks_exist",
                    "topic_label": "Error Recovery Topic"
                }
    
    async def _retrieve_and_filter_memories(
        self,
        query: str,
        day_id: str,
        candidates: Optional[List[MemoryCandidate]] = None
    ) -> List[MemoryCandidate]:
        """
        TASK 2: Memory retrieval + 2-key filtering (Vector similarity + LLM).
        
        Implements 2-key filtering to kill false positives:
        - KEY 1: Vertex similarity score (semantic)
        - KEY 2: Original query text (verbatim or summary)
        
        Args:
            query: User query text
            candidates: Optional pre-fetched candidates (if None, performs vector search)
        
        Returns:
            List of MemoryCandidate objects (filtered by LLM)
        """
        # Tracer span removed - telemetry deleted
        
        # If no candidates provided, perform vector search via crawler
        if not candidates:
            logger.info(f"Governor: Performing vector search for query: '{query[:100]}...'")
            
            # Use crawler to perform vector search
            try:
                from hmlr.memory.models import Intent, QueryType
                
                # Create intent for crawler (Intent is a dataclass)
                # Pass query as keywords for vector search
                intent = Intent(
                    keywords=query.lower().split(),  # Use query words as keywords
                    query_type=QueryType.CHAT,
                    raw_query=query
                )
                
                # Retrieve contexts from crawler (this searches all embeddings)
                retrieved_context = self.crawler.retrieve_context(
                    intent=intent,
                    current_day_id=day_id,
                    max_results=20,  # Get top 20 candidates for filtering
                    window=None  # Search all time periods
                )
                
                logger.debug(f"Crawler found {len(retrieved_context.contexts)} candidates")
                
                # Convert crawler results to MemoryCandidate objects
                candidates = []
                for ctx in retrieved_context.contexts:
                    # Extract memory ID
                    mem_id = ctx.get('turn_id') or ctx.get('block_id') or ctx.get('summary_id') or 'unknown'
                    
                    # Build content preview
                    user_msg = ctx.get('user_message', '')
                    ai_resp = ctx.get('assistant_response', '')
                    content = ctx.get('content', '')
                    
                    if user_msg or ai_resp:
                        preview = f"User: {user_msg}\nAI: {ai_resp}"
                    elif content:
                        preview = content
                    else:
                        preview = str(ctx)
                    
                    # Truncate preview
                    preview = preview[:500] + "..." if len(preview) > 500 else preview
                    
                    # Create MemoryCandidate
                    candidates.append(MemoryCandidate(
                        memory_id=mem_id,
                        content_preview=preview,
                        score=ctx.get('similarity', 0.0),
                        source_type=ctx.get('source_type', 'turn'),
                        full_object=ctx
                    ))
                    
                if candidates:
                    logger.debug(f"Converted to {len(candidates)} MemoryCandidate objects")
                    for i, cand in enumerate(candidates[:3], 1):
                        logger.debug(f"  [{i}] {cand.memory_id}: score={cand.score:.3f}")
                else:
                    logger.info("No candidates found in vector search")
                    return []
                
            except Exception as e:
                logger.error(f"Vector search failed: {e}", exc_info=True)
                # We raise a custom error here instead of returning empty list if it's critical,
                # but for resilience we often want to fallback.
                # However, this is a "Silent Failure" pattern if not carefully handled.
                # Let's log it as a critical system error but allow fallback to prevent crash.
                return []
        
        if not candidates:
            return []
        
        # Fetch original queries for 2-key filtering
        enriched_candidates = []
        for cand in candidates:
            # Extract original query from full_object
            original_query = cand.full_object.get('original_query', '')
            
            # If original query >1k tokens, fetch gardener summary instead
            # (Placeholder logic - implement token counting later)
            if len(original_query) > 4000:  # Rough heuristic: 1 token ≈ 4 chars
                # TODO: Fetch gardener summary from storage
                original_query = original_query[:1000] + "... [truncated]"
            
            enriched_candidates.append({
                "index": len(enriched_candidates),
                "memory_id": cand.memory_id,
                "similarity": cand.score,
                "original_query": original_query,
                "content": cand.content_preview,
                "metadata": {
                    "source_type": cand.source_type,
                    "timestamp": cand.full_object.get('timestamp', 'unknown')
                }
            })
            
        # Build 2-key filtering prompt
        candidates_text = ""
        for ec in enriched_candidates:
            candidates_text += f"[{ec['index']}] Similarity: {ec['similarity']:.2f}\n"
            candidates_text += f"   Original Query: \"{ec['original_query'][:200]}...\"\n"
            candidates_text += f"   Content: {ec['content'][:300]}...\n"
            candidates_text += f"   Metadata: {json.dumps(ec['metadata'])}\n\n"
        
        # Debug logging removed - was writing to debug_llm_flow.txt
        # If debugging needed, use logger.debug() with DEBUG_MODE env variable
        logger.debug(f"Memory candidates for filtering: {len(enriched_candidates)} total")
            
        filter_prompt = f"""You are a memory filter using 2-key validation.

CURRENT QUERY: "{query}"

MEMORY CANDIDATES:
{candidates_text}

TASK: Select ONLY memories that are truly relevant to the current query.

KEY 1 (Similarity Score): Semantic similarity from embeddings (0.0-1.0)
KEY 2 (Original Query): The actual query that created this memory

IMPORTANT: High similarity does NOT guarantee relevance!
Example:
- "I love Python" vs "I hate Python" = 95% similarity but OPPOSITE meaning
- "Python advantages" vs "Python disadvantages" = High similarity but different intent

Use BOTH keys to filter out false positives.

Return JSON:
{{
    "relevant_indices": [0, 2, 5],
    "reasoning": "<brief explanation of why others were filtered out>"
}}
"""
        try:
            # Use GPT-4.1 mini for filtering
            logger.info(f"Governor: Running 2-key memory filter on {len(enriched_candidates)} candidates")
            
            response = await self.api_client.query_external_api_async(
                filter_prompt,
                model=model_config.get_lattice_model()
            )
            
            # Parse JSON
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
                relevant_indices = data.get("relevant_indices", [])
                reasoning = data.get("reasoning", "")
                
                # Filter candidates
                filtered = [candidates[idx] for idx in relevant_indices if 0 <= idx < len(candidates)]
                
                # Log filter results
                logger.info(f"Memory filter: {len(filtered)}/{len(candidates)} selected")
                logger.debug(f"Selected indices: {relevant_indices}")
                logger.debug(f"Reasoning: {reasoning}")
                
                # Debug logging: Use logger.debug() instead of file writes
                logger.debug(
                    f"Memory filter results: {len(filtered)}/{len(candidates)} selected. "
                    f"Indices: {relevant_indices}"
                )
                
                if filtered:
                    logger.debug(f"Approved memories: {[m.memory_id for m in filtered]}")
                else:
                    logger.info("No memories approved by Governor")
                
                logger.info(
                    f"Memory filter: {len(filtered)}/{len(candidates)} relevant. "
                    f"Reason: {reasoning[:100]}"
                )
                return filtered
            
            # Fallback: Return all candidates if parsing fails
            logger.warning("Failed to parse memory filter JSON, returning all candidates")
            return candidates
            
        except Exception as e:
            logger.error(f"Memory filtering failed: {e}", exc_info=True)
            logger.warning("FALLBACK: Returning all candidates due to LatticeGovernorError")
            # Fail open: returning candidates is safer than returning nothing
            return candidates
    
    def _lookup_facts(self, query: str) -> List[Dict[str, Any]]:
        """
        TASK 3: Fact store lookup (synchronous SQLite query).
        
        Args:
            query: User query text
        
        Returns:
            List of fact dictionaries from fact_store
        """
        # Tracer span removed - telemetry deleted
        
        # Extract keywords from query
        words = re.findall(r'\b[A-Z]{2,}\b|\b\w+\b', query)
        unique_words = list(set(words))[:5]
        
        facts = []
        for word in unique_words:
            # Try exact match (only method available)
            fact = self.storage.query_fact_store(word)
            if fact and fact not in facts:
                facts.append(fact)
        
        logger.info(f"Fact lookup: Found {len(facts)} matching facts")
        return facts
    
    def _retrieve_dossiers(self, query: str) -> List[Dict[str, Any]]:
        """
        TASK 4: Dossier retrieval (synchronous semantic search via embeddings).
        
        Args:
            query: User query text
        
        Returns:
            List of dossier dictionaries with metadata and facts
        """
        if not self.dossier_retriever:
            return []
        
        # Tracer span removed - telemetry deleted
            
        # Retrieve ALL matching dossiers above threshold (no top_k limit)
        dossiers = self.dossier_retriever.retrieve_relevant_dossiers(
            query=query,
            top_k=None,  # No limit - return all matching dossiers
            threshold=model_config.MIN_SIMILARITY_THRESHOLD
        )
        
        logger.info(f"Dossier retrieval: Found {len(dossiers)} relevant dossiers")
        return dossiers
    
    def _check_fact_store(self, query: str) -> List[Dict[str, Any]]:
        """
        Check fact_store for exact keyword matches
        
        Args:
            query: User query text
        
        Returns:
            List of matching facts (empty if none found)
        """
        # Tracer span removed - telemetry deleted
        
        # Extract potential keywords from query (simple word extraction)
        words = re.findall(r'\b[A-Z]{2,}\b|\b\w+\b', query)
        unique_words = list(set(words))[:5]  # Check up to 5 keywords
        
        results = []
        for word in unique_words:
            # Try exact match (only method available)
            fact = self.storage.query_fact_store(word)
            if fact and fact not in results:
                results.append(fact)
        
        return results
    
    def _check_daily_ledger(self, query: str) -> List[Dict[str, Any]]:
        """
        Check daily_ledger for same-day Bridge Blocks
        
        Args:
            query: User query text
        
        Returns:
            List of Bridge Blocks from today (empty if none found)
        """
        # Tracer span removed - telemetry deleted
        
        # Get all active blocks (cross-day continuity)
        today_blocks = self.storage.get_active_bridge_blocks()
        
        if not today_blocks:
            return []
        
        # Return ALL same-day blocks - let the Governor (LLM) decide relevance
        # Rationale: Vector similarity can match without lexical overlap
        # Example: "serverless services" relates to "AWS EC2" discussion
        # even though no keywords match
        return today_blocks
    
    def _format_bridge_block(self, content: Dict[str, Any]) -> str:
        """
        Format Bridge Block content for LLM preview.
        
        Args:
            content: Bridge Block content_json dictionary
        
        Returns:
            Formatted preview string
        """
        topic = content.get('topic_label', 'Unknown Topic')
        summary = content.get('summary', '')[:200]
        open_loops = content.get('open_loops', [])
        decisions = content.get('decisions_made', [])
        
        preview = f"[BRIDGE BLOCK] Topic: {topic}\n"
        preview += f"Summary: {summary}...\n"
        
        if open_loops:
            preview += f"Open Loops: {', '.join(open_loops[:3])}\n"
        
        if decisions:
            preview += f"Decisions: {', '.join(decisions[:3])}\n"
        
        return preview
