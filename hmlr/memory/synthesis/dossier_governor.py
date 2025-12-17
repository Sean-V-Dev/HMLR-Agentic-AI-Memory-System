"""
Dossier Governor - Write-Side Fact Routing

The DossierGovernor decides where incoming facts should be stored:
1. Search for candidate dossiers using Multi-Vector Voting
2. LLM decides: append to existing dossier or create new one
3. Update dossier summary incrementally
4. Track provenance for all changes

Architecture:
- Multi-Vector Voting: Each fact searches independently, dossiers with most
  hits "bubble up" as top candidates
- LLM Decision: Given new facts + top candidates, LLM chooses append target
  or decides to create new dossier
- Incremental Updates: Summaries updated with "old summary + new facts"

Author: CognitiveLattice Team
Created: 2025-12-15 (Phase 3: Dossier System)
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


class DossierGovernor:
    """
    Write-side governor for routing facts to dossiers.
    
    The DossierGovernor receives fact packets from the Gardener and decides
    whether to append them to existing dossiers or create new ones. This is
    the core intelligence layer that enables incremental fact aggregation.
    
    Key Features:
    - Multi-Vector Voting: Search with ALL facts, not just the first one
    - LLM-driven routing: Semantic understanding of fact relationships
    - Incremental summaries: Build narratives as facts accumulate
    - Full provenance: Track every operation for debugging and analysis
    """
    
    def __init__(self, storage, dossier_storage, llm_client, id_generator):
        """
        Initialize dossier governor.
        
        Args:
            storage: Storage instance (for dossier CRUD)
            dossier_storage: DossierEmbeddingStorage instance (for vector search)
            llm_client: LLM client for routing decisions and summaries
            id_generator: ID generator for creating unique IDs
        """
        self.storage = storage
        self.dossier_storage = dossier_storage
        self.llm_client = llm_client
        self.id_generator = id_generator
        logger.info("DossierGovernor initialized")
    
    async def process_fact_packet(self, fact_packet: Dict[str, Any]) -> str:
        """
        Route fact packet to appropriate dossier.
        
        This is the main entry point. Takes a semantically grouped set of facts
        and decides where they should be stored.
        
        Args:
            fact_packet: Dictionary with:
                - cluster_label: Theme/topic name (e.g., "Dietary Preferences")
                - facts: List of fact strings
                - source_block_id: Bridge block ID these facts came from
                - timestamp: When facts were extracted
        
        Returns:
            dossier_id where facts were added or created
        
        Example:
            fact_packet = {
                'cluster_label': 'Vegetarian Diet',
                'facts': ['User is strictly vegetarian', 'User avoids meat'],
                'source_block_id': 'block_001',
                'timestamp': '2025-12-15T10:30:00'
            }
            dossier_id = await governor.process_fact_packet(fact_packet)
        """
        cluster_label = fact_packet['cluster_label']
        facts = fact_packet['facts']
        source_block_id = fact_packet['source_block_id']
        
        logger.info(f"Processing fact packet: {cluster_label} ({len(facts)} facts)")
        
        # 1. Search for candidate dossiers using Multi-Vector Voting
        candidates = self._find_candidate_dossiers(facts, top_k=5)
        
        # 2. LLM decides: append to existing or create new
        if candidates:
            logger.debug(f"Found {len(candidates)} candidate dossiers")
            decision = await self._llm_decide_routing(facts, candidates)
            
            if decision['action'] == 'append':
                dossier_id = decision['target_dossier_id']
                logger.info(f"LLM decided: APPEND to dossier {dossier_id}")
                await self._append_to_dossier(dossier_id, facts, source_block_id)
                return dossier_id
            else:
                logger.info(f"LLM decided: CREATE new dossier")
        else:
            logger.info(f"No candidate dossiers found, creating new")
        
        # 3. No suitable dossier found, create new
        dossier_id = await self._create_new_dossier(cluster_label, facts, source_block_id)
        return dossier_id
    
    def _find_candidate_dossiers(self, facts: List[str], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Multi-Vector Voting: Search using ALL facts and rank by hit frequency.
        
        This is the key innovation that solves the "vague fact" problem. Instead
        of searching with just the first fact, we search with ALL facts and count
        which dossiers get the most hits. This causes the correct dossier to
        "bubble up" even when some facts are generic.
        
        Algorithm:
        1. Search for each fact individually against all dossier facts
        2. Tally which dossiers get the most hits
        3. Sort by hit count (primary) and score sum (tiebreaker)
        4. Return top K candidates
        
        Example Scenario:
            Incoming facts: ["It is fast", "Tartarus encryption is robust"]
            
            "It is fast" matches 100 random dossiers (vague)
            "Tartarus encryption" matches dos_tartarus (specific)
            
            Vote tally:
            - dos_tartarus: 1 hit (from specific fact)
            - dos_random_1: 1 hit (from vague fact)
            - dos_random_2: 1 hit (from vague fact)
            ... 98 more with 1 hit each
            
            But if there are 5 facts and 3 of them match dos_tartarus,
            then dos_tartarus gets 3 hits and wins decisively.
        
        Args:
            facts: List of fact strings to search with
            top_k: Maximum number of candidates to return
        
        Returns:
            List of candidate dossier dictionaries with vote metadata
        """
        vote_tally = {}  # {dossier_id: {'score_sum': 0.0, 'hits': 0}}
        
        logger.debug(f"Multi-Vector Voting: searching with {len(facts)} facts")
        
        # 1. Search for EVERY fact in the packet
        for i, fact in enumerate(facts, 1):
            results = self.dossier_storage.search_similar_facts(
                query=fact,
                top_k=10,  # Cast a wider net per fact
                threshold=0.4  # Consistent with memory search threshold
            )
            
            logger.debug(f"  Fact {i}: found {len(results)} matches")
            
            # 2. Tally the votes
            for fact_id, dossier_id, score in results:
                if dossier_id not in vote_tally:
                    vote_tally[dossier_id] = {'score_sum': 0.0, 'hits': 0}
                
                vote_tally[dossier_id]['hits'] += 1
                vote_tally[dossier_id]['score_sum'] += score
        
        if not vote_tally:
            logger.debug("  No matches found across all facts")
            return []
        
        # 3. Sort by Hit Count first (primary), then Score Sum (tiebreaker)
        # This causes dossiers with more matching facts to "bubble up"
        sorted_dossiers = sorted(
            vote_tally.items(),
            key=lambda item: (item[1]['hits'], item[1]['score_sum']),
            reverse=True
        )
        
        logger.debug(f"  Vote results: {len(sorted_dossiers)} dossiers ranked")
        for dossier_id, stats in sorted_dossiers[:3]:
            logger.debug(f"    {dossier_id}: {stats['hits']} hits, score {stats['score_sum']:.2f}")
        
        # 4. Fetch full details for top K dossiers
        candidates = []
        for dossier_id, stats in sorted_dossiers[:top_k]:
            dossier = self.storage.get_dossier(dossier_id)
            if dossier:
                dossier_facts = self.storage.get_dossier_facts(dossier_id)
                candidates.append({
                    'dossier_id': dossier_id,
                    'title': dossier['title'],
                    'summary': dossier['summary'],
                    'facts': [f['fact_text'] for f in dossier_facts],
                    'vote_hits': stats['hits'],  # How many facts matched
                    'vote_score': stats['score_sum']  # Total similarity score
                })
        
        return candidates
    
    async def _llm_decide_routing(self, new_facts: List[str], 
                                   candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        LLM decides whether to append or create new dossier.
        
        The LLM sees:
        - The new facts to be stored
        - Top candidate dossiers (with their summaries and existing facts)
        - Vote metadata (how many hits each candidate got)
        
        The LLM decides:
        - APPEND: If new facts belong to an existing dossier
        - CREATE: If new facts form a distinct topic
        
        Args:
            new_facts: List of fact strings to route
            candidates: List of candidate dossier dictionaries
        
        Returns:
            Dictionary: {'action': 'append'|'create', 'target_dossier_id': '...'}
        """
        # Format candidates for LLM
        candidates_summary = []
        for c in candidates:
            candidates_summary.append({
                'dossier_id': c['dossier_id'],
                'title': c['title'],
                'summary': c['summary'],
                'vote_hits': c['vote_hits'],
                'existing_facts': c['facts'][:5]  # Show first 5 facts
            })
        
        prompt = f"""You are a fact routing system. Decide whether new facts should be appended to an existing dossier or create a new dossier.

NEW FACTS TO ROUTE:
{json.dumps(new_facts, indent=2)}

CANDIDATE DOSSIERS (ranked by Multi-Vector Voting):
{json.dumps(candidates_summary, indent=2)}

DECISION RULES:
1. If new facts semantically belong to an existing dossier (same topic, related concepts), APPEND
2. If new facts form a distinct topic that doesn't fit existing dossiers, CREATE
3. Consider the vote_hits: higher hits mean stronger semantic relationship
4. Facts don't need to be identical - look for conceptual relationships

Return JSON:
- To append: {{"action": "append", "target_dossier_id": "dos_xxx"}}
- To create new: {{"action": "create"}}

Decision:"""
        
        try:
            response = self.llm_client.query_external_api(
                query=prompt,
                model="gpt-4.1-mini"
            )
            
            # Extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                decision = json.loads(json_match.group(0))
                logger.info(f"LLM routing decision: {decision}")
                return decision
            else:
                logger.warning("No JSON found in LLM response, defaulting to CREATE")
                return {'action': 'create'}
        
        except Exception as e:
            logger.error(f"LLM routing decision failed: {e}, defaulting to CREATE")
            return {'action': 'create'}
    
    async def _append_to_dossier(self, dossier_id: str, facts: List[str], 
                                 source_block_id: str) -> None:
        """
        Add facts to existing dossier and update summary.
        
        Steps:
        1. Add each fact to dossier_facts table
        2. Embed each fact for future searches
        3. Log provenance for each fact
        4. Update dossier summary incrementally
        5. Update last_modified timestamp
        
        Args:
            dossier_id: Target dossier ID
            facts: List of fact strings to add
            source_block_id: Bridge block that contributed these facts
        """
        logger.info(f"Appending {len(facts)} facts to dossier {dossier_id}")
        
        # 1. Add each fact
        for fact_text in facts:
            fact_id = self.id_generator.generate_id("fact")
            
            # Store fact in database
            success = self.storage.add_fact_to_dossier(
                dossier_id=dossier_id,
                fact_id=fact_id,
                fact_text=fact_text,
                source_block_id=source_block_id,
                source_turn_id=None,  # Can be enhanced later with turn tracking
                confidence=1.0
            )
            
            if not success:
                logger.error(f"Failed to add fact {fact_id} to dossier")
                continue
            
            # 2. Embed fact for future searches
            self.dossier_storage.save_fact_embedding(fact_id, dossier_id, fact_text)
            
            # 3. Log provenance
            prov_id = self.id_generator.generate_id("prov")
            self.storage.add_provenance_entry(
                dossier_id=dossier_id,
                operation="fact_added",
                provenance_id=prov_id,
                source_block_id=source_block_id,
                details=json.dumps({"fact_id": fact_id, "fact_text": fact_text[:100]})
            )
            
            logger.debug(f"  Added fact {fact_id}: {fact_text[:50]}...")
        
        # 4. Update dossier summary (incremental)
        await self._update_dossier_summary(dossier_id, facts, source_block_id)
        
        logger.info(f"Successfully appended {len(facts)} facts to dossier {dossier_id}")
    
    async def _create_new_dossier(self, title: str, facts: List[str], 
                                  source_block_id: str) -> str:
        """
        Create new dossier with initial facts.
        
        Steps:
        1. Generate dossier ID and initial summary
        2. Create dossier in database
        3. Add all facts to dossier
        4. Embed all facts for searching
        5. Log creation provenance
        
        Args:
            title: Dossier title (from cluster_label)
            facts: Initial facts for this dossier
            source_block_id: Bridge block that contributed these facts
        
        Returns:
            New dossier ID
        """
        dossier_id = self.id_generator.generate_id("dos")
        logger.info(f"Creating new dossier: {dossier_id} - {title}")
        
        # 1. Generate summaries
        summary = await self._generate_summary(facts, title)
        search_summary = await self._generate_search_summary(facts, title, summary)
        
        # 2. Create dossier
        success = self.storage.create_dossier(
            dossier_id=dossier_id,
            title=title,
            summary=summary,
            search_summary=search_summary
        )
        
        if not success:
            logger.error(f"Failed to create dossier {dossier_id}")
            return None
        
        # 3. Embed search summary for broad retrieval
        self.dossier_storage.save_dossier_search_embedding(dossier_id, search_summary)
        
        # 4. Add facts
        for fact_text in facts:
            fact_id = self.id_generator.generate_id("fact")
            
            self.storage.add_fact_to_dossier(
                dossier_id=dossier_id,
                fact_id=fact_id,
                fact_text=fact_text,
                source_block_id=source_block_id,
                confidence=1.0
            )
            
            # Embed fact (for fine-grained matching in Multi-Vector Voting)
            self.dossier_storage.save_fact_embedding(fact_id, dossier_id, fact_text)
            
            logger.debug(f"  Added fact {fact_id}: {fact_text[:50]}...")
        
        # 5. Log provenance
        prov_id = self.id_generator.generate_id("prov")
        self.storage.add_provenance_entry(
            dossier_id=dossier_id,
            operation="created",
            provenance_id=prov_id,
            source_block_id=source_block_id,
            details=json.dumps({"num_facts": len(facts), "title": title})
        )
        
        logger.info(f"Created dossier {dossier_id} with {len(facts)} facts")
        return dossier_id
    
    async def _update_dossier_summary(self, dossier_id: str, new_facts: List[str],
                                     source_block_id: str) -> None:
        """
        Incrementally update dossier summary with new facts.
        
        Strategy: "Old Summary + New Facts → Updated Summary"
        This allows the summary to evolve as facts accumulate, building
        causal chains and narratives over time.
        
        Args:
            dossier_id: Target dossier ID
            new_facts: New facts being added
            source_block_id: Source of new facts
        """
        dossier = self.storage.get_dossier(dossier_id)
        old_summary = dossier['summary']
        
        prompt = f"""Update this dossier summary with new facts. Build causal chains where possible.

OLD SUMMARY:
{old_summary}

NEW FACTS BEING ADDED:
{json.dumps(new_facts, indent=2)}

INSTRUCTIONS:
1. Integrate new facts into the existing narrative
2. Build causal chains where facts relate (e.g., "Because X, therefore Y")
3. Do NOT create duplicates of existing information
4. Keep summary concise but comprehensive (2-4 sentences)

UPDATED SUMMARY:"""
        
        try:
            new_summary = self.llm_client.query_external_api(
                query=prompt,
                model="gpt-4.1-mini"
            )
            
            # Clean up response
            new_summary = new_summary.strip()
            if new_summary.startswith("UPDATED SUMMARY:"):
                new_summary = new_summary[16:].strip()
            
            # Update in database
            self.storage.update_dossier_summary(dossier_id, new_summary)
            
            # Log provenance
            prov_id = self.id_generator.generate_id("prov")
            self.storage.add_provenance_entry(
                dossier_id=dossier_id,
                operation="summary_updated",
                provenance_id=prov_id,
                source_block_id=source_block_id,
                details=json.dumps({"num_new_facts": len(new_facts)})
            )
            
            logger.debug(f"  Updated summary for dossier {dossier_id}")
        
        except Exception as e:
            logger.error(f"Failed to update dossier summary: {e}")
    
    async def _generate_summary(self, facts: List[str], title: str) -> str:
        """
        Generate initial summary for a new dossier.
        
        Args:
            facts: Initial facts for the dossier
            title: Dossier title
        
        Returns:
            Summary text
        """
        prompt = f"""Generate a concise summary for a new fact dossier.

TITLE: {title}

FACTS:
{json.dumps(facts, indent=2)}

Generate a 2-3 sentence summary that:
1. Captures the essence of these facts
2. Identifies any causal relationships
3. Sets context for future facts

SUMMARY:"""
        
        try:
            summary = self.llm_client.query_external_api(
                query=prompt,
                model="gpt-4.1-mini"
            )
            
            summary = summary.strip()
            if summary.startswith("SUMMARY:"):
                summary = summary[8:].strip()
            
            return summary
        
        except Exception as e:
            logger.error(f"Failed to generate summary: {e}")
            # Fallback: concatenate facts
            return f"{title}: " + "; ".join(facts[:3])
    
    async def _generate_search_summary(self, facts: List[str], title: str, summary: str) -> str:
        """
        Generate search-optimized summary for broad topic retrieval.
        
        This text is specifically designed to match general queries. It should:
        - Be broader and more general than individual facts
        - Include topic keywords and related concepts
        - Capture the "what this is about" at a high level
        
        Example:
            Facts: ["Tesla Model 3 is an electric sedan", "Has 300 mile range"]
            Search Summary: "User's Tesla Model 3 electric vehicle - sedan for daily commuting and transportation with long range capability and autopilot features"
        
        This allows broad queries like "which car for family trip" to match, then
        the LLM can examine the specific facts to make the final decision.
        
        Args:
            facts: All facts in the dossier
            title: Dossier title
            summary: Already-generated detailed summary
        
        Returns:
            Search-optimized summary text
        """
        prompt = f"""Generate a SEARCH-OPTIMIZED summary for semantic retrieval.

TITLE: {title}
SUMMARY: {summary}

FACTS:
{json.dumps(facts, indent=2)}

Create a search summary that:
1. Uses BROAD, GENERAL language (not narrow/specific)
2. Includes TOPIC KEYWORDS and related concepts
3. Captures "what this is about" at a high level
4. Would match general queries about this subject
5. Keep it to 2-3 sentences maximum

Example:
Input: Facts about "Tesla Model 3 is electric", "has 300 mile range"
Output: "User's Tesla Model 3 electric vehicle for personal transportation, daily commuting and travel. Electric sedan with long-range capability, autopilot technology, and modern features for efficient driving."

SEARCH SUMMARY:"""
        
        try:
            search_summary = self.llm_client.query_external_api(
                query=prompt,
                model="gpt-4.1-mini"
            )
            
            search_summary = search_summary.strip()
            if search_summary.startswith("SEARCH SUMMARY:"):
                search_summary = search_summary[15:].strip()
            
            logger.debug(f"Generated search summary: {search_summary[:80]}...")
            return search_summary
        
        except Exception as e:
            logger.error(f"Failed to generate search summary: {e}")
            # Fallback: use title + summary
            return f"{title}. {summary}"
