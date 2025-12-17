"""
Manual Gardener - Phase 2 Refactored: Fact Classifier and Router

The Gardener's NEW Job (Post-Phase 2):
1. Load bridge block (completed topic conversation)
2. Load existing facts from fact_store (extracted by FactScrubber during conversation)
3. Classify facts using THREE HEURISTICS:
   - Environment Test: Global settings (Python version, OS, language)
   - Constraint Test: Rules that forbid/mandate behaviors
   - Definition Test: Temporary aliases or status markers
4. Apply tags to block_metadata table (stored ONCE per block)
5. Group remaining facts semantically (facts that don't match tag patterns)
6. Route fact groups to DossierGovernor for dossier creation
7. Delete bridge block

Key Changes from Phase 1:
- NO MORE CHUNKING: Facts already extracted by FactScrubber
- NO MORE EMBEDDING: Done at fact level in dossier system
- NO MORE gardened_memory: Replaced by block_metadata + dossiers
- Gardener is now a CLASSIFIER, not a CHUNKER

Dual Output System:
- Sticky Meta Tags: Scope/validity/environment (block_metadata table)
- Dossier Facts: Narrative/causal chains (routed to DossierGovernor)
"""

import re
import json
from typing import List, Dict, Any
from datetime import datetime


class ManualGardener:
    """
    Phase 2 Refactored Gardener: Fact Classifier and Router
    
    NO MORE CHUNKING: Facts already extracted by FactScrubber during conversation.
    
    New responsibilities:
    1. Classify facts using three heuristics (Environment, Constraint, Definition)
    2. Apply sticky meta tags to block_metadata table
    3. Group remaining facts semantically
    4. Route fact groups to DossierGovernor
    """
    
    def __init__(self, storage, embedding_storage, llm_client, dossier_governor=None, dossier_storage=None):
        """
        Initialize gardener.
        
        Args:
            storage: Storage instance (for block_metadata operations)
            embedding_storage: EmbeddingStorage instance (deprecated for Phase 2)
            llm_client: LLM client for fact classification and grouping
            dossier_governor: DossierGovernor instance for fact routing
            dossier_storage: DossierEmbeddingStorage instance
        """
        self.storage = storage
        self.embedding_storage = embedding_storage  # Kept for backward compatibility, not used in Phase 2
        self.llm_client = llm_client
        self.dossier_governor = dossier_governor
        self.dossier_storage = dossier_storage
    
    async def process_bridge_block(self, block_id: str) -> Dict[str, Any]:
        """
        Process a Bridge Block with Phase 2 Dual-Output Flow.
        
        Phase 2 Flow:
        1. Load bridge block
        2. Load existing facts from fact_store (extracted by FactScrubber)
        3. Classify facts using THREE HEURISTICS:
           - Environment Test: Global settings (Python version, OS, etc.)
           - Constraint Test: Rules that forbid/mandate behaviors
           - Definition Test: Temporary aliases or status markers
        4. Apply tags to block_metadata table (stored ONCE per block)
        5. Group remaining facts semantically (facts that don't match tag patterns)
        6. Route fact groups to DossierGovernor for dossier creation
        7. Delete bridge block
        
        Args:
            block_id: Bridge Block ID
        
        Returns:
            Processing summary with stats
        """
        print(f"\n🌱 Gardener Phase 2: Processing Block {block_id}")
        
        # 1. Load Bridge Block
        block_data = self._load_bridge_block(block_id)
        if not block_data:
            print(f"   ❌ Block not found")
            return {"status": "error", "message": "Block not found"}
        
        topic_label = block_data.get('topic_label', 'Unknown Topic')
        print(f"   📋 Topic: {topic_label}")
        
        # 2. Load existing facts from fact_store (extracted by FactScrubber during conversation)
        print(f"\n   📋 Loading facts from fact_store...")
        existing_facts = self.storage.get_facts_for_block(block_id)
        
        if not existing_facts:
            print(f"   ⚠️  No facts found for {block_id} - skipping")
            self._delete_bridge_block(block_id)
            return {
                "status": "success",
                "block_id": block_id,
                "message": "No facts to process"
            }
        
        print(f"   ✅ Found {len(existing_facts)} facts from conversation")
        
        # Show sample facts
        for i, fact in enumerate(existing_facts[:5]):  # Show first 5
            fact_value = fact.get('value', '')
            print(f"      • {fact_value[:80]}...")
        if len(existing_facts) > 5:
            print(f"      ... and {len(existing_facts) - 5} more")
        
        # 3. TAGGING PASS: Classify facts using three heuristics
        print(f"\n   🏷️  Classifying facts (Environment/Constraint/Definition heuristics)...")
        classification = await self._classify_facts_for_tagging(existing_facts)
        
        # 4. Apply tags to block metadata (NOT to chunks)
        global_tags = classification.get('global_tags', [])
        section_rules = classification.get('section_rules', [])
        
        if global_tags or section_rules:
            self.storage.save_block_metadata(
                block_id=block_id,
                global_tags=global_tags,
                section_rules=section_rules
            )
            print(f"   ✅ Applied {len(global_tags)} global tags, {len(section_rules)} section rules")
            
            # Show tags
            for tag in global_tags[:3]:  # Show first 3
                print(f"      [Global] {tag}")
            for rule in section_rules[:3]:  # Show first 3
                print(f"      [Section] {rule.get('rule', 'unknown')}")
        else:
            print(f"   ℹ️  No tags identified for this block")
        
        # 5. DOSSIER PASS: ALL facts go to dossiers (dual-output system!)
        # Even tagged facts should be in dossiers for narrative context
        all_fact_texts = [f.get('value', '') for f in existing_facts]
        dossier_count = 0
        
        if all_fact_texts and self.dossier_governor:
            print(f"\n   🗂️  Processing {len(all_fact_texts)} facts into dossiers...")
            
            # Prepare facts for semantic grouping
            fact_list = []
            for fact_text in all_fact_texts:
                # Find original fact to get metadata
                original_fact = next((f for f in existing_facts if f.get('value') == fact_text), {})
                fact_list.append({
                    'text': fact_text,
                    'key': original_fact.get('key', ''),
                    'timestamp': original_fact.get('timestamp', datetime.now().isoformat()),
                    'turn_id': original_fact.get('turn_id', '')
                })
            
            # 6. Route groups to dossier governor (async)
            fact_groups = await self._group_facts_semantically(fact_list)
            
            for group in fact_groups:
                fact_packet = {
                    'cluster_label': group['label'],
                    'facts': group['facts'],
                    'source_block_id': block_id,
                    'timestamp': group.get('timestamp', datetime.now().isoformat())
                }
                
                try:
                    dossier_id = await self.dossier_governor.process_fact_packet(fact_packet)
                    if dossier_id:
                        print(f"      ✅ Dossier: {dossier_id} ({group['label']})")
                        dossier_count += 1
                except Exception as e:
                    print(f"      ⚠️  Failed: {group['label']}: {e}")
            
            print(f"   ✅ Created/updated {dossier_count} dossiers")
        elif not self.dossier_governor:
            print(f"   ⚠️  Dossier system unavailable")
        elif not dossier_facts:
            print(f"   ℹ️  All facts classified as tags, no dossier facts")
        
        # 7. Delete processed bridge block from active memory
        self._delete_bridge_block(block_id)
        
        print(f"\n✅ Gardener Phase 2: Block {block_id} processed successfully!")
        
        return {
            "status": "success",
            "block_id": block_id,
            "topic_label": topic_label,
            "facts_processed": len(existing_facts),
            "tags_applied": len(global_tags) + len(section_rules),
            "dossiers_created": dossier_count
        }
    
    def _load_bridge_block(self, block_id: str) -> Dict[str, Any]:
        """Load Bridge Block from daily_ledger."""
        cursor = self.storage.conn.cursor()
        cursor.execute("""
            SELECT content_json FROM daily_ledger 
            WHERE block_id = ?
        """, (block_id,))
        
        row = cursor.fetchone()
        if not row:
            return None
        
        return json.loads(row[0])
    
    async def _classify_facts_for_tagging(self, facts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Classify facts using THREE HEURISTICS (Phase 2 core logic).
        
        Heuristics:
        1. ENVIRONMENT TEST: Is this a global setting/version/language?
           Examples: "Using Python 3.9", "On Windows", "Project uses TypeScript"
           → Global tag (applies to all retrieved content)
        
        2. CONSTRAINT TEST: Does this forbid or mandate something?
           Examples: "Never use eval()", "Always check permissions first"
           → Constraint tag (global or section-specific)
        
        3. DEFINITION TEST: Is this a temporary renaming or status marker?
           Examples: "Call the server Box A", "Old API is deprecated"
           → Alias/status tag (section-specific with turn range)
        
        Args:
            facts: List of fact dictionaries from fact_store
        
        Returns:
            {
                "global_tags": ["env: python-3.9", "os: windows"],
                "section_rules": [{"start_turn": 10, "end_turn": 15, "rule": "no-eval"}],
                "dossier_facts": ["User prefers dark mode", "User works remotely"]
            }
        """
        if not facts:
            return {"global_tags": [], "section_rules": [], "dossier_facts": []}
        
        # Format facts for LLM
        facts_formatted = []
        for fact in facts:
            facts_formatted.append({
                "text": fact.get('value', ''),
                "turn_id": fact.get('turn_id', '')
            })
        
        prompt = f"""Analyze these facts extracted from a conversation and classify them using THREE heuristics:

Facts:
{json.dumps(facts_formatted, indent=2)}

HEURISTICS:

1. ENVIRONMENT TEST: Global settings, versions, languages, OS?
   Examples: "Using Python 3.9" → env: python-3.9
             "On Windows" → os: windows
             "Project uses TypeScript" → lang: typescript
   → Tag as GLOBAL (applies to entire conversation)

2. CONSTRAINT TEST: Rules that FORBID or MANDATE something?
   Examples: "Never use eval()" → no-eval
             "Always check permissions" → check-permissions
             "Must validate input" → validate-input
   → Tag as CONSTRAINT (global or section-specific)

3. DEFINITION TEST: Temporary aliases, renamings, status markers?
   Examples: "Call the server Box A" → server=Box A (turn range)
             "Old API is deprecated" → status: deprecated (turn range)
             "Refer to database as DB1" → database=DB1 (turn range)
   → Tag as ALIAS/STATUS (section-specific with turn range)

IMPORTANT: Facts that don't match any of these patterns go to "dossier_facts".
These are narrative facts (preferences, history, context) that belong in dossiers.

Return JSON:
{{
  "global_tags": ["env: python-3.9", "os: windows"],
  "section_rules": [
    {{"start_turn": 10, "end_turn": 15, "rule": "no-eval"}},
    {{"start_turn": 5, "end_turn": 8, "rule": "server=Box A"}}
  ],
  "dossier_facts": ["User prefers dark mode", "User works remotely"]
}}

Classification:"""
        
        try:
            response = self.llm_client.query_external_api(
                query=prompt,
                model="gpt-4.1-mini"
            )
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                classification = json.loads(json_match.group(0))
                
                # Validate structure
                if 'global_tags' not in classification:
                    classification['global_tags'] = []
                if 'section_rules' not in classification:
                    classification['section_rules'] = []
                if 'dossier_facts' not in classification:
                    classification['dossier_facts'] = []
                
                return classification
            else:
                print(f"   ⚠️  No JSON found in classification response, using fallback")
                # Fallback: all facts go to dossiers
                return {
                    "global_tags": [],
                    "section_rules": [],
                    "dossier_facts": [f.get('value', '') for f in facts]
                }
        
        except Exception as e:
            print(f"   ⚠️  Classification failed: {e}, using fallback")
            return {
                "global_tags": [],
                "section_rules": [],
                "dossier_facts": [f.get('value', '') for f in facts]
            }
    
    def _delete_bridge_block(self, block_id: str):
        """
        Delete processed bridge block from daily_ledger.
        
        Args:
            block_id: Bridge block ID to delete
        """
        cursor = self.storage.conn.cursor()
        cursor.execute("DELETE FROM daily_ledger WHERE block_id = ?", (block_id,))
        self.storage.conn.commit()
        print(f"   🗑️  Deleted bridge block {block_id} from active memory")
    
    async def _group_facts_semantically(self, facts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Group related facts by semantic theme using LLM.
        
        This is Phase 2 preparation for Phase 3's dossier routing.
        Groups facts that belong together conceptually, preparing them
        for the DossierGovernor to decide which dossier they should join.
        
        Args:
            facts: List of fact dictionaries with 'text' and 'turn_id' fields
        
        Returns:
            List of fact groups: [{"label": "...", "facts": [...], "timestamp": "..."}]
        
        Example:
            Input: [
                {"text": "User is vegetarian", "turn_id": "turn_001"},
                {"text": "User avoids meat", "turn_id": "turn_001"},
                {"text": "User works with Python", "turn_id": "turn_002"}
            ]
            Output: [
                {
                    "label": "Dietary Preferences",
                    "facts": ["User is vegetarian", "User avoids meat"],
                    "timestamp": "2025-12-15T10:30:00"
                },
                {
                    "label": "Programming",
                    "facts": ["User works with Python"],
                    "timestamp": "2025-12-15T10:31:00"
                }
            ]
        """
        if not facts:
            return []
        
        # Format facts for LLM
        facts_text = json.dumps(facts, indent=2)
        
        prompt = f"""Given these facts extracted from a conversation, group related facts by semantic theme.

Facts:
{facts_text}

For each group, provide:
1. A concise label (2-5 words) describing the theme
2. The facts that belong to that group
3. The earliest timestamp from facts in the group

Return as JSON array:
[
  {{
    "label": "Theme Name",
    "facts": ["fact text 1", "fact text 2"],
    "timestamp": "ISO timestamp"
  }}
]

Groups:"""
        
        try:
            response = self.llm_client.query_external_api(
                query=prompt,
                model="gpt-4.1-mini"
            )
            
            # Extract JSON from response
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                groups = json.loads(json_match.group(0))
                print(f"   📦 Grouped {len(facts)} facts into {len(groups)} semantic clusters")
                return groups
            else:
                print(f"   ⚠️  No JSON found in grouping response, creating single group")
                # Fallback: put all facts in one group
                return [{
                    "label": "General Facts",
                    "facts": [f['text'] for f in facts],
                    "timestamp": facts[0].get('timestamp', datetime.now().isoformat())
                }]
        
        except Exception as e:
            print(f"   ⚠️  Semantic grouping failed: {e}, creating single group")
            return [{
                "label": "General Facts",
                "facts": [f['text'] for f in facts],
                "timestamp": facts[0].get('timestamp', datetime.now().isoformat())
            }]
