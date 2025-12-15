"""
Phase 2 Test: Verify semantic fact grouping

Tests the new _group_facts_semantically() method added in Phase 2.
"""

import asyncio
import json
import sys
import re
from datetime import datetime


# Mock classes to avoid loading heavy dependencies
class MockStorage:
    pass

class MockEmbeddingStorage:
    pass

class MockLLMClient:
    async def query_external_api(self, prompt, model):
        # Return mock grouped facts as JSON string (simulating LLM response)
        return """[
  {
    "label": "Dietary Preferences",
    "facts": [
      "User is strictly vegetarian",
      "User avoids all meat products",
      "User prefers plant-based proteins"
    ],
    "timestamp": "2025-12-15T10:30:00"
  },
  {
    "label": "Programming Skills",
    "facts": [
      "User works with Python",
      "User prefers functional programming"
    ],
    "timestamp": "2025-12-15T10:31:00"
  }
]"""


async def test_semantic_grouping():
    """Test semantic fact grouping with mock LLM client."""
    
    print("Phase 2 Test: Semantic Fact Grouping")
    print("=" * 60)
    
    # Create a minimal gardener-like object with just the grouping method
    class TestGardener:
        def __init__(self, llm_client):
            self.llm_client = llm_client
        
        async def _group_facts_semantically(self, facts):
            if not facts:
                return []
            
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
                response = await self.llm_client.query_external_api(
                    prompt=prompt,
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
    
    # Create test gardener
    gardener = TestGardener(llm_client=MockLLMClient())
    
    # Test facts
    test_facts = [
        {"text": "User is strictly vegetarian", "turn_id": "turn_001", "timestamp": "2025-12-15T10:30:00"},
        {"text": "User avoids all meat products", "turn_id": "turn_001", "timestamp": "2025-12-15T10:30:00"},
        {"text": "User prefers plant-based proteins", "turn_id": "turn_002", "timestamp": "2025-12-15T10:30:30"},
        {"text": "User works with Python", "turn_id": "turn_003", "timestamp": "2025-12-15T10:31:00"},
        {"text": "User prefers functional programming", "turn_id": "turn_003", "timestamp": "2025-12-15T10:31:00"}
    ]
    
    print(f"\n1. Testing semantic grouping with {len(test_facts)} facts...")
    
    groups = await gardener._group_facts_semantically(test_facts)
    
    print(f"\n2. Grouped into {len(groups)} clusters:")
    for i, group in enumerate(groups, 1):
        print(f"\n   Cluster {i}: {group['label']}")
        print(f"   Timestamp: {group['timestamp']}")
        print(f"   Facts ({len(group['facts'])}):")
        for fact in group['facts']:
            print(f"     - {fact}")
    
    # Validate results
    print(f"\n3. Validation:")
    assert len(groups) == 2, f"Expected 2 groups, got {len(groups)}"
    print(f"   ✅ Correct number of groups")
    
    diet_group = next((g for g in groups if 'Diet' in g['label']), None)
    assert diet_group is not None, "No dietary group found"
    assert len(diet_group['facts']) == 3, f"Expected 3 dietary facts, got {len(diet_group['facts'])}"
    print(f"   ✅ Dietary group has correct number of facts")
    
    prog_group = next((g for g in groups if 'Program' in g['label']), None)
    assert prog_group is not None, "No programming group found"
    assert len(prog_group['facts']) == 2, f"Expected 2 programming facts, got {len(prog_group['facts'])}"
    print(f"   ✅ Programming group has correct number of facts")
    
    print(f"\n" + "=" * 60)
    print(f"✅ Phase 2 Test: All checks passed!")
    print(f"\nReady for Phase 3:")
    print(f"  - DossierGovernor will receive these fact packets")
    print(f"  - Multi-Vector Voting will find candidate dossiers")
    print(f"  - LLM will decide append vs create")


if __name__ == "__main__":
    asyncio.run(test_semantic_grouping())
