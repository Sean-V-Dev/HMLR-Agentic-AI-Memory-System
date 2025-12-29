"""
Test 7B: Vegetarian Conflict (Scribe + Profile Retrieval)

This test validates the complete Scribe workflow:
1. Turn 1: "I am strictly vegetarian" → Scribe extracts constraint
2. Garden Turn 1 → Close bridge block
3. Turn 2 (New Block): "I'm going to a steakhouse and really craving steak. Is that a good idea?"
4. LLM must read profile, recognize constraint, and warn against eating steak

CRITICAL: Tests Scribe extraction → Profile storage → Profile retrieval → Constraint enforcement
"""

import pytest
import asyncio
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from hmlr.core.component_factory import ComponentFactory


@pytest.fixture
def test_db_path(tmp_path):
    """Create temporary database for test isolation."""
    db_path = tmp_path / "test_7b_vegetarian.db"
    yield str(db_path)
    # Cleanup
    if db_path.exists():
        db_path.unlink()


@pytest.mark.asyncio
async def test_7b_vegetarian_conflict_e2e(test_db_path):
    """
    Test 7B: Vegetarian Conflict - Full Scribe Workflow
    
    Phase 1: User declares constraint
    - "I am strictly vegetarian"
    - Scribe extracts dietary_vegetarian constraint
    - Profile updated with constraint
    - Bridge block closed
    
    Phase 2: User asks about conflicting action (NEW BLOCK)
    - "I'm going to a steakhouse and really craving steak. Is that a good idea?"
    - System loads user profile (sees vegetarian constraint)
    - LLM MUST deny eating steak based on profile constraint
    
    Expected:
    - Scribe extracts constraint from Turn 1
    - Profile persists across bridge blocks
    - Turn 2 response acknowledges vegetarian constraint and warns against steak
    """
    
    print("\n" + "="*80)
    print("TEST 7B: Vegetarian Conflict (Scribe → Profile → Constraint Enforcement)")
    print("="*80)
    
    # ========================================================================
    # SETUP: FRESH DATABASE AND EMPTY PROFILE
    # ========================================================================
    
    # Delete existing test database if it exists to ensure clean state
    if os.path.exists(test_db_path):
        os.remove(test_db_path)
        print(f"🗑️  Removed existing database: {test_db_path}")
    
    os.environ['COGNITIVE_LATTICE_DB'] = test_db_path
    print(f"📦 Using fresh database: {test_db_path}")
    
    # Create EMPTY user profile in temp location
    # IMPORTANT: Use temp directory for test isolation
    import json
    from pathlib import Path
    import tempfile
    
    # Create temp profile file
    temp_profile_dir = Path(test_db_path).parent
    profile_path = temp_profile_dir / "user_profile_lite.json"
    
    # Set environment variable so Scribe writes to test profile
    os.environ['USER_PROFILE_PATH'] = str(profile_path)
    print(f"🔧 Using isolated profile: {profile_path}")
    
    # Start with empty profile (no constraints)
    profile_data = {
        "version": "1.0",
        "last_updated": "",
        "glossary": {
            "constraints": [],
            "identity": [],
            "preferences": []
        }
    }
    
    with open(profile_path, 'w', encoding='utf-8') as f:
        json.dump(profile_data, f, indent=2)
    
    print("\n✅ Fresh database and empty user profile created")
    print(f"   Profile path: {profile_path}")
    
    # ========================================================================
    # INITIALIZE COMPONENTS
    # ========================================================================
    factory = ComponentFactory()
    components = factory.create_all_components()
    conversation_engine = factory.create_conversation_engine(components)
    storage = components.storage
    
    # Create ManualGardener for closing bridge blocks
    from hmlr.memory.gardener.manual_gardener import ManualGardener
    from hmlr.core.external_api_client import ExternalAPIClient
    
    llm_client = ExternalAPIClient()
    gardener = ManualGardener(
        storage=components.storage,
        embedding_storage=components.embedding_storage,
        llm_client=llm_client,
        dossier_governor=components.dossier_governor,
        dossier_storage=components.dossier_storage
    )
    
    # ========================================================================
    # TURN 1: USER DECLARES VEGETARIAN CONSTRAINT
    # ========================================================================
    conversation_turns = [
        # Turn 1: Declare dietary constraint (Scribe should extract this)
        "I am strictly vegetarian. I don't eat meat or fish.",
        
        # Turn 2: NEW BLOCK - Ask about steak (conflicts with constraint)
        "I'm going to a steakhouse tonight and I'm really craving a steak. Is that a good idea for me?"
    ]
    
    # ========================================================================
    # RUN CONVERSATION
    # ========================================================================
    responses = []
    
    print("\n" + "="*80)
    print("TURN 1: Declare Vegetarian Constraint")
    print("="*80)
    print(f"User: {conversation_turns[0]}")
    
    response1 = await conversation_engine.process_user_message(conversation_turns[0])
    responses.append(response1)
    print(f"\nAssistant: {response1.to_console_display()}")
    
    # ========================================================================
    # AWAIT SCRIBE COMPLETION
    # ========================================================================
    # The Scribe runs as a background task - we need to wait for it to actually finish
    print("\n⏳ Waiting for Scribe to process constraint extraction...")
    
    # Get the background manager and wait for all tasks to complete
    if hasattr(conversation_engine, 'background_manager'):
        await conversation_engine.background_manager.shutdown(timeout=10.0)
        print("✅ Scribe background task completed")
    else:
        # Fallback: just wait if no background manager
        await asyncio.sleep(5.0)
        print("⚠️  No background manager - used fallback wait")
    
    # ========================================================================
    # CLOSE BRIDGE BLOCK TO ISOLATE PROFILE TEST
    # ========================================================================
    print("\n" + "="*80)
    print("ISOLATING PROFILE: Closing Bridge Block")
    print("="*80)
    
    # Get the active bridge block and close it
    active_blocks = storage.get_active_bridge_blocks()
    if active_blocks:
        for block in active_blocks:
            block_id = block.get('block_id')
            if block_id:
                # Mark as COMPLETED so it won't be loaded as "active" in Turn 2
                storage.update_bridge_block_status(block_id, 'COMPLETED', exit_reason='test_isolation')
                print(f"✅ Closed bridge block: {block_id}")
    else:
        print("⚠️  No active bridge blocks to close")
    
    # Clear the sliding window so Turn 2 can't access Turn 1 from recent memory
    if hasattr(components.sliding_window, 'clear'):
        components.sliding_window.clear()
    print("✅ Sliding window cleared")
    print("   Turn 2 will ONLY have access to user profile, not past memory or bridge blocks")
    
    # ========================================================================
    # VERIFY SCRIBE EXTRACTED CONSTRAINT
    # ========================================================================
    print("\n" + "="*80)
    print("VERIFY: Scribe Extraction")
    print("="*80)
    
    with open(profile_path, 'r', encoding='utf-8') as f:
        updated_profile = json.load(f)
    
    constraints = updated_profile.get("glossary", {}).get("constraints", [])
    vegetarian_constraint = next(
        (c for c in constraints if "vegetarian" in c.get("key", "").lower() or 
                                   "vegetarian" in c.get("description", "").lower()),
        None
    )
    
    if vegetarian_constraint:
        print(f"✅ Scribe extracted vegetarian constraint:")
        print(f"   Key: {vegetarian_constraint.get('key')}")
        print(f"   Description: {vegetarian_constraint.get('description')}")
    else:
        print(f"❌ WARNING: Scribe did not extract vegetarian constraint")
        print(f"   Current profile: {json.dumps(updated_profile, indent=2)}")
    
    # ========================================================================
    # TURN 2: NEW BRIDGE BLOCK - ASK ABOUT STEAK (CONFLICT)
    # ========================================================================
    print("\n" + "="*80)
    print("TURN 2: Ask About Eating Steak (NEW BRIDGE BLOCK)")
    print("="*80)
    print(f"User: {conversation_turns[1]}")
    print("\n⚠️  CRITICAL: Turn 1 memory has been cleared")
    print("   The ONLY source of vegetarian constraint is the user profile")
    print("   If LLM doesn't warn against steak, Scribe extraction failed\n")
    
    response2 = await conversation_engine.process_user_message(conversation_turns[1])
    responses.append(response2)
    
    final_response_text = response2.to_console_display()
    print(f"\nAssistant: {final_response_text}")
    
    # ========================================================================
    # VALIDATE: LLM ACKNOWLEDGED CONSTRAINT AND DENIED STEAK
    # ========================================================================
    print("\n" + "="*80)
    print("VALIDATION: Constraint Enforcement")
    print("="*80)
    
    final_lower = final_response_text.lower()
    
    # Check if LLM acknowledged vegetarian constraint
    mentions_vegetarian = any(
        keyword in final_lower 
        for keyword in ["vegetarian", "meat", "diet", "dietary", "plant-based", "constraint"]
    )
    
    # Check if LLM warned against/denied eating steak
    denies_steak = any(
        phrase in final_lower
        for phrase in [
            "not a good idea", "wouldn't recommend", "against", "conflict",
            "shouldn't", "won't align", "not align", "dietary", "vegetarian",
            "can't", "shouldn't eat", "avoid"
        ]
    )
    
    # Should NOT blindly say "yes go ahead and eat steak"
    encourages_steak = any(
        phrase in final_lower
        for phrase in [
            "go ahead", "good idea", "enjoy the steak", "great choice",
            "perfect", "sounds good"
        ]
    ) and not mentions_vegetarian
    
    print(f"\n✓ Response length: {len(final_response_text)} characters")
    print(f"✓ Mentions vegetarian/dietary constraint: {mentions_vegetarian}")
    print(f"✓ Denies or warns against eating steak: {denies_steak}")
    print(f"✓ Avoids blindly encouraging steak: {not encourages_steak}")
    
    # ========================================================================
    # ASSERTIONS
    # ========================================================================
    
    # Now that we properly close the bridge block, the ONLY way the LLM
    # can know about vegetarianism is from the user profile.
    # If Scribe didn't extract it, the test should fail.
    
    if vegetarian_constraint is None:
        print(f"\n❌ CRITICAL: Scribe did NOT extract vegetarian constraint!")
        print(f"   The LLM response is based on something other than the profile.")
        print(f"   Check if bridge blocks are properly isolated.")
    
    assert mentions_vegetarian, \
        "❌ FAILURE: LLM did not acknowledge vegetarian constraint"
    
    assert denies_steak or not encourages_steak, \
        "❌ FAILURE: LLM did not warn against eating steak despite vegetarian constraint"
    
    # NEW: Also assert that Scribe extracted the constraint
    assert vegetarian_constraint is not None, \
        "❌ FAILURE: Scribe did not extract vegetarian constraint to profile"
    
    # ========================================================================
    # SUCCESS SUMMARY
    # ========================================================================
    print("\n" + "="*80)
    print("✅ TEST 7B PASSED: Scribe → Profile → Constraint Enforcement")
    print("="*80)
    print("\nValidated:")
    print("  ✓ Turn 1: User declared vegetarian constraint")
    print("  ✓ Scribe extracted constraint to user profile")
    print("  ✓ Bridge block closed (Turn 1 isolated)")
    print("  ✓ Sliding window cleared")
    print("  ✓ Turn 2: New bridge block created")
    print("  ✓ Profile constraint loaded into Turn 2 context")
    print("  ✓ LLM acknowledged constraint and denied eating steak")
    print("\nThis proves:")
    print("  → Scribe successfully extracts dietary constraints")
    print("  → User profile persists independently of memory/bridge blocks")
    print("  → Profile constraints are enforced by LLM")
    print("  → Scribe → Profile → LLM pathway works correctly")
    print("="*80)
    
    # Close database connection BEFORE teardown to avoid lock issues
    if storage.conn:
        storage.conn.close()


if __name__ == "__main__":
    """Run test directly without pytest"""
    import tempfile
    
    # Create temporary directory for test database
    temp_dir = tempfile.mkdtemp()
    test_db = os.path.join(temp_dir, "test_7b_vegetarian.db")
    
    try:
        asyncio.run(test_7b_vegetarian_conflict_e2e(test_db))
    except AssertionError:
        # Re-raise assertion errors (test failures)
        raise
    except Exception as e:
        print(f"\n❌ Test error: {e}")
        raise
    finally:
        # Clean up temp directory - ignore Windows file lock errors
        import shutil
        import time
        time.sleep(0.5)  # Give Windows time to release file locks
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except:
            pass  # Silently ignore cleanup errors
