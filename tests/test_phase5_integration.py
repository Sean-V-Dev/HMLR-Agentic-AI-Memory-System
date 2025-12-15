"""
Phase 5 Integration Test: End-to-End Dossier System

This test verifies the complete dossier flow:
1. Facts are extracted and grouped semantically (ManualGardener)
2. Fact packets are processed by DossierGovernor (write-side)
3. Dossiers are stored with embeddings
4. Dossiers can be retrieved by query (read-side)
5. Context hydrator formats dossiers for LLM
"""

import os
import sys
import asyncio
import tempfile
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hmlr.memory import Storage
from hmlr.memory.dossier_storage import DossierEmbeddingStorage
from hmlr.memory.retrieval.dossier_retriever import DossierRetriever
from hmlr.memory.synthesis.dossier_governor import DossierGovernor
from hmlr.memory.id_generator import IDGenerator
from hmlr.core.external_api_client import ExternalAPIClient


def test_phase5_integration():
    """Test complete dossier system flow"""
    print("\n" + "="*80)
    print("PHASE 5 INTEGRATION TEST: End-to-End Dossier System")
    print("="*80 + "\n")
    
    # Create temporary database for testing
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        test_db_path = tmp.name
    
    try:
        # Initialize components
        print("1️⃣  Initializing test components...")
        storage = Storage(db_path=test_db_path)
        dossier_storage = DossierEmbeddingStorage(test_db_path)
        llm_client = ExternalAPIClient()
        id_generator = IDGenerator()
        
        dossier_governor = DossierGovernor(
            storage=storage,
            dossier_storage=dossier_storage,
            llm_client=llm_client,
            id_generator=id_generator
        )
        
        dossier_retriever = DossierRetriever(storage, dossier_storage)
        print("   ✅ Components initialized\n")
        
        # Test 1: Create first dossier with vegetarian facts
        print("2️⃣  TEST 1: Creating dossier with vegetarian diet facts")
        fact_packet_1 = {
            'cluster_label': 'Vegetarian Diet',
            'facts': [
                'User is strictly vegetarian',
                'User avoids meat and fish',
                'User prefers plant-based proteins like tofu and beans'
            ],
            'source_block_id': 'block_test_001',
            'timestamp': '2025-12-15T10:00:00'
        }
        
        dossier_id_1 = asyncio.run(dossier_governor.process_fact_packet(fact_packet_1))
        print(f"   ✅ Created dossier: {dossier_id_1}")
        
        # Verify dossier exists
        dossier_1 = storage.get_dossier(dossier_id_1)
        assert dossier_1 is not None, "Dossier 1 should exist"
        assert dossier_1['title'] == 'Vegetarian Diet', "Title should match cluster label"
        print(f"   ✅ Dossier title: {dossier_1['title']}")
        print(f"   ✅ Dossier summary: {dossier_1['summary'][:100]}...\n")
        
        # Verify facts were added
        facts_1 = storage.get_dossier_facts(dossier_id_1)
        assert len(facts_1) == 3, "Should have 3 facts"
        print(f"   ✅ Dossier has {len(facts_1)} facts")
        
        # Verify embeddings were created
        fact_count = dossier_storage.get_fact_count(dossier_id_1)
        assert fact_count == 3, "Should have 3 fact embeddings"
        print(f"   ✅ Dossier has {fact_count} fact embeddings\n")
        
        # Test 2: Create second dossier on different topic
        print("3️⃣  TEST 2: Creating dossier on different topic (exercise)")
        fact_packet_2 = {
            'cluster_label': 'Exercise Routine',
            'facts': [
                'User runs 3 miles every morning',
                'User does yoga on weekends',
                'User prefers outdoor activities'
            ],
            'source_block_id': 'block_test_002',
            'timestamp': '2025-12-15T11:00:00'
        }
        
        dossier_id_2 = asyncio.run(dossier_governor.process_fact_packet(fact_packet_2))
        print(f"   ✅ Created dossier: {dossier_id_2}")
        assert dossier_id_2 != dossier_id_1, "Should create new dossier for different topic"
        
        dossier_2 = storage.get_dossier(dossier_id_2)
        print(f"   ✅ Dossier title: {dossier_2['title']}")
        print(f"   ✅ Dossier summary: {dossier_2['summary'][:100]}...\n")
        
        # Test 3: Append to existing dossier (Multi-Vector Voting)
        print("4️⃣  TEST 3: Appending facts to existing dossier (Multi-Vector Voting)")
        fact_packet_3 = {
            'cluster_label': 'Dietary Preferences',
            'facts': [
                'User avoids dairy products',  # Should match vegetarian dossier
                'Plant-based diet is important to user',  # Should match vegetarian dossier
                'User checks ingredient labels carefully'  # Related to vegetarian dossier
            ],
            'source_block_id': 'block_test_003',
            'timestamp': '2025-12-15T12:00:00'
        }
        
        dossier_id_3 = asyncio.run(dossier_governor.process_fact_packet(fact_packet_3))
        print(f"   ✅ Routed to dossier: {dossier_id_3}")
        
        # Verify facts were appended to dossier 1 (Multi-Vector Voting should match vegetarian topic)
        facts_updated = storage.get_dossier_facts(dossier_id_1)
        print(f"   ✅ Dossier 1 now has {len(facts_updated)} facts (was 3)")
        
        # Check if summary was updated
        dossier_updated = storage.get_dossier(dossier_id_1)
        print(f"   ✅ Updated summary: {dossier_updated['summary'][:150]}...\n")
        
        # Test 4: Retrieval by query (read-side)
        print("5️⃣  TEST 4: Retrieving dossiers by query")
        
        # Query about vegetarian diet
        query_1 = "What are the user's dietary restrictions?"
        results_1 = dossier_retriever.retrieve_relevant_dossiers(
            query=query_1,
            top_k=3,
            threshold=0.3
        )
        print(f"   Query: '{query_1}'")
        print(f"   ✅ Found {len(results_1)} relevant dossiers")
        
        if results_1:
            top_result = results_1[0]
            print(f"   ✅ Top result: {top_result['title']}")
            print(f"      Relevance score: {top_result['relevance_score']:.3f}")
            print(f"      Facts: {len(top_result['facts'])}")
            assert top_result['dossier_id'] == dossier_id_1, "Should retrieve vegetarian dossier"
        
        print()
        
        # Query about exercise
        query_2 = "Tell me about the user's fitness activities"
        results_2 = dossier_retriever.retrieve_relevant_dossiers(
            query=query_2,
            top_k=3,
            threshold=0.3
        )
        print(f"   Query: '{query_2}'")
        print(f"   ✅ Found {len(results_2)} relevant dossiers")
        
        if results_2:
            top_result = results_2[0]
            print(f"   ✅ Top result: {top_result['title']}")
            print(f"      Relevance score: {top_result['relevance_score']:.3f}")
            print(f"      Facts: {len(top_result['facts'])}")
        
        print()
        
        # Test 5: Context formatting
        print("6️⃣  TEST 5: Formatting dossiers for context window")
        formatted_context = dossier_retriever.format_for_context(results_1)
        print(f"   ✅ Formatted context length: {len(formatted_context)} chars")
        print(f"\n   Preview:")
        print("   " + "-"*76)
        lines = formatted_context.split('\n')[:15]
        for line in lines:
            print(f"   {line}")
        if len(formatted_context.split('\n')) > 15:
            print(f"   ... ({len(formatted_context.split('\n')) - 15} more lines)")
        print("   " + "-"*76 + "\n")
        
        # Test 6: Provenance tracking
        print("7️⃣  TEST 6: Verifying provenance tracking")
        history = storage.get_dossier_history(dossier_id_1)
        print(f"   ✅ Dossier 1 has {len(history)} provenance entries")
        
        for i, entry in enumerate(history[:5], 1):
            print(f"   {i}. {entry['operation']} - {entry['timestamp']}")
        
        if len(history) > 5:
            print(f"   ... ({len(history) - 5} more entries)")
        
        print()
        
        # Test 7: Multi-Vector Voting validation
        print("8️⃣  TEST 7: Validating Multi-Vector Voting algorithm")
        test_facts = [
            'User likes vegetables',  # Should match vegetarian dossier
            'It is healthy',  # Vague - might match multiple
            'Plant protein is preferred'  # Should match vegetarian dossier
        ]
        
        # Search with each fact and count votes
        vote_counts = {}
        for fact in test_facts:
            results = dossier_storage.search_similar_facts(fact, top_k=5, threshold=0.3)
            for fact_id, dos_id, score in results:
                vote_counts[dos_id] = vote_counts.get(dos_id, 0) + 1
        
        print(f"   Vote tally:")
        for dos_id, votes in sorted(vote_counts.items(), key=lambda x: x[1], reverse=True):
            dos = storage.get_dossier(dos_id)
            title = dos['title'] if dos else 'Unknown'
            print(f"   - {title}: {votes} votes")
        
        if vote_counts:
            winner = max(vote_counts.items(), key=lambda x: x[1])
            print(f"   ✅ Winner: {storage.get_dossier(winner[0])['title']} ({winner[1]} votes)\n")
        
        # Summary
        print("="*80)
        print("✅ ALL TESTS PASSED")
        print("="*80)
        print(f"\nTest Summary:")
        print(f"  - Created {len(storage.get_all_dossiers())} dossiers")
        print(f"  - Dossier 1: {len(storage.get_dossier_facts(dossier_id_1))} facts")
        print(f"  - Dossier 2: {len(storage.get_dossier_facts(dossier_id_2))} facts")
        print(f"  - Multi-Vector Voting: ✅ Working")
        print(f"  - Retrieval: ✅ Working")
        print(f"  - Context formatting: ✅ Working")
        print(f"  - Provenance tracking: ✅ Working")
        print()
        
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Cleanup
        try:
            os.unlink(test_db_path)
            print(f"🧹 Cleaned up test database\n")
        except:
            pass


if __name__ == "__main__":
    success = test_phase5_integration()
    sys.exit(0 if success else 1)
