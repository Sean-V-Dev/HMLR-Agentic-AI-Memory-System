"""
Test current retrieval system with multiple car-related queries
to verify if ALL 5 car dossiers are being retrieved and passed to the Governor.

Goal: Verify that domain-level retrieval is working (or not) before making changes.
"""

import asyncio
import os
from hmlr.memory.retrieval.dossier_retriever import DossierRetriever
from hmlr.memory.dossier_storage import DossierEmbeddingStorage

# Test queries - all about cars the user owns
TEST_QUERIES = [
    "Which of my cars would be best for a family road trip?",
    "Which of my cars should I take to the race track?",
    "Which of my cars would be best to drive around the city?",
    "Which of my cars is most fuel efficient?",
    "Which of my cars has the most cargo space?",
]

async def test_retrieval_coverage():
    """Test if retrieval returns all 5 car dossiers for various car queries."""
    
    db_path = "test_integration_phase2.db"
    
    if not os.path.exists(db_path):
        print(f"❌ Database not found: {db_path}")
        print("Run test_integration_phase2.py first to create test data")
        return
    
    # Initialize components
    from hmlr.memory.storage import Storage
    
    storage = Storage(db_path)
    dossier_storage = DossierEmbeddingStorage(db_path)
    retriever = DossierRetriever(storage=storage, dossier_storage=dossier_storage)
    
    # First, get ground truth - how many car dossiers exist?
    print("=" * 80)
    print("GROUND TRUTH: All Dossiers in Database")
    print("=" * 80)
    
    all_dossiers = storage.get_all_dossiers()
    print(f"\nTotal dossiers in database: {len(all_dossiers)}")
    
    for i, dossier in enumerate(all_dossiers, 1):
        print(f"\n{i}. {dossier.get('title', 'Untitled')}")
        print(f"   Summary: {dossier.get('summary', 'No summary')[:100]}...")
        
        # Get facts for this dossier
        facts = storage.get_dossier_facts(dossier['dossier_id'])
        print(f"   Facts: {len(facts)}")
    
    print("\n" + "=" * 80)
    print("RETRIEVAL TESTS: Do all car queries return ALL 5 dossiers?")
    print("=" * 80)
    
    # Test each query
    for query_num, query in enumerate(TEST_QUERIES, 1):
        print(f"\n\n{'=' * 80}")
        print(f"TEST {query_num}: {query}")
        print("=" * 80)
        
        # Retrieve dossiers with timing
        import time
        start_time = time.time()
        retrieved = retriever.retrieve_relevant_dossiers(query, top_k=10)
        search_time = time.time() - start_time
        
        print(f"\n📊 Retrieved: {len(retrieved)} dossiers in {search_time:.4f}s ({search_time*1000:.1f}ms)")
        print(f"   Query embedding + search time: {search_time:.4f}s")
        
        if len(retrieved) == 0:
            print("❌ NO DOSSIERS RETRIEVED - Governor gets nothing!")
            continue
        
        # Show what was retrieved with detailed scoring
        for i, dossier in enumerate(retrieved, 1):
            hit_count = dossier.get('hit_count', 0)
            max_sim = dossier.get('max_similarity', 0.0)
            print(f"\n{i}. {dossier.get('title', 'Untitled')}")
            print(f"   Hits: {hit_count} facts matched | Max similarity: {max_sim:.4f}")
            print(f"   Summary: {dossier.get('summary', 'No summary')[:80]}...")
        
        # Check coverage
        if len(retrieved) == len(all_dossiers):
            print(f"\n✅ SUCCESS: All {len(all_dossiers)} dossiers retrieved - Governor sees full inventory")
        else:
            print(f"\n⚠️  PARTIAL: Only {len(retrieved)}/{len(all_dossiers)} dossiers retrieved")
            print(f"   Missing: {len(all_dossiers) - len(retrieved)} dossiers")
            
            # Show which ones are missing
            retrieved_titles = {d.get('title', 'Untitled') for d in retrieved}
            all_titles = {d.get('title', 'Untitled') for d in all_dossiers}
            missing = all_titles - retrieved_titles
            
            if missing:
                print(f"   Not retrieved: {', '.join(missing)}")
    
    # Summary
    print("\n\n" + "=" * 80)
    print("SUMMARY: Expected Behavior vs Actual")
    print("=" * 80)
    print("\n🎯 EXPECTED:")
    print("   All car-related queries should return ALL 5 car dossiers")
    print("   Governor sees complete inventory and chooses best match")
    print("   Query specifics (family/race/city) handled by Governor, not retrieval")
    
    print("\n📈 ACTUAL:")
    print(f"   Total dossiers: {len(all_dossiers)}")
    print("   Test the results above to see if retrieval is domain-aware")
    
    print("\n💡 INTERPRETATION:")
    print("   If queries return different subsets: Need domain-based retrieval")
    print("   If queries return all 5 dossiers: Current system already works!")
    print("   If queries return 0-2 dossiers: Retrieval too narrow")

if __name__ == "__main__":
    asyncio.run(test_retrieval_coverage())
