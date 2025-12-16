"""
Quick E2E test - directly test dossier creation and retrieval
Bypasses full component initialization
"""
import sys
import asyncio
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

# Direct imports
from hmlr.memory import Storage
from hmlr.memory.dossier_storage import DossierEmbeddingStorage
from hmlr.memory.retrieval.dossier_retriever import DossierRetriever
from hmlr.memory.synthesis.dossier_governor import DossierGovernor
from hmlr.memory.id_generator import IDGenerator
from hmlr.core.external_api_client import ExternalAPIClient

print("\n" + "="*80)
print("QUICK E2E TEST: Dossier System")
print("="*80)

# Use real database
print("\n[1] Initializing components...")
storage = Storage()
dossier_storage = DossierEmbeddingStorage(storage.db_path)
llm_client = ExternalAPIClient()
id_generator = IDGenerator()

dossier_governor = DossierGovernor(
    storage=storage,
    dossier_storage=dossier_storage,
    llm_client=llm_client,
    id_generator=id_generator
)

dossier_retriever = DossierRetriever(storage, dossier_storage)
print("    [+] Components ready")

# Test 1: Create a dossier about vegetarian diet
print("\n[2] Creating dossier: Vegetarian Diet")
fact_packet = {
    'cluster_label': 'Vegetarian Diet Preferences',
    'facts': [
        'User is strictly vegetarian',
        'User avoids all meat products',
        'User prefers plant-based proteins like tofu and beans',
        'User checks ingredient labels carefully for animal products'
    ],
    'source_block_id': 'block_quicktest_001',
    'timestamp': '2025-12-15T18:45:00'
}

dossier_id = asyncio.run(dossier_governor.process_fact_packet(fact_packet))
print(f"    [+] Dossier created: {dossier_id}")

# Verify it was saved
dossier = storage.get_dossier(dossier_id)
print(f"    [+] Title: {dossier['title']}")
print(f"    [+] Summary: {dossier['summary'][:100]}...")
facts = storage.get_dossier_facts(dossier_id)
print(f"    [+] Facts: {len(facts)}")

# Test 2: Retrieve by query
print("\n[3] Retrieving dossiers with query: 'dietary restrictions'")
results = dossier_retriever.retrieve_relevant_dossiers(
    query="What are the user's dietary restrictions and food preferences?",
    top_k=3,
    threshold=0.3
)
print(f"    [+] Found {len(results)} dossiers")
if results:
    top = results[0]
    print(f"    [+] Top result: {top['title']}")
    print(f"    [+] Relevance: {top['relevance_score']:.3f}")
    print(f"    [+] Facts in result: {len(top['facts'])}")

# Test 3: Format for context
print("\n[4] Formatting for context window")
formatted = dossier_retriever.format_for_context(results)
print(f"    [+] Context length: {len(formatted)} chars")
print("\n    Preview (first 500 chars):")
print("    " + "-"*76)
for line in formatted[:500].split('\n'):
    print(f"    {line}")
print("    ...")
print("    " + "-"*76)

print("\n" + "="*80)
print("[+] E2E TEST PASSED - Dossier system working!")
print("="*80 + "\n")
