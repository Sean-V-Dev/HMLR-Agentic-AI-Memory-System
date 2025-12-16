"""
Test dossier retrieval - verify search works
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from hmlr.memory import Storage
from hmlr.memory.dossier_storage import DossierEmbeddingStorage
from hmlr.memory.retrieval.dossier_retriever import DossierRetriever

print("\n" + "="*80)
print("DOSSIER RETRIEVAL TEST")
print("="*80)

# Initialize
print("\n[1] Initializing...")
storage = Storage()
dossier_storage = DossierEmbeddingStorage(storage.db_path)
retriever = DossierRetriever(storage, dossier_storage)
print("    [+] Ready")

# Test queries
queries = [
    "What are the user's dietary restrictions?",
    "Tell me about vegetarian preferences",
    "Does the user eat meat?",
    "Food preferences and diet"
]

for i, query in enumerate(queries, 1):
    print(f"\n[{i}] Query: '{query}'")
    results = retriever.retrieve_relevant_dossiers(query, top_k=3, threshold=0.3)
    print(f"    [+] Found {len(results)} dossiers")
    
    for j, dos in enumerate(results, 1):
        print(f"\n    Result {j}:")
        print(f"      Title: {dos['title']}")
        print(f"      Relevance: {dos['relevance_score']:.3f}")
        print(f"      Facts: {len(dos['facts'])}")
        print(f"      First 2 facts:")
        for fact in dos['facts'][:2]:
            print(f"        - {fact['fact_text']}")

# Test formatting
print("\n" + "="*80)
print("CONTEXT FORMATTING TEST")
print("="*80 + "\n")

results = retriever.retrieve_relevant_dossiers("dietary preferences", top_k=2, threshold=0.3)
formatted = retriever.format_for_context(results)

print(f"Formatted context ({len(formatted)} chars):\n")
print(formatted)

print("\n" + "="*80)
print("[+] RETRIEVAL TEST COMPLETE")
print("="*80 + "\n")
