"""Test embedding search directly"""
import sys
sys.path.insert(0, '.')

from hmlr.core.component_factory import ComponentFactory

# Initialize components
factory = ComponentFactory()
components = factory.create_all_components()

# Test search
query = "mountains in china"
print(f"Testing search for: '{query}'\n")

results = components.embedding_storage.search_similar(
    query=query,
    top_k=10,
    min_similarity=0.3
)

print(f"Results found: {len(results)}\n")

for i, result in enumerate(results, 1):
    print(f"[{i}] Turn ID: {result.get('turn_id')}")
    print(f"    Similarity: {result.get('similarity', 0):.4f}")
    print(f"    Text: {result.get('text', '')[:100]}...")
    print()
