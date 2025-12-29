"""
Compare BGE embedding models: small vs base vs large

Tests retrieval accuracy across 3 BGE variants with realistic HMLR content.
Goal: Determine if bge-small or bge-base can replace bge-large without
losing retrieval quality.
"""

import time
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

# BGE Model variants (reordered: large first to test GPU warm-up hypothesis)
MODELS = [
    {
        "name": "bge-large",
        "model_id": "BAAI/bge-large-en-v1.5",
        "dims": 1024,
        "expected_speed": "SLOWER (current)"
    },
    {
        "name": "bge-base",
        "model_id": "BAAI/bge-base-en-v1.5",
        "dims": 768,
        "expected_speed": "MEDIUM"
    },
    {
        "name": "bge-small",
        "model_id": "BAAI/bge-small-en-v1.5",
        "dims": 384,
        "expected_speed": "FASTEST"
    }
]

# Test corpus: Realistic conversation content
TEST_SENTENCES = [
    "I work at Tesla as a senior engineer. My main project is optimizing battery thermal management systems.",
    "My favorite restaurant is Tartarus Bar & Grill downtown. They have amazing vegetarian options and craft cocktails.",
    "I'm planning a vacation to Japan in March. Really interested in visiting Kyoto temples and trying authentic ramen.",
    "I drive a 2018 Honda Civic. It's reliable but I've been thinking about upgrading to an EV soon.",
    "My daughter's birthday is next week. She's turning 7 and loves dinosaurs, so I'm planning a museum trip.",
    "I have a severe peanut allergy. I always carry an EpiPen and avoid Thai restaurants.",
    "I'm renovating my kitchen. Looking at quartz countertops and considering a gas range upgrade.",
    "I play guitar in a local indie rock band called Cerberus. We mostly do covers but working on originals.",
    "My dog Max is a golden retriever. He's 3 years old and loves going to the dog park on weekends.",
    "I'm learning Python for data analysis. Currently working through pandas tutorials and Kaggle competitions."
]

# Test queries: Vague but on-point
TEST_QUERIES = [
    "Where does he work?",  # Should match: Tesla engineer
    "What restaurants does he like?",  # Should match: Tartarus Bar & Grill
    "Any upcoming trips planned?",  # Should match: Japan vacation
    "What kind of car?",  # Should match: Honda Civic
    "Tell me about his kids",  # Should match: daughter's birthday
    "Any food allergies or restrictions?",  # Should match: peanut allergy
    "What home improvement projects?",  # Should match: kitchen renovation
    "Does he play any instruments?",  # Should match: guitar in band
    "Any pets?",  # Should match: dog Max
    "What's he learning lately?",  # Should match: Python/data analysis
]

def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors."""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def test_model(model_info):
    """Test a single model variant."""
    print(f"\n{'=' * 80}")
    print(f"Testing: {model_info['name']} ({model_info['dims']}D)")
    print(f"Expected Speed: {model_info['expected_speed']}")
    print("=" * 80)
    
    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        print(f"Using GPU: {gpu_name}")
        
        # Show GPU memory before loading
        torch.cuda.empty_cache()
        mem_before = torch.cuda.memory_allocated() / 1024**2
        print(f"GPU memory before: {mem_before:.1f} MB")
    else:
        print("Using CPU")
    
    # Load model
    print(f"\nLoading {model_info['model_id']}...")
    start_load = time.time()
    model = SentenceTransformer(model_info['model_id'], device=device)
    load_time = time.time() - start_load
    
    if device == 'cuda':
        mem_after = torch.cuda.memory_allocated() / 1024**2
        mem_used = mem_after - mem_before
        print(f"✓ Loaded in {load_time:.2f}s (GPU memory: +{mem_used:.1f} MB)")
    else:
        print(f"✓ Loaded in {load_time:.2f}s")
    
    # Embed corpus
    print(f"\nEmbedding {len(TEST_SENTENCES)} sentences...")
    start_embed = time.time()
    sentence_embeddings = model.encode(
        TEST_SENTENCES, 
        convert_to_numpy=True,
        show_progress_bar=False,
        batch_size=8
    )
    embed_time = time.time() - start_embed
    print(f"✓ Embedded in {embed_time:.3f}s ({embed_time/len(TEST_SENTENCES):.4f}s per sentence)")
    
    # Test each query
    print(f"\nTesting {len(TEST_QUERIES)} queries...")
    results = []
    
    for i, query in enumerate(TEST_QUERIES, 1):
        # Embed query
        query_embedding = model.encode(
            query, 
            convert_to_numpy=True,
            show_progress_bar=False
        )
        
        # Calculate similarity to all sentences
        similarities = [cosine_similarity(query_embedding, sent_emb) 
                       for sent_emb in sentence_embeddings]
        
        # Get top match
        top_idx = np.argmax(similarities)
        top_score = similarities[top_idx]
        top_sentence = TEST_SENTENCES[top_idx]
        
        results.append({
            "query": query,
            "top_match": top_sentence,
            "score": top_score,
            "all_scores": similarities
        })
        
        print(f"  {i}/{len(TEST_QUERIES)} queries processed", end='\r')
    
    print(f"  ✓ All {len(TEST_QUERIES)} queries completed")
    
    return {
        "model": model_info,
        "load_time": load_time,
        "embed_time": embed_time,
        "results": results
    }

def print_model_results(test_result):
    """Print detailed results for a model."""
    model = test_result['model']
    results = test_result['results']
    
    print(f"\n{'=' * 80}")
    print(f"RESULTS: {model['name']}")
    print("=" * 80)
    
    print(f"\n⏱️  Performance:")
    print(f"   Load time: {test_result['load_time']:.2f}s")
    print(f"   Embed time: {test_result['embed_time']:.3f}s for {len(TEST_SENTENCES)} sentences")
    
    print(f"\n🎯 Retrieval Quality:")
    strong_matches = sum(1 for r in results if r['score'] >= 0.5)
    medium_matches = sum(1 for r in results if 0.4 <= r['score'] < 0.5)
    weak_matches = sum(1 for r in results if r['score'] < 0.4)
    
    print(f"   Strong matches (≥0.5): {strong_matches}/{len(results)}")
    print(f"   Medium matches (0.4-0.5): {medium_matches}/{len(results)}")
    print(f"   Weak matches (<0.4): {weak_matches}/{len(results)}")
    
    avg_score = np.mean([r['score'] for r in results])
    print(f"   Average similarity: {avg_score:.4f}")
    
    print(f"\n📋 Query Details:")
    for i, result in enumerate(results, 1):
        score_emoji = "✅" if result['score'] >= 0.5 else "⚠️" if result['score'] >= 0.4 else "❌"
        print(f"\n{i}. {score_emoji} {result['query']}")
        print(f"   Score: {result['score']:.4f}")
        print(f"   Match: {result['top_match'][:80]}...")

def compare_models(all_results):
    """Print side-by-side comparison."""
    print(f"\n\n{'=' * 80}")
    print("SIDE-BY-SIDE COMPARISON")
    print("=" * 80)
    
    print(f"\n{'Model':<15} {'Dims':<8} {'Load':<10} {'Embed':<12} {'Avg Score':<12} {'Strong':<8}")
    print("-" * 80)
    
    for result in all_results:
        model = result['model']
        avg_score = np.mean([r['score'] for r in result['results']])
        strong = sum(1 for r in result['results'] if r['score'] >= 0.5)
        
        print(f"{model['name']:<15} "
              f"{model['dims']:<8} "
              f"{result['load_time']:.2f}s{'':<5} "
              f"{result['embed_time']:.3f}s{'':<6} "
              f"{avg_score:.4f}{'':<7} "
              f"{strong}/{len(result['results'])}")
    
    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)
    
    # Find best performer
    best_avg = max(np.mean([r['score'] for r in res['results']]) for res in all_results)
    best_model = next(res for res in all_results 
                     if np.mean([r['score'] for r in res['results']]) == best_avg)
    
    fastest_model = min(all_results, key=lambda r: r['embed_time'])
    
    print(f"\n🏆 Best Accuracy: {best_model['model']['name']} (avg: {best_avg:.4f})")
    print(f"⚡ Fastest: {fastest_model['model']['name']} ({fastest_model['embed_time']:.3f}s)")
    
    # Check if smaller model is "good enough"
    small_avg = np.mean([r['score'] for r in all_results[0]['results']])
    large_avg = np.mean([r['score'] for r in all_results[2]['results']])
    diff = large_avg - small_avg
    
    print(f"\n📊 Accuracy Drop (large → small): {diff:.4f}")
    
    if diff < 0.02:
        print("   ✅ NEGLIGIBLE - bge-small is just as accurate")
        print("   → Recommend switching to bge-small for 3x speed boost")
    elif diff < 0.05:
        print("   ⚠️  MINOR - bge-small slightly worse but acceptable")
        print("   → Consider bge-base as middle ground")
    else:
        print("   ❌ SIGNIFICANT - bge-large clearly better")
        print("   → Stick with bge-large for accuracy")

def main():
    print("=" * 80)
    print("BGE MODEL COMPARISON: small vs base vs large")
    print("=" * 80)
    print(f"\nTesting {len(MODELS)} models with:")
    print(f"  - {len(TEST_SENTENCES)} corpus sentences")
    print(f"  - {len(TEST_QUERIES)} vague queries")
    print("\nGoal: Can we use a smaller/faster model without losing accuracy?")
    
    all_results = []
    
    for model_info in MODELS:
        result = test_model(model_info)
        print_model_results(result)
        all_results.append(result)
    
    compare_models(all_results)

if __name__ == "__main__":
    main()
