"""
Test different embedding models to see if retrieval improves.

Tests 4 models:
- Fast tier: snowflake-arctic-embed-xs, snowflake-arctic-embed-s
- Accurate tier: BAAI/bge-large-en-v1.5, snowflake-arctic-embed-l

For each model:
1. Clear existing embeddings
2. Re-embed all dossier search_summaries
3. Run domain retrieval tests
4. Show results
"""

import asyncio
import os
import sqlite3
from sentence_transformers import SentenceTransformer
from hmlr.memory.storage import Storage
from hmlr.memory.dossier_storage import DossierEmbeddingStorage
from hmlr.memory.retrieval.dossier_retriever import DossierRetriever
import time

# Models to test
MODELS_TO_TEST = [
    {
        "name": "snowflake-arctic-embed-xs",
        "model_id": "Snowflake/snowflake-arctic-embed-xs",
        "speed": "VERY FAST",
        "dims": 384
    },
    {
        "name": "snowflake-arctic-embed-s",
        "model_id": "Snowflake/snowflake-arctic-embed-s",
        "speed": "FAST",
        "dims": 384
    },
    {
        "name": "bge-large-en-v1.5",
        "model_id": "BAAI/bge-large-en-v1.5",
        "speed": "SLOWER",
        "dims": 1024
    },
    {
        "name": "snowflake-arctic-embed-l",
        "model_id": "Snowflake/snowflake-arctic-embed-l",
        "speed": "SLOWER",
        "dims": 1024
    }
]

TEST_QUERIES = [
    "Which of my cars would be best for a family road trip?",
    "Which of my cars should I take to the race track?",
    "Which of my cars would be best to drive around the city?",
    "Which of my cars is most fuel efficient?",
    "Which of my cars has the most cargo space?",
]

def clear_embeddings(db_path):
    """Clear all dossier search embeddings"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM dossier_search_embeddings")
    conn.commit()
    conn.close()
    print("   ✅ Cleared existing embeddings")

def re_embed_all_dossiers(db_path, model):
    """Re-embed all dossiers with the given model"""
    print(f"   Embedding all dossiers with {model['name']}...")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get all dossiers with search_summaries
    cursor.execute("SELECT dossier_id, search_summary FROM dossiers WHERE search_summary IS NOT NULL")
    dossiers = cursor.fetchall()
    
    print(f"   Found {len(dossiers)} dossiers to embed")
    
    # Load model
    start_load = time.time()
    embedding_model = SentenceTransformer(model['model_id'])
    load_time = time.time() - start_load
    print(f"   Model loaded in {load_time:.2f}s")
    
    # Embed each dossier
    start_embed = time.time()
    for dossier_id, search_summary in dossiers:
        if not search_summary:
            continue
        
        # Create embedding
        embedding = embedding_model.encode(search_summary, convert_to_numpy=True)
        embedding_blob = embedding.tobytes()
        
        # Save to database
        cursor.execute("""
            INSERT INTO dossier_search_embeddings (dossier_id, embedding, created_at)
            VALUES (?, ?, datetime('now'))
        """, (dossier_id, embedding_blob))
    
    conn.commit()
    embed_time = time.time() - start_embed
    
    # Verify
    cursor.execute("SELECT COUNT(*) FROM dossier_search_embeddings")
    count = cursor.fetchone()[0]
    
    conn.close()
    
    print(f"   ✅ Embedded {count} dossiers in {embed_time:.2f}s ({embed_time/len(dossiers):.3f}s per dossier)")
    
    return embedding_model

def test_retrieval_with_model(db_path, model, embedding_model):
    """Test retrieval with the given model"""
    
    storage = Storage(db_path)
    
    # Create custom DossierEmbeddingStorage with this model
    dossier_storage = DossierEmbeddingStorage(db_path)
    dossier_storage.model = embedding_model  # Replace model
    
    retriever = DossierRetriever(storage=storage, dossier_storage=dossier_storage)
    
    # Get ground truth
    all_dossiers = storage.get_all_dossiers()
    total_dossiers = len(all_dossiers)
    
    results = []
    
    print(f"\n   Testing {len(TEST_QUERIES)} queries...")
    
    for query in TEST_QUERIES:
        retrieved = retriever.retrieve_relevant_dossiers(query, top_k=10)
        
        result = {
            "query": query,
            "retrieved_count": len(retrieved),
            "total": total_dossiers,
            "retrieved_titles": [d.get('title', 'Untitled') for d in retrieved]
        }
        results.append(result)
        
        status = "✅ ALL" if len(retrieved) == total_dossiers else f"⚠️  {len(retrieved)}/{total_dossiers}"
        print(f"      {status} | {query[:50]}...")
    
    return results

def print_model_results(model, results):
    """Print detailed results for a model"""
    print(f"\n{'=' * 100}")
    print(f"RESULTS: {model['name']} ({model['speed']}, {model['dims']}D)")
    print("=" * 100)
    
    total_queries = len(results)
    perfect_queries = sum(1 for r in results if r['retrieved_count'] == r['total'])
    
    print(f"\n📊 Overall: {perfect_queries}/{total_queries} queries returned ALL dossiers")
    
    for i, result in enumerate(results, 1):
        status = "✅" if result['retrieved_count'] == result['total'] else "❌"
        print(f"\n{i}. {status} {result['query']}")
        print(f"   Retrieved: {result['retrieved_count']}/{result['total']}")
        
        if result['retrieved_count'] < result['total']:
            # Show what was retrieved
            print(f"   Got: {', '.join(result['retrieved_titles'])}")

async def main():
    db_path = "test_integration_phase2.db"
    
    if not os.path.exists(db_path):
        print(f"❌ Database not found: {db_path}")
        print("Run test_integration_phase2.py first to create test data")
        return
    
    print("=" * 100)
    print("EMBEDDING MODEL COMPARISON TEST")
    print("=" * 100)
    print("\nTesting if better embedding models improve domain-level retrieval")
    print(f"Testing {len(MODELS_TO_TEST)} models with {len(TEST_QUERIES)} queries each\n")
    
    all_model_results = []
    
    for i, model in enumerate(MODELS_TO_TEST, 1):
        print(f"\n{'#' * 100}")
        print(f"MODEL {i}/{len(MODELS_TO_TEST)}: {model['name']}")
        print(f"Speed: {model['speed']} | Dimensions: {model['dims']}D")
        print("#" * 100)
        
        # Step 1: Clear existing embeddings
        print("\n1. Clearing existing embeddings...")
        clear_embeddings(db_path)
        
        # Step 2: Re-embed with new model
        print(f"\n2. Re-embedding with {model['name']}...")
        embedding_model = re_embed_all_dossiers(db_path, model)
        
        # Step 3: Test retrieval
        print("\n3. Testing retrieval...")
        results = test_retrieval_with_model(db_path, model, embedding_model)
        
        # Step 4: Show results
        print_model_results(model, results)
        
        all_model_results.append({
            "model": model,
            "results": results
        })
    
    # Final comparison
    print("\n\n" + "=" * 100)
    print("FINAL COMPARISON: Which model retrieves ALL 5 dossiers most often?")
    print("=" * 100)
    
    for model_result in all_model_results:
        model = model_result['model']
        results = model_result['results']
        
        perfect_queries = sum(1 for r in results if r['retrieved_count'] == r['total'])
        total_queries = len(results)
        
        score = f"{perfect_queries}/{total_queries}"
        bar = "█" * perfect_queries + "░" * (total_queries - perfect_queries)
        
        print(f"\n{model['name']:30} {score:6} {bar}")
        print(f"{'':30} Speed: {model['speed']:10} Dims: {model['dims']:4}D")
    
    print("\n" + "=" * 100)
    print("INTERPRETATION:")
    print("=" * 100)
    
    best_score = max(sum(1 for r in mr['results'] if r['retrieved_count'] == r['total']) 
                     for mr in all_model_results)
    
    if best_score == len(TEST_QUERIES):
        print("✅ At least one model retrieves ALL dossiers for ALL queries!")
        print("   → Better embeddings SOLVE the problem")
        print("   → Use the best-performing model")
    elif best_score > 0:
        print("⚠️  Some models perform better than others, but none are perfect")
        print("   → Better embeddings HELP but don't fully solve the problem")
        print("   → May still need domain-based retrieval")
    else:
        print("❌ All models fail to retrieve all dossiers")
        print("   → This is NOT an embedding quality issue")
        print("   → MUST implement domain-based retrieval architecture")

if __name__ == "__main__":
    asyncio.run(main())
