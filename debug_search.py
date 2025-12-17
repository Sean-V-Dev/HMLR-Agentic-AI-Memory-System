"""Debug dossier search to find why retrieval returns 0 results."""
import sqlite3
import numpy as np
from sentence_transformers import SentenceTransformer

db_path = "test_integration_phase2.db"
query = "Which of my cars would be best for a family road trip?"
threshold = 0.4

print("=" * 80)
print("DOSSIER SEARCH DEBUG")
print("=" * 80)

# Load model
print("\n📦 Loading embedding model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("✅ Model loaded")

# Encode query
print(f"\n🔍 Query: {query}")
query_embedding = model.encode(query)
print(f"   Query embedding shape: {query_embedding.shape}")

# Get all embedded facts
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

cursor.execute("""
    SELECT dfe.fact_id, dfe.dossier_id, dfe.embedding, df.fact_text, d.title
    FROM dossier_fact_embeddings dfe
    JOIN dossier_facts df ON dfe.fact_id = df.fact_id
    JOIN dossiers d ON dfe.dossier_id = d.dossier_id
""")
facts = cursor.fetchall()

print(f"\n📊 Total embedded facts: {len(facts)}")

# Compute similarities
print(f"\n🎯 Computing similarities (threshold={threshold}):")
results = []

for fact_id, dossier_id, embedding_blob, fact_text, title in facts:
    # Decode embedding
    fact_embedding = np.frombuffer(embedding_blob, dtype=np.float32)
    
    # Compute cosine similarity
    similarity = np.dot(query_embedding, fact_embedding) / (
        np.linalg.norm(query_embedding) * np.linalg.norm(fact_embedding)
    )
    
    results.append((fact_id, dossier_id, similarity, fact_text, title))
    
    # Print all scores
    match_indicator = "✅" if similarity >= threshold else "  "
    print(f"{match_indicator} {similarity:.4f} | [{title}] {fact_text[:60]}...")

# Sort by similarity
results.sort(key=lambda x: x[2], reverse=True)

print(f"\n{'='*80}")
print("TOP 5 MATCHES:")
print(f"{'='*80}")
for i, (fact_id, dossier_id, similarity, fact_text, title) in enumerate(results[:5], 1):
    print(f"\n{i}. Score: {similarity:.4f}")
    print(f"   Dossier: {title}")
    print(f"   Fact: {fact_text}")

# Check what would be returned
above_threshold = [r for r in results if r[2] >= threshold]
print(f"\n{'='*80}")
print(f"📈 THRESHOLD ANALYSIS:")
print(f"{'='*80}")
print(f"  Threshold: {threshold}")
print(f"  Facts above threshold: {len(above_threshold)}/{len(results)}")
if above_threshold:
    print(f"  Highest score: {above_threshold[0][2]:.4f}")
    unique_dossiers = len(set(r[1] for r in above_threshold))
    print(f"  Unique dossiers: {unique_dossiers}")
else:
    print(f"  ⚠️  NO FACTS ABOVE THRESHOLD!")
    print(f"  Highest score: {results[0][2]:.4f}")
    print(f"  Recommended threshold: {results[0][2] - 0.05:.2f}")

conn.close()
