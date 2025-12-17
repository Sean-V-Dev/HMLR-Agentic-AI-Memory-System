"""Debug search_summary retrieval."""
import sqlite3
import numpy as np
from sentence_transformers import SentenceTransformer

db_path = "test_integration_phase2.db"
query = "Which of my cars would be best for a family road trip?"

print("=" * 80)
print("SEARCH_SUMMARY DEBUG")
print("=" * 80)

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')
query_embedding = model.encode(query)

# Get all dossiers with search_summaries
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

cursor.execute("""
    SELECT d.dossier_id, d.title, d.search_summary, dse.embedding
    FROM dossiers d
    JOIN dossier_search_embeddings dse ON d.dossier_id = dse.dossier_id
""")
dossiers = cursor.fetchall()

print(f"\n🔍 Query: {query}")
print(f"📊 Total dossiers: {len(dossiers)}\n")

results = []
for dossier_id, title, search_summary, embedding_blob in dossiers:
    dossier_embedding = np.frombuffer(embedding_blob, dtype=np.float32)
    similarity = np.dot(query_embedding, dossier_embedding) / (
        np.linalg.norm(query_embedding) * np.linalg.norm(dossier_embedding)
    )
    results.append((dossier_id, title, similarity, search_summary))

results.sort(key=lambda x: x[2], reverse=True)

print("🎯 SIMILARITY SCORES:")
print("=" * 80)
for dossier_id, title, similarity, search_summary in results:
    match = "✅" if similarity >= 0.3 else "  "
    print(f"{match} {similarity:.4f} | {title}")
    print(f"          Search Summary: {search_summary[:120]}...")
    print()

conn.close()
