"""
Test to verify if embeddings are being created for gardened chunks
"""
import sqlite3

conn = sqlite3.connect('memory/cognitive_lattice_memory.db')
cursor = conn.cursor()

# Check current state
cursor.execute('SELECT COUNT(*) FROM gardened_memory')
gardened_count = cursor.fetchone()[0]

cursor.execute('SELECT COUNT(*) FROM embeddings WHERE turn_id LIKE "turn_%"')
gardened_embedding_count = cursor.fetchone()[0]

print(f"Gardened chunks in database: {gardened_count}")
print(f"Embeddings for gardened chunks (turn_id LIKE 'turn_%'): {gardened_embedding_count}")

if gardened_count > 0 and gardened_embedding_count == 0:
    print("\n❌ PROBLEM: Gardened chunks exist but have NO embeddings!")
    print("   This explains why vector search returns 0 results.")
    print("\n   Solution: Re-run the gardener to create embeddings for existing chunks")

conn.close()
