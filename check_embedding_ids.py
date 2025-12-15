import sqlite3

conn = sqlite3.connect('memory/cognitive_lattice_memory.db')
cursor = conn.cursor()

# Show sample embeddings for gardened chunks
cursor.execute('''SELECT embedding_id, turn_id FROM embeddings WHERE turn_id LIKE "turn_%" LIMIT 10''')
print("Embeddings for gardened chunks:")
for row in cursor.fetchall():
    print(f"  embedding_id: {row[0]}")
    print(f"  turn_id: {row[1]}\n")

conn.close()
