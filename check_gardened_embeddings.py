import sqlite3

conn = sqlite3.connect('memory/cognitive_lattice_memory.db')
cursor = conn.cursor()

# Get gardened chunk IDs
cursor.execute('SELECT chunk_id FROM gardened_memory LIMIT 5')
gardened_ids = [row[0] for row in cursor.fetchall()]
print('Sample gardened chunk IDs:')
for gid in gardened_ids:
    print(f'  {gid}')

print('\n' + '='*80 + '\n')

# Check embeddings table structure
cursor.execute('PRAGMA table_info(embeddings)')
columns = cursor.fetchall()
print('Embeddings table structure:')
for col in columns:
    print(f'  {col[1]} ({col[2]})')

print('\n' + '='*80 + '\n')

# Check if any of these IDs are in embeddings
print('Checking embeddings for gardened chunks:')
for gid in gardened_ids:
    cursor.execute('SELECT embedding_id FROM embeddings WHERE embedding_id = ?', (gid,))
    result = cursor.fetchone()
    print(f'  {gid}: {"✓ HAS EMBEDDING" if result else "✗ NO EMBEDDING"}')

print('\n' + '='*80 + '\n')

# Show sample embeddings
cursor.execute('SELECT embedding_id FROM embeddings LIMIT 10')
print('Sample embedding IDs in database:')
for row in cursor.fetchall():
    print(f'  {row[0]}')

conn.close()
