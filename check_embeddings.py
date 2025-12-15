import sqlite3

conn = sqlite3.connect('memory/cognitive_lattice_memory.db')
cursor = conn.cursor()

# Check embedding tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '%embedding%'")
tables = cursor.fetchall()
print('Embedding-related tables:')
for table in tables:
    print(f'  {table[0]}')
    cursor.execute(f'SELECT COUNT(*) FROM {table[0]}')
    print(f'    Rows: {cursor.fetchone()[0]}')

print('\n' + '='*80)

# Check gardened memory
cursor.execute('SELECT COUNT(*) FROM gardened_memory')
print(f'\nTotal gardened memories: {cursor.fetchone()[0]}')

# Check if embeddings exist for gardened chunks
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
all_tables = [t[0] for t in cursor.fetchall()]
print(f'\nAll tables in database: {", ".join(all_tables)}')

conn.close()
