import sqlite3

conn = sqlite3.connect('hmlr/memory/cognitive_lattice_memory.db')
c = conn.cursor()
c.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
tables = c.fetchall()

print(f'\nDatabase tables ({len(tables)}):')
for t in tables:
    print(f'  - {t[0]}')

# Check for dossier tables specifically
dossier_tables = [t[0] for t in tables if 'dossier' in t[0].lower()]
print(f'\nDossier tables: {len(dossier_tables)}')
for t in dossier_tables:
    print(f'  ✅ {t}')

conn.close()
