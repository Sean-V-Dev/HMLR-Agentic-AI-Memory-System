import sqlite3

conn = sqlite3.connect('hmlr/memory/cognitive_lattice_memory.db')
c = conn.cursor()

# Check dossiers
c.execute("SELECT COUNT(*) FROM dossiers")
dossier_count = c.fetchone()[0]
print(f'\n📊 Dossier System Status:')
print(f'   Dossiers: {dossier_count}')

c.execute("SELECT COUNT(*) FROM dossier_facts")
fact_count = c.fetchone()[0]
print(f'   Facts: {fact_count}')

c.execute("SELECT COUNT(*) FROM dossier_provenance")
prov_count = c.fetchone()[0]
print(f'   Provenance entries: {prov_count}')

# Show dossiers if any exist
if dossier_count > 0:
    print(f'\n📂 Existing Dossiers:')
    c.execute("SELECT dossier_id, title, summary, created_at, last_updated FROM dossiers")
    for row in c.fetchall():
        print(f'\n   ID: {row[0]}')
        print(f'   Title: {row[1]}')
        print(f'   Summary: {row[2][:100]}...' if len(row[2]) > 100 else f'   Summary: {row[2]}')
        print(f'   Created: {row[3]}')
        print(f'   Updated: {row[4]}')
        
        # Get facts for this dossier
        c.execute("SELECT COUNT(*) FROM dossier_facts WHERE dossier_id = ?", (row[0],))
        facts = c.fetchone()[0]
        print(f'   Facts: {facts}')
else:
    print(f'\n❌ No dossiers found - system not yet used')

conn.close()
