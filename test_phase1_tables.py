"""Quick test to verify Phase 1 database tables were created."""
import sqlite3

db_path = 'hmlr/memory/cognitive_lattice_memory.db'
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Check for dossier tables
cursor.execute("""
    SELECT name FROM sqlite_master 
    WHERE type='table' AND name LIKE 'dossier%'
    ORDER BY name
""")

tables = cursor.fetchall()

print("Phase 1 Test: Dossier Tables")
print("=" * 50)

if tables:
    print(f"✅ Found {len(tables)} dossier tables:")
    for table in tables:
        print(f"   - {table[0]}")
        
        # Get column count
        cursor.execute(f"PRAGMA table_info({table[0]})")
        columns = cursor.fetchall()
        print(f"     ({len(columns)} columns)")
else:
    print("❌ No dossier tables found - tables need to be created")
    print("   Run main.py to initialize database")

conn.close()
