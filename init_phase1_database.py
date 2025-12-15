"""Initialize database with Phase 1 dossier tables."""
from hmlr.memory.storage import Storage

print("Initializing database with Phase 1 dossier tables...")
print("=" * 60)

storage = Storage('hmlr/memory/cognitive_lattice_memory.db')

print("\n✅ Database initialized successfully!")
print(f"   Path: {storage.db_path}")

# Verify tables were created
import sqlite3
conn = sqlite3.connect(storage.db_path)
cursor = conn.cursor()

cursor.execute("""
    SELECT name FROM sqlite_master 
    WHERE type='table' AND name LIKE 'dossier%'
    ORDER BY name
""")

tables = cursor.fetchall()
print(f"\n📊 Created {len(tables)} dossier tables:")
for table in tables:
    cursor.execute(f"PRAGMA table_info({table[0]})")
    columns = cursor.fetchall()
    print(f"   ✅ {table[0]} ({len(columns)} columns)")

conn.close()
storage.close()

print("\n" + "=" * 60)
print("Phase 1 foundation complete! Ready for Phase 2.")
