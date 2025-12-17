"""
Inspect dossiers created by test_13_hydra_dossier_e2e.py
"""

import sqlite3
import sys
from pathlib import Path

# Use the actual database path from the test output
test_db = Path(r"C:\Users\seanv\AppData\Local\Temp\tmphcxomz43\test_13_hydra_dossier_e2e.db")

if not test_db.exists():
    print(f"❌ Database not found: {test_db}")
    sys.exit(1)

print(f"📂 Database: {test_db}\n")

conn = sqlite3.connect(test_db)
cursor = conn.cursor()

# Get all dossiers
cursor.execute("""
    SELECT dossier_id, title, summary, created_at
    FROM dossiers
    ORDER BY created_at
""")

dossiers = cursor.fetchall()

print(f"{'='*100}")
print(f"DOSSIERS CREATED: {len(dossiers)}")
print(f"{'='*100}\n")

for i, (dossier_id, title, summary, created_at) in enumerate(dossiers, 1):
    print(f"\n{'─'*100}")
    print(f"DOSSIER {i}: {title}")
    print(f"{'─'*100}")
    print(f"ID: {dossier_id}")
    print(f"Created: {created_at}")
    print(f"\nSummary:")
    print(f"{summary}")
    
    # Get facts for this dossier from dossier_facts table
    cursor.execute("""
        SELECT df.fact_id, fs.key, fs.value, fs.category, fs.created_at
        FROM dossier_facts df
        JOIN fact_store fs ON df.fact_id = fs.fact_id
        WHERE df.dossier_id = ?
        ORDER BY fs.created_at
    """, (dossier_id,))
    
    facts = cursor.fetchall()
    
    print(f"\nFacts ({len(facts)}):")
    for j, (fact_id, key, value, category, fact_created) in enumerate(facts, 1):
        print(f"  {j}. [{category or 'uncategorized'}] {key}: {value}")
        print(f"     (ID: {fact_id}, Created: {fact_created})")
    
    # Get fact embeddings count
    cursor.execute("""
        SELECT COUNT(*)
        FROM dossier_fact_embeddings
        WHERE dossier_id = ?
    """, (dossier_id,))
    
    embedding_count = cursor.fetchone()[0]
    print(f"\nEmbeddings: {embedding_count} fact-level embeddings")

print(f"\n{'='*100}")
print("SUMMARY")
print(f"{'='*100}")

# Total facts across all dossiers
cursor.execute("SELECT COUNT(*) FROM fact_store")
total_facts = cursor.fetchone()[0]

# Total embeddings
cursor.execute("SELECT COUNT(*) FROM dossier_fact_embeddings")
total_embeddings = cursor.fetchone()[0]

print(f"Total facts in database: {total_facts}")
print(f"Total fact embeddings: {total_embeddings}")
print(f"Total dossiers: {len(dossiers)}")

# Check for bridge blocks (should be closed)
cursor.execute("SELECT COUNT(*) FROM bridge_blocks WHERE status = 'ACTIVE'")
active_blocks = cursor.fetchone()[0]

cursor.execute("SELECT COUNT(*) FROM bridge_blocks WHERE status = 'CLOSED'")
closed_blocks = cursor.fetchone()[0]

conn.close()
