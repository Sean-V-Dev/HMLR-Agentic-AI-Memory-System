"""
Quick check of test database structure
"""

import sqlite3
import tempfile
from pathlib import Path

temp_dir = Path(tempfile.gettempdir())
possible_dbs = list(temp_dir.glob("**/test_13_hydra_dossier_e2e.db"))
if possible_dbs:
    test_db = possible_dbs[0]
    print(f"📂 Database: {test_db}\n")
    
    conn = sqlite3.connect(test_db)
    cursor = conn.cursor()
    
    # List all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    
    print("Tables in database:")
    for table in tables:
        print(f"  - {table[0]}")
        
        # Count rows
        cursor.execute(f"SELECT COUNT(*) FROM {table[0]}")
        count = cursor.fetchone()[0]
        print(f"    Rows: {count}")
    
    print("\n" + "="*80)
    print("FACT STORE SAMPLE:")
    print("="*80)
    cursor.execute("SELECT fact_id, fact_text, fact_type, source_block_id FROM fact_store LIMIT 10")
    for row in cursor.fetchall():
        print(f"\n{row[0]}")
        print(f"  Type: {row[2]}")
        print(f"  Block: {row[3]}")
        print(f"  Text: {row[1][:100]}...")
    
    conn.close()
