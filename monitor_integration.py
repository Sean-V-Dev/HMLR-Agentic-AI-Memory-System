"""
Monitor integration test progress in real-time
"""

import sqlite3
import time
import os

db_path = "test_integration_phase2.db"

print("🔄 Monitoring Integration Test Progress")
print("=" * 80)
print("(Press Ctrl+C to stop)\n")

last_fact_count = 0
last_dossier_count = 0
last_metadata_count = 0

try:
    while True:
        if not os.path.exists(db_path):
            print("⏳ Waiting for database to be created...")
            time.sleep(2)
            continue
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Count facts
        cursor.execute("SELECT COUNT(*) FROM fact_store")
        fact_count = cursor.fetchone()[0]
        
        # Count dossiers
        cursor.execute("SELECT COUNT(*) FROM dossiers")
        dossier_count = cursor.fetchone()[0]
        
        # Count metadata
        cursor.execute("SELECT COUNT(*) FROM block_metadata")
        metadata_count = cursor.fetchone()[0]
        
        # Count bridge blocks
        cursor.execute("SELECT COUNT(*) FROM daily_ledger")
        ledger_count = cursor.fetchone()[0]
        
        # Show updates if changed
        if (fact_count != last_fact_count or 
            dossier_count != last_dossier_count or 
            metadata_count != last_metadata_count):
            
            print(f"\r📊 Facts: {fact_count} | Dossiers: {dossier_count} | "
                  f"Tagged Blocks: {metadata_count} | Bridge Blocks: {ledger_count}", end='', flush=True)
            
            last_fact_count = fact_count
            last_dossier_count = dossier_count
            last_metadata_count = metadata_count
        
        conn.close()
        time.sleep(1)

except KeyboardInterrupt:
    print("\n\n✅ Monitoring stopped")
