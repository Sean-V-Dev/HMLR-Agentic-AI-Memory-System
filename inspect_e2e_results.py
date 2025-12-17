"""
Quick E2E Test Results Viewer

Inspects the test database to see what was created during the E2E test.
"""

import sqlite3
import json

db_path = "test_e2e_phase2.db"

print("🔍 E2E Test Results")
print("=" * 80)

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Check facts
print("\n📋 FACTS IN fact_store:")
print("-" * 80)
cursor.execute("SELECT key, value, source_block_id FROM fact_store ORDER BY fact_id")
facts = cursor.fetchall()
print(f"Total: {len(facts)} facts\n")
for key, value, block_id in facts[:10]:  # Show first 10
    print(f"  [{block_id}] {value}")
if len(facts) > 10:
    print(f"  ... and {len(facts) - 10} more")

# Check dossiers
print(f"\n\n🗂️  DOSSIERS:")
print("-" * 80)
cursor.execute("SELECT dossier_id, title, summary FROM dossiers")
dossiers = cursor.fetchall()
print(f"Total: {len(dossiers)} dossiers\n")
for dos_id, title, summary in dossiers:
    print(f"\n  📁 {title} ({dos_id})")
    print(f"     {summary[:100]}...")
    
    # Count facts in dossier
    cursor.execute("SELECT fact_text FROM dossier_facts WHERE dossier_id = ?", (dos_id,))
    dos_facts = cursor.fetchall()
    print(f"     Facts: {len(dos_facts)}")
    for fact_text, in dos_facts[:3]:
        print(f"        • {fact_text[:80]}")
    if len(dos_facts) > 3:
        print(f"        ... and {len(dos_facts) - 3} more")

# Check block_metadata
print(f"\n\n🏷️  BLOCK METADATA (Tags):")
print("-" * 80)
cursor.execute("SELECT block_id, global_tags, section_rules FROM block_metadata")
metadata = cursor.fetchall()
print(f"Total: {len(metadata)} blocks with tags\n")
for block_id, global_tags_json, section_rules_json in metadata:
    global_tags = json.loads(global_tags_json) if global_tags_json else []
    section_rules = json.loads(section_rules_json) if section_rules_json else []
    
    print(f"  📦 {block_id}")
    if global_tags:
        print(f"     Global: {', '.join(global_tags)}")
    if section_rules:
        print(f"     Sections: {len(section_rules)} rules")

# Check daily_ledger (should be empty after gardening)
print(f"\n\n🌉 DAILY LEDGER (Bridge Blocks):")
print("-" * 80)
cursor.execute("SELECT COUNT(*) FROM daily_ledger")
count = cursor.fetchone()[0]
print(f"Remaining blocks: {count} (should be 0 if all gardened)")

if count > 0:
    cursor.execute("SELECT block_id, status FROM daily_ledger")
    for block_id, status in cursor.fetchall():
        print(f"  • {block_id}: {status}")

# Check metadata_staging
print(f"\n\n📝 METADATA STAGING (Turns):")
print("-" * 80)
cursor.execute("SELECT turn_id, user_message FROM metadata_staging ORDER BY turn_sequence")
turns = cursor.fetchall()
print(f"Total: {len(turns)} turns\n")
for turn_id, user_msg in turns:
    print(f"  • {turn_id}: {user_msg[:60]}...")

conn.close()

print(f"\n" + "=" * 80)
print("✅ Inspection Complete")
print(f"\nDatabase: {db_path}")
