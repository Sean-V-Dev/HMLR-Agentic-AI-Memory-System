"""
Test Phase 2 Refactored Gardener

Tests the dual-output system:
1. Sticky Meta Tags -> block_metadata table
2. Fact Packets -> DossierGovernor -> Dossiers
"""

import sys
import sqlite3
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from hmlr.memory.storage import Storage
from hmlr.memory.embeddings.embedding_manager import EmbeddingStorage
from hmlr.memory.gardener.manual_gardener import ManualGardener
from hmlr.memory.synthesis.dossier_governor import DossierGovernor
from hmlr.memory.dossier_storage import DossierEmbeddingStorage
from hmlr.core.external_api_client import ExternalAPIClient

def test_phase2_gardener():
    """Test Phase 2 refactored gardener"""
    print("🧪 Phase 2 Gardener Test")
    print("=" * 80)
    
    # Initialize components
    db_path = "hmlr/memory/cognitive_lattice_memory.db"
    storage = Storage(db_path)
    embedding_storage = EmbeddingStorage(db_path)
    
    # Initialize dossier system
    dossier_storage = DossierEmbeddingStorage(
        db_path=db_path,
        model_name="all-MiniLM-L6-v2"
    )
    
    llm_client = ExternalAPIClient()
    
    # Initialize ID generator
    from hmlr.memory.id_generator import IDGenerator
    id_generator = IDGenerator()
    
    dossier_governor = DossierGovernor(
        storage=storage,
        dossier_storage=dossier_storage,
        llm_client=llm_client,
        id_generator=id_generator
    )
    
    # Initialize gardener
    gardener = ManualGardener(
        storage=storage,
        embedding_storage=embedding_storage,
        llm_client=llm_client,
        dossier_governor=dossier_governor,
        dossier_storage=dossier_storage
    )
    
    # Check for existing bridge blocks
    print("\n1. Checking for Bridge Blocks...")
    print("-" * 80)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("SELECT block_id, content_json FROM daily_ledger WHERE block_type = 'bridge_block'")
    blocks_raw = cursor.fetchall()
    
    # Parse topic labels from content_json
    import json
    blocks = []
    for block_id, content_json in blocks_raw:
        content = json.loads(content_json)
        topic_label = content.get('topic_label', 'Unknown')
        blocks.append((block_id, topic_label))
    
    if not blocks:
        print("   ⚠️  No bridge blocks found in daily_ledger")
        print("   Creating a test block...")
        
        # Create a test bridge block with facts
        test_block_id = "test_block_phase2_001"
        
        # First, create some facts in fact_store
        from datetime import datetime
        test_facts = [
            {
                'key': 'environment',
                'value': 'User is working with Python 3.9',
                'turn_id': 'turn_001',
                'timestamp': datetime.now().isoformat()
            },
            {
                'key': 'constraint',
                'value': 'Never use eval() in this project',
                'turn_id': 'turn_002',
                'timestamp': datetime.now().isoformat()
            },
            {
                'key': 'preference',
                'value': 'User prefers dark mode in VS Code',
                'turn_id': 'turn_003',
                'timestamp': datetime.now().isoformat()
            },
            {
                'key': 'alias',
                'value': 'Refer to the production server as Box A',
                'turn_id': 'turn_004',
                'timestamp': datetime.now().isoformat()
            },
            {
                'key': 'history',
                'value': 'User has been working on this project for 6 months',
                'turn_id': 'turn_005',
                'timestamp': datetime.now().isoformat()
            }
        ]
        
        # Store facts in fact_store
        import json
        for fact in test_facts:
            cursor.execute("""
                INSERT OR IGNORE INTO fact_store 
                (key, value, turn_id, block_id, timestamp)
                VALUES (?, ?, ?, ?, ?)
            """, (
                fact['key'],
                fact['value'],
                fact['turn_id'],
                test_block_id,
                fact['timestamp']
            ))
        
        # Create bridge block in daily_ledger
        block_content = {
            'topic_label': 'Test Topic for Phase 2',
            'turns': [
                {
                    'turn_id': 'turn_001',
                    'user_message': 'What version of Python am I using?',
                    'ai_response': 'You are working with Python 3.9'
                },
                {
                    'turn_id': 'turn_002',
                    'user_message': 'Any coding rules I should follow?',
                    'ai_response': 'Never use eval() in this project for security'
                },
                {
                    'turn_id': 'turn_003',
                    'user_message': 'What are my preferences?',
                    'ai_response': 'You prefer dark mode in VS Code'
                }
            ]
        }
        
        cursor.execute("""
            INSERT OR IGNORE INTO daily_ledger
            (block_id, block_type, content_json, created_at)
            VALUES (?, ?, ?, ?)
        """, (
            test_block_id,
            'bridge_block',
            json.dumps(block_content),
            datetime.now().isoformat()
        ))
        
        conn.commit()
        print(f"   ✅ Created test block: {test_block_id}")
        print(f"   ✅ Added {len(test_facts)} test facts")
        blocks = [(test_block_id, 'Test Topic for Phase 2')]
    
    conn.close()
    
    print(f"\n   Found {len(blocks)} bridge block(s):")
    for block_id, topic in blocks:
        print(f"      • {block_id}: {topic}")
    
    # Process first bridge block
    if blocks:
        block_id = blocks[0][0]
        print(f"\n2. Processing Block: {block_id}")
        print("-" * 80)
        
        result = gardener.process_bridge_block(block_id)
        
        print(f"\n3. Processing Result:")
        print("-" * 80)
        print(f"   Status: {result.get('status')}")
        print(f"   Facts Processed: {result.get('facts_processed', 0)}")
        print(f"   Tags Applied: {result.get('tags_applied', 0)}")
        print(f"   Dossiers Created: {result.get('dossiers_created', 0)}")
        
        # Check block_metadata table
        print(f"\n4. Checking block_metadata Table:")
        print("-" * 80)
        
        metadata = storage.get_block_metadata(block_id)
        print(f"   Global Tags: {metadata.get('global_tags', [])}")
        print(f"   Section Rules: {metadata.get('section_rules', [])}")
        
        # Check dossiers
        print(f"\n5. Checking Dossiers:")
        print("-" * 80)
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT dossier_id, title, summary 
            FROM dossiers 
            ORDER BY last_updated DESC
            LIMIT 5
        """)
        
        dossiers = cursor.fetchall()
        print(f"   Found {len(dossiers)} dossiers:")
        for dos_id, title, summary in dossiers:
            print(f"      • {dos_id}: {title}")
            print(f"        {summary[:80]}...")
        
        conn.close()
        
        print(f"\n{'=' * 80}")
        print("✅ Phase 2 Gardener Test Complete!")
        
    else:
        print("\n   ⚠️  No blocks to process")

if __name__ == "__main__":
    test_phase2_gardener()
