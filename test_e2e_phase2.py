"""
End-to-End Phase 2 Test: Multi-Car Scenario

Tests the complete flow:
1. Conversation Turn → Fact Extraction → Bridge Block → Gardening → Dossiers
2. Repeat for 5 turns (5 different cars)
3. Turn 6: Multi-hop query requiring info from all 5 cars
4. Retrieve and validate context assembly

Scenario:
- Turn 1: User's Tesla Model 3 (electric, autopilot)
- Turn 2: User's Ford F-150 (truck, hauling)
- Turn 3: User's Honda Civic (daily commuter, gas-efficient)
- Turn 4: User's Porsche 911 (weekend car, sports car)
- Turn 5: User's Toyota Sienna (family minivan)
- Turn 6: "Which of my cars would be best for a road trip?"
  (Needs: comfort, range, cargo space - requires info from multiple cars)
"""

import sys
import os
import sqlite3
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from hmlr.memory.storage import Storage
from hmlr.memory.embeddings.embedding_manager import EmbeddingStorage
from hmlr.memory.gardener.manual_gardener import ManualGardener
from hmlr.memory.synthesis.dossier_governor import DossierGovernor
from hmlr.memory.dossier_storage import DossierEmbeddingStorage
from hmlr.memory.retrieval.dossier_retriever import DossierRetriever
from hmlr.memory.retrieval.context_assembler import ContextAssembler
from hmlr.core.external_api_client import ExternalAPIClient
from hmlr.memory.id_generator import IDGenerator


class E2ETestHarness:
    """Test harness for end-to-end Phase 2 validation"""
    
    def __init__(self, test_db_path: str):
        """Initialize test harness with fresh database"""
        self.test_db_path = test_db_path
        
        # Remove existing test database
        if os.path.exists(test_db_path):
            os.remove(test_db_path)
            print(f"   🗑️  Removed existing test database")
        
        # Initialize components
        print(f"   🔧 Initializing components...")
        self.storage = Storage(test_db_path)
        self.embedding_storage = EmbeddingStorage(test_db_path)
        self.dossier_storage = DossierEmbeddingStorage(test_db_path, model_name="all-MiniLM-L6-v2")
        self.llm_client = ExternalAPIClient()
        self.id_generator = IDGenerator()
        
        self.dossier_governor = DossierGovernor(
            storage=self.storage,
            dossier_storage=self.dossier_storage,
            llm_client=self.llm_client,
            id_generator=self.id_generator
        )
        
        self.gardener = ManualGardener(
            storage=self.storage,
            embedding_storage=self.embedding_storage,
            llm_client=self.llm_client,
            dossier_governor=self.dossier_governor,
            dossier_storage=self.dossier_storage
        )
        
        self.dossier_retriever = DossierRetriever(
            storage=self.storage,
            dossier_storage=self.dossier_storage
        )
        
        self.context_assembler = ContextAssembler(self.storage)
        
        print(f"   ✅ Components initialized")
    
    def simulate_turn(self, turn_num: int, user_message: str, ai_response: str, 
                     expected_facts: List[str]) -> Dict:
        """
        Simulate a single conversation turn atomically.
        
        Flow:
        1. Create turn in metadata_staging (simulating conversation)
        2. Extract facts and store in fact_store
        3. Create bridge block in daily_ledger
        4. Garden the bridge block (tags + dossiers)
        
        Returns:
            Dict with turn_id, block_id, facts, tags, dossiers
        """
        print(f"\n{'=' * 80}")
        print(f"TURN {turn_num}: {user_message[:50]}...")
        print(f"{'=' * 80}")
        
        turn_id = f"turn_{turn_num:03d}"
        block_id = f"block_{turn_num:03d}"
        timestamp = datetime.now().isoformat()
        
        # Step 1: Store turn in metadata_staging (simulates conversation)
        print(f"\n   1️⃣  Storing turn in metadata_staging...")
        conn = sqlite3.connect(self.test_db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO metadata_staging 
            (turn_id, turn_sequence, session_id, day_id, timestamp, 
             user_message, assistant_response)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            turn_id,
            turn_num,
            "test_session_001",
            "day_20251216",
            timestamp,
            user_message,
            ai_response
        ))
        conn.commit()
        print(f"      ✅ Turn stored: {turn_id}")
        
        # Step 2: Extract and store facts (simulates FactScrubber)
        print(f"\n   2️⃣  Extracting facts (simulating FactScrubber)...")
        for i, fact in enumerate(expected_facts):
            cursor.execute("""
                INSERT INTO fact_store 
                (key, value, category, source_block_id, created_at)
                VALUES (?, ?, ?, ?, ?)
            """, (
                f"car_fact_{turn_num}_{i}",
                fact,
                "vehicle_info",
                block_id,
                timestamp
            ))
        conn.commit()
        print(f"      ✅ Extracted {len(expected_facts)} facts")
        for fact in expected_facts:
            print(f"         • {fact}")
        
        # Step 3: Create bridge block in daily_ledger
        print(f"\n   3️⃣  Creating bridge block...")
        block_content = {
            'topic_label': f'Car Discussion {turn_num}',
            'turns': [{
                'turn_id': turn_id,
                'user_message': user_message,
                'ai_response': ai_response
            }]
        }
        
        cursor.execute("""
            INSERT INTO daily_ledger
            (block_id, content_json, created_at, status)
            VALUES (?, ?, ?, ?)
        """, (
            block_id,
            json.dumps(block_content),
            timestamp,
            'COMPLETE'  # Mark as complete so gardener processes it
        ))
        conn.commit()
        conn.close()
        print(f"      ✅ Bridge block created: {block_id}")
        
        # Step 4: Garden the bridge block (Phase 2 dual-output)
        print(f"\n   4️⃣  Gardening bridge block (Phase 2)...")
        result = self.gardener.process_bridge_block(block_id)
        
        # Verify results
        print(f"\n   📊 Gardening Results:")
        print(f"      Status: {result.get('status')}")
        print(f"      Facts Processed: {result.get('facts_processed', 0)}")
        print(f"      Tags Applied: {result.get('tags_applied', 0)}")
        print(f"      Dossiers Created: {result.get('dossiers_created', 0)}")
        
        # Check what tags were created
        metadata = self.storage.get_block_metadata(block_id)
        print(f"\n   🏷️  Tags:")
        for tag in metadata.get('global_tags', [])[:3]:
            print(f"      [Global] {tag}")
        for rule in metadata.get('section_rules', [])[:3]:
            print(f"      [Section] {rule.get('rule', 'unknown')}")
        
        return {
            'turn_id': turn_id,
            'block_id': block_id,
            'facts_count': len(expected_facts),
            'tags_count': result.get('tags_applied', 0),
            'dossiers_count': result.get('dossiers_created', 0)
        }
    
    def retrieve_for_query(self, query: str, top_k: int = 5) -> Dict:
        """
        Retrieve relevant context for a query.
        
        Args:
            query: User query
            top_k: Number of dossiers to retrieve
        
        Returns:
            Dict with retrieved dossiers and formatted context
        """
        print(f"\n{'=' * 80}")
        print(f"RETRIEVAL: {query}")
        print(f"{'=' * 80}")
        
        # Retrieve relevant dossiers
        print(f"\n   🔍 Retrieving relevant dossiers...")
        dossiers = self.dossier_retriever.retrieve_relevant_dossiers(
            query=query,
            top_k=top_k
        )
        
        print(f"   ✅ Retrieved {len(dossiers)} dossiers")
        for dos in dossiers:
            print(f"      • {dos.get('title', 'Untitled')}")
            print(f"        Score: {dos.get('score', 0):.3f}")
            print(f"        Facts: {len(dos.get('facts', []))}")
        
        # Format context using ContextAssembler
        print(f"\n   📝 Formatting context...")
        formatted_context = self.context_assembler.hydrate_dossiers_with_facts(dossiers)
        
        print(f"\n   📄 Formatted Context Preview:")
        print("-" * 80)
        print(formatted_context[:500] + "..." if len(formatted_context) > 500 else formatted_context)
        print("-" * 80)
        
        return {
            'dossiers': dossiers,
            'formatted_context': formatted_context,
            'dossier_count': len(dossiers)
        }
    
    def inspect_database(self):
        """Inspect final database state"""
        print(f"\n{'=' * 80}")
        print(f"DATABASE INSPECTION")
        print(f"{'=' * 80}")
        
        conn = sqlite3.connect(self.test_db_path)
        cursor = conn.cursor()
        
        # Count facts
        cursor.execute("SELECT COUNT(*) FROM fact_store")
        fact_count = cursor.fetchone()[0]
        print(f"\n   📋 fact_store: {fact_count} facts")
        
        # Count dossiers
        cursor.execute("SELECT COUNT(*) FROM dossiers")
        dossier_count = cursor.fetchone()[0]
        print(f"   🗂️  dossiers: {dossier_count} dossiers")
        
        # List dossiers
        cursor.execute("SELECT dossier_id, title, summary FROM dossiers")
        dossiers = cursor.fetchall()
        for dos_id, title, summary in dossiers:
            print(f"\n      • {title} ({dos_id})")
            print(f"        {summary[:100]}...")
            
            # Count facts in dossier
            cursor.execute("SELECT COUNT(*) FROM dossier_facts WHERE dossier_id = ?", (dos_id,))
            fact_count = cursor.fetchone()[0]
            print(f"        Facts: {fact_count}")
        
        # Count block metadata
        cursor.execute("SELECT COUNT(*) FROM block_metadata")
        metadata_count = cursor.fetchone()[0]
        print(f"\n   🏷️  block_metadata: {metadata_count} blocks with tags")
        
        # Count remaining blocks (should be 0 after gardening)
        cursor.execute("SELECT COUNT(*) FROM daily_ledger")
        ledger_count = cursor.fetchone()[0]
        print(f"   🌉 daily_ledger blocks: {ledger_count} (should be 0 after gardening)")
        
        conn.close()


def run_e2e_test():
    """Run complete end-to-end test"""
    print("\n" + "=" * 80)
    print("🧪 END-TO-END PHASE 2 TEST: Multi-Car Scenario")
    print("=" * 80)
    
    # Initialize test harness
    test_db_path = "test_e2e_phase2.db"
    harness = E2ETestHarness(test_db_path)
    
    # Define 5 turns about different cars
    turns = [
        {
            'user_message': "Tell me about my Tesla Model 3",
            'ai_response': "Your Tesla Model 3 is an electric sedan with autopilot capabilities. "
                          "It has a range of about 300 miles per charge and seats 5 people comfortably. "
                          "You use it primarily for your daily commute.",
            'expected_facts': [
                "User owns a Tesla Model 3",
                "Tesla Model 3 is electric with autopilot",
                "Tesla Model 3 has 300 mile range",
                "Tesla Model 3 seats 5 people",
                "User uses Tesla Model 3 for daily commute"
            ]
        },
        {
            'user_message': "What about my Ford F-150 truck?",
            'ai_response': "Your Ford F-150 is a full-size pickup truck with a powerful V8 engine. "
                          "It has excellent towing capacity (up to 13,000 lbs) and a large cargo bed. "
                          "You use it for hauling equipment and home improvement projects.",
            'expected_facts': [
                "User owns a Ford F-150 truck",
                "Ford F-150 has V8 engine and towing capacity",
                "Ford F-150 can tow up to 13,000 lbs",
                "Ford F-150 has large cargo bed",
                "User uses F-150 for hauling and home projects"
            ]
        },
        {
            'user_message': "Remind me about my Honda Civic",
            'ai_response': "Your Honda Civic is a compact sedan that's very fuel-efficient, getting about 40 MPG. "
                          "It's reliable and low-maintenance, making it perfect for your daily commute. "
                          "The backseat is a bit cramped for long trips though.",
            'expected_facts': [
                "User owns a Honda Civic",
                "Honda Civic is fuel-efficient at 40 MPG",
                "Honda Civic is reliable and low-maintenance",
                "Honda Civic used for daily commute",
                "Honda Civic backseat is cramped for long trips"
            ]
        },
        {
            'user_message': "What do you know about my Porsche 911?",
            'ai_response': "Your Porsche 911 is a high-performance sports car with a turbocharged engine. "
                          "It's extremely fun to drive on winding roads and tracks. "
                          "You keep it in the garage most of the time and only drive it on weekends.",
            'expected_facts': [
                "User owns a Porsche 911",
                "Porsche 911 is a sports car with turbocharged engine",
                "Porsche 911 is fun to drive on winding roads",
                "User drives Porsche 911 only on weekends",
                "Porsche 911 kept in garage most of the time"
            ]
        },
        {
            'user_message': "Tell me about my Toyota Sienna minivan",
            'ai_response': "Your Toyota Sienna is a family minivan with three rows of seating for 8 passengers. "
                          "It has plenty of cargo space, especially with the seats folded down. "
                          "The ride is very comfortable for long trips, and it's great for family vacations.",
            'expected_facts': [
                "User owns a Toyota Sienna minivan",
                "Toyota Sienna seats 8 passengers with three rows",
                "Toyota Sienna has plenty of cargo space",
                "Toyota Sienna is comfortable for long trips",
                "Toyota Sienna is great for family vacations"
            ]
        }
    ]
    
    # Process each turn atomically
    turn_results = []
    for i, turn_data in enumerate(turns, start=1):
        result = harness.simulate_turn(
            turn_num=i,
            user_message=turn_data['user_message'],
            ai_response=turn_data['ai_response'],
            expected_facts=turn_data['expected_facts']
        )
        turn_results.append(result)
    
    # Summary after all turns
    print(f"\n{'=' * 80}")
    print(f"TURNS SUMMARY")
    print(f"{'=' * 80}")
    print(f"\nProcessed {len(turn_results)} turns:")
    total_facts = sum(r['facts_count'] for r in turn_results)
    total_tags = sum(r['tags_count'] for r in turn_results)
    total_dossiers = sum(r['dossiers_count'] for r in turn_results)
    
    print(f"   Total Facts: {total_facts}")
    print(f"   Total Tags: {total_tags}")
    print(f"   Total Dossiers: {total_dossiers}")
    
    # Inspect database
    harness.inspect_database()
    
    # Turn 6: Multi-hop query requiring info from multiple cars
    print(f"\n\n{'=' * 80}")
    print(f"TURN 6: Multi-Hop Query")
    print(f"{'=' * 80}")
    
    query = "Which of my cars would be best for a road trip with my family?"
    
    print(f"\n   ❓ Query: {query}")
    print(f"\n   🎯 Expected: Should retrieve info about:")
    print(f"      • Toyota Sienna (comfortable, seats 8, family vacations)")
    print(f"      • Tesla Model 3 (300 mile range, seats 5)")
    print(f"      • Honda Civic (fuel-efficient but cramped backseat)")
    print(f"      • Maybe Ford F-150 (cargo space but not comfortable)")
    
    retrieval_result = harness.retrieve_for_query(query, top_k=5)
    
    # Analyze retrieval quality
    print(f"\n   📊 Retrieval Analysis:")
    retrieved_titles = [d.get('title', '') for d in retrieval_result['dossiers']]
    print(f"      Retrieved Dossiers: {retrieved_titles}")
    
    # Check if key cars were retrieved
    has_sienna = any('sienna' in title.lower() or 'minivan' in title.lower() 
                     for title in retrieved_titles)
    has_tesla = any('tesla' in title.lower() for title in retrieved_titles)
    has_civic = any('civic' in title.lower() for title in retrieved_titles)
    
    print(f"\n   ✅ Quality Check:")
    print(f"      Sienna (best choice): {'✅ Found' if has_sienna else '❌ Missing'}")
    print(f"      Tesla (good range): {'✅ Found' if has_tesla else '❌ Missing'}")
    print(f"      Civic (cramped): {'✅ Found' if has_civic else '❌ Missing'}")
    
    # Final summary
    print(f"\n{'=' * 80}")
    print(f"✅ END-TO-END TEST COMPLETE")
    print(f"{'=' * 80}")
    
    print(f"\nTest Database: {test_db_path}")
    print(f"   • {total_facts} facts extracted")
    print(f"   • {total_dossiers} dossiers created")
    print(f"   • {total_tags} tags applied")
    print(f"   • {retrieval_result['dossier_count']} dossiers retrieved for multi-hop query")
    
    if has_sienna and (has_tesla or has_civic):
        print(f"\n   🎉 SUCCESS: Multi-hop retrieval working!")
        print(f"   Retrieved relevant information from multiple turns.")
    else:
        print(f"\n   ⚠️  PARTIAL: Some relevant dossiers not retrieved.")
        print(f"   May need to tune retrieval parameters or add more facts.")


if __name__ == "__main__":
    run_e2e_test()
