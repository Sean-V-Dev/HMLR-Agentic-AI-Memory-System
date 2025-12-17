"""
TRUE End-to-End Phase 2 Integration Test

This test uses the ACTUAL system components to watch data flow through:
1. ConversationManager stores turn
2. FactScrubber extracts facts from turn
3. Facts stored in fact_store
4. Bridge blocks created in daily_ledger
5. Gardener processes blocks (Phase 2):
   - Loads facts from fact_store
   - Classifies: tags vs dossier facts
   - Applies tags to block_metadata
   - Groups dossier facts semantically
6. DossierGovernor decides:
   - Create 1 dossier or multiple?
   - Merge with existing dossiers?
7. Retrieval uses tags + dossiers
8. ContextAssembler formats for LLM

Multi-Car Scenario:
- Turn 1: Tesla (electric, 300mi, autopilot)
- Turn 2: F-150 (truck, towing, cargo)
- Turn 3: Civic (fuel-efficient, cramped)
- Turn 4: Porsche (sports, weekend only)
- Turn 5: Sienna (minivan, family trips)
- Query: "Best car for family road trip?"
"""

import sys
import os
import sqlite3
import json
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Dict, List

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from hmlr.memory.storage import Storage
from hmlr.memory.embeddings.embedding_manager import EmbeddingStorage
from hmlr.memory.fact_scrubber import FactScrubber
from hmlr.memory.gardener.manual_gardener import ManualGardener
from hmlr.memory.synthesis.dossier_governor import DossierGovernor
from hmlr.memory.dossier_storage import DossierEmbeddingStorage
from hmlr.memory.retrieval.dossier_retriever import DossierRetriever
from hmlr.memory.retrieval.context_assembler import ContextAssembler
from hmlr.core.external_api_client import ExternalAPIClient
from hmlr.memory.id_generator import IDGenerator


def print_section(title: str):
    """Print a section header"""
    print(f"\n{'=' * 80}")
    print(f"{title}")
    print(f"{'=' * 80}")


def print_step(step_num: int, title: str):
    """Print a step header"""
    print(f"\n   {step_num}️⃣  {title}")
    print(f"   {'-' * 76}")


class IntegrationTestHarness:
    """Test harness that uses real system components"""
    
    def __init__(self, test_db_path: str):
        """Initialize with fresh database and real components"""
        self.test_db_path = test_db_path
        
        # Remove existing test database
        if os.path.exists(test_db_path):
            os.remove(test_db_path)
            print(f"   🗑️  Removed existing test database")
        
        # Initialize ALL real components
        print(f"   🔧 Initializing real system components...")
        self.storage = Storage(test_db_path)
        self.embedding_storage = EmbeddingStorage(test_db_path)
        self.llm_client = ExternalAPIClient()
        self.id_generator = IDGenerator()
        
        # FactScrubber (extracts facts from turns)
        self.fact_scrubber = FactScrubber(
            storage=self.storage,
            api_client=self.llm_client
        )
        
        # Dossier system - using largest model for best accuracy
        self.dossier_storage = DossierEmbeddingStorage(
            db_path=test_db_path,
            model_name="Snowflake/snowflake-arctic-embed-l"  # 1024D, most accurate
        )
        
        self.dossier_governor = DossierGovernor(
            storage=self.storage,
            dossier_storage=self.dossier_storage,
            llm_client=self.llm_client,
            id_generator=self.id_generator
        )
        
        # Gardener (Phase 2 refactored)
        self.gardener = ManualGardener(
            storage=self.storage,
            embedding_storage=self.embedding_storage,
            llm_client=self.llm_client,
            dossier_governor=self.dossier_governor,
            dossier_storage=self.dossier_storage
        )
        
        # Retrieval
        self.dossier_retriever = DossierRetriever(
            storage=self.storage,
            dossier_storage=self.dossier_storage
        )
        
        self.context_assembler = ContextAssembler(self.storage)
        
        print(f"   ✅ All components initialized")
    
    async def process_turn(self, turn_num: int, user_message: str, 
                          ai_response: str, garden_immediately: bool = False) -> Dict:
        """
        Process a single conversation turn through the REAL system.
        
        Flow:
        1. Store turn in metadata_staging
        2. FactScrubber extracts facts (LLM call)
        3. Facts saved to fact_store
        4. Create bridge block
        5. Optionally garden immediately (watch flow in real-time!)
        
        Args:
            turn_num: Turn number
            user_message: User's message
            ai_response: AI's response
            garden_immediately: If True, garden the block right away
        
        Returns:
            Dict with turn stats
        """
        print_section(f"TURN {turn_num}")
        
        turn_id = f"turn_{turn_num:03d}"
        block_id = f"block_{turn_num:03d}"
        span_id = f"span_001"
        timestamp = datetime.now().isoformat()
        
        # Step 1: Store turn in metadata_staging
        print_step(1, "Store turn in metadata_staging")
        
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
        conn.close()
        
        print(f"      ✅ Turn stored: {turn_id}")
        print(f"      User: {user_message[:60]}...")
        print(f"      AI: {ai_response[:60]}...")
        
        # Step 2: FactScrubber extracts facts from USER message ONLY
        print_step(2, "FactScrubber extracts facts (real LLM call)")
        
        # Create dummy chunks (FactScrubber needs them for linking)
        chunks = []
        
        try:
            facts = await self.fact_scrubber.extract_and_save(
                turn_id=turn_id,
                message_text=user_message,  # Only user message, not AI response!
                chunks=chunks,
                span_id=span_id,
                block_id=block_id
            )
            
            print(f"      ✅ Extracted {len(facts)} facts")
            for fact in facts:
                print(f"         [{fact.category}] {fact.key}: {fact.value[:60]}...")
        
        except Exception as e:
            print(f"      ⚠️  Fact extraction failed: {e}")
            facts = []
        
        # Step 3: Create bridge block (always)
        print_step(3, "Create bridge block in daily_ledger")
        
        conn = sqlite3.connect(self.test_db_path)
        cursor = conn.cursor()
        
        block_content = {
            'topic_label': f'Car Discussion Turn {turn_num}',
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
            'COMPLETE'
        ))
        conn.commit()
        conn.close()
        
        print(f"      ✅ Bridge block created: {block_id}")
        
        # Step 4: Garden immediately if requested
        garden_result = None
        if garden_immediately:
            print_step(4, "Garden block immediately (watch the flow!)")
            garden_result = await self.garden_block(block_id)
        
        return {
            'turn_id': turn_id,
            'block_id': block_id,
            'facts_extracted': len(facts),
            'gardened': garden_immediately,
            'garden_result': garden_result
        }
    
    async def garden_block(self, block_id: str) -> Dict:
        """
        Garden a bridge block through Phase 2 system.
        
        Flow:
        1. Gardener loads facts from fact_store
        2. Classifies facts (tags vs dossier facts) - LLM call
        3. Applies tags to block_metadata
        4. Groups dossier facts semantically - LLM call
        5. DossierGovernor routes facts - LLM decides merge/create
        
        Returns:
            Dict with gardening stats
        """
        print_section(f"GARDENING: {block_id}")
        
        print_step(1, "Gardener processes bridge block (Phase 2)")
        print(f"      This will:")
        print(f"      • Load facts from fact_store")
        print(f"      • Classify facts (Environment/Constraint/Definition heuristics)")
        print(f"      • Apply tags to block_metadata")
        print(f"      • Group dossier facts semantically")
        print(f"      • Route to DossierGovernor")
        
        result = await self.gardener.process_bridge_block(block_id)
        
        print_step(2, "Gardening Results")
        print(f"      Status: {result.get('status')}")
        print(f"      Facts Processed: {result.get('facts_processed', 0)}")
        print(f"      Tags Applied: {result.get('tags_applied', 0)}")
        print(f"      Dossiers Created/Updated: {result.get('dossiers_created', 0)}")
        
        # Show what was created
        metadata = self.storage.get_block_metadata(block_id)
        if metadata.get('global_tags') or metadata.get('section_rules'):
            print_step(3, "Tags Applied")
            for tag in metadata.get('global_tags', [])[:5]:
                print(f"         [Global] {tag}")
            for rule in metadata.get('section_rules', [])[:5]:
                print(f"         [Section] {rule.get('rule', 'unknown')}")
        
        return result
    
    def inspect_dossiers(self):
        """Inspect current dossier state"""
        print_section("DOSSIER INSPECTION")
        
        conn = sqlite3.connect(self.test_db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT dossier_id, title, summary FROM dossiers ORDER BY created_at")
        dossiers = cursor.fetchall()
        
        print(f"\n   Total Dossiers: {len(dossiers)}")
        print(f"   {'─' * 76}")
        
        for dos_id, title, summary in dossiers:
            print(f"\n   📁 {title}")
            print(f"      ID: {dos_id}")
            print(f"      Summary: {summary[:100]}...")
            
            # Count facts
            cursor.execute("SELECT COUNT(*) FROM dossier_facts WHERE dossier_id = ?", (dos_id,))
            fact_count = cursor.fetchone()[0]
            print(f"      Facts: {fact_count}")
            
            # Show first few facts
            cursor.execute("SELECT fact_text FROM dossier_facts WHERE dossier_id = ? LIMIT 3", (dos_id,))
            for fact_text, in cursor.fetchall():
                print(f"         • {fact_text[:70]}...")
        
        conn.close()
    
    def retrieve_and_format(self, query: str, top_k: int = 5) -> Dict:
        """Retrieve dossiers and format context"""
        print_section(f"RETRIEVAL: {query}")
        
        print_step(1, "DossierRetriever searches for relevant dossiers")
        
        dossiers = self.dossier_retriever.retrieve_relevant_dossiers(
            query=query,
            top_k=top_k
        )
        
        print(f"      ✅ Retrieved {len(dossiers)} dossiers")
        for dos in dossiers:
            print(f"         • {dos.get('title', 'Untitled')} (score: {dos.get('score', 0):.3f})")
        
        print_step(2, "ContextAssembler formats for LLM")
        
        formatted_context = self.context_assembler.hydrate_dossiers_with_facts(dossiers)
        
        print(f"\n      Context Preview:")
        print(f"      {'-' * 72}")
        preview = formatted_context[:400] if formatted_context else "(empty)"
        print(f"{preview}...")
        print(f"      {'-' * 72}")
        
        return {
            'dossiers': dossiers,
            'formatted_context': formatted_context
        }


async def run_integration_test():
    """Run complete integration test"""
    print_section("🧪 TRUE END-TO-END INTEGRATION TEST")
    print("\nThis test uses REAL system components:")
    print("  • FactScrubber (LLM extraction)")
    print("  • ManualGardener (Phase 2 classification)")
    print("  • DossierGovernor (intelligent routing)")
    print("  • DossierRetriever (vector search)")
    print("  • ContextAssembler (group-by-block)")
    
    # Initialize
    test_db_path = "test_integration_phase2.db"
    harness = IntegrationTestHarness(test_db_path)
    
    # Define conversation turns where USER tells US facts (not asking questions)
    turns = [
        {
            'user': "I just bought a Tesla Model 3! It's an electric sedan with autopilot, seats 5 people, and has a 300 mile range. I use it for my daily commute to work.",
            'ai': "Congratulations on the new Tesla! That's a great choice for commuting. The 300 mile range should handle your daily needs well."
        },
        {
            'user': "I also have a Ford F-150 pickup truck with a V8 engine. It can tow up to 13,000 lbs and has a large cargo bed. I use it for hauling equipment and home projects.",
            'ai': "The F-150 is a workhorse! That towing capacity is impressive. Sounds like you have the right tool for heavy-duty tasks."
        },
        {
            'user': "My Honda Civic is super reliable and gets 40 MPG. It's my daily commuter but the backseat is pretty cramped for long trips.",
            'ai': "Great fuel efficiency on the Civic! Though I can see how the backseat space might be limiting for longer journeys."
        },
        {
            'user': "I have a Porsche 911 sports car with a turbocharged engine. It's incredibly fun on winding roads! I only drive it on weekends and keep it garaged the rest of the time.",
            'ai': "What a fun weekend car! Keeping it garaged is smart to preserve it. Those turbocharged engines are a blast."
        },
        {
            'user': "My Toyota Sienna minivan seats 8 passengers across three rows. It has plenty of cargo space and is very comfortable for long family trips and vacations.",
            'ai': "Perfect for family road trips! That 8-passenger capacity and cargo space make it ideal for vacations with the whole family."
        }
    ]
    
    # Process turns 1-5 (garden immediately after each turn)
    print_section("TURN-BY-TURN PROCESSING")
    print("   Each turn: Extract Facts → Create Block → Garden → Watch DossierGovernor!\n")
    
    for i, turn_data in enumerate(turns, start=1):
        print(f"\n{'='*70}")
        print(f"🚗 TURN {i}: {turn_data['user'][:50]}...")
        print(f"{'='*70}")
        
        result = await harness.process_turn(
            turn_num=i,
            user_message=turn_data['user'],
            ai_response=turn_data['ai'],
            garden_immediately=True  # Garden after EACH turn atomically!
        )
        
        print(f"\n✅ Turn {i} complete:")
        print(f"   - Facts extracted: {result['facts_extracted']}")
        if result['gardened'] and result['garden_result']:
            gr = result['garden_result']
            print(f"   - Dossiers created: {gr.get('dossiers_created', 0)}")
            print(f"   - Tags applied: {gr.get('tags_applied', 0)}")
    
    # Inspect final state
    harness.inspect_dossiers()
    
    # Test retrieval
    query = "Which of my cars would be best for a family road trip?"
    retrieval_result = harness.retrieve_and_format(query, top_k=5)
    
    # Final summary
    print_section("✅ INTEGRATION TEST COMPLETE")
    
    print(f"\nDatabase: {test_db_path}")
    print(f"\n📊 Statistics:")
    
    conn = sqlite3.connect(test_db_path)
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM fact_store")
    fact_count = cursor.fetchone()[0]
    print(f"   • Facts extracted: {fact_count}")
    
    cursor.execute("SELECT COUNT(*) FROM dossiers")
    dossier_count = cursor.fetchone()[0]
    print(f"   • Dossiers created: {dossier_count}")
    
    cursor.execute("SELECT COUNT(*) FROM block_metadata")
    metadata_count = cursor.fetchone()[0]
    print(f"   • Blocks with tags: {metadata_count}")
    
    cursor.execute("SELECT COUNT(*) FROM daily_ledger")
    ledger_count = cursor.fetchone()[0]
    print(f"   • Remaining bridge blocks: {ledger_count} (should be 0)")
    
    conn.close()
    
    print(f"\n🎯 Key Questions Answered:")
    print(f"   1. Did FactScrubber extract facts? {('✅ Yes' if fact_count > 0 else '❌ No')}")
    print(f"   2. Did DossierGovernor create dossiers? {('✅ Yes' if dossier_count > 0 else '❌ No')}")
    print(f"   3. Did gardener apply tags? {('✅ Yes' if metadata_count > 0 else '⚠️  No (all classified as dossier facts)')}")
    print(f"   4. Were bridge blocks deleted? {('✅ Yes' if ledger_count == 0 else '❌ No')}")
    print(f"   5. One dossier or multiple? {dossier_count} dossier(s)")
    
    if dossier_count == 1:
        print(f"      → DossierGovernor grouped all car facts into ONE dossier")
    elif dossier_count == 5:
        print(f"      → DossierGovernor created SEPARATE dossiers per car")
    else:
        print(f"      → DossierGovernor created {dossier_count} dossiers (mixed strategy)")


if __name__ == "__main__":
    asyncio.run(run_integration_test())
