"""
Test FactScrubber on frankenstein.txt (17k tokens).

Verifies that large documents automatically chunk and extract facts
from all chunks with deduplication.
"""

import asyncio
import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

# Enable INFO logging to see chunking messages
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')

from hmlr.memory.fact_scrubber import FactScrubber
from hmlr.memory.chunking.chunk_engine import ChunkEngine
from hmlr.core.external_api_client import ExternalAPIClient
from hmlr.memory.storage import Storage


async def test_frankenstein_extraction():
    """Test FactScrubber with large document (frankenstein.txt)."""
    print("=" * 80)
    print("FRANKENSTEIN.TXT FACT EXTRACTION TEST (17k tokens)")
    print("=" * 80)
    print()
    
    # Load frankenstein.txt
    frank_path = Path(__file__).parent.parent / "frankenstein.txt"
    
    if not frank_path.exists():
        print(f"[FAIL] File not found: {frank_path}")
        return
    
    text = frank_path.read_text(encoding='utf-8')
    print(f"Loaded frankenstein.txt:")
    print(f"  Characters: {len(text):,}")
    print(f"  Estimated tokens: {len(text) // 4:,}")
    print()
    
    # Initialize components
    print("Initializing components...")
    api_client = ExternalAPIClient()
    chunk_engine = ChunkEngine()
    storage = Storage(db_path=":memory:")
    fact_scrubber = FactScrubber(
        storage=storage,
        api_client=api_client
    )
    print("[OK] Components initialized")
    print()
    
    # Create chunks (for linking facts to evidence)
    print("Creating chunks...")
    chunks = chunk_engine.chunk_turn(
        text=text,
        turn_id="test_frankenstein",
        span_id=None
    )
    print(f"[OK] Created {len(chunks)} chunks")
    print()
    
    # Extract facts (will auto-chunk the text)
    print("=" * 80)
    print("EXTRACTING FACTS (will auto-chunk into ~2 chunks)...")
    print("=" * 80)
    print()
    
    try:
        facts = await fact_scrubber.extract_and_save(
            turn_id="test_frankenstein",
            message_text=text,
            chunks=chunks,
            span_id=None,
            block_id=None
        )
        
        print()
        print("=" * 80)
        print("EXTRACTION RESULTS")
        print("=" * 80)
        print()
        
        if facts:
            print(f"[OK] Extracted {len(facts)} unique facts from large document")
            print()
            
            # Show first 10 facts
            print("First 10 facts:")
            for i, fact in enumerate(facts[:10], 1):
                print(f"{i}. {fact.key}: {fact.value[:80]}..." if len(fact.value) > 80 else f"{i}. {fact.key}: {fact.value}")
            
            if len(facts) > 10:
                print(f"... and {len(facts) - 10} more facts")
        else:
            print("[FAIL] NO FACTS EXTRACTED")
    
    except Exception as e:
        print()
        print("=" * 80)
        print("[FAIL] EXTRACTION FAILED")
        print("=" * 80)
        print()
        print(f"Error: {e}")
        import traceback
        print()
        print("Full traceback:")
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_frankenstein_extraction())
