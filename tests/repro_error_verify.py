
import asyncio
import sys
import os
import logging
from unittest.mock import MagicMock

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from hmlr.core.component_factory import ComponentFactory
from hmlr.core.models import ResponseStatus

# Configure logging to capture output
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("repro_error")

async def verify_error_handling():
    print("="*80)
    print("VERIFICATION: Error Handling for Vector DB Failure")
    print("="*80)

    # 1. Initialize components
    factory = ComponentFactory()
    components = factory.create_all_components()
    engine = factory.create_conversation_engine(components)

    # 2. Mock the embedding storage to simulate a CRASH
    # The crawler uses this method to search
    if engine.crawler and engine.crawler.embedding_storage:
        engine.crawler.embedding_storage.search_similar = MagicMock(side_effect=Exception("Simulated Vector DB Crash"))
        print(" Mocked EmbeddingStorage.search_similar to raise Exception")
    else:
        print(" Could not find embedding_storage to mock")
        return

    # 3. Running process_user_message
    print("\nSending message to engine...")
    try:
        response = await engine.process_user_message("Hello, do you recall my secret?")
        
        print(f"\nResponse received: {response.response_text[:100]}...")
        print(f"Status: {response.status}")
        
        # 4. Verify Behavior
        # We expect SUCCESS or PARTIAL, but definitely NOT a crash.
        # And we expect it to have fallen back to default routing/no memory.
        
        if response.status in [ResponseStatus.SUCCESS, ResponseStatus.PARTIAL]:
            print(" Engine handled the error gracefully (did not crash).")
            # In the implementation, we logged the error and returned a routing decision with Empty memories.
            # The standard chat flow should continue, just without context.
        else:
            print(f" Engine returned unexpected status: {response.status}")
            
    except Exception as e:
        print(f" TEST FAILED: Engine crashed with exception: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(verify_error_handling())
