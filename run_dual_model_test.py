"""
Quick script to re-run integration test with new dual-model setup.

Steps:
1. Delete old database (to force re-embedding)
2. Run test_integration_phase2.py to create dossiers
3. Run test_domain_retrieval.py to verify all 5 dossiers retrieved

This will use:
- Write (gardening): snowflake-arctic-embed-l (1024D, accurate)
- Read (queries): snowflake-arctic-embed-xs (384D, fast)
"""

import os
import subprocess
import sys

def main():
    db_path = "test_integration_phase2.db"
    
    print("=" * 100)
    print("RE-RUNNING INTEGRATION TEST WITH DUAL-MODEL SETUP")
    print("=" * 100)
    print("\nWrite model (gardening): snowflake-arctic-embed-l (1024D, accurate)")
    print("Read model (queries):     snowflake-arctic-embed-xs (384D, fast)\n")
    
    # Step 1: Delete old database
    if os.path.exists(db_path):
        print(f"1. Deleting old database: {db_path}")
        os.remove(db_path)
        print("   ✅ Deleted\n")
    else:
        print(f"1. No existing database found\n")
    
    # Step 2: Run integration test (creates dossiers with new embeddings)
    print("2. Running integration test (creates dossiers with new embeddings)")
    print("   This will take a few minutes (downloading models + LLM calls)...\n")
    
    result = subprocess.run(
        [sys.executable, "test_integration_phase2.py"],
        capture_output=False,
        text=True
    )
    
    if result.returncode != 0:
        print("\n❌ Integration test failed!")
        return
    
    print("\n   ✅ Integration test completed\n")
    
    # Step 3: Run domain retrieval test
    print("3. Running domain retrieval test (fact-level with hit-count voting)")
    print("   Testing 5 queries to verify all dossiers retrieved...\n")
    
    result = subprocess.run(
        [sys.executable, "test_domain_retrieval.py"],
        capture_output=False,
        text=True
    )
    
    if result.returncode != 0:
        print("\n❌ Domain retrieval test failed!")
        return
    
    print("\n" + "=" * 100)
    print("COMPLETE: Dual-model integration test finished")
    print("=" * 100)

if __name__ == "__main__":
    main()
