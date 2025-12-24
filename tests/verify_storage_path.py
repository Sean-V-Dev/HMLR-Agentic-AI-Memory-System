import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from hmlr.memory import Storage

def check_storage_path():
    print("--- Storage Path Verification ---")
    
    # 1. Default (no args, no env)
    # Ensure env var is cleared for test
    if 'HMLR_DB_PATH' in os.environ:
        del os.environ['HMLR_DB_PATH']
        
    s = Storage()
    print(f"Default Path: {s.db_path}")
    expected_default = str(Path.home() / ".hmlr" / "cognitive_lattice_memory.db")
    if s.db_path == expected_default:
        print("PASS: Default path is correct (User Home)")
    else:
        print(f"FAIL: Expected {expected_default}, got {s.db_path}")

    # 2. Env Var
    os.environ['HMLR_DB_PATH'] = "./env_test.db"
    s2 = Storage()
    print(f"Env Path: {s2.db_path}")
    if s2.db_path == "./env_test.db":
        print("PASS: Env var path respected")
    else:
        print("FAIL: Env var ignored")

    # Clean up (don't leave env var set in process if it matters, though it's local)

if __name__ == "__main__":
    check_storage_path()
