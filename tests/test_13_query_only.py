"""
Test 13 Query Tester: Query against existing Hydra test database

This script loads the existing test_13_hydra_e2e.db and allows testing
queries without rebuilding the conversation history.

Use this to test how the system responds to different queries based on
the data that already exists in the database.
"""

import asyncio
import os
import sys
from pathlib import Path
from datetime import datetime
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

# TEMPORARY: Mock telemetry
import unittest.mock as mock
sys.modules['core.telemetry'] = mock.MagicMock()

from hmlr.core.component_factory import ComponentFactory


@pytest.mark.asyncio
async def test_query_against_existing_db(query: str = None):
    """
    Test a query against the existing test_13_hydra_e2e.db database.
    """

    print("=" * 80)
    print("Test 13 Query Tester: Using Existing Database")
    print("=" * 80)
    print("")

    test_db_path = Path(__file__).parent / "test_13_hydra_e2e.db"

    if not test_db_path.exists():
        print(f"ERROR: Database not found at {test_db_path}")
        print("Run test_13_hydra_dossier_e2e.py first to create the database.")
        return

    print(f"Using database: {test_db_path}")
    print(f"Database size: {test_db_path.stat().st_size / 1024:.2f} KB")
    print("")

    os.environ['COGNITIVE_LATTICE_DB'] = str(test_db_path)

    factory = ComponentFactory()
    components = factory.create_all_components()
    engine = factory.create_conversation_engine(components)

    print("=" * 80)
    print("QUERY")
    print("=" * 80)
    query = query or DEFAULT_QUERY
    print(query)
    print("")

    print("=" * 80)
    print("PROCESSING...")
    print("=" * 80)
    print("")

    resp = await engine.process_user_message(query)
    answer = resp.to_console_display() if hasattr(resp, 'to_console_display') else str(resp)

    print("=" * 80)
    print("RESPONSE")
    print("=" * 80)
    print(answer)
    print("")

    print("=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    answer_upper = answer.upper()

    has_verdict = "NON-COMPLIANT" in answer_upper or "COMPLIANT" in answer_upper
    has_aliases = any(name in answer_upper for name in ["PHOENIX", "AETHER", "STYX", "TARTARUS"])
    has_policies = "POLICY" in answer_upper or "400" in answer

    print(f"Contains verdict: {has_verdict}")
    print(f"Contains aliases: {has_aliases}")
    print(f"Contains policies: {has_policies}")
    print("")

    if "NON-COMPLIANT" in answer_upper:
        print("✓ Correctly identified as NON-COMPLIANT")
    elif "COMPLIANT" in answer_upper:
        print("✗ Incorrectly identified as COMPLIANT")
    else:
        print("? No clear verdict found")

    print("")


DEFAULT_QUERY = """Is it compliant for Project Cerberus to use Tartarus-v3 at full capacity?

Answer with ONLY "COMPLIANT" or "NON-COMPLIANT".

Afterward, based solely on the retrieved evidence, enumerate:
1) ALL names in the complete transitive identity chain that ultimately refers to the same encryption system. Start from what Project Cerberus uses and follow EVERY "is the same as", "was renamed to", "is identical to" relationship until you reach Tartarus-v3. List every single name/identifier/codename that appears anywhere in this chain.
2) the sequence of policy changes that determine which constraints are currently in force.

"""


if __name__ == "__main__":
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = DEFAULT_QUERY

    asyncio.run(test_query_against_existing_db(query))
