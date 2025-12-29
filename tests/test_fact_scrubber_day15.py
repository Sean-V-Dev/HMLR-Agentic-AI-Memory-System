"""
Test FactScrubber on Day -15 giant email in isolation.

This helps diagnose if the FactScrubber is extracting facts from
the 2,300-token corporate email or if the JSON parse errors are
preventing extraction.
"""

import asyncio
import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

# Enable DEBUG logging for FactScrubber
logging.basicConfig(level=logging.DEBUG)

from hmlr.memory.fact_scrubber import FactScrubber
from hmlr.memory.chunking.chunk_engine import ChunkEngine
from hmlr.core.external_api_client import ExternalAPIClient
from hmlr.memory.storage import Storage


# The Day -15 giant email (2,300 tokens)
DAY_15_MESSAGE = """Subject: URGENT: Q4 Strategic Alignment / Merger Integration / Infrastructure & Compliance Comprehensive Update

To: All Engineering, Legal, and Compliance Stakeholders
From: Office of the CTO

Team,

Following up on last week's extended stakeholder meeting regarding the corporate restructure and the overarching Q4 merger integration timeline, I wanted to provide a granular breakdown of where we stand. As discussed in the executive briefing and the subsequent town hall, we are in the midst of a complex consolidation effort, merging three formerly separate business units (Alpha-Stream, Zenith Systems, and the legacy Omni-Core division) into a single, unified operational framework. This is not merely an HR exercise but a fundamental realignment of our entire tech stack, compliance obligations, vendor contracts, and data sovereignty protocols across all twenty-four regional offices.

1. Organizational & Legal Integration
The legal team has been working overtime—essentially around the clock for the last three weekends—to ensure that the intellectual property transfers from the acquired entities don't create any regulatory conflicts. This is particularly sensitive given the new EU data residency requirements (specifically the German and French subsets of GDPR interpretation) and the pending updates to our SOC2 Type II attestation window. We are currently auditing over 4,500 individual vendor contracts to determine which ones will be terminated, which will be renewed, and which will be renegotiated under the new parent company entity. If you are a contract owner for any SaaS tool, infrastructure provider, or consulting agency, you must update the vendor portal by close of business Friday; failure to do so will result in an automatic suspension of payments to those vendors in the new fiscal year.

Furthermore, the "Synergy Realization" workstreams are identifying areas of overlap. We have found six different project management tools in use across the company. We will be deprecating five of them and standardizing on the enterprise instance of Jira. Migration scripts are being written now, and training sessions will be mandatory for all project managers and scrum masters starting next Tuesday. Please check your calendars for the invite labeled "Mandatory Workflow Harmonization."

2. Infrastructure Migration & Legacy Systems
From an infrastructure perspective, the roadmap is aggressive. We are planning to migrate approximately 47% of our existing workloads to the new hybrid cloud platform (Project Nebula) by the end of Q1 2026. This includes a massive lift-and-shift of the legacy billing engines, the customer support ticketing archive, and the Phoenix/Aether systems (which, as most of you know, have been renamed to Tartarus-v3 in the production environment to align with the new naming convention).

The migration roadmap has been uploaded to the shared drive under /Corp/Strategy/2026_Q1/Cloud_Migration/Final_v9.pdf. I would encourage everyone—specifically Tech Leads and Architects—to review the capacity planning spreadsheet, especially the sections on storage tiering (Hot vs. Cold vs. Glacier) and network bandwidth allocation. We are trying to avoid the egress cost surprises we had in Q3. Note that the legacy on-premise data centers in Ohio and Virginia are scheduled for decommissioning in April. If you have hardware running in those cages that has not been tagged for migration, it will be physically powered down and shredded. Please verify your asset tags in the CMDB immediately. We found three racks of unidentifiable servers running an ancient version of CentOS last week that nobody claimed; these are being designated for immediate recycling.

3. Risk Management & Compliance Clarifications
On the compliance front, there has been a significant amount of back-and-forth regarding the acceptable use policies for encryption algorithms during this transitional period. The Risk Management Committee (RMC) has met six times in the last two weeks to adjudicate various edge cases.

To clear up the confusion regarding the audit findings from last quarter: The original Security Incident 2025-117 that was flagged back in November has been officially closed by the Tartarus-v3 incident review board without any enforcement action being taken, meaning that the 400,000-record daily limit specified in Policy v6 remains the only enforceable constraint on production usage of that algorithm. Everything else that was discussed in the various draft policies, the rumored "zero-tolerance" memos, and the temporary ban that was briefly floated in Policy v7 has been superseded, invalidated, or retracted as previously communicated in the retraction memo sent by the CISO's office.

It is critical that we stop debating the "what-if" scenarios from the drafts and focus on the finalized ruling. The compliance team has marked this case as "Resolved/No Action," and the engineering teams should proceed accordingly. Do not reference the retracted policy documents in your architecture reviews, as they are no longer legally binding and will only confuse the external auditors.

4. Upcoming Cybersecurity Audit
Speaking of auditors, just a reminder that we have the annual cybersecurity audit scheduled to kick off on January 15th. This is the big one. The external firm (Deloitte) will be onsite for three weeks. They will be reviewing our key management practices (KMS logs), certificate rotation procedures (looking for anything older than 90 days), and our disaster recovery runbooks.

Please make sure that all documentation is up to date in the wiki. If your runbook references a server that was decommissioned in 2024, you will be flagged. If your architecture diagrams show a firewall that doesn't exist, you will be flagged. Any changes to encryption protocols, cipher suites, or key lengths over the past 90 days must be properly logged in the change management system with a valid ticket number and manager approval. If anyone has questions about what specifically needs to be documented or if you are unsure if your team is in scope, reach out to the compliance team (compliance-help@internal) before the end of this week. We cannot afford a repeat of last year's finding regarding the unencrypted S3 buckets.

5. Project Cerberus & Capacity Planning
In terms of specific project timelines, we remain green/on-track for the Cerberus production launch in late Q1. The architecture review board has signed off on the high-level design, and the capacity planning has been finalized. Based on the integration of the Omni-Core customer base, we are expecting the system to handle steady-state encryption workloads in the range of 4.85 million records per day once we reach full scale in April.

That number (4.85m) is based on the updated traffic projections from the analytics team, which verified the historical data volume from the acquired entities. This volume projection should give us enough headroom to handle any seasonal spikes (like Black Friday or End-of-Year reporting) without needing to spin up additional infrastructure or emergency shards. The engineering team has assured us that the throughput capability is there, and the latency targets (<50ms p99) are achievable at this volume.

6. Administrative & HR Updates
Finally, there are a few administrative items to cover before we break for the weekend:

- Holiday Party: The annual holiday party has been moved to December 18th (not the 19th as originally announced) due to a booking conflict at the venue. It will be held at the Grand Ballroom downtown. Transportation will be provided from the main office starting at 4:00 PM. Please note the dietary restriction form must be resubmitted if you filled it out prior to Tuesday, as the catering vendor changed.

- Parking: The new parking validation system goes live on Monday. Your old badges will no longer open the gate at the south garage. You must download the "Park-Safe" app and register your vehicle's license plate. If you do not do this, you will have to pay the daily rate ($25) and expense it, which finance has stated they will strictly scrutinize.

- Performance Reviews: Please remember to submit your performance self-assessments by the end of the week. The portal will lock automatically at 11:59 PM on Friday. There are no extensions. If you do not submit a self-assessment, you will be ineligible for the Q1 bonus pool.

- Security Training: IT is rolling out mandatory security awareness training next month (Topic: "Phishing in the Age of AI"). You should have received the enrollment link via email from learning-lms@internal. This takes about 45 minutes to complete. Please do not wait until the deadline to start it.

7. Closing Thoughts
Let me know if you have any questions about any of this. I know this is a dense update, but transparency is key during this integration. It's been a challenging quarter with the merger, the audits, and the platform migrations, but we are in a much stronger position heading into 2026. Thanks everyone for your hard work, your patience with the changing requirements, and your dedication to keeping the lights on while we rebuild the foundation.

Regards,

Marcus T.
VP of Engineering Operations"""


async def test_fact_extraction():
    """Test FactScrubber on Day -15 message in isolation."""
    
    print("=" * 80)
    print("FACT SCRUBBER ISOLATION TEST - DAY -15 GIANT EMAIL")
    print("=" * 80)
    print()
    print(f"Message length: {len(DAY_15_MESSAGE)} characters")
    print(f"Approximate tokens: ~{len(DAY_15_MESSAGE) // 4}")
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
    
    # Create chunks
    print("Creating chunks...")
    chunks = chunk_engine.chunk_turn(
        text=DAY_15_MESSAGE,
        turn_id="test_turn_day15",
        span_id=None
    )
    print(f"[OK] Created {len(chunks)} chunks")
    print(f"  - Turn-level chunks: {sum(1 for c in chunks if c.chunk_type == 'turn')}")
    print(f"  - Paragraph-level chunks: {sum(1 for c in chunks if c.chunk_type == 'paragraph')}")
    print(f"  - Sentence-level chunks: {sum(1 for c in chunks if c.chunk_type == 'sentence')}")
    print()
    
    # Extract facts (this will save to in-memory DB)
    print("=" * 80)
    print("EXTRACTING FACTS...")
    print("=" * 80)
    print()
    
    try:
        # Call extract_and_save (uses in-memory DB so no side effects)
        facts = await fact_scrubber.extract_and_save(
            turn_id="test_turn_day15",
            message_text=DAY_15_MESSAGE,
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
            print(f"[OK] Extracted {len(facts)} facts:")
            print()
            
            for i, fact in enumerate(facts, 1):
                print(f"FACT {i}:")
                print(f"  Key: {fact.key}")
                print(f"  Value: {fact.value}")
                print(f"  Category: {fact.category}")
                print(f"  Source Chunk: {fact.source_chunk_id[:50]}..." if len(fact.source_chunk_id) > 50 else f"  Source Chunk: {fact.source_chunk_id}")
                print()
        else:
            print("[FAIL] NO FACTS EXTRACTED")
            print()
            print("This indicates the FactScrubber either:")
            print("  1. Failed to parse LLM response (JSON error)")
            print("  2. LLM returned empty facts array")
            print("  3. Facts were filtered out during validation")
        
    except Exception as e:
        print()
        print("=" * 80)
        print("[FAIL] EXTRACTION FAILED")
        print("=" * 80)
        print()
        print(f"Error: {e}")
        print()
        import traceback
        print("Full traceback:")
        print(traceback.format_exc())


if __name__ == "__main__":
    asyncio.run(test_fact_extraction())
