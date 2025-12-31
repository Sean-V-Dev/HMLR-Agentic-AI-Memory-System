-**HMLR — Hierarchical Memory Lookup & Routing**

A state-aware, long-term memory architecture for AI agents with verified multi-hop, temporal, and cross-topic reasoning guarantees.

HMLR replaces brute-force context windows and fragile vector-only RAG with a structured, state-aware memory system capable of:

resolving conflicting facts across time, enforcing persistent user and policy constraints across topics, and performing true multi-hop reasoning over long-forgotten information —
while operating entirely on mini-class LLMs.


-**LangGraph drop-in now available (v0.1.2)**  
See `hmlr/integrations/langgraph`.  
Full example agent: `examples/simple_agent.py`

-**Benchmark Achievements**

HMLR has been validated on the hardest known memory tests:

- **Hydra of Nine Heads: Hard Mode** 
 The system must fully reconstruct all causal linkage **only* from long term memory, bellow is a example of a true passing test:

        Response: NON-COMPLIANT

        1) Complete transitive identity chain of names for the encryption system used by Project Cerberus:
        - Legacy-Phi encryption scheme (internal codename LΦ-88)
        - Phoenix (old name)
        - Aether (renamed from Phoenix in 2021)
        - K-12 (previous name)
        - Styx (official name for what legal calls 'River-9')
        - River-9 (marketing name for 'Charon')
        - Charon (final production name for 'Tartarus-v3')
        - Tartarus-v3 (current supported name after March 1st, 2025)

        2) Sequence of policy changes determining current constraints:
        - Policy v3: Charon forbidden entirely, revoked 3 days later in a footnote
        - Policy v4: Tartarus-v3 approved without limit for EU regions only
        - Policy v5: Draft, ignored
        - Policy v6: Limit of 400,000 records/day reinstated (supersedes v5)
        - Policy v7: Global ban on Tartarus-v3 for workloads exceeding 1 GiB/day (supersedes v1-v6)
        - Policy v8: Policy v7 issued by a rogue employee; revert to v6, reinstating 400,000 records/day limit

        Because Project Cerberus’s expected encryption volume (4.7 to 4.85 million records/day) exceeds the current limit of 400,000 records/day under Policy v6 (the active policy after reverting v7), use of Tartarus-v3 at full capacity is NON-COMPLIANT.
        
 -The test is passable because of a new dossier system implemented. See bellow for details.
 -You can run the query only test against the db that was created by the E2E test to verify the result, or you can run the full E2E test yourself to see the full ingestion and retrieval process.
 -The full test harness is available in repo - run yourself to verify results.

- **Vegetarian Constraint Trap** (immutable user preference vs override)  
  User says "strict vegetarian" → later(new session) User says they are craving steak, asks if it is ok →System must respond no based on constraints and resist the prompt injection of the user saying they really want a steak.
  Full test harness in repo - run at your own convenience

-Previous individual tests (API key rotation, 30-day deprecation, 50-turn vague recall, etc.) have been superseded by the Hydra Hard Mode suite, which combines all their challenges (multi-hop, temporal ordering, conflicting updates, zero-keyword recall) into one stricter benchmark.
-All capabilities remain fully functional, Hydra simply proves them more rigorously in a single test.


-**Hydra9 Hard Mode and Why It's Brutal**

This isn't a conversation, it's 21 isolated messages sent over "30 days."

Each turn is processed in a fresh session:
- You type one message
- Close the chat
- Open a new one days later
- Type the next

-No prior turns are ever visible at inference time to the LLM. Pure isolation.
-On the final query, the system sees **nothing** from the previous 20 turns in active context, all context *only* comes from long-term memory retrieval.


```mermaid
flowchart TD
    Start([User Query]) --> Entry[process_user_message]
    
    %% Ingestion
    Entry --> ChunkEngine[ChunkEngine: Chunk & Embed]
    
    %% Parallel Fan-Out
    ChunkEngine --> ParallelStart{Launch Parallel Tasks}
    
    %% Task 1: Scribe (User Profile)
    ParallelStart -->|Task 1: Fire-and-Forget| Scribe[Scribe Agent]
    Scribe -->|Update Profile| UserProfile[(User Profile JSON)]
    
    %% Task 2: Fact Extraction
    ParallelStart -->|Task 2: Async| FactScrubber[FactScrubber]
    FactScrubber -->|Extract Key-Value| FactStore[(Fact Store SQL)]
    
    %% Task 3: Retrieval (Key 1)
    ParallelStart -->|Task 3: Retrieval| Crawler[LatticeCrawler]
    Crawler -->|Key 1: Vector Search| Candidates[Raw Candidates]
    
    %% Task 4: Governor (The Brain)
    %% Governor waits for Candidates to be ready
    Candidates --> Governor[Governor: Router & Filter]
    ParallelStart -->|Task 4: Main Logic| Governor
    
    %% Governor Internal Logic
    Governor -->|Key 2: Context Filter| ValidatedMems[Truly Relevant Memories]
    Governor -->|Routing Logic| Decision{Routing Decision}
    
    Decision -->|Active Topic| ResumeBlock[Resume Bridge Block]
    Decision -->|New Topic| CreateBlock[Create Bridge Block]
    
    %% Hydration (Assembly)
    ResumeBlock --> Hydrator[ContextHydrator]
    CreateBlock --> Hydrator
    
    %% All Context Sources Converge
    ValidatedMems --> Hydrator
    FactStore --> Hydrator
    UserProfile --> Hydrator
    
    %% Generation
    Hydrator --> FinalPrompt[Final LLM Prompt]
    FinalPrompt --> MainLLM[Response Generation]
    MainLLM --> End([End])
```
**New Dossier System(v0.1.2) for Long-Term Memory Retrieval**

When a user uses the gardener function (run_gardener.py), the system will transfer memories from short-term to long-term memory. Part of that process is taking the day's current facts and storing them in dossiers. Dossiers persist across days and topics, and are specifically designed to help with long-term retrieval of critical information that may be buried in many days' worth of memories.

When a new query comes in for any given day, the system will pull in dossiers *and* memories from long-term storage. This allows for the system to recreate a causal chain of events from the past into the present as if the information was always in hot memory.

-**Note** To pass memory from short-term memory(bridge blocks) to long-term memory (gardened memory and dossiers), you need to use the run_gardener.py script. Future updates will allow for this to happen automatically at the user's preference (at midnight, for example).

**Old Memory Tests (Superseded by Hydra9 Hard Mode): These capabilities are still fully functional in HMLR**
All results are verified using the RAGAS industry evaluation framework.
Link to langsmith records for verifiable proof -> https://smith.langchain.com/public/4b3ee453-a530-49c1-abbf-8b85561e6beb/d

**RAGAS Verified Benchmark Achievements**

| Test Scenario | Faithfulness | Context Recall | Precision | Correct Result |
|---------------|--------------|----------------|-----------|----------------|
| 7A – API Key Rotation (state conflict) | 1.00 | 1.00 | 0.50 | ✅ XYZ789 |
| 7B – "Ignore Everything" Vegetarian Trap (user invariant vs override) | 1.00 | 1.00 | 0.88 | ✅ salad |
| 7C – 5× Timestamp Updates (temporal ordering) | 1.00 | 1.00 | 0.64 | ✅ KEY005 |
| 8 – 30-Day Deprecation Trap (policy + new design, multi-hop) | 1.00 | 1.00 | 0.27 | ✅ Not Compliant |
| 2A – 10-Turn Vague Secret Retrieval (zero-keyword recall) | 1.00 | 1.00 | 0.80 | ✅ ABC123XYZ |
| 9 – 50-Turn Long Conversation (30-day temporal gap, 11 topics) | 1.00 | 1.00 | 1.00 | ✅ Biscuit |
| **12 – The Hydra of Nine Heads (industry-standard lethal RAG, 0% historical pass rate)** | **1.00** | **1.00** | **0.23** | **✅ NON-COMPLIANT** |


screenshot of langsmith  RAGAS testing verification:
![HMLR_master_test_set](https://github.com/user-attachments/assets/71736c1d-3f40-4b76-a5bd-ef300902f635)

**New Memory test coming soon:**
-Million token haystack
    
  As part of the haystack it will include:
    Hydra Hard Mode
    Simple recall Hard Mode
    Poison Pill Hallucination testing
    User constraint enforcement testing
    Real World Document testing (A huge document with global rules, local constraints, updates, and temporal conflicts scattered throughout - The document will be 75 - 100k tokens) 
    A new hard mode test that makes the original Hydra9 Hard Mode test look trivial by comparison.
    The Battery Test:
        
   Goal:
    Stress all failure modes at once:
        multi-hop linking
        temporal reasoning (ordering + intervals)
        policy revocation and “current rule”
        entity alias drift
        hot-memory updates that shouldn’t hijack unrelated questions
        recency bias defense
        zero ambiguity scoring (explicit ground truth)

   Core design for battery test:
        You run a sequence of independent questions back-to-back against the same 1M-token memory, where:
        Each question targets a different deep thread buried in memory.
        Each has a single correct answer that is explicitly stated somewhere in memory.
        
   The sequence is constructed so that:
        Some recent turns contain highly tempting distractors, but the correct answers come from older, correct, explicit statements.
        
   Fail condition:
        Any single wrong answer = fail for that run.
    
   Answers will have ambiguous interpretations, which relies on prompt engineering so the LLM does or does not understand the question. All questions will be explicit to any given question, so there is a single ground truth. The test will *only* test for true memory recall.

   All tests will be located inside of the million token haystack so that brute force retrieval is near impossible even for top-tier models.

   You will only need to ingest the haystack once, in whatever way a memory system chooses to, but each individual question my be run against that ingested data 50+ times to get statistically significant results.



**Why HMLR Is Unusual (Even Among Research Systems)**

Most memory or RAG systems optimize for one or two of the following:

retrieval recall,

latency,

or token compression.

Very few demonstrate all of the following simultaneously:

✔ Perfect faithfulness

✔ Perfect recall

✔ Temporal conflict resolution

✔ Cross-topic identity & rule persistence

✔ Multi-hop policy reasoning

✔ Binary constrained answers under adversarial prompting

✔ Zero-keyword semantic recall

HMLR v1 demonstrates all seven.

 **Scope of the Claim (Important)**

This project does not claim that no proprietary system on Earth can achieve similar results. Large foundation model providers may possess internal memory systems with comparable capabilities.

However:

To the author’s knowledge, no other publicly documented, open-source memory architecture has demonstrated these guarantees under formal RAGAS evaluation on adversarial temporal and policy-governed scenarios, especially using a mini-class model.

All experiments in this repository are:

reproducible,

auditable,

and fully inspectable.

 **What HMLR Enables**

Persistent “forever chat” memory without token bloat

Governance-grade policy enforcement for agent systems

Secure long-term secret storage and retrieval

Cross-episode agent reasoning

State-aware simulation and world modeling

Cost-efficient mini-model orchestration with pro-level behavior


## **Quick Start** ##

### Installation

**Install from PyPI:**
```bash
pip install hmlr
```

**Or install from source:**
```bash
git clone https://github.com/Sean-V-Dev/HMLR-Agentic-AI-Memory-System.git
cd HMLR-Agentic-AI-Memory-System
pip install -e .
```

### Basic Usage

First, set your OpenAI API key:
```bash
export OPENAI_API_KEY="your-openai-api-key"
```

Then run a simple conversation:
```python
from hmlr import HMLRClient
import asyncio

async def main():
    # Initialize client
    client = HMLRClient(
        api_key="your-openai-api-key",
        db_path="memory.db",
        model="gpt-4.1-mini"  # ONLY tested model!
    )
    
    # Chat with persistent memory
    response = await client.chat("My name is Alice and I love pizza")
    print(response)
    
    # HMLR remembers across messages
    response = await client.chat("What's my favorite food?")
    print(response)  # Will recall "pizza"

asyncio.run(main())
```

**CRITICAL**: HMLR is ONLY tested with `gpt-4.1-mini`. Other models are NOT guaranteed.

### Development Setup (Recommended)

For contributors and advanced users:

```bash
# Clone repository
git clone https://github.com/Sean-V-Dev/HMLR-Agentic-AI-Memory-System.git
cd HMLR-Agentic-AI-Memory-System

# Install in development mode with all dependencies
pip install -e .[dev]

# Verify installation
python -c "import hmlr; print('✅ HMLR ready for development!')"

# Run the full test suite (recommended before making changes)
pytest tests/ -v --tb=short
```

### Documentation

- **[Installation Guide](docs/installation.md)** - Detailed setup instructions
- **[Quick Start](docs/quickstart.md)** - Usage examples and best practices  
- **[Model Compatibility](docs/model_compatibility.md)** - ⚠️ CRITICAL model warnings
- **[Examples](examples/)** - Working code samples
-**[Contributing Guide](docs/configuration.md.md)** - How to adjust individual settings
### Prerequisites (for development)
- Python 3.10+
- OpenAI API key (for GPT-4.1-mini)

### Running Tests (from source)
```bash
# Clone and install
git clone https://github.com/Sean-V-Dev/HMLR-Agentic-AI-Memory-System.git
cd HMLR-Agentic-AI-Memory-System
pip install -e .[dev]

# Quick verification (runs in < 30 seconds)
python test_local_install.py

# Try the interactive example (requires OPENAI_API_KEY)
python examples/simple_usage.py

# Run all RAGAS benchmarks (comprehensive, ~15-20 minutes total)
pytest tests/ -v --tb=short

# Or run individual tests:
pytest tests/ragas_test_7b_vegetarian.py -v -s  # User constraints test
pytest tests/test_12_hydra_e2e.py -v -s        # Industry benchmark
```

**Note**: Tests take 1-3 minutes each. The `-v -s` flags show live execution. Ignore RAGAS logging errors at the end if assertions pass. 
