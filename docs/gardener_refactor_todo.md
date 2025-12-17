# Manual Gardener Refactor - Phase 2 TODO List

**Date:** December 15, 2025 (Updated: December 16, 2025)  
**Current File:** `hmlr/memory/gardener/manual_gardener.py`  
**Status:** Partially complete - needs major refactoring

---

## ✅ EMBEDDING MODEL: Unified Across All Systems

**Decision (Dec 16, 2025):** Use `Snowflake/snowflake-arctic-embed-l` (1024D) everywhere.

**Rationale:**
- Query embedding can be shared across memory and dossier retrieval
- Fast enough for production (~60ms per query)
- Excellent accuracy (0.75+ cosine similarity)
- No dimension mismatch issues

**Updated Files:**
- `hmlr/memory/embeddings/embedding_manager.py` - Memory/topics (384D → 1024D)
- `hmlr/memory/dossier_storage.py` - Dossier facts (default: 1024D)

**Performance Impact:**
- Single query embedding reused for all retrieval systems
- Write-side (gardening): Can be slow, user not waiting
- Read-side (queries): 55-68ms per query (acceptable)

---

## 🔮 FUTURE ENHANCEMENT: Long Query Chunking (NOT FOR PHASE 2)

**Problem Identified:** 
Long queries (like pasted emails or lengthy questions) can overwhelm the embedding model's context window or cause accuracy loss due to excessive tokens diluting the semantic signal.

**Example:**
- Hydra test email with 9 different requests
- One critical detail buried in a parenthetical
- Entire email as single embedding: parenthetical gets lost in noise OR cut off

**Solution Architecture:**
Treat exceptionally long queries like memory storage - chunk and embed separately.

```
Long Query (>512 tokens):
├─ Detect: Query length exceeds threshold
├─ Chunk: Split into paragraphs/sentences (same as memory storage)
├─ Embed: Each chunk gets its own embedding
├─ Search: Query with ALL chunk embeddings in parallel
└─ Aggregate: Collect results from all chunks (Multi-Vector Voting)
```

**Performance Strategy:**
- **Cloud:** Fire parallel serverless compute (Lambda, Cloud Functions)
  - Wait for slowest chunk to return (~same time as single query)
- **Local:** Parallelize with ThreadPoolExecutor or asyncio
  - Modern CPUs can handle multiple embedding tasks concurrently
  - Model inference is often I/O bound, not CPU bound
  - Parallelism factor: 4-8 chunks simultaneously on typical hardware

**Implementation Notes:**
- Reuse existing chunking logic from memory storage (paragraph/sentence level)
- Snowflake-l model: 512 token max input → chunk at ~400 tokens for safety
- Aggregate results: Deduplicate by dossier_id, rank by total hits
- Only applies to queries >512 tokens (rare in natural usage)
- Users pasting large content pay the time cost (acceptable)

**Priority:** Post-Phase 2 (once core dossier system is stable)

**Related Files:**
- `hmlr/memory/chunking/` - existing chunking logic
- `hmlr/memory/retrieval/dossier_retriever.py` - add chunk-aware query handling

---

## What Currently EXISTS (But Shouldn't)

### ❌ Code to DELETE:

1. **HierarchicalChunker class** (lines 25-129)
   - Entire class creates hierarchical chunks
   - This work already happened during conversation
   - Completely obsolete with dossier system

2. **Chunking code in `process_bridge_block()`** (lines ~195-210)
   ```python
   # 2. Chunk all turns hierarchically
   print(f"\n   🔪 Chunking turns...")
   all_chunks = []
   for turn in turns:
       chunks = self.chunker.chunk_turn(turn_id, user_msg, ai_resp)
       all_chunks.extend(chunks)
   ```

3. **Turn summary generation** (lines ~212-222)
   ```python
   # 3. Generate summaries for large turns
   print(f"\n   📝 Generating turn summaries...")
   for chunk in all_chunks:
       if chunk.chunk_type == 'turn' and chunk.text == "[SUMMARY NEEDED]":
           summary = self._generate_turn_summary(full_text, chunk.chunk_id)
   ```

4. **Embedding creation loop** (lines ~230-240)
   ```python
   # 5. Embed all chunks
   print(f"\n   🔍 Creating embeddings...")
   for chunk in all_chunks:
       num_embeddings = self.embedding_storage.save_turn_embeddings(...)
   ```

5. **gardened_memory storage** (lines ~245-250)
   ```python
   # 6. Store chunks with global tags in long-term memory
   self._store_chunks_with_tags(block_id, all_chunks, existing_facts)
   ```

6. **`_extract_global_tags()` method** (if exists)
   - Was extracting facts from entire conversation
   - Should use facts from `fact_store` instead

7. **`_reconstruct_full_topic()` method** (if exists)
   - Was building full conversation text for LLM
   - No longer needed

8. **`_store_chunks_with_tags()` method** (if exists)
   - Stored in gardened_memory table
   - Table is now obsolete

---

## What Currently EXISTS (And Should STAY)

### ✅ Code to KEEP:

1. **`_group_facts_semantically()` method** (lines ~417-505)
   - Groups related facts by theme
   - Used for dossier routing
   - Keep as-is

2. **`_delete_bridge_block()` method** (if exists)
   - Deletes processed bridge block from daily_ledger
   - Still needed at end of process

3. **`__init__()` parameters**
   - storage, embedding_storage, llm_client
   - dossier_governor, dossier_storage
   - All correct

---

## What NEEDS TO BE ADDED

### 🆕 New Functionality Required:

#### 1. **`_classify_facts_for_tagging()` method**
**Purpose:** Apply the three heuristics to classify facts

**Heuristics:**
- **Environment Test:** "I am using Python 3.9" → `{"type": "global", "tag": "env: python-3.9"}`
- **Constraint Test:** "Never use eval()" → `{"type": "constraint", "tag": "no-eval", "scope": "section"}`
- **Definition Test:** "Call the server Box A" → `{"type": "alias", "tag": "server=Box A", "turn_range": [3, 8]}`

**Output:**
```python
{
    "global_tags": ["env: python-3.9", "os: windows"],
    "section_rules": [
        {"start_turn": 2, "end_turn": 5, "rule": "no-eval"},
        {"start_turn": 6, "end_turn": 10, "rule": "server=Box A"}
    ],
    "dossier_facts": ["User prefers dark mode", "User works remotely"]
}
```

**Implementation:**
```python
async def _classify_facts_for_tagging(self, facts: List[Dict]) -> Dict:
    """
    Use LLM to classify facts by the three heuristics.
    Returns tags vs dossier-facts.
    """
    prompt = f"""Analyze these facts and classify them:

Facts (extracted from user statements):
{json.dumps([{{'text': f.get('value'), 'turn_id': f.get('turn_id')}} for f in facts], indent=2)}

Apply THREE heuristics:

1. ENVIRONMENT TEST: Settings, version, language?
   Examples: "Using Python 3.9", "On Windows"
   → Global tag

2. CONSTRAINT TEST: Forbids/mandates something?
   Examples: "Never use eval()", "Always check permissions"
   → Constraint tag (global or section)

3. DEFINITION TEST: Renames/defines temporarily?
   Examples: "Call the server Box A", "Old API is deprecated"
   → Alias/status tag (section)

Return JSON:
{{
  "global_tags": ["env: python-3.9", ...],
  "section_rules": [{{"start_turn": X, "end_turn": Y, "rule": "..."}}],
  "dossier_facts": ["facts that don't match above"]
}}
"""
    
    response = await self.llm_client.query_external_api(
        prompt=prompt,
        model="gpt-4.1-mini"
    )
    
    # Parse JSON from response
    return json.loads(response)
```

#### 2. **Update `process_bridge_block()` to dual-output flow**

**New Flow:**
```python
async def process_bridge_block(self, block_id: str) -> Dict[str, Any]:
    """
    Dual output:
    1. Sticky Meta Tags → block_metadata table
    2. Fact Packets → DossierGovernor
    """
    
    # 1. Load bridge block
    block_data = self._load_bridge_block(block_id)
    
    # 2. Load existing facts from fact_store (FactScrubber already extracted them)
    existing_facts = self.storage.get_facts_for_block(block_id)
    if not existing_facts:
        print(f"   ⚠️  No facts found for {block_id}")
        return
    
    print(f"   📋 Loaded {len(existing_facts)} facts from fact_store")
    
    # 3. TAGGING PASS: Classify facts using three heuristics
    classification = await self._classify_facts_for_tagging(existing_facts)
    
    # 4. Apply tags to block metadata (NOT to chunks)
    if classification['global_tags'] or classification['section_rules']:
        self.storage.save_block_metadata(
            block_id=block_id,
            global_tags=classification['global_tags'],
            section_rules=classification['section_rules']
        )
        print(f"   🏷️  Applied {len(classification['global_tags'])} global tags")
    
    # 5. DOSSIER PASS: Group remaining facts semantically
    dossier_facts = classification['dossier_facts']
    if dossier_facts and self.dossier_governor:
        fact_groups = await self._group_facts_semantically(dossier_facts)
        
        # 6. Route each group to dossier governor
        dossier_count = 0
        for group in fact_groups:
            fact_packet = {
                'cluster_label': group['label'],
                'facts': group['facts'],
                'source_block_id': block_id,
                'timestamp': group.get('timestamp', datetime.now().isoformat())
            }
            
            try:
                dossier_id = await self.dossier_governor.process_fact_packet(fact_packet)
                if dossier_id:
                    print(f"      ✅ Created/updated dossier: {dossier_id}")
                    dossier_count += 1
            except Exception as e:
                print(f"      ⚠️  Failed to create dossier: {e}")
    
    # 7. Delete processed bridge block
    self._delete_bridge_block(block_id)
    
    print(f"   ✅ Gardening complete: {block_id}")
    
    return {
        "status": "success",
        "block_id": block_id,
        "facts_processed": len(existing_facts),
        "tags_applied": len(classification['global_tags']),
        "dossiers_created": dossier_count
    }
```

#### 3. **Storage methods for block_metadata table**

**Add to `hmlr/memory/storage.py`:**

```python
def save_block_metadata(self, block_id: str, global_tags: List[str], 
                       section_rules: List[Dict]) -> None:
    """
    Save sticky meta tags for a bridge block.
    Tags are stored ONCE per block, not per chunk.
    """
    cursor = self.conn.cursor()
    cursor.execute("""
        INSERT OR REPLACE INTO block_metadata 
        (block_id, global_tags, section_rules, created_at)
        VALUES (?, ?, ?, ?)
    """, (
        block_id,
        json.dumps(global_tags),
        json.dumps(section_rules),
        datetime.now().isoformat()
    ))
    self.conn.commit()

def get_block_metadata(self, block_id: str) -> Dict:
    """
    Retrieve sticky meta tags for a bridge block.
    Used during read-side chunk hydration.
    """
    cursor = self.conn.cursor()
    cursor.execute("""
        SELECT global_tags, section_rules 
        FROM block_metadata 
        WHERE block_id = ?
    """, (block_id,))
    
    row = cursor.fetchone()
    if not row:
        return {'global_tags': [], 'section_rules': []}
    
    return {
        'global_tags': json.loads(row[0]),
        'section_rules': json.loads(row[1])
    }
```

#### 4. **Database schema for block_metadata table**

**Add to `storage.py` in `_initialize_database()`:**

```python
# === BLOCK METADATA TABLE (Phase 2) ===
# Stores sticky meta tags at bridge block level (not per chunk)
cursor.execute("""
    CREATE TABLE IF NOT EXISTS block_metadata (
        block_id TEXT PRIMARY KEY,
        global_tags TEXT,  -- JSON: ["env: python-3.9", "os: windows"]
        section_rules TEXT,  -- JSON: [{"start_turn": 10, "end_turn": 15, "rule": "no-eval"}]
        created_at TEXT NOT NULL,
        FOREIGN KEY (block_id) REFERENCES daily_ledger(block_id) ON DELETE CASCADE
    )
""")
cursor.execute("CREATE INDEX IF NOT EXISTS idx_block_metadata_block ON block_metadata(block_id)")
```

---

## Read-Side: Context Assembler (Separate File)

### 🆕 New File: `hmlr/memory/retrieval/context_assembler.py`

**Purpose:** Hydrate retrieved chunks with their block metadata WITHOUT duplicating tags

**Key Method:**
```python
def hydrate_chunks_with_metadata(self, chunks: List[Dict]) -> str:
    """
    Group chunks by block_id and inject metadata ONCE per block.
    
    Input: 5 chunks from block_55
    Output:
        ### Context Block: block_55
        Active Rules: [env: python-3.9], [os: windows]
        
        Chunk 1: "Run the command"
        Chunk 2: "Check the logs"
        Chunk 3: "Wait for confirmation"
        [DEPRECATED] Chunk 4: "Old API call"
        Chunk 5: "Verify results"
    
    Token Cost: Tags paid ONCE, not 5 times
    """
    
    # Group chunks by block_id
    blocks = {}
    for chunk in chunks:
        block_id = chunk.get('block_id')
        if block_id not in blocks:
            blocks[block_id] = {
                'metadata': self.storage.get_block_metadata(block_id),
                'chunks': []
            }
        blocks[block_id]['chunks'].append(chunk)
    
    # Build context string
    context_parts = []
    for block_id, data in blocks.items():
        # Header with tags (ONCE per block)
        context_parts.append(f"\n### Context Block: {block_id}")
        
        if data['metadata']['global_tags']:
            context_parts.append(f"Active Rules: {', '.join(data['metadata']['global_tags'])}")
        
        # Chunks (NO repeated tags)
        for chunk in data['chunks']:
            # Check if chunk falls in section rule range
            turn_id = chunk.get('turn_id')
            section_tag = self._get_section_tag(turn_id, data['metadata']['section_rules'])
            
            if section_tag:
                context_parts.append(f"  [{section_tag}] {chunk['text']}")
            else:
                context_parts.append(f"  {chunk['text']}")
        
        context_parts.append("")  # Blank line between blocks
    
    return "\n".join(context_parts)

def _get_section_tag(self, turn_id: str, section_rules: List[Dict]) -> str:
    """Check if turn_id falls within a section rule range."""
    # Extract turn number from turn_id (e.g., "turn_20251211_081709" → needs mapping)
    # This depends on how you track turn order within a block
    for rule in section_rules:
        # Implementation depends on your turn_id format
        # May need turn_index stored with chunk
        pass
    return None
```

---

## Testing Checklist

### ✅ Phase 2 Tests to Create:

1. **test_fact_classification.py**
   - Test Environment heuristic
   - Test Constraint heuristic
   - Test Definition heuristic
   - Verify correct classification of mixed facts

2. **test_block_metadata_storage.py**
   - Test save_block_metadata()
   - Test get_block_metadata()
   - Verify JSON serialization

3. **test_gardener_no_chunking.py**
   - Verify no HierarchicalChunker calls
   - Verify no embedding loops
   - Verify no gardened_memory inserts

4. **test_context_assembler.py**
   - Test group-by-block logic
   - Verify tags appear once per block
   - Verify section tags applied correctly

---

## Summary

**DELETE:**
- HierarchicalChunker class
- All chunking code
- Embedding creation loops
- gardened_memory storage
- _extract_global_tags() method
- _reconstruct_full_topic() method

**KEEP:**
- _group_facts_semantically()
- _delete_bridge_block()
- __init__ parameters

**ADD:**
- _classify_facts_for_tagging() method
- Update process_bridge_block() to dual-output
- save_block_metadata() in storage
- get_block_metadata() in storage
- block_metadata table schema
- context_assembler.py file (read-side)

**KEY INSIGHT:**
The gardener is NO LONGER a chunker/embedder. It is a **Fact Classifier and Router**.
- Classify facts → Tags (scope/validity)
- Group facts → Dossiers (narrative)
- Store tags → block_metadata (referenced, not duplicated)
- Delete block → facts live in fact_store, tags in block_metadata, dossiers in dossier system
