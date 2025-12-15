# Dossier System Implementation Roadmap

**Created:** December 15, 2025  
**Status:** Planning Phase Complete  
**Purpose:** Implement write-side and read-side dossier governors for incremental fact aggregation

---

## Executive Summary

The dossier system addresses the current limitation where facts extracted from bridge blocks are stored as isolated chunks with duplicated metadata. The new architecture creates a meta-layer at the turn level that aggregates semantically related facts into dossiers with causal chains.

**Key Components:**
- **Write-Side:** DossierGovernor routes facts to existing dossiers or creates new ones
- **Read-Side:** DossierRetriever performs parallel search alongside topic/memory retrieval
- **Storage:** Normalized schema with separate tables for dossier headers, facts, and provenance

**Value Proposition:**
- Facts that arrive separately across multiple turns build into coherent narratives
- Eliminates metadata duplication (currently global_tags copied to every chunk)
- Enables causal chain reasoning ("because X, therefore Y")
- Provides richer context window for LLM via fact aggregations

---

## Architectural Decisions

### Core Design Choices

| Decision Point | Choice | Rationale |
|---------------|--------|-----------|
| **Dossier Title** | Use `cluster_label` from gardener | Avoids extra LLM call, leverages existing semantic grouping |
| **Embedding Strategy** | Fact-level embeddings (each fact individually) | Enables granular search, retrieve full dossier on any match |
| **Deduplication** | LLM-driven during append ("do not create duplicates") | Leverages LLM's semantic understanding vs rigid string matching |
| **Permissions** | Default `{"access": "full"}` | Future-proofing for multi-user scenarios |
| **History Tracking** | Yes, via `dossier_provenance` table | Essential for understanding fact evolution |
| **Similarity Threshold** | 0.4 | Matches existing memory search threshold |
| **Token Budget** | 3000 initially | Start high for testing, optimize later |
| **Cross-Dossier Facts** | Single fact = one dossier | Simplifies v1, can extend later |

---

## Database Schema

### Table: `dossiers`

Stores dossier headers and metadata.

```sql
CREATE TABLE IF NOT EXISTS dossiers (
    dossier_id TEXT PRIMARY KEY,           -- Format: dos_YYYYMMDD_HHMMSS
    title TEXT NOT NULL,                   -- From cluster_label
    summary TEXT,                          -- LLM-generated summary of all facts
    created_at TEXT NOT NULL,              -- ISO timestamp
    last_updated TEXT NOT NULL,            -- ISO timestamp
    permissions TEXT DEFAULT '{"access": "full"}',  -- JSON permissions object
    status TEXT DEFAULT 'active'           -- active, archived, deleted
);

CREATE INDEX idx_dossiers_updated ON dossiers(last_updated);
CREATE INDEX idx_dossiers_status ON dossiers(status);
```

### Table: `dossier_facts`

Stores individual facts within dossiers.

```sql
CREATE TABLE IF NOT EXISTS dossier_facts (
    fact_id TEXT PRIMARY KEY,              -- Format: fact_YYYYMMDD_HHMMSS_XXX
    dossier_id TEXT NOT NULL,              -- Foreign key to dossiers
    fact_text TEXT NOT NULL,               -- The actual fact content
    fact_type TEXT,                        -- observation, preference, goal, relationship, etc.
    added_at TEXT NOT NULL,                -- ISO timestamp when fact added
    source_block_id TEXT,                  -- Bridge block that contributed this fact
    source_turn_id TEXT,                   -- Turn within bridge block
    confidence REAL DEFAULT 1.0,           -- Confidence score (0-1)
    FOREIGN KEY (dossier_id) REFERENCES dossiers(dossier_id) ON DELETE CASCADE
);

CREATE INDEX idx_dossier_facts_dossier ON dossier_facts(dossier_id);
CREATE INDEX idx_dossier_facts_added ON dossier_facts(added_at);
CREATE INDEX idx_dossier_facts_source_block ON dossier_facts(source_block_id);
```

### Table: `dossier_provenance`

Tracks the history of dossier updates for provenance.

```sql
CREATE TABLE IF NOT EXISTS dossier_provenance (
    provenance_id TEXT PRIMARY KEY,        -- Format: prov_YYYYMMDD_HHMMSS_XXX
    dossier_id TEXT NOT NULL,              -- Foreign key to dossiers
    operation TEXT NOT NULL,               -- created, fact_added, fact_removed, summary_updated
    timestamp TEXT NOT NULL,               -- ISO timestamp
    source_block_id TEXT,                  -- Bridge block that triggered update
    source_turn_id TEXT,                   -- Turn that triggered update
    details TEXT,                          -- JSON with operation-specific details
    FOREIGN KEY (dossier_id) REFERENCES dossiers(dossier_id) ON DELETE CASCADE
);

CREATE INDEX idx_provenance_dossier ON dossier_provenance(dossier_id);
CREATE INDEX idx_provenance_timestamp ON dossier_provenance(timestamp);
```

---

## Implementation Phases

### Phase 1: Foundation - Database & Storage Layer

**Goal:** Create database tables and fact-level embedding storage class.

#### Tasks:

1. **Update `hmlr/memory/storage.py`**
   - Add dossier table creation in `_initialize_database()`
   - Add methods: `create_dossier()`, `add_fact_to_dossier()`, `get_dossier()`, `get_dossier_facts()`, `update_dossier_summary()`

2. **Create `hmlr/memory/dossier_storage.py`**
   - Implement `DossierEmbeddingStorage` class
   - Methods: `save_fact_embedding()`, `search_similar_facts()`, `get_dossier_by_fact_id()`
   - Uses same embedding model (all-MiniLM-L6-v2, 384D)

3. **Add Provenance Tracking**
   - Methods in storage: `add_provenance_entry()`, `get_dossier_history()`

#### Code Example: `DossierEmbeddingStorage`

```python
# hmlr/memory/dossier_storage.py
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Tuple, Dict, Any

class DossierEmbeddingStorage:
    """Manages fact-level embeddings for dossier retrieval."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self._initialize_table()
    
    def _initialize_table(self):
        """Create dossier_fact_embeddings table."""
        import sqlite3
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS dossier_fact_embeddings (
                fact_id TEXT PRIMARY KEY,
                dossier_id TEXT NOT NULL,
                embedding BLOB NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_dfe_dossier ON dossier_fact_embeddings(dossier_id)")
        conn.commit()
        conn.close()
    
    def save_fact_embedding(self, fact_id: str, dossier_id: str, fact_text: str) -> None:
        """Embed and store a single fact."""
        embedding = self.model.encode(fact_text)
        embedding_blob = embedding.tobytes()
        
        import sqlite3
        from datetime import datetime
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO dossier_fact_embeddings 
            (fact_id, dossier_id, embedding, created_at)
            VALUES (?, ?, ?, ?)
        """, (fact_id, dossier_id, embedding_blob, datetime.now().isoformat()))
        conn.commit()
        conn.close()
    
    def search_similar_facts(self, query: str, top_k: int = 10, 
                            threshold: float = 0.4) -> List[Tuple[str, str, float]]:
        """Search for similar facts, return (fact_id, dossier_id, score)."""
        query_embedding = self.model.encode(query)
        
        import sqlite3
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT fact_id, dossier_id, embedding FROM dossier_fact_embeddings")
        
        results = []
        for fact_id, dossier_id, embedding_blob in cursor.fetchall():
            fact_embedding = np.frombuffer(embedding_blob, dtype=np.float32)
            similarity = np.dot(query_embedding, fact_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(fact_embedding)
            )
            if similarity >= threshold:
                results.append((fact_id, dossier_id, float(similarity)))
        
        conn.close()
        results.sort(key=lambda x: x[2], reverse=True)
        return results[:top_k]
    
    def get_dossier_by_fact_id(self, fact_id: str) -> str:
        """Get dossier_id for a given fact_id."""
        import sqlite3
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT dossier_id FROM dossier_fact_embeddings WHERE fact_id = ?", (fact_id,))
        result = cursor.fetchone()
        conn.close()
        return result[0] if result else None
```

#### Test Checkpoint:

```python
# Manually insert test dossier
storage.create_dossier("dos_test", "Vegetarian Diet", "Test summary")
storage.add_fact_to_dossier("dos_test", "fact_001", "User is strictly vegetarian")

# Test embedding search
dossier_storage = DossierEmbeddingStorage(db_path)
dossier_storage.save_fact_embedding("fact_001", "dos_test", "User is strictly vegetarian")
results = dossier_storage.search_similar_facts("dietary preferences")
assert len(results) > 0
assert results[0][1] == "dos_test"
```

---

### Phase 2: Gardener Refactor

**Goal:** Remove hierarchical chunking, add semantic fact grouping, prepare facts for dossier routing.

#### Tasks:

1. **Remove `HierarchicalChunker` class** from `manual_gardener.py`
2. **Modify `process_bridge_block()`** to accept pre-chunked facts
3. **Add `_group_facts_semantically()`** method using LLM
4. **Update output format** to produce fact packets for DossierGovernor

#### Code Changes:

```python
# hmlr/memory/gardener/manual_gardener.py

async def process_bridge_block(self, block_id: str):
    """Process bridge block facts and route to dossier governor."""
    
    # 1. Retrieve bridge block
    block = self.storage.get_bridge_block(block_id)
    if not block:
        print(f"No bridge block found with ID: {block_id}")
        return
    
    # 2. Extract facts from block (already turn-level chunks)
    facts = []
    for turn in block['turns']:
        turn_facts = self._extract_facts_from_turn(turn)
        facts.extend(turn_facts)
    
    # 3. Group facts semantically
    fact_groups = await self._group_facts_semantically(facts)
    
    # 4. Route each group to dossier governor
    for group in fact_groups:
        fact_packet = {
            'cluster_label': group['label'],
            'facts': group['facts'],
            'source_block_id': block_id,
            'timestamp': group['timestamp']
        }
        await self.dossier_governor.process_fact_packet(fact_packet)
    
    # 5. Delete processed bridge block
    self._delete_bridge_block(block_id)

async def _group_facts_semantically(self, facts: List[Dict]) -> List[Dict]:
    """Use LLM to group related facts by topic/theme."""
    
    prompt = f"""Given these facts extracted from a conversation:

{json.dumps(facts, indent=2)}

Group related facts by semantic theme. For each group, provide:
1. A concise label (2-5 words)
2. The facts that belong to that group
3. The earliest timestamp from facts in the group

Return as JSON array: [{{"label": "...", "facts": [...], "timestamp": "..."}}]
"""
    
    response = await self.llm_client.query(
        prompt=prompt,
        model="gpt-4.1-mini",
        response_format={"type": "json_object"}
    )
    
    return json.loads(response)['groups']
```

#### Test Checkpoint:

```python
# Process test bridge block with facts about vegetarian diet
facts = [
    {"text": "User is strictly vegetarian", "turn_id": "turn_001"},
    {"text": "User avoids all meat products", "turn_id": "turn_001"},
    {"text": "User mentioned dietary restrictions", "turn_id": "turn_002"}
]
groups = await gardener._group_facts_semantically(facts)
assert len(groups) >= 1
assert groups[0]['label'] in ["Vegetarian Diet", "Dietary Preferences"]
```

---

### Phase 3: Write-Side DossierGovernor

**Goal:** Implement routing logic that decides whether to append facts to existing dossiers or create new ones.

#### Tasks:

1. **Create `hmlr/memory/synthesis/dossier_governor.py`**
2. **Implement `process_fact_packet()`** main entry point
3. **Implement `_find_candidate_dossiers()`** using embedding search
4. **Implement `_llm_decide_routing()`** for append vs create decision
5. **Implement `_append_to_dossier()`** and `_create_new_dossier()`**
6. **Add provenance tracking** for all operations

#### Code Example:

```python
# hmlr/memory/synthesis/dossier_governor.py
import json
from datetime import datetime
from typing import Dict, List, Any

class DossierGovernor:
    """Write-side governor for routing facts to dossiers."""
    
    def __init__(self, storage, dossier_storage, llm_client, id_generator):
        self.storage = storage
        self.dossier_storage = dossier_storage
        self.llm_client = llm_client
        self.id_generator = id_generator
    
    async def process_fact_packet(self, fact_packet: Dict[str, Any]) -> str:
        """
        Route fact packet to appropriate dossier.
        Returns dossier_id where facts were added/created.
        """
        cluster_label = fact_packet['cluster_label']
        facts = fact_packet['facts']
        source_block_id = fact_packet['source_block_id']
        
        # 1. Search for candidate dossiers using Multi-Vector Voting
        candidates = self._find_candidate_dossiers(facts, top_k=5)
        
        # 2. LLM decides: append to existing or create new
        if candidates:
            decision = await self._llm_decide_routing(facts, candidates)
            
            if decision['action'] == 'append':
                dossier_id = decision['target_dossier_id']
                await self._append_to_dossier(dossier_id, facts, source_block_id)
                return dossier_id
        
        # 3. No suitable dossier found, create new
        dossier_id = await self._create_new_dossier(cluster_label, facts, source_block_id)
        return dossier_id
    
    def _find_candidate_dossiers(self, facts: List[str], top_k: int = 5) -> List[Dict]:
        """
        Multi-Vector Voting: Search using ALL facts and rank by hit frequency.
        
        Algorithm:
        1. Search for each fact individually against all dossier facts
        2. Tally which dossiers get the most hits
        3. Sort by hit count (dossiers that match multiple facts bubble up)
        4. Return top K candidates
        
        Example:
        - If 5 facts are incoming and 3 of them match facts in Dossier_A,
          while only 1 fact matches Dossier_B, then Dossier_A wins the vote.
        - This solves the "vague fact" problem where a generic statement like
          "It is fast" might match many dossiers, but specific facts will cause
          the correct dossier to rise to the top.
        """
        vote_tally = {}  # {dossier_id: {'score_sum': 0.0, 'hits': 0}}
        
        # 1. Search for EVERY fact in the packet
        for fact in facts:
            results = self.dossier_storage.search_similar_facts(
                query=fact,
                top_k=10,  # Cast a wider net per fact
                threshold=0.4  # Consistent with memory search threshold
            )
            
            # 2. Tally the votes
            for fact_id, dossier_id, score in results:
                if dossier_id not in vote_tally:
                    vote_tally[dossier_id] = {'score_sum': 0.0, 'hits': 0}
                
                vote_tally[dossier_id]['hits'] += 1
                vote_tally[dossier_id]['score_sum'] += score
        
        # 3. Sort by Hit Count first (primary), then Score Sum (tiebreaker)
        # This causes dossiers with more matching facts to "bubble up"
        sorted_dossiers = sorted(
            vote_tally.items(),
            key=lambda item: (item[1]['hits'], item[1]['score_sum']),
            reverse=True
        )
        
        # 4. Fetch full details for top K dossiers
        candidates = []
        for dossier_id, stats in sorted_dossiers[:top_k]:
            dossier = self.storage.get_dossier(dossier_id)
            if dossier:
                dossier_facts = self.storage.get_dossier_facts(dossier_id)
                candidates.append({
                    'dossier_id': dossier_id,
                    'title': dossier['title'],
                    'summary': dossier['summary'],
                    'facts': [f['fact_text'] for f in dossier_facts],
                    'vote_hits': stats['hits'],  # How many facts matched
                    'vote_score': stats['score_sum']  # Total similarity score
                })
        
        return candidates
    
    async def _llm_decide_routing(self, new_facts: List[str], 
                                   candidates: List[Dict]) -> Dict:
        """LLM decides whether to append or create new."""
        
        prompt = f"""Given these new facts to store:

NEW FACTS:
{json.dumps(new_facts, indent=2)}

CANDIDATE DOSSIERS:
{json.dumps(candidates, indent=2)}

Decide:
- If new facts belong to an existing dossier, return: {{"action": "append", "target_dossier_id": "dos_xxx"}}
- If new facts form a distinct topic, return: {{"action": "create"}}

Consider semantic similarity and whether facts build a causal chain.
"""
        
        response = await self.llm_client.query(
            prompt=prompt,
            model="gpt-4.1-mini",
            response_format={"type": "json_object"}
        )
        
        return json.loads(response)
    
    async def _append_to_dossier(self, dossier_id: str, facts: List[str], 
                                 source_block_id: str) -> None:
        """Add facts to existing dossier, update summary."""
        
        # 1. Add each fact
        for fact_text in facts:
            fact_id = self.id_generator.generate_id("fact")
            self.storage.add_fact_to_dossier(
                dossier_id=dossier_id,
                fact_id=fact_id,
                fact_text=fact_text,
                source_block_id=source_block_id
            )
            
            # Embed fact
            self.dossier_storage.save_fact_embedding(fact_id, dossier_id, fact_text)
            
            # Log provenance
            self.storage.add_provenance_entry(
                dossier_id=dossier_id,
                operation="fact_added",
                source_block_id=source_block_id,
                details=json.dumps({"fact_id": fact_id})
            )
        
        # 2. Update dossier summary (incremental)
        await self._update_dossier_summary(dossier_id, facts)
    
    async def _create_new_dossier(self, title: str, facts: List[str], 
                                  source_block_id: str) -> str:
        """Create new dossier with facts."""
        
        dossier_id = self.id_generator.generate_id("dos")
        
        # 1. Generate initial summary
        summary = await self._generate_summary(facts)
        
        # 2. Create dossier
        self.storage.create_dossier(
            dossier_id=dossier_id,
            title=title,
            summary=summary
        )
        
        # 3. Add facts
        for fact_text in facts:
            fact_id = self.id_generator.generate_id("fact")
            self.storage.add_fact_to_dossier(
                dossier_id=dossier_id,
                fact_id=fact_id,
                fact_text=fact_text,
                source_block_id=source_block_id
            )
            
            # Embed fact
            self.dossier_storage.save_fact_embedding(fact_id, dossier_id, fact_text)
        
        # 4. Log provenance
        self.storage.add_provenance_entry(
            dossier_id=dossier_id,
            operation="created",
            source_block_id=source_block_id,
            details=json.dumps({"num_facts": len(facts)})
        )
        
        return dossier_id
    
    async def _update_dossier_summary(self, dossier_id: str, new_facts: List[str]) -> None:
        """Incrementally update dossier summary with new facts."""
        
        dossier = self.storage.get_dossier(dossier_id)
        old_summary = dossier['summary']
        
        prompt = f"""Update this dossier summary with new facts:

OLD SUMMARY:
{old_summary}

NEW FACTS:
{json.dumps(new_facts, indent=2)}

Generate updated summary that incorporates new facts. Build causal chains where possible.
Do not create duplicates of existing information.
"""
        
        new_summary = await self.llm_client.query(
            prompt=prompt,
            model="gpt-4.1-mini"
        )
        
        self.storage.update_dossier_summary(dossier_id, new_summary)
        
        # Log provenance
        self.storage.add_provenance_entry(
            dossier_id=dossier_id,
            operation="summary_updated",
            source_block_id=None,
            details=json.dumps({"num_new_facts": len(new_facts)})
        )
```

#### Test Checkpoint:

```python
# Test 1: Create new dossier
fact_packet = {
    'cluster_label': 'Vegetarian Diet',
    'facts': ['User is strictly vegetarian', 'User avoids meat'],
    'source_block_id': 'block_001',
    'timestamp': datetime.now().isoformat()
}
dossier_id = await dossier_governor.process_fact_packet(fact_packet)
assert dossier_id is not None

# Test 2: Append to existing dossier
fact_packet2 = {
    'cluster_label': 'Vegetarian Diet',
    'facts': ['User prefers plant-based proteins'],
    'source_block_id': 'block_002',
    'timestamp': datetime.now().isoformat()
}
dossier_id2 = await dossier_governor.process_fact_packet(fact_packet2)
assert dossier_id2 == dossier_id  # Should append to same dossier

# Test 3: Multi-Vector Voting (the "vague fact" scenario)
fact_packet3 = {
    'cluster_label': 'Dietary Preferences',
    'facts': [
        'It is healthy',  # Vague - might match many dossiers
        'User avoids all animal products',  # Specific - matches vegetarian dossier
        'Plant-based diet has benefits'  # Specific - matches vegetarian dossier
    ],
    'source_block_id': 'block_003',
    'timestamp': datetime.now().isoformat()
}
dossier_id3 = await dossier_governor.process_fact_packet(fact_packet3)
# Should still route to vegetarian dossier because 2/3 facts matched it
assert dossier_id3 == dossier_id

# Verify voting system worked
candidates = dossier_governor._find_candidate_dossiers(fact_packet3['facts'], top_k=5)
assert candidates[0]['dossier_id'] == dossier_id  # Top candidate by vote count
assert candidates[0]['vote_hits'] >= 2  # At least 2 facts matched

# Verify provenance
history = storage.get_dossier_history(dossier_id)
assert len(history) >= 2  # created + fact_added
```

---

### Phase 4: Read-Side Dossier Retrieval

**Goal:** Integrate dossier search into existing retrieval pipeline as 3rd parallel governor call.

#### Tasks:

1. **Create `hmlr/memory/retrieval/dossier_retriever.py`**
2. **Implement `retrieve_relevant_dossiers()`** using fact embedding search
3. **Add `format_for_context()`** to prepare dossiers for LLM context
4. **Modify `lattice.py` (TheGovernor)** to add 3rd parallel call
5. **Update context window** in `conversation_engine.py` to include dossiers

#### Code Example:

```python
# hmlr/memory/retrieval/dossier_retriever.py
from typing import List, Dict, Any

class DossierRetriever:
    """Read-side retriever for dossier search."""
    
    def __init__(self, storage, dossier_storage):
        self.storage = storage
        self.dossier_storage = dossier_storage
    
    def retrieve_relevant_dossiers(self, query: str, top_k: int = 3, 
                                   threshold: float = 0.4) -> List[Dict]:
        """Search for relevant dossiers based on fact embeddings."""
        
        # 1. Search fact embeddings
        fact_results = self.dossier_storage.search_similar_facts(
            query=query,
            top_k=top_k * 2,  # Get more facts to dedupe by dossier
            threshold=threshold
        )
        
        # 2. Deduplicate by dossier_id, get full dossiers
        dossier_ids = list(set([r[1] for r in fact_results]))[:top_k]
        
        dossiers = []
        for dossier_id in dossier_ids:
            dossier = self.storage.get_dossier(dossier_id)
            if dossier:
                facts = self.storage.get_dossier_facts(dossier_id)
                dossiers.append({
                    'dossier_id': dossier_id,
                    'title': dossier['title'],
                    'summary': dossier['summary'],
                    'facts': facts,
                    'score': max([r[2] for r in fact_results if r[1] == dossier_id])
                })
        
        return sorted(dossiers, key=lambda x: x['score'], reverse=True)
    
    def format_for_context(self, dossiers: List[Dict]) -> str:
        """Format dossiers for LLM context window."""
        
        if not dossiers:
            return ""
        
        formatted = "=== FACT DOSSIERS ===\n\n"
        for dossier in dossiers:
            formatted += f"## {dossier['title']}\n"
            formatted += f"Summary: {dossier['summary']}\n\n"
            formatted += "Facts:\n"
            for fact in dossier['facts']:
                formatted += f"  - {fact['fact_text']} (added: {fact['added_at']})\n"
            formatted += f"\n(Score: {dossier['score']:.2f})\n\n"
        
        return formatted
```

#### Integration in `lattice.py`:

```python
# hmlr/memory/retrieval/lattice.py

async def _retrieve_and_filter_memories(self, query: str, k: int = 5):
    """Retrieve memories, topics, and dossiers in parallel."""
    
    # Parallel retrieval
    topic_task = asyncio.create_task(self._retrieve_topics(query, k))
    memory_task = asyncio.create_task(self._retrieve_memory_chunks(query, k))
    dossier_task = asyncio.create_task(self._retrieve_dossiers(query, k=3))  # NEW
    
    topics, memory_chunks, dossiers = await asyncio.gather(
        topic_task, memory_task, dossier_task
    )
    
    # Filter each type
    filtered_topics = await self._filter_topics(query, topics) if topics else []
    filtered_memories = await self._filter_memories(query, memory_chunks) if memory_chunks else []
    filtered_dossiers = await self._filter_dossiers(query, dossiers) if dossiers else []
    
    return {
        'topics': filtered_topics,
        'memories': filtered_memories,
        'dossiers': filtered_dossiers
    }

async def _retrieve_dossiers(self, query: str, k: int = 3):
    """Retrieve relevant dossiers."""
    return self.dossier_retriever.retrieve_relevant_dossiers(
        query=query,
        top_k=k,
        threshold=0.4
    )

async def _filter_dossiers(self, query: str, dossiers: List[Dict]) -> List[Dict]:
    """LLM filters dossiers for relevance."""
    
    if not dossiers:
        return []
    
    prompt = f"""User query: {query}

Retrieved fact dossiers:
{json.dumps([{'title': d['title'], 'summary': d['summary']} for d in dossiers], indent=2)}

Which dossiers are relevant to answering the user's query?
Return JSON: {{"relevant_dossier_ids": ["dos_xxx", ...]}}
"""
    
    response = await self.llm_client.query(
        prompt=prompt,
        model="gpt-4.1-mini",
        response_format={"type": "json_object"}
    )
    
    relevant_ids = json.loads(response)['relevant_dossier_ids']
    return [d for d in dossiers if d['dossier_id'] in relevant_ids]
```

#### Update Context in `conversation_engine.py`:

```python
# hmlr/core/conversation_engine.py

async def _handle_chat(self, user_message: str):
    """Handle chat with dossier context."""
    
    # Retrieve all context types
    retrieval_result = await self.governor._retrieve_and_filter_memories(
        query=user_message,
        k=5
    )
    
    # Format context
    topic_context = self.governor._format_topics(retrieval_result['topics'])
    memory_context = self.governor._format_memories(retrieval_result['memories'])
    dossier_context = self.dossier_retriever.format_for_context(
        retrieval_result['dossiers']
    )
    
    # Build full context
    full_context = f"""{topic_context}

{memory_context}

{dossier_context}"""
    
    # Query LLM
    response = await self.governor.api_client.query(
        prompt=full_context + f"\n\nUser: {user_message}\nAssistant:",
        model="gpt-4.1-mini"
    )
    
    return response
```

#### Test Checkpoint:

```python
# Test dossier retrieval
query = "What are the user's dietary restrictions?"
dossiers = dossier_retriever.retrieve_relevant_dossiers(query, top_k=3)
assert len(dossiers) > 0
assert any('vegetarian' in d['title'].lower() for d in dossiers)

# Test context formatting
context = dossier_retriever.format_for_context(dossiers)
assert "=== FACT DOSSIERS ===" in context
assert "Vegetarian Diet" in context

# Test integrated retrieval
retrieval_result = await governor._retrieve_and_filter_memories(query, k=5)
assert 'dossiers' in retrieval_result
assert len(retrieval_result['dossiers']) > 0
```

---

### Phase 5: Integration & Testing

**Goal:** Update component factory, integrate into main flow, create comprehensive tests.

#### Tasks:

1. **Update `hmlr/core/component_factory.py`** to initialize DossierGovernor and DossierRetriever
2. **Modify `run_gardener.py`** to use new gardener flow
3. **Update `main.py`** to display dossier context (optional debug)
4. **Create `tests/test_dossier_system.py`** with E2E tests
5. **Create `tests/test_incremental_fact_building.py`** (Hydra9-style scenario)

#### Component Factory Updates:

```python
# hmlr/core/component_factory.py

def create_components(config_path: str):
    """Initialize all components including dossier system."""
    
    # Existing components
    storage = MemoryStorage(db_path)
    embedding_storage = EmbeddingStorage(db_path)
    llm_client = LlamaClient(api_key)
    id_generator = IDGenerator()
    
    # NEW: Dossier components
    dossier_storage = DossierEmbeddingStorage(db_path)
    dossier_governor = DossierGovernor(
        storage=storage,
        dossier_storage=dossier_storage,
        llm_client=llm_client,
        id_generator=id_generator
    )
    dossier_retriever = DossierRetriever(
        storage=storage,
        dossier_storage=dossier_storage
    )
    
    # Update gardener with dossier_governor
    gardener = ManualGardener(
        storage=storage,
        embedding_storage=embedding_storage,
        dossier_storage=dossier_storage,
        dossier_governor=dossier_governor,
        llm_client=llm_client,
        id_generator=id_generator
    )
    
    # Update governor with dossier_retriever
    governor = TheGovernor(
        crawler=crawler,
        embedding_storage=embedding_storage,
        storage=storage,
        dossier_retriever=dossier_retriever,  # NEW
        llm_client=llm_client
    )
    
    return ComponentBundle(
        storage=storage,
        embedding_storage=embedding_storage,
        dossier_storage=dossier_storage,  # NEW
        dossier_governor=dossier_governor,  # NEW
        dossier_retriever=dossier_retriever,  # NEW
        gardener=gardener,
        governor=governor,
        # ... other components
    )
```

#### E2E Test Example:

```python
# tests/test_dossier_system.py
import pytest
import asyncio
from hmlr.core.component_factory import create_components

@pytest.mark.asyncio
async def test_incremental_fact_building():
    """Test that facts arriving separately build into single dossier."""
    
    components = create_components("config/test_config.json")
    
    # Simulate bridge block 1: User mentions vegetarian diet
    bridge_block_1 = {
        'block_id': 'block_001',
        'turns': [{
            'turn_id': 'turn_001',
            'user': 'I am strictly vegetarian',
            'assistant': 'I understand you follow a vegetarian diet.'
        }]
    }
    components.storage.save_bridge_block(bridge_block_1)
    
    # Process block 1
    await components.gardener.process_bridge_block('block_001')
    
    # Verify dossier created
    dossiers = components.storage.get_all_dossiers()
    assert len(dossiers) == 1
    diet_dossier = dossiers[0]
    assert 'vegetarian' in diet_dossier['title'].lower()
    
    # Simulate bridge block 2: User adds more dietary info (separate conversation)
    bridge_block_2 = {
        'block_id': 'block_002',
        'turns': [{
            'turn_id': 'turn_002',
            'user': 'I also avoid eggs and dairy',
            'assistant': 'So you follow a vegan diet?'
        }]
    }
    components.storage.save_bridge_block(bridge_block_2)
    
    # Process block 2
    await components.gardener.process_bridge_block('block_002')
    
    # Verify facts appended to SAME dossier (not created new one)
    dossiers = components.storage.get_all_dossiers()
    assert len(dossiers) == 1  # Still only 1 dossier
    
    # Verify facts were appended
    facts = components.storage.get_dossier_facts(diet_dossier['dossier_id'])
    assert len(facts) >= 2
    fact_texts = [f['fact_text'] for f in facts]
    assert any('vegetarian' in t.lower() for t in fact_texts)
    assert any('eggs' in t.lower() or 'dairy' in t.lower() for t in fact_texts)
    
    # Verify provenance tracking
    history = components.storage.get_dossier_history(diet_dossier['dossier_id'])
    assert len(history) >= 3  # created + 2x fact_added
    assert history[0]['operation'] == 'created'
    assert history[-1]['operation'] == 'fact_added'

@pytest.mark.asyncio
async def test_dossier_retrieval_in_conversation():
    """Test that dossiers appear in context and influence responses."""
    
    components = create_components("config/test_config.json")
    
    # Create test dossier
    await _setup_vegetarian_dossier(components)
    
    # Query that should retrieve dossier
    user_message = "What can I cook for dinner tonight?"
    
    # Retrieve context
    retrieval_result = await components.governor._retrieve_and_filter_memories(
        query=user_message,
        k=5
    )
    
    # Verify dossier retrieved
    assert 'dossiers' in retrieval_result
    assert len(retrieval_result['dossiers']) > 0
    
    diet_dossier = retrieval_result['dossiers'][0]
    assert 'vegetarian' in diet_dossier['title'].lower()
    
    # Format context
    context = components.dossier_retriever.format_for_context(
        retrieval_result['dossiers']
    )
    
    # Verify context formatting
    assert "=== FACT DOSSIERS ===" in context
    assert "vegetarian" in context.lower()
```

---

### Phase 6: Data Migration

**Goal:** Migrate existing bridge blocks and gardened memory to new dossier structure.

#### Tasks:

1. **Create `scripts/migrate_to_dossiers.py`** migration script
2. **Process all bridge blocks** through new gardener flow
3. **Validate migration** - check dossier counts, fact counts, provenance
4. **Archive old data** - move pre-migration data to backup tables

#### Migration Script:

```python
# scripts/migrate_to_dossiers.py
import asyncio
from hmlr.core.component_factory import create_components

async def migrate_existing_data():
    """Migrate existing bridge blocks to dossier system."""
    
    components = create_components("config/user_profile_lite.json")
    
    # Get all unprocessed bridge blocks
    bridge_blocks = components.storage.get_all_bridge_blocks()
    
    print(f"Found {len(bridge_blocks)} bridge blocks to migrate")
    
    for i, block in enumerate(bridge_blocks):
        block_id = block['block_id']
        print(f"Processing block {i+1}/{len(bridge_blocks)}: {block_id}")
        
        try:
            await components.gardener.process_bridge_block(block_id)
            print(f"  ✓ Processed successfully")
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    # Validate migration
    dossiers = components.storage.get_all_dossiers()
    print(f"\nMigration complete:")
    print(f"  - {len(dossiers)} dossiers created")
    
    total_facts = sum(
        len(components.storage.get_dossier_facts(d['dossier_id'])) 
        for d in dossiers
    )
    print(f"  - {total_facts} facts stored")
    
    # Show sample dossiers
    print("\nSample dossiers:")
    for dossier in dossiers[:5]:
        print(f"  - {dossier['title']}: {dossier['summary'][:100]}...")

if __name__ == "__main__":
    asyncio.run(migrate_existing_data())
```

#### Validation Queries:

```sql
-- Check dossier counts
SELECT COUNT(*) FROM dossiers;

-- Check fact distribution
SELECT 
    d.title,
    COUNT(df.fact_id) as num_facts
FROM dossiers d
LEFT JOIN dossier_facts df ON d.dossier_id = df.dossier_id
GROUP BY d.dossier_id
ORDER BY num_facts DESC;

-- Check provenance tracking
SELECT 
    dossier_id,
    operation,
    COUNT(*) as count
FROM dossier_provenance
GROUP BY dossier_id, operation;

-- Check embedding coverage
SELECT 
    COUNT(DISTINCT df.fact_id) as facts_with_embeddings,
    (SELECT COUNT(*) FROM dossier_facts) as total_facts
FROM dossier_fact_embeddings dfe
JOIN dossier_facts df ON dfe.fact_id = df.fact_id;
```

---

## Testing Strategy

### Unit Tests

- `test_dossier_storage.py` - Database operations
- `test_dossier_embedding_storage.py` - Fact embedding search
- `test_dossier_governor.py` - Routing logic
- `test_dossier_retriever.py` - Retrieval and formatting

### Integration Tests

- `test_gardener_to_dossier.py` - End-to-end write path
- `test_dossier_retrieval_integration.py` - End-to-end read path

### E2E Tests

- `test_incremental_fact_building.py` - Hydra9-style scenario
- `test_dossier_context_in_conversation.py` - Full conversation flow

### Performance Tests

- Token usage with 3 parallel governor calls
- Embedding search latency
- Database query performance with large dossier counts

---

## Rollout Plan

### Phase 1-2: Week 1
- Database schema
- Storage classes
- Gardener refactor

### Phase 3: Week 2
- DossierGovernor implementation
- Unit tests

### Phase 4: Week 2-3
- DossierRetriever implementation
- Integration with lattice.py

### Phase 5: Week 3
- Component factory updates
- E2E tests

### Phase 6: Week 4
- Data migration
- Production deployment

---

## Open Questions & Future Enhancements

### V1 Scope Exclusions

- Cross-dossier facts (single fact can link to multiple dossiers)
- Dossier merging (when two dossiers discovered to be related)
- Dossier archival/expiration policies
- Multi-user permissions enforcement

### Future Considerations

1. **Dossier Evolution Tracking:** Track how dossier summaries change over time
2. **Fact Confidence Decay:** Lower confidence scores for older facts
3. **Dossier Relationships:** Graph structure linking related dossiers
4. **Automated Dossier Cleanup:** Archive stale or redundant dossiers
5. **Dossier Export:** JSON export for external analysis

---

## References

- Original Requirements: Conversation with user, December 15, 2025
- Existing Architecture: `hmlr/memory/retrieval/lattice.py`, `hmlr/memory/gardener/manual_gardener.py`
- Database Schema: `hmlr/memory/storage.py`
- Test Scenarios: `tests/test_12_hydra_e2e.py` (Hydra9 scenario)

---

**Document Status:** Planning Phase Complete  
**Next Action:** Begin Phase 1 implementation  
**Last Updated:** December 15, 2025
