"""
Phase 11.3: Fact Scrubber - Parallel fact extraction with chunk linking.

The FactScrubber extracts "hard facts" (definitions, acronyms, secrets, entities)
from conversation turns and links them to sentence-level chunks for precise provenance.

Key Features:
- Parallel LLM extraction (async, non-blocking)
- Precise linking to sentence chunks (evidence_snippet)
- Categorical organization (Definition, Acronym, Secret, Entity)
- Fast exact-match retrieval via indexed fact_store

Usage:
    scrubber = FactScrubber(storage, llm_client)
    await scrubber.extract_and_save(turn_id, message_text, chunks)
    facts = scrubber.query_facts(query="HMLR")
"""

import json
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

from memory.storage import Storage


@dataclass
class Fact:
    """
    Represents a hard fact extracted from conversation.
    
    Attributes:
        key: The identifier (e.g., "HMLR", "user_name", "API_KEY")
        value: The fact content (e.g., "Hierarchical Memory Lookup & Routing")
        category: Classification (Definition, Acronym, Secret, Entity)
        evidence_snippet: 10-20 words of context around the fact
        source_chunk_id: Sentence chunk ID containing the fact (highest precision)
        source_paragraph_id: Paragraph chunk ID for broader context
        source_block_id: Bridge block ID (if archived)
        source_span_id: Conversation span ID
        created_at: ISO-8601 timestamp
    """
    key: str
    value: str
    category: str  # Definition | Acronym | Secret | Entity
    evidence_snippet: str
    source_chunk_id: Optional[str] = None
    source_paragraph_id: Optional[str] = None
    source_block_id: Optional[str] = None
    source_span_id: Optional[str] = None
    created_at: str = ""
    
    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(asdict(self), indent=2)
    
    @staticmethod
    def from_json(json_str: str) -> 'Fact':
        """Deserialize from JSON string."""
        data = json.loads(json_str)
        return Fact(**data)


class FactScrubber:
    """
    Extracts hard facts from conversation turns using LLM prompting.
    
    The scrubber identifies:
    1. Definitions and acronyms (e.g., "HMLR = Hierarchical Memory...")
    2. Entity relationships (e.g., "John is the CEO of X")
    3. Secrets/keys/credentials (e.g., "API key is abc123")
    4. Factual statements (e.g., "User prefers Python over JavaScript")
    
    Facts are linked to sentence-level chunks for precise provenance tracking.
    """
    
    # LLM Prompt Template for fact extraction
    EXTRACTION_PROMPT = """Extract ONLY hard facts from this message.

CATEGORIES:
1. Definition - Definitions of terms or concepts
2. Acronym - Acronym expansions (e.g., "API = Application Programming Interface")
3. Secret - Credentials, API keys, passwords, tokens
4. Entity - Relationships between entities (e.g., "John is CEO of X")

RULES:
- Ignore general conversation or opinions
- Extract only verifiable, referenceable facts
- For acronyms, include the full expansion
- For secrets, include the key/value pair
- For entities, include the relationship type

MESSAGE:
{message}

Return JSON in this exact format:
{{
  "facts": [
    {{
      "key": "concise identifier (2-4 words)",
      "value": "the fact itself (complete sentence or phrase)",
      "category": "Definition|Acronym|Secret|Entity",
      "evidence_snippet": "exact 10-20 word quote containing the fact"
    }}
  ]
}}

If no facts found, return: {{"facts": []}}
"""
    
    def __init__(self, storage: Storage, api_client=None):
        """
        Initialize the FactScrubber.
        
        Args:
            storage: Storage instance for database operations
            api_client: ExternalAPIClient for LLM-based extraction (optional for testing)
        """
        self.storage = storage
        self.api_client = api_client
        self._ensure_fact_store_exists()
    
    def _ensure_fact_store_exists(self):
        """Ensure fact_store table exists with all required columns."""
        # This is handled by the Phase 11.5 migration, but we verify here
        cursor = self.storage.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS fact_store (
                fact_id INTEGER PRIMARY KEY AUTOINCREMENT,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                category TEXT,
                source_span_id TEXT,
                source_chunk_id TEXT,
                source_paragraph_id TEXT,
                source_block_id TEXT,
                evidence_snippet TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (source_span_id) REFERENCES spans(span_id) ON DELETE SET NULL
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_fact_key ON fact_store(key)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_fact_chunk ON fact_store(source_chunk_id)")
        self.storage.conn.commit()
    
    async def extract_and_save(
        self,
        turn_id: str,
        message_text: str,
        chunks: List[Dict[str, Any]],
        span_id: Optional[str] = None,
        block_id: Optional[str] = None
    ) -> List[Fact]:
        """
        Extract facts from message and save to fact_store with chunk links.
        
        Args:
            turn_id: Turn identifier
            message_text: User message text
            chunks: List of sentence chunks from ChunkEngine (contains chunk_id, text, parent)
            span_id: Current span ID (optional)
            block_id: Bridge block ID if archived (optional)
        
        Returns:
            List of extracted Fact objects
        
        Performance Target: <500ms (parallel, non-blocking)
        """
        if not self.api_client:
            # Fallback: Use heuristic extraction (for testing or no-LLM mode)
            return self._heuristic_extract(message_text, chunks, span_id, block_id)
        
        try:
            # Call LLM for fact extraction using ExternalAPIClient
            prompt = self.EXTRACTION_PROMPT.format(message=message_text)
            
            # Use GPT-4.1-mini for fast, cheap extraction
            # ExternalAPIClient.query_external_api returns the content string directly
            response_content = self.api_client.query_external_api(
                query=prompt,
                model="gpt-4.1-mini",  # Fast and cheap for fact extraction
                max_tokens=500
            )
            
            # Parse JSON response
            facts_data = self._parse_llm_response(response_content)
            
            # Link facts to chunks and save
            facts = []
            for fact_dict in facts_data.get("facts", []):
                fact = self._create_fact_with_chunk_link(
                    fact_dict, chunks, span_id, block_id
                )
                if fact:
                    self._save_fact(fact)
                    facts.append(fact)
            
            return facts
        
        except Exception as e:
            print(f"[FactScrubber] LLM extraction failed: {e}, using fallback")
            return self._heuristic_extract(message_text, chunks, span_id, block_id)
    
    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM JSON response, handling markdown code blocks."""
        # Strip markdown code blocks if present
        response = response.strip()
        if response.startswith("```"):
            lines = response.split("\n")
            response = "\n".join(lines[1:-1])  # Remove first and last line
        
        try:
            return json.loads(response)
        except json.JSONDecodeError as e:
            print(f"[FactScrubber] JSON parse error: {e}")
            return {"facts": []}
    
    def _create_fact_with_chunk_link(
        self,
        fact_dict: Dict[str, Any],
        chunks: List[Any],
        span_id: Optional[str],
        block_id: Optional[str]
    ) -> Optional[Fact]:
        """
        Create Fact object and link to the sentence chunk containing evidence.
        
        Strategy:
        1. Use evidence_snippet to find matching sentence chunk
        2. Extract parent paragraph chunk ID
        3. Link to block_id if provided
        """
        evidence = fact_dict.get("evidence_snippet", "")
        
        # Find the sentence chunk containing the evidence
        source_chunk_id = None
        source_paragraph_id = None
        
        for chunk in chunks:
            # Handle both dict and Chunk dataclass objects
            chunk_type = chunk.chunk_type if hasattr(chunk, 'chunk_type') else chunk.get("chunk_type")
            chunk_text = chunk.text_verbatim if hasattr(chunk, 'text_verbatim') else chunk.get("text_verbatim", "")
            
            if chunk_type == "sentence":
                # Fuzzy match: Check if evidence is contained in chunk or vice versa
                # Remove all periods for comparison (ChunkEngine may add periods for abbreviations)
                evidence_clean = evidence.replace('.', '').replace(' ', '').lower()
                chunk_text_clean = chunk_text.replace('.', '').replace(' ', '').lower()
                
                if (evidence_clean in chunk_text_clean or 
                    chunk_text_clean in evidence_clean):
                    source_chunk_id = chunk.chunk_id if hasattr(chunk, 'chunk_id') else chunk.get("chunk_id")
                    source_paragraph_id = chunk.parent_chunk_id if hasattr(chunk, 'parent_chunk_id') else chunk.get("parent_chunk_id")
                    break
        
        # Create Fact object
        fact = Fact(
            key=fact_dict.get("key", ""),
            value=fact_dict.get("value", ""),
            category=fact_dict.get("category", "Definition"),
            evidence_snippet=evidence,
            source_chunk_id=source_chunk_id,
            source_paragraph_id=source_paragraph_id,
            source_block_id=block_id,
            source_span_id=span_id,
            created_at=datetime.now().isoformat() + "Z"
        )
        
        return fact if fact.key and fact.value else None
    
    def _heuristic_extract(
        self,
        message_text: str,
        chunks: List[Any],
        span_id: Optional[str],
        block_id: Optional[str]
    ) -> List[Fact]:
        """
        Fallback heuristic fact extraction (no LLM).
        
        Patterns:
        - Acronym: "X = Y" or "X stands for Y"
        - Definition: "X is Y" (proper noun capitalization)
        - Secret: "key", "password", "token" keywords
        """
        facts = []
        
        # Pattern 1: Acronym expansion (e.g., "HMLR = Hierarchical Memory..." or "FACT5 = Test...")
        acronym_pattern = r'([A-Z][A-Z0-9]+)\s*=\s*(.+?)(?:\.|$)'
        for match in re.finditer(acronym_pattern, message_text):
            acronym = match.group(1)
            expansion = match.group(2).strip()
            
            # Create fact dict and use _create_fact_with_chunk_link for proper linking
            fact_dict = {
                "key": acronym,
                "value": expansion,
                "category": "Acronym",
                "evidence_snippet": match.group(0)[:50]
            }
            
            fact = self._create_fact_with_chunk_link(fact_dict, chunks, span_id, block_id)
            if fact:
                facts.append(fact)
                self._save_fact(fact)
        
        # Pattern 2: "stands for" (e.g., "HMLR stands for...")
        stands_for_pattern = r'([A-Z][A-Z0-9]+)\s+stands for\s+(.+?)(?:\.|$)'
        for match in re.finditer(stands_for_pattern, message_text, re.IGNORECASE):
            acronym = match.group(1)
            expansion = match.group(2).strip()
            
            fact_dict = {
                "key": acronym,
                "value": expansion,
                "category": "Acronym",
                "evidence_snippet": match.group(0)[:50]
            }
            
            fact = self._create_fact_with_chunk_link(fact_dict, chunks, span_id, block_id)
            if fact:
                facts.append(fact)
                self._save_fact(fact)
        
        return facts
    
    def _save_fact(self, fact: Fact):
        """Persist fact to fact_store table."""
        cursor = self.storage.conn.cursor()
        cursor.execute("""
            INSERT INTO fact_store (
                key, value, category, evidence_snippet,
                source_chunk_id, source_paragraph_id, source_block_id, source_span_id,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            fact.key,
            fact.value,
            fact.category,
            fact.evidence_snippet,
            fact.source_chunk_id,
            fact.source_paragraph_id,
            fact.source_block_id,
            fact.source_span_id,
            fact.created_at
        ))
        self.storage.conn.commit()
    
    def query_facts(self, query: str, limit: int = 10) -> List[Fact]:
        """
        Query fact_store for exact keyword matches.
        
        Args:
            query: Search query (e.g., "HMLR", "API_KEY")
            limit: Maximum number of results
        
        Returns:
            List of matching Fact objects, sorted by recency
        
        Performance Target: <50ms (indexed lookup)
        """
        cursor = self.storage.conn.cursor()
        
        # Case-insensitive search on key or value
        cursor.execute("""
            SELECT 
                key, value, category, evidence_snippet,
                source_chunk_id, source_paragraph_id, source_block_id, source_span_id,
                created_at
            FROM fact_store
            WHERE key LIKE ? OR value LIKE ?
            ORDER BY created_at DESC
            LIMIT ?
        """, (f"%{query}%", f"%{query}%", limit))
        
        facts = []
        for row in cursor.fetchall():
            fact = Fact(
                key=row[0],
                value=row[1],
                category=row[2],
                evidence_snippet=row[3],
                source_chunk_id=row[4],
                source_paragraph_id=row[5],
                source_block_id=row[6],
                source_span_id=row[7],
                created_at=row[8]
            )
            facts.append(fact)
        
        return facts
    
    def get_fact_by_key(self, key: str) -> Optional[Fact]:
        """
        Get the most recent fact for an exact key match.
        
        Args:
            key: Exact key (e.g., "HMLR")
        
        Returns:
            Most recent Fact object or None
        """
        cursor = self.storage.conn.cursor()
        cursor.execute("""
            SELECT 
                key, value, category, evidence_snippet,
                source_chunk_id, source_paragraph_id, source_block_id, source_span_id,
                created_at
            FROM fact_store
            WHERE key = ?
            ORDER BY created_at DESC
            LIMIT 1
        """, (key,))
        
        row = cursor.fetchone()
        if not row:
            return None
        
        return Fact(
            key=row[0],
            value=row[1],
            category=row[2],
            evidence_snippet=row[3],
            source_chunk_id=row[4],
            source_paragraph_id=row[5],
            source_block_id=row[6],
            source_span_id=row[7],
            created_at=row[8]
        )
    
    def get_facts_by_category(self, category: str, limit: int = 50) -> List[Fact]:
        """
        Get all facts in a category.
        
        Args:
            category: Fact category (Definition, Acronym, Secret, Entity)
            limit: Maximum number of results
        
        Returns:
            List of Fact objects
        """
        cursor = self.storage.conn.cursor()
        cursor.execute("""
            SELECT 
                key, value, category, evidence_snippet,
                source_chunk_id, source_paragraph_id, source_block_id, source_span_id,
                created_at
            FROM fact_store
            WHERE category = ?
            ORDER BY created_at DESC
            LIMIT ?
        """, (category, limit))
        
        facts = []
        for row in cursor.fetchall():
            fact = Fact(
                key=row[0],
                value=row[1],
                category=row[2],
                evidence_snippet=row[3],
                source_chunk_id=row[4],
                source_paragraph_id=row[5],
                source_block_id=row[6],
                source_span_id=row[7],
                created_at=row[8]
            )
            facts.append(fact)
        
        return facts
