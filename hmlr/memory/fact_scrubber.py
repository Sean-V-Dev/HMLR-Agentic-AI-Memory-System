"""

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
import logging
import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from hmlr.core.model_config import model_config

logger = logging.getLogger(__name__)

from hmlr.memory.storage import Storage


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
    source_turn_id: Optional[str] = None
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
5. Task - A task, plan, or reminder the user intends to complete

RULES:
- Ignore general conversation or opinions
- Extract only verifiable, referenceable facts
- For acronyms, include the full expansion
- For secrets, include the key/value pair
- For entities, include the relationship type
- Extract a Task only if the message clearly expresses the user's own intent or request
- The task must involve an action the user plans, needs, or wants to do
- If a time/date is mentioned for a task, include it in the task value
- Ignore hypothetical examples, third-party statements, or quoted speech

TEMPORAL INFORMATION (CRITICAL):
- If the message contains explicit dates, timestamps, or temporal markers (e.g., "[Date: 2023/01/05]", "on Monday", "last week", "in March 2022"), include the temporal context in the fact value
- For events, appointments, or actions, always capture WHEN they happened if a date/time is mentioned
- Examples:
  * "User visited their parents" → "User visited the art museum on 2023/01/05"
  * "Meeting scheduled" → "Meeting scheduled for March 20, 2023 at 2:00 pm"
  * "Started project X" → "Started project X in January 2023"
  * "Remind me to set an appointment" → "User needs to set an appointment with their doctor next week"
  * "I plan to renew my license in March" → "User plans to renew their license in March"
- The temporal information is part of the fact itself, not separate metadata

MESSAGE:
{message}

Return JSON in this exact format:
{{
  "facts": [
    {{
      "key": "concise identifier (2-4 words)",
      "value": "the fact itself INCLUDING temporal information if present (complete sentence or phrase)",
      "category": "Definition|Acronym|Secret|Entity|Task",
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
    
    def _estimate_tokens(self, text: str) -> int:
        """Quick token estimation (4 chars ≈ 1 token)."""
        return len(text) // 4
    
    def _chunk_large_text_for_extraction(
        self,
        text: str,
        chunk_size_tokens: int = 10000,
        overlap_tokens: int = 500
    ) -> List[Dict[str, Any]]:
        """
        Split large text into chunks for fact extraction.
        
        Strategy: Each chunk = 10k content tokens. Overlap = 500 tokens between chunks.
        Tax: ~5% for documents >10k tokens (500 overlap / 10k ≈ 5%)
        Threshold: Only chunk if >10k tokens (keeps overhead low for normal messages)
        
        Args:
            text: Full text to chunk
            chunk_size_tokens: Target content size per chunk (10k tokens)
            overlap_tokens: Overlap between chunks (500 for context)
        
        Returns:
            List of chunk dicts with 'text', 'start_char', 'end_char', 'chunk_index'
        """
        estimated_tokens = self._estimate_tokens(text)
        
        # Only chunk if >10k tokens (keeps overhead low)
        if estimated_tokens <= 10000:
            return [{
                'text': text,
                'start_char': 0,
                'end_char': len(text),
                'chunk_index': 0,
                'total_chunks': 1
            }]
        
        # Convert token counts to character counts (rough estimate)
        chunk_size_chars = chunk_size_tokens * 4
        overlap_chars = overlap_tokens * 4
        
        chunks = []
        start_char = 0
        chunk_index = 0
        
        while start_char < len(text):
            end_char = min(start_char + chunk_size_chars, len(text))
            
            # Try to break at sentence boundary (. ! ?) if not at end
            if end_char < len(text):
                # Look back up to 500 chars for a sentence boundary
                search_start = max(end_char - 500, start_char)
                last_period = text.rfind('.', search_start, end_char)
                last_exclaim = text.rfind('!', search_start, end_char)
                last_question = text.rfind('?', search_start, end_char)
                
                boundary = max(last_period, last_exclaim, last_question)
                if boundary > search_start:
                    end_char = boundary + 1  # Include the punctuation
            
            chunk_text = text[start_char:end_char]
            
            chunks.append({
                'text': chunk_text,
                'start_char': start_char,
                'end_char': end_char,
                'chunk_index': chunk_index,
                'total_chunks': None  # Will update after loop
            })
            
            # Move to next chunk with overlap (unless this is the last chunk)
            if end_char >= len(text):
                break
                
            start_char = end_char - overlap_chars
            chunk_index += 1
        
        # Update total_chunks count
        total = len(chunks)
        for chunk in chunks:
            chunk['total_chunks'] = total
        
        logger.info(f"Split {estimated_tokens:,} token text into {total} chunks (~{estimated_tokens//total:,} tokens each)")
        return chunks
    
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
        
        For large texts (>10k tokens), automatically chunks the text with 500-token
        overlap to prevent JSON truncation while keeping overhead under 10%.
        
        Args:
            turn_id: Turn identifier
            message_text: User message text
            chunks: List of sentence chunks from ChunkEngine (contains chunk_id, text, parent)
            span_id: Current span ID (optional)
            block_id: Bridge block ID if archived (optional)
        
        Returns:
            List of extracted Fact objects
        
        Performance Target: <500ms per chunk (parallel, non-blocking)
        """
        if not self.api_client:
            return self._heuristic_extract(message_text, chunks, span_id, block_id)
        
        try:
            # Split text into chunks if >10k tokens
            text_chunks = self._chunk_large_text_for_extraction(message_text)
            
            if len(text_chunks) > 1:
                logger.info(f"Extracting facts from {len(text_chunks)} chunks in parallel")
            
            # Create extraction tasks for all chunks (parallel execution)
            async def extract_from_chunk(text_chunk):
                chunk_text = text_chunk['text']
                chunk_idx = text_chunk['chunk_index']
                total_chunks = text_chunk['total_chunks']
                
                if total_chunks > 1:
                    logger.debug(f"Processing chunk {chunk_idx + 1}/{total_chunks} ({self._estimate_tokens(chunk_text):,} tokens)")
                
                # Call LLM for fact extraction
                prompt = self.EXTRACTION_PROMPT.format(message=chunk_text)
                
                response_content = await self.api_client.query_external_api_async(
                    query=prompt,
                    model=model_config.get_synthesis_model(),
                    max_tokens=model_config.FACT_EXTRACTION_MAX_TOKENS
                )
                
                # Parse JSON response
                return self._parse_llm_response(response_content)
            
            # Execute all chunks in parallel
            results = await asyncio.gather(*[extract_from_chunk(chunk) for chunk in text_chunks])
            
            # Process results and deduplicate
            all_facts = []
            seen_facts = set()  # (key, value) tuples for deduplication
            
            for facts_data in results:
                for fact_dict in facts_data.get("facts", []):
                    fact = self._create_fact_with_chunk_link(
                        fact_dict, chunks, span_id, block_id, turn_id
                    )
                    if fact:
                        fact_key = (fact.key, fact.value)
                        if fact_key not in seen_facts:
                            seen_facts.add(fact_key)
                            self._save_fact(fact)
                            all_facts.append(fact)
            
            if len(text_chunks) > 1:
                logger.info(f"Extracted {len(all_facts)} unique facts from {len(text_chunks)} chunks")
            
            return all_facts
        
        except Exception as e:
            logger.warning(f"LLM extraction failed: {e}, using fallback", exc_info=True)
            return self._heuristic_extract(message_text, chunks, span_id, block_id)
    
    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM JSON response, handling markdown code blocks and malformed output."""
        # Strip markdown code blocks if present
        response = response.strip()
        if response.startswith("```"):
            lines = response.split("\n")
            response = "\n".join(lines[1:-1])  # Remove first and last line
        
        try:
            return json.loads(response)
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse error: {e}")
            logger.debug(f"Failed response length: {len(response)} chars")
            logger.debug(f"Failed response (first 1000 chars): {response[:1000]}")
            logger.debug(f"Failed response (last 500 chars): {response[-500:]}")
            
            # Attempt to extract partial facts array using regex
            try:
                import re
                # Look for facts array pattern
                facts_match = re.search(r'"facts"\s*:\s*\[(.*?)\]', response, re.DOTALL)
                if facts_match:
                    facts_json = '[' + facts_match.group(1) + ']'
                    # Try to parse just the facts array
                    facts_array = json.loads(facts_json)
                    logger.info(f"Recovered {len(facts_array)} facts from partial JSON")
                    return {"facts": facts_array}
            except Exception as recovery_error:
                logger.debug(f"Recovery attempt failed: {recovery_error}")
            
            # Last resort: return empty facts array
            logger.warning("Returning empty facts array due to unrecoverable JSON error")
            return {"facts": []}
    
    def _create_fact_with_chunk_link(
        self,
        fact_dict: Dict[str, Any],
        chunks: List[Any],
        span_id: Optional[str],
        block_id: Optional[str],
        turn_id: Optional[str] = None
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
            source_turn_id=turn_id,
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
                source_chunk_id, source_paragraph_id, source_block_id, 
                source_turn_id, source_span_id, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            fact.key,
            fact.value,
            fact.category,
            fact.evidence_snippet,
            fact.source_chunk_id,
            fact.source_paragraph_id,
            fact.source_block_id,
            fact.source_turn_id,
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
                source_chunk_id, source_paragraph_id, source_block_id, 
                source_turn_id, source_span_id, created_at
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
                source_turn_id=row[7],
                source_span_id=row[8],
                created_at=row[9]
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
                source_chunk_id, source_paragraph_id, source_block_id, 
                source_turn_id, source_span_id, created_at
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
            source_turn_id=row[7],
            source_span_id=row[8],
            created_at=row[9]
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
                source_chunk_id, source_paragraph_id, source_block_id, 
                source_turn_id, source_span_id, created_at
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
                source_turn_id=row[7],
                source_span_id=row[8],
                created_at=row[9]
            )
            facts.append(fact)
        
        return facts
