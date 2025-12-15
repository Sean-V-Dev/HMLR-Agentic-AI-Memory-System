"""
Manual Gardener - Moves Bridge Blocks to Long-term Memory

The Gardener's Job:
1. Take completed Bridge Block (topic conversation)
2. Hierarchically chunk: Turn → Paragraph → Sentence
3. Embed each level with proper IDs
4. Extract global meta-tags from ENTIRE topic
5. Store in long-term memory with tags

Design Philosophy:
- Sentence/Paragraph: Store verbatim text (low token count)
- Turn: Store summary if >250 tokens, else verbatim
- Global Tags: LLM reads entire topic, extracts rules/constraints
- Tag Adhesion: Tags stick to ALL retrieved pieces from this topic
"""

import re
import json
from typing import List, Dict, Any, Tuple
from datetime import datetime
from dataclasses import dataclass

@dataclass
class Chunk:
    """Hierarchical chunk with ID and verbatim text"""
    chunk_id: str
    chunk_type: str  # 'sentence', 'paragraph', 'turn'
    text: str  # Verbatim or summary
    parent_id: str = None
    token_count: int = 0


class HierarchicalChunker:
    """
    Creates hierarchical chunk structure:
    turn_001
      ├─ paragraph_001
      │   ├─ sentence_001
      │   └─ sentence_002
      └─ paragraph_002
          ├─ sentence_003
          └─ sentence_004
    """
    
    TURN_SUMMARY_THRESHOLD = 250  # tokens
    
    def __init__(self):
        self.chunks: List[Chunk] = []
    
    def chunk_turn(self, turn_id: str, user_message: str, assistant_response: str) -> List[Chunk]:
        """
        Chunk a single turn hierarchically.
        
        Args:
            turn_id: Turn identifier (e.g., 't_20251204_001')
            user_message: User's query
            assistant_response: AI's response
        
        Returns:
            List of hierarchical chunks
        """
        chunks = []
        
        # Combine user + AI for full turn text
        full_turn = f"User: {user_message}\n\nAssistant: {assistant_response}"
        turn_tokens = self._estimate_tokens(full_turn)
        
        # Split assistant response into paragraphs
        paragraphs = self._split_paragraphs(assistant_response)
        
        # Process each paragraph
        for p_idx, para_text in enumerate(paragraphs):
            para_id = f"{turn_id}_p{p_idx:03d}"
            
            # Split paragraph into sentences
            sentences = self._split_sentences(para_text)
            
            # Create sentence chunks
            for s_idx, sent_text in enumerate(sentences):
                sent_id = f"{para_id}_s{s_idx:03d}"
                chunks.append(Chunk(
                    chunk_id=sent_id,
                    chunk_type='sentence',
                    text=sent_text.strip(),
                    parent_id=para_id,
                    token_count=self._estimate_tokens(sent_text)
                ))
            
            # Create paragraph chunk (verbatim)
            chunks.append(Chunk(
                chunk_id=para_id,
                chunk_type='paragraph',
                text=para_text.strip(),
                parent_id=turn_id,
                token_count=self._estimate_tokens(para_text)
            ))
        
        # Create turn chunk (summary if >250 tokens, else verbatim)
        if turn_tokens > self.TURN_SUMMARY_THRESHOLD:
            turn_text = "[SUMMARY NEEDED]"  # Gardener will generate this
        else:
            turn_text = full_turn
        
        chunks.append(Chunk(
            chunk_id=turn_id,
            chunk_type='turn',
            text=turn_text,
            parent_id=None,
            token_count=turn_tokens
        ))
        
        return chunks
    
    def _split_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs."""
        # Split on double newlines or more
        paragraphs = re.split(r'\n\n+', text)
        return [p.strip() for p in paragraphs if p.strip()]
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split paragraph into sentences."""
        # Simple sentence splitter (handles periods, question marks, exclamation points)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimate (1 token ≈ 4 chars)."""
        return len(text) // 4


class ManualGardener:
    """
    Manual Gardener for Test 8.
    
    Takes a Bridge Block and processes it into long-term memory with:
    - Hierarchical embeddings
    - Global meta-tags
    """
    
    def __init__(self, storage, embedding_storage, llm_client, dossier_governor=None, dossier_storage=None):
        """
        Initialize gardener.
        
        Args:
            storage: Storage instance
            embedding_storage: EmbeddingStorage instance
            llm_client: LLM client for meta-tag extraction
            dossier_governor: DossierGovernor instance (Phase 3+)
            dossier_storage: DossierEmbeddingStorage instance (Phase 3+)
        """
        self.storage = storage
        self.embedding_storage = embedding_storage
        self.llm_client = llm_client
        self.chunker = HierarchicalChunker()
        self.dossier_governor = dossier_governor
        self.dossier_storage = dossier_storage
    
    def process_bridge_block(self, block_id: str) -> Dict[str, Any]:
        """
        Process a Bridge Block into long-term memory.
        
        Steps:
        1. Load block from daily_ledger
        2. Chunk hierarchically
        3. Generate summaries for large turns
        4. Extract global meta-tags from entire topic
        5. Embed all chunks
        6. Store in long-term memory
        
        Args:
            block_id: Bridge Block ID
        
        Returns:
            Processing summary with stats
        """
        print(f"\n🌱 Gardener: Processing Block {block_id}")
        
        # 1. Load Bridge Block
        block_data = self._load_bridge_block(block_id)
        if not block_data:
            print(f"   ❌ Block not found")
            return {"status": "error", "message": "Block not found"}
        
        topic_label = block_data.get('topic_label', 'Unknown Topic')
        turns = block_data.get('turns', [])
        
        print(f"   📋 Topic: {topic_label}")
        print(f"   📝 Turns: {len(turns)}")
        
        # 2. Chunk all turns hierarchically
        print(f"\n   🔪 Chunking turns...")
        all_chunks = []
        
        for turn in turns:
            turn_id = turn.get('turn_id', 'unknown')
            user_msg = turn.get('user_message', '')
            ai_resp = turn.get('ai_response', '')
            
            chunks = self.chunker.chunk_turn(turn_id, user_msg, ai_resp)
            all_chunks.extend(chunks)
            
            # Count chunks by type
            sentences = sum(1 for c in chunks if c.chunk_type == 'sentence')
            paragraphs = sum(1 for c in chunks if c.chunk_type == 'paragraph')
            print(f"      {turn_id}: {paragraphs} paragraphs, {sentences} sentences")
        
        print(f"   ✅ Total chunks: {len(all_chunks)}")
        
        # 3. Generate summaries for large turns
        print(f"\n   📝 Generating turn summaries...")
        for chunk in all_chunks:
            if chunk.chunk_type == 'turn' and chunk.text == "[SUMMARY NEEDED]":
                # Find all child chunks
                turn_paragraphs = [c for c in all_chunks 
                                  if c.chunk_type == 'paragraph' and c.parent_id == chunk.chunk_id]
                full_text = "\n\n".join([p.text for p in turn_paragraphs])
                
                # Generate summary
                summary = self._generate_turn_summary(full_text, chunk.chunk_id)
                chunk.text = summary
                print(f"      ✅ {chunk.chunk_id}: {len(summary)} chars")
        
        # 4. Extract global meta-tags
        print(f"\n   🏷️  Extracting global meta-tags from entire topic...")
        full_topic_text = self._reconstruct_full_topic(block_data)
        global_tags = self._extract_global_tags(full_topic_text, topic_label)
        
        print(f"   ✅ Extracted {len(global_tags)} global tags:")
        for tag in global_tags:
            print(f"      • {tag['type']}: {tag['value']}")
        
        # 5. Embed all chunks
        print(f"\n   🔍 Creating embeddings...")
        embedding_count = 0
        
        for chunk in all_chunks:
            # Embed the verbatim text
            num_embeddings = self.embedding_storage.save_turn_embeddings(
                chunk.chunk_id, 
                [chunk.text]  # Single chunk containing verbatim text
            )
            embedding_count += num_embeddings
        
        print(f"   ✅ Created {embedding_count} embeddings")
        
        # 6. Store chunks with global tags in long-term memory
        print(f"\n   💾 Storing in long-term memory...")
        self._store_chunks_with_tags(block_id, all_chunks, global_tags)
        print(f"   ✅ Stored {len(all_chunks)} chunks with {len(global_tags)} global tags")
        
        # 7. Delete processed bridge block from active memory
        self._delete_bridge_block(block_id)
        
        print(f"\n✅ Gardener: Block {block_id} processed successfully!")
        
        return {
            "status": "success",
            "block_id": block_id,
            "topic_label": topic_label,
            "chunks_created": len(all_chunks),
            "embeddings_created": embedding_count,
            "global_tags": len(global_tags),
            "tags": global_tags
        }
    
    def _load_bridge_block(self, block_id: str) -> Dict[str, Any]:
        """Load Bridge Block from daily_ledger."""
        cursor = self.storage.conn.cursor()
        cursor.execute("""
            SELECT content_json FROM daily_ledger 
            WHERE block_id = ?
        """, (block_id,))
        
        row = cursor.fetchone()
        if not row:
            return None
        
        return json.loads(row[0])
    
    def _generate_turn_summary(self, full_text: str, turn_id: str) -> str:
        """
        Generate summary for a large turn (>250 tokens).
        
        Args:
            full_text: Full turn text
            turn_id: Turn identifier
        
        Returns:
            Summary text
        """
        prompt = f"""Summarize this conversation turn concisely (2-3 sentences):

{full_text}

Summary:"""
        
        try:
            response = self.llm_client.query_external_api(prompt, model="gpt-4.1-mini")
            return response.strip()
        except Exception as e:
            print(f"      ⚠️  Summary generation failed for {turn_id}: {e}")
            return full_text[:500] + "..."  # Fallback truncation
    
    def _reconstruct_full_topic(self, block_data: Dict[str, Any]) -> str:
        """Reconstruct full topic conversation for meta-tag extraction."""
        turns = block_data.get('turns', [])
        full_text = f"Topic: {block_data.get('topic_label', 'Unknown')}\n\n"
        
        for turn in turns:
            full_text += f"User: {turn.get('user_message', '')}\n"
            full_text += f"Assistant: {turn.get('ai_response', '')}\n\n"
        
        return full_text
    
    def _extract_global_tags(self, full_topic_text: str, topic_label: str) -> List[Dict[str, str]]:
        """
        Extract global meta-tags from entire topic.
        
        These tags apply to ALL chunks from this topic.
        
        Args:
            full_topic_text: Full conversation text
            topic_label: Topic name
        
        Returns:
            List of global tags
        """
        prompt = f"""Read this entire conversation and extract global meta-tags that apply to the ENTIRE topic.

Focus on:
- Global rules/policies mentioned
- Deprecations/safety warnings
- Project constraints
- Key decisions made
- Important facts that should "stick" to any retrieved piece

Conversation:
{full_topic_text}

Return JSON array of tags in format:
{{"type": "global_rule|deprecation|constraint|decision|fact", "value": "brief statement"}}

Tags:"""
        
        try:
            response = self.llm_client.query_external_api(prompt, model="gpt-4.1-mini")
            
            # Extract JSON
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                tags = json.loads(json_match.group(0))
                return tags
            else:
                print(f"      ⚠️  No JSON found in tag extraction response")
                return []
        
        except Exception as e:
            print(f"      ⚠️  Tag extraction failed: {e}")
            return []
    
    def _store_chunks_with_tags(self, block_id: str, chunks: List[Chunk], global_tags: List[Dict[str, str]]):
        """
        Store chunks and global tags in long-term memory.
        
        Note: For now, storing in a simple table. In full HMLR v2, this would go to
        spans/summaries tables with proper hierarchical structure.
        """
        cursor = self.storage.conn.cursor()
        
        # Create gardened_memory table if not exists
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS gardened_memory (
                chunk_id TEXT PRIMARY KEY,
                block_id TEXT NOT NULL,
                chunk_type TEXT NOT NULL,
                text_content TEXT NOT NULL,
                parent_id TEXT,
                token_count INTEGER,
                global_tags TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Store each chunk with global tags
        tags_json = json.dumps(global_tags)
        
        for chunk in chunks:
            cursor.execute("""
                INSERT OR REPLACE INTO gardened_memory
                (chunk_id, block_id, chunk_type, text_content, parent_id, token_count, global_tags)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                chunk.chunk_id,
                block_id,
                chunk.chunk_type,
                chunk.text,
                chunk.parent_id,
                chunk.token_count,
                tags_json
            ))
        
        self.storage.conn.commit()
    
    def _delete_bridge_block(self, block_id: str):
        """
        Delete processed bridge block from daily_ledger.
        
        Args:
            block_id: Bridge block ID to delete
        """
        cursor = self.storage.conn.cursor()
        cursor.execute("DELETE FROM daily_ledger WHERE block_id = ?", (block_id,))
        self.storage.conn.commit()
        print(f"   🗑️  Deleted bridge block {block_id} from active memory")
    
    async def _group_facts_semantically(self, facts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Group related facts by semantic theme using LLM.
        
        This is Phase 2 preparation for Phase 3's dossier routing.
        Groups facts that belong together conceptually, preparing them
        for the DossierGovernor to decide which dossier they should join.
        
        Args:
            facts: List of fact dictionaries with 'text' and 'turn_id' fields
        
        Returns:
            List of fact groups: [{"label": "...", "facts": [...], "timestamp": "..."}]
        
        Example:
            Input: [
                {"text": "User is vegetarian", "turn_id": "turn_001"},
                {"text": "User avoids meat", "turn_id": "turn_001"},
                {"text": "User works with Python", "turn_id": "turn_002"}
            ]
            Output: [
                {
                    "label": "Dietary Preferences",
                    "facts": ["User is vegetarian", "User avoids meat"],
                    "timestamp": "2025-12-15T10:30:00"
                },
                {
                    "label": "Programming",
                    "facts": ["User works with Python"],
                    "timestamp": "2025-12-15T10:31:00"
                }
            ]
        """
        if not facts:
            return []
        
        # Format facts for LLM
        facts_text = json.dumps(facts, indent=2)
        
        prompt = f"""Given these facts extracted from a conversation, group related facts by semantic theme.

Facts:
{facts_text}

For each group, provide:
1. A concise label (2-5 words) describing the theme
2. The facts that belong to that group
3. The earliest timestamp from facts in the group

Return as JSON array:
[
  {{
    "label": "Theme Name",
    "facts": ["fact text 1", "fact text 2"],
    "timestamp": "ISO timestamp"
  }}
]

Groups:"""
        
        try:
            response = await self.llm_client.query_external_api(
                prompt=prompt,
                model="gpt-4.1-mini"
            )
            
            # Extract JSON from response
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                groups = json.loads(json_match.group(0))
                print(f"   📦 Grouped {len(facts)} facts into {len(groups)} semantic clusters")
                return groups
            else:
                print(f"   ⚠️  No JSON found in grouping response, creating single group")
                # Fallback: put all facts in one group
                return [{
                    "label": "General Facts",
                    "facts": [f['text'] for f in facts],
                    "timestamp": facts[0].get('timestamp', datetime.now().isoformat())
                }]
        
        except Exception as e:
            print(f"   ⚠️  Semantic grouping failed: {e}, creating single group")
            return [{
                "label": "General Facts",
                "facts": [f['text'] for f in facts],
                "timestamp": facts[0].get('timestamp', datetime.now().isoformat())
            }]
