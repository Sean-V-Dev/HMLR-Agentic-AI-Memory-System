"""
HMLR v1 - Hydrator

The Hydrator is responsible for:
1. Taking a list of approved memory IDs (from The Governor).
2. Fetching the full content of these memories.
3. Hydrating Bridge Blocks from daily_ledger:
   - Active Bridge Block (current topic): Load full conversation turns (verbatim)
   - Inactive Bridge Blocks (other topics): Include lightweight metadata summaries
4. Enforcing the token budget (Context Window Management).
5. Formatting the final context string for the LLM.
"""

import logging
import json
from typing import List, Dict, Any, Optional
from hmlr.memory.storage import Storage
from hmlr.memory.models import ConversationTurn

logger = logging.getLogger(__name__)

class Hydrator:
    def __init__(self, storage: Storage, token_limit: int = 2000):
        self.storage = storage
        self.token_limit = token_limit

    def hydrate(self, approved_memory_ids: List[str], query: Optional[str] = None) -> List[ConversationTurn]:
        """
        Fetch full content for approved IDs.
        
        Hydrating Bridge Blocks from daily_ledger:
        - Active Bridge Block (current topic): Load full conversation turns (verbatim)
        - Inactive Bridge Blocks (other topics): Include lightweight metadata summaries
        
        Args:
            approved_memory_ids: List of memory IDs approved by Governor
            query: Optional user query for determining active Bridge Block
        
        Returns:
            List of ConversationTurn objects (includes hydrated bridge blocks)
        """
        hydrated_memories = []
        bridge_blocks = []
        
        for mem_id in approved_memory_ids:
            # Check if this is a Bridge Block ID
            if mem_id.startswith('bridge_block_') or mem_id.startswith('bb_'):
                # Extract Bridge Block from daily_ledger
                bridge_block = self._get_bridge_block(mem_id)
                if bridge_block:
                    bridge_blocks.append(bridge_block)
                else:
                    logger.warning(f"Hydrator could not find Bridge Block: {mem_id}")
            else:
                # Standard turn retrieval
                turn = self.storage.get_turn_by_id(mem_id)
                if turn:
                    hydrated_memories.append(turn)
                else:
                    logger.warning(f"Hydrator could not find memory ID: {mem_id}")
        
        # Hydrate Bridge Blocks
        if bridge_blocks:
            active_block, inactive_blocks = self._identify_active_block(bridge_blocks, query)
            
            # Debug
            logger.debug(f"Active block: {active_block['block_id'] if active_block else None}")
            logger.debug(f"Inactive blocks: {[b['block_id'] for b in inactive_blocks]}")
            logger.debug(f"Query: {query}")
            
            # Hydrate active block (full conversation turns)
            if active_block:
                logger.info(f"Hydrating active Bridge Block: {active_block['block_id']}")
                active_turns = self._hydrate_bridge_block_verbatim(active_block)
                hydrated_memories.extend(active_turns)
            
            # Add metadata summaries for inactive blocks (lightweight)
            for block in inactive_blocks:
                logger.info(f"Adding metadata for inactive Bridge Block: {block['block_id']}")
                metadata_turn = self._create_metadata_placeholder(block)
                hydrated_memories.append(metadata_turn)
        
        return hydrated_memories
    
    def _get_bridge_block(self, block_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a Bridge Block from daily_ledger by ID.
        
        Args:
            block_id: Bridge Block ID
        
        Returns:
            Bridge Block dict with content, metadata, etc.
        """
        cursor = self.storage.conn.cursor()
        cursor.execute("""
            SELECT block_id, content_json, span_id, created_at, status, exit_reason
            FROM daily_ledger
            WHERE block_id = ?
        """, (block_id,))
        
        row = cursor.fetchone()
        if not row:
            return None
        
        try:
            content = json.loads(row[1])
            return {
                'block_id': row[0],
                'content': content,
                'span_id': row[2],
                'created_at': row[3],
                'status': row[4],
                'exit_reason': row[5]
            }
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse Bridge Block {row[0]}: {e}")
            return None
    
    def _identify_active_block(
        self, 
        bridge_blocks: List[Dict[str, Any]], 
        query: Optional[str]
    ) -> tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Identify which Bridge Block is active (matches current query).
        
        Strategy:
        1. If query provided, match against topic_label and keywords
        2. If no match, use most recent block as active
        3. All others are inactive
        
        Args:
            bridge_blocks: List of Bridge Block dicts
            query: User query (optional)
        
        Returns:
            (active_block, inactive_blocks) tuple
        """
        if not bridge_blocks:
            return None, []
        
        # Sort by created_at (most recent first)
        sorted_blocks = sorted(
            bridge_blocks, 
            key=lambda b: b['created_at'], 
            reverse=True
        )
        
        # If no query, use most recent as active
        if not query:
            return sorted_blocks[0], sorted_blocks[1:]
        
        # Try to match query against block topics/keywords
        query_lower = query.lower()
        for block in sorted_blocks:
            content = block.get('content', {})
            topic_label = content.get('topic_label', '').lower()
            keywords = [kw.lower() for kw in content.get('keywords', [])]
            
            # Check if query matches topic or keywords
            if topic_label and topic_label in query_lower:
                return block, [b for b in sorted_blocks if b != block]
            
            if any(kw in query_lower for kw in keywords):
                return block, [b for b in sorted_blocks if b != block]
        
        # No match found - use most recent as active
        return sorted_blocks[0], sorted_blocks[1:]
    
    def _hydrate_bridge_block_verbatim(self, bridge_block: Dict[str, Any]) -> List[ConversationTurn]:
        """
        Hydrate a Bridge Block with full conversation turns (verbatim).
        
        Args:
            bridge_block: Bridge Block dict with span_id
        
        Returns:
            List of ConversationTurn objects from the span
        """
        span_id = bridge_block.get('span_id')
        if not span_id:
            logger.warning(f"Bridge Block {bridge_block['block_id']} has no span_id")
            return []
        
        # Get span to retrieve turn_ids
        span = self.storage.get_span(span_id)
        if not span:
            logger.warning(f"Span {span_id} not found for Bridge Block {bridge_block['block_id']}")
            return []
        
        # Retrieve all turns in the span
        turns = []
        for turn_id in span.turn_ids:
            turn = self.storage.get_turn_by_id(turn_id)
            if turn:
                turns.append(turn)
            else:
                logger.warning(f"Turn {turn_id} not found in span {span_id}")
        
        logger.info(f"Hydrated {len(turns)} turns from Bridge Block {bridge_block['block_id']}")
        return turns
    
    def _create_metadata_placeholder(self, bridge_block: Dict[str, Any]) -> ConversationTurn:
        """
        Create a lightweight metadata placeholder for an inactive Bridge Block.
        
        This allows the LLM to be aware of other same-day topics without
        loading full conversation history (saves context window).
        
        Args:
            bridge_block: Bridge Block dict
        
        Returns:
            ConversationTurn with metadata summary
        """
        from datetime import datetime
        
        content = bridge_block.get('content', {})
        topic_label = content.get('topic_label', 'Unknown Topic')
        summary = content.get('summary', 'No summary available')[:200]  # Truncate
        open_loops = content.get('open_loops', [])[:3]  # First 3 loops
        decisions = content.get('decisions_made', [])[:3]  # First 3 decisions
        keywords = content.get('keywords', [])[:5]  # First 5 keywords
        
        # Format metadata as a compact summary
        metadata_text = f"""[Bridge Block: {topic_label}]
Summary: {summary}
Keywords: {', '.join(keywords)}
Open Loops: {', '.join(open_loops) if open_loops else 'None'}
Decisions: {', '.join(decisions) if decisions else 'None'}
Status: {bridge_block.get('status', 'UNKNOWN')}"""
        
        # Create a pseudo-turn for the metadata
        return ConversationTurn(
            turn_id=bridge_block['block_id'],  # Use block_id as turn_id
            session_id="bridge_block_metadata",
            day_id=content.get('span_id', 'unknown')[:15],  # Extract day from span
            timestamp=datetime.fromisoformat(bridge_block['created_at']),
            turn_sequence=0,
            user_message=f"[Topic Reference: {topic_label}]",
            assistant_response=metadata_text,
            span_id=bridge_block.get('span_id', '')
        )

    def build_context_string(self, memories: List[ConversationTurn]) -> str:
        """
        Format hydrated memories into a context string.
        Enforces token budget.
        """
        current_tokens = 0
        
        # Sort memories by timestamp (oldest first) for coherent reading
        memories.sort(key=lambda x: x.timestamp)
        
        retrieved_text = "### Relevant Past Memories:\n"
        added_count = 0
        
        for mem in memories:
            # Check if this is a Bridge Block metadata placeholder
            is_metadata = (mem.session_id == "bridge_block_metadata")
            
            if is_metadata:
                # Compact format for inactive Bridge Blocks
                entry = f"\n{mem.assistant_response}\n\n"
            else:
                # Standard format for conversation turns
                entry = f"[{mem.timestamp.strftime('%Y-%m-%d %H:%M')}] [ref:{mem.turn_id}]\nUser: {mem.user_message}\nAI: {mem.assistant_response}\n\n"
            
            # Simple char count approximation: 1 token ~= 4 chars
            entry_tokens = len(entry) / 4
            
            if current_tokens + entry_tokens > self.token_limit:
                logger.info(f"Hydrator hit token limit ({self.token_limit}). Stopped adding memories.")
                break
            
            retrieved_text += entry
            current_tokens += entry_tokens
            added_count += 1
            
        if added_count == 0:
            return ""

        return retrieved_text
