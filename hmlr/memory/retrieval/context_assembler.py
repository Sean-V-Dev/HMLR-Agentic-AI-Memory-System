"""
Context Assembler - Group-by-Block Hydration

This module implements the Group-by-Block pattern for retrieving chunks with sticky meta tags.

Key Insight: Tags are stored ONCE per block in block_metadata table.
Chunks reference block_id to get their tags (pointer model, not duplication).

Hydration Pattern:
1. Retrieve chunks from various sources
2. Group chunks by block_id
3. Fetch block_metadata ONCE per block (not per chunk)
4. Format as headers with chunks underneath

Token Savings Example:

- NEW WAY (group-by-block with header):
  ### Context Block: block_55
  Active Rules: [env: python-3.9], [os: windows]
  
  - "Run the command"
  - "Check the logs"
  - "Wait for confirmation"
  Cost: Tags paid ONCE = 1/3rd token cost

"""

from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class ContextAssembler:
    """
    Assembles context from retrieved chunks using Group-by-Block pattern.
    """
    
    def __init__(self, storage):
        """
        Initialize context assembler.
        
        Args:
            storage: Storage instance for fetching block metadata
        """
        self.storage = storage
    
    def hydrate_chunks_with_metadata(self, chunks: List[Dict[str, Any]], include_headers: bool = True) -> str:

        if not chunks:
            return ""
        
        # Group chunks by block_id
        blocks = {}
        for chunk in chunks:
            block_id = chunk.get('block_id')
            if not block_id:
                # Chunk without block_id, add to "untagged" group
                block_id = "_untagged"
            
            if block_id not in blocks:
                blocks[block_id] = {
                    'metadata': None,  # Will fetch lazily
                    'chunks': []
                }
            blocks[block_id]['chunks'].append(chunk)
        
        # Build context string
        context_parts = []
        
        for block_id, data in blocks.items():
            if block_id == "_untagged":
                # No metadata for untagged chunks
                context_parts.append("\n### Untagged Context")
                for chunk in data['chunks']:
                    context_parts.append(f"  {chunk.get('text', '')}")
                context_parts.append("")
                continue
            
            # Fetch metadata ONCE per block (lazy loading)
            if data['metadata'] is None:
                data['metadata'] = self.storage.get_block_metadata(block_id)
            
            metadata = data['metadata']
            
            # Header with block ID and tags (ONCE per block)
            if include_headers:
                context_parts.append(f"\n### Context Block: {block_id}")
                
                # Global tags (apply to all chunks in block)
                if metadata.get('global_tags'):
                    tags_str = ', '.join([f"[{tag}]" for tag in metadata['global_tags']])
                    context_parts.append(f"Active Rules: {tags_str}")
                
                context_parts.append("")  # Blank line after header
            
            # Chunks (NO repeated tags)
            for chunk in data['chunks']:
                # Check if chunk falls in section rule range
                turn_id = chunk.get('turn_id', '')
                section_tag = self._get_section_tag_for_turn(turn_id, metadata.get('section_rules', []))
                
                # Format chunk with section tag if applicable
                chunk_text = chunk.get('text', '')
                if section_tag:
                    context_parts.append(f"  [{section_tag}] {chunk_text}")
                else:
                    context_parts.append(f"  {chunk_text}")
            
            context_parts.append("")  # Blank line between blocks
        
        return "\n".join(context_parts)
    
    def _get_section_tag_for_turn(self, turn_id: str, section_rules: List[Dict]) -> str:

        if not turn_id or not section_rules:
            return None
        
        # For each section rule, check if turn_id falls in range
        for rule in section_rules:
            start_turn = rule.get('start_turn')
            end_turn = rule.get('end_turn')
            rule_text = rule.get('rule', '')
            
            # Simple string comparison (works if turn_ids are sortable)
            # More sophisticated version would extract turn sequence numbers
            if start_turn and end_turn:
                if start_turn <= turn_id <= end_turn:
                    return rule_text
        
        return None
    
    def hydrate_dossiers_with_facts(self, dossiers: List[Dict[str, Any]]) -> str:

        if not dossiers:
            return ""
        
        context_parts = ["\n## Relevant Dossiers\n"]
        
        for dossier in dossiers:
            dossier_id = dossier.get('dossier_id', 'unknown')
            title = dossier.get('title', 'Untitled Dossier')
            summary = dossier.get('summary', 'No summary available')
            facts = dossier.get('facts', [])
            last_updated = dossier.get('last_updated', '')
            
            context_parts.append(f"### Dossier: {title}")
            context_parts.append(f"Summary: {summary}")
            context_parts.append("")
            
            if facts:
                context_parts.append("Facts:")
                for fact in facts:
                    fact_text = fact.get('fact_text', '') if isinstance(fact, dict) else fact
                    context_parts.append(f"- {fact_text}")
                context_parts.append("")
            
            if last_updated:
                context_parts.append(f"Last Updated: {last_updated}")
            
            context_parts.append("")  # Blank line between dossiers
        
        return "\n".join(context_parts)
    
    def assemble_full_context(self, 
                             chunks: List[Dict[str, Any]], 
                             dossiers: List[Dict[str, Any]],
                             max_tokens: int = 4000) -> str:
        """
        Assemble complete context from both chunks and dossiers.
        
        Args:
            chunks: Retrieved chunks (may include gardened_memory chunks)
            dossiers: Retrieved dossiers
            max_tokens: Maximum token budget for context
        
        Returns:
            Formatted context string ready for LLM
        
        Note: This is a simple implementation. A production version would:
        - Estimate tokens more accurately
        - Trim context if over budget
        - Prioritize by relevance scores
        """
        context_parts = []
        
        # Add dossiers first (highest-level context)
        if dossiers:
            dossier_context = self.hydrate_dossiers_with_facts(dossiers)
            context_parts.append(dossier_context)
        
        # Add chunks with group-by-block hydration
        if chunks:
            chunk_context = self.hydrate_chunks_with_metadata(chunks)
            context_parts.append(chunk_context)
        
        full_context = "\n".join(context_parts)
        
        # Simple token estimate (1 token ≈ 4 chars)
        estimated_tokens = len(full_context) // 4
        
        if estimated_tokens > max_tokens:
            logger.warning(f"Context exceeds token budget: {estimated_tokens} > {max_tokens}")
            # Truncate to budget (simple version - better would prioritize by relevance)
            char_limit = max_tokens * 4
            full_context = full_context[:char_limit] + "\n\n[Context truncated due to token limit]"
        
        return full_context