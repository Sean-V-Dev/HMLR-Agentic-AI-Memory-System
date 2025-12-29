"""
Dossier Retriever - Read-Side Fact Aggregation

The DossierRetriever searches for relevant dossiers and formats them for LLM context.
This is the read-side complement to DossierGovernor (write-side).

Architecture:
- Search: Fact-level embedding search with deduplication by dossier
- Retrieval: Return full dossiers (all facts) when any fact matches
- Formatting: Structured context with title, summary, and facts

"""

import json
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class DossierRetriever:
    """
    Read-side retriever for dossier search.
    
    The DossierRetriever is called during query processing to find relevant
    fact aggregations. It searches at the fact level but returns full dossiers,
    providing rich context for the LLM to generate responses.
    
    Integration:
    - Called in parallel with topic and memory retrieval
    - Results filtered by LLM governor
    - Formatted dossiers added to context window
    """
    
    def __init__(self, storage, dossier_storage):
        """
        Initialize dossier retriever.
        
        Args:
            storage: Storage instance (for retrieving full dossiers)
            dossier_storage: DossierEmbeddingStorage instance (for vector search)
        """
        self.storage = storage
        self.dossier_storage = dossier_storage
        logger.info("DossierRetriever initialized")
    
    def retrieve_relevant_dossiers(
        self,
        query: str,
        top_k: int = 3,
        threshold: float = 0.4
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant dossiers based on fact embeddings.
        
        Process:
        1. Search fact embeddings using query
        2. Deduplicate by dossier_id (multiple facts from same dossier)
        3. Get full dossier details for each matched dossier
        4. Return top K dossiers with all their facts
        
        Args:
            query: Query text to search for
            top_k: Maximum number of dossiers to return (None = return all matching)
            threshold: Minimum similarity score (0-1, default 0.4)
        
        Returns:
            List of dossier dictionaries with structure:
            [
                {
                    'dossier_id': 'dos_xxx',
                    'title': 'Dossier Title',
                    'summary': 'Summary text',
                    'facts': [
                        {'fact_id': 'fact_xxx', 'fact_text': '...', 'added_at': '...'},
                        ...
                    ],
                    'score': 0.85  # Highest similarity score from any fact
                }
            ]
        
        Example:
            retriever = DossierRetriever(storage, dossier_storage)
            dossiers = retriever.retrieve_relevant_dossiers(
                "What are the user's dietary restrictions?",
                top_k=3
            )
            # Returns vegetarian diet dossier with all related facts
        """
        logger.debug(f"Searching for dossiers matching query: '{query[:50]}...'")
        
        # 1. Search fact-level embeddings (MULTI-VECTOR VOTING)
        # Each fact match votes for its parent dossier
        fact_results = self.dossier_storage.search_similar_facts(
            query=query,
            top_k=100,  # Get many facts to enable hit-count voting
            threshold=0.4
        )
        
        if not fact_results:
            logger.debug("No matching facts found")
            return []
        
        # 2. Aggregate by dossier (Multi-Vector Voting)
        # Count how many facts from each dossier matched
        dossier_hits = {}
        dossier_max_scores = {}
        
        for fact_id, dossier_id, score in fact_results:
            if dossier_id not in dossier_hits:
                dossier_hits[dossier_id] = 0
                dossier_max_scores[dossier_id] = 0.0
            dossier_hits[dossier_id] += 1
            dossier_max_scores[dossier_id] = max(dossier_max_scores[dossier_id], score)
        
        logger.debug(f"Found {len(fact_results)} matching facts across {len(dossier_hits)} dossiers")
        
        # 3. Rank dossiers by hit count (primary) and max score (secondary)
        sorted_dossiers = sorted(
            dossier_hits.items(),
            key=lambda x: (x[1], dossier_max_scores[x[0]]),  # Sort by hits, then score
            reverse=True
        )
        
        # Apply top_k limit if specified, otherwise return all
        top_dossiers = sorted_dossiers[:top_k] if top_k else sorted_dossiers
        
        # 4. Get full dossier details
        dossiers = []
        for dossier_id, hit_count in top_dossiers:
            dossier = self.storage.get_dossier(dossier_id)
            if dossier:
                facts = self.storage.get_dossier_facts(dossier_id)
                max_score = dossier_max_scores[dossier_id]
                dossiers.append({
                    'dossier_id': dossier_id,
                    'title': dossier['title'],
                    'summary': dossier['summary'],
                    'facts': facts,  # Full fact objects with metadata
                    'hit_count': hit_count,  # How many facts matched
                    'max_similarity': max_score,  # Highest fact similarity
                    'created_at': dossier['created_at'],
                    'last_updated': dossier['last_updated']
                })
                logger.debug(f"  Retrieved dossier: {dossier['title']} ({len(facts)} facts, {hit_count} hits, max score: {max_score:.3f})")
        
        logger.info(f"Retrieved {len(dossiers)} relevant dossiers")
        return dossiers
    
    def format_for_context(self, dossiers: List[Dict[str, Any]]) -> str:
        """
        Format dossiers for LLM context window.
        
        Creates a structured, readable format that the LLM can easily parse.
        Includes title, summary, and all facts with timestamps.
        
        Args:
            dossiers: List of dossier dictionaries from retrieve_relevant_dossiers()
        
        Returns:
            Formatted string ready for LLM context
        
        Example Output:
            === FACT DOSSIERS ===
            
            ## Vegetarian Diet
            Summary: User follows a strict vegetarian lifestyle, avoiding all 
            animal products and preferring plant-based proteins.
            
            Facts:
              - User is strictly vegetarian (added: 12/15/2025 10:30:00)
              - User avoids all meat products (added: 12/15/2025 10:30:00)
              - User prefers plant-based proteins (added: 12/15/2025 10:31:00)
            
            (Score: 0.85, Last updated: 12/15/2025 10:31:00)
        """
        if not dossiers:
            return ""
        
        formatted = "=== FACT DOSSIERS ===\n\n"
        
        for dossier in dossiers:
            formatted += f"## {dossier['title']}\n"
            
            # Add summary
            if dossier['summary']:
                formatted += f"Summary: {dossier['summary']}\n\n"
            
            # Add facts
            formatted += "Facts:\n"
            for fact in dossier['facts']:
                fact_text = fact['fact_text']
                added_at = fact['added_at']
                
                # Format timestamp if available
                try:
                    from datetime import datetime
                    dt = datetime.fromisoformat(added_at)
                    timestamp_str = dt.strftime("%m/%d/%Y %H:%M:%S")
                except (ValueError, TypeError):
                    timestamp_str = added_at
                
                formatted += f"  - {fact_text} (added: {timestamp_str})\n"
            
            # Add metadata
            formatted += f"\n(Score: {dossier['score']:.2f}, Last updated: {dossier['last_updated']})\n\n"
        
        logger.debug(f"Formatted {len(dossiers)} dossiers for context ({len(formatted)} chars)")
        return formatted
    
    def get_dossier_by_id(self, dossier_id: str) -> Dict[str, Any]:
        """
        Retrieve a specific dossier by ID.
        
        Useful for direct access or debugging.
        
        Args:
            dossier_id: Dossier ID to retrieve
        
        Returns:
            Dossier dictionary with facts, or None if not found
        """
        dossier = self.storage.get_dossier(dossier_id)
        if not dossier:
            return None
        
        facts = self.storage.get_dossier_facts(dossier_id)
        return {
            'dossier_id': dossier_id,
            'title': dossier['title'],
            'summary': dossier['summary'],
            'facts': facts,
            'created_at': dossier['created_at'],
            'last_updated': dossier['last_updated'],
            'status': dossier['status']
        }
    
    def get_all_dossiers(self, status: str = 'active') -> List[Dict[str, Any]]:
        """
        Get all dossiers with given status.
        
        Useful for debugging, analysis, or bulk operations.
        
        Args:
            status: Filter by status ('active', 'archived', etc.)
        
        Returns:
            List of all dossiers with basic metadata
        """
        dossiers = self.storage.get_all_dossiers(status=status)
        result = []
        
        for dossier in dossiers:
            fact_count = len(self.storage.get_dossier_facts(dossier['dossier_id']))
            result.append({
                'dossier_id': dossier['dossier_id'],
                'title': dossier['title'],
                'summary': dossier['summary'][:100] + '...' if len(dossier['summary']) > 100 else dossier['summary'],
                'fact_count': fact_count,
                'last_updated': dossier['last_updated'],
                'status': dossier['status']
            })
        
        return result


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("DossierRetriever Test")
    print("=" * 60)
    print("\nThis module requires Storage and DossierEmbeddingStorage.")
    print("Run integration tests with test_phase4_retrieval.py")
    print("\nKey Methods:")
    print("  - retrieve_relevant_dossiers(query, top_k=3)")
    print("  - format_for_context(dossiers)")
    print("  - get_dossier_by_id(dossier_id)")
    print("  - get_all_dossiers(status='active')")
