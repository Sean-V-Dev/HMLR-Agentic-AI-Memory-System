"""
Retrieval system for long-horizon memory.

Components:
- LatticeCrawler: Searches day nodes and retrieves relevant context
- ContextHydrator: Builds LLM prompts with retrieved context
"""

from .crawler import LatticeCrawler

__all__ = ['LatticeCrawler']
