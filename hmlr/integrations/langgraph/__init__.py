"""
HMLR LangGraph Integration

This package provides LangGraph nodes for integrating HMLR
as a long-term memory layer in your AI agents.

Quick Start:
    from hmlr.integrations.langgraph import hmlr_memory_node
    
    # Add to your LangGraph graph
    graph.add_node("memory", hmlr_memory_node)
"""

from .nodes import (
    hmlr_memory_node,
    hmlr_chat_node,
    create_hmlr_graph,
)
from .state import HMLRState
from .client import HMLRClientManager

__all__ = [
    "hmlr_memory_node",
    "hmlr_chat_node",
    "create_hmlr_graph",
    "HMLRState",
    "HMLRClientManager",
]
