"""
State Schema for HMLR LangGraph Integration

This module defines the TypedDict for state that flows through the graph.
"""

from typing import TypedDict, List, Optional, Annotated, Any
from typing_extensions import Required

# Check if langgraph is available
try:
    from langgraph.graph.message import add_messages
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    # Fallback - just use regular list
    def add_messages(left, right):
        return left + right


class HMLRState(TypedDict, total=False):
    """
    State schema for HMLR-enhanced LangGraph agents.
    
    This state flows through the graph and gets enriched at each node.
    
    Required Fields:
        messages: The conversation history (LangGraph standard)
        
    Identity Fields:
        user_id: Cross-thread user identity (for profile lookup)
        session_id: Within-thread session (for conversation continuity)
        
    HMLR-Populated Fields (set by hmlr_memory_node):
        hmlr_context: Formatted context string for LLM prompt
        hmlr_memories: Raw memory items retrieved
        user_profile: User profile/constraints string
        dossiers: Retrieved dossier items
        
    Metadata:
        contexts_retrieved: Count of memories found
        hmlr_healthy: Whether HMLR initialized correctly
        hmlr_error: Error message if something failed
    """
    
    # === Required: Conversation Messages ===
    messages: Required[Annotated[List[dict], add_messages]]
    
    # === Identity ===
    user_id: str
    session_id: str
    
    # === HMLR Context (populated by memory node) ===
    hmlr_context: Optional[str]
    hmlr_memories: Optional[List[dict]]
    user_profile: Optional[str]
    dossiers: Optional[List[dict]]
    
    # === Metadata ===
    contexts_retrieved: int
    hmlr_healthy: bool
    hmlr_error: Optional[str]


class SimpleHMLRState(TypedDict, total=False):
    """
    Minimal state for simple use cases.
    
    Use this if you just want to add memory to an existing agent
    without changing much of your state structure.
    """
    
    # The user's latest message
    user_message: str
    
    # The assistant's response
    assistant_response: str
    
    # Session tracking
    session_id: str
    
    # HMLR context (set by memory node)
    hmlr_context: Optional[str]
    user_profile: Optional[str]
