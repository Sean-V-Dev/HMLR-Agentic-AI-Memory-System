"""
LangGraph Nodes for HMLR Memory Integration

These nodes can be added to any LangGraph StateGraph to give your
agent long-term memory capabilities.

Quick Start:
    from langgraph.graph import StateGraph
    from hmlr.integrations.langgraph import hmlr_memory_node, HMLRState
    
    graph = StateGraph(HMLRState)
    graph.add_node("memory", hmlr_memory_node)
    graph.add_node("llm", your_llm_node)
    graph.add_edge("memory", "llm")
"""

import logging
from typing import Any, Dict

from .client import get_client_manager
from .state import HMLRState

logger = logging.getLogger(__name__)


def _extract_config(config: Dict[str, Any]) -> dict:
    """Extract HMLR config from LangGraph config."""
    configurable = config.get("configurable", {})
    return {
        "hmlr_db_path": configurable.get("hmlr_db_path"),
        "hmlr_profile_path": configurable.get("hmlr_profile_path"),
        "openai_api_key": configurable.get("openai_api_key"),
    }


def _get_latest_user_message(state: HMLRState) -> str:
    """Extract the latest user message from state."""
    messages = state.get("messages", [])
    
    # Handle both dict format and LangChain message format
    for msg in reversed(messages):
        if isinstance(msg, dict):
            if msg.get("role") == "user":
                return msg.get("content", "")
        elif hasattr(msg, "type") and msg.type == "human":
            return msg.content
    
    # Fallback to user_message field (SimpleHMLRState)
    return state.get("user_message", "")


async def hmlr_memory_node(state: HMLRState, config: Dict[str, Any] = None) -> dict:
    """
    LangGraph node that retrieves HMLR context for the current message.
    
    This node:
    1. Takes the latest user message from state
    2. Queries HMLR for relevant memories, facts, and user profile
    3. Returns enriched state with context ready for LLM
    
    Add this node BEFORE your LLM node to give it memory context.
    
    Args:
        state: Current graph state with messages
        config: LangGraph config with optional HMLR settings:
            - configurable.hmlr_db_path: Path to HMLR database
            - configurable.hmlr_profile_path: Path to user profile
            - configurable.session_id: Session ID for this conversation
            
    Returns:
        Updated state with:
            - hmlr_context: Formatted context string for LLM
            - hmlr_memories: Raw memory items
            - user_profile: User profile string
            - hmlr_healthy: Health status
    """
    config = config or {}
    hmlr_config = _extract_config(config)
    
    # Get session_id from config or state
    session_id = (
        config.get("configurable", {}).get("session_id") or
        config.get("configurable", {}).get("thread_id") or
        state.get("session_id") or
        "default_session"
    )
    
    try:
        # Get HMLR engine
        manager = get_client_manager()
        engine = manager.get_engine(hmlr_config, raise_on_error=False)
        
        if engine is None:
            logger.warning("HMLR engine not available")
            return {
                "hmlr_context": None,
                "hmlr_memories": [],
                "user_profile": None,
                "hmlr_healthy": False,
                "hmlr_error": "HMLR engine not initialized",
                "contexts_retrieved": 0,
            }
        
        # Get the user message
        user_message = _get_latest_user_message(state)
        if not user_message:
            return {
                "hmlr_context": None,
                "hmlr_healthy": True,
                "contexts_retrieved": 0,
            }
        
        # Query HMLR for context
        # Use governor.govern() for retrieval without chat response
        governor = engine.governor
        day_id = engine.conversation_mgr.current_day
        
        if governor:
            routing_decision, memories, facts, dossiers = await governor.govern(
                user_message, day_id
            )
            
            # Get user profile
            user_profile = None
            if engine.user_profile_manager:
                user_profile = engine.user_profile_manager.get_user_profile_context()
            
            # Format context
            context_parts = []
            
            if user_profile and user_profile.strip():
                context_parts.append(f"USER PROFILE:\n{user_profile}")
            
            if memories:
                memory_text = "\n".join([
                    f"- {m.get('summary', m.get('content', str(m)))}"
                    for m in memories[:10]  # Limit to 10
                ])
                context_parts.append(f"RELEVANT MEMORIES:\n{memory_text}")
            
            if facts:
                fact_text = "\n".join([f"- {f}" for f in facts[:5]])
                context_parts.append(f"KNOWN FACTS:\n{fact_text}")
            
            if dossiers:
                dossier_text = "\n".join([
                    f"- {d.get('entity_name', 'Unknown')}: {d.get('summary', '')}"
                    for d in dossiers[:5]
                ])
                context_parts.append(f"ENTITIES:\n{dossier_text}")
            
            hmlr_context = "\n\n".join(context_parts) if context_parts else None
            
            return {
                "hmlr_context": hmlr_context,
                "hmlr_memories": memories,
                "user_profile": user_profile,
                "dossiers": dossiers,
                "hmlr_healthy": True,
                "contexts_retrieved": len(memories) + len(facts) + len(dossiers),
            }
        else:
            return {
                "hmlr_context": None,
                "hmlr_healthy": False,
                "hmlr_error": "Governor not available (API issue?)",
                "contexts_retrieved": 0,
            }
            
    except Exception as e:
        logger.error(f"HMLR memory node error: {e}", exc_info=True)
        return {
            "hmlr_context": None,
            "hmlr_healthy": False,
            "hmlr_error": str(e),
            "contexts_retrieved": 0,
        }


async def hmlr_chat_node(state: HMLRState, config: Dict[str, Any] = None) -> dict:
    """
    LangGraph node that does FULL HMLR chat processing.
    
    This node:
    1. Retrieves memory context
    2. Calls the LLM with context
    3. Persists the conversation turn
    4. Returns the response
    
    Use this if you want HMLR to handle the entire chat, including the LLM call.
    Use hmlr_memory_node instead if you want to use your own LLM.
    
    Args:
        state: Current graph state
        config: LangGraph config
        
    Returns:
        Updated state with assistant response added to messages
    """
    config = config or {}
    hmlr_config = _extract_config(config)
    
    session_id = (
        config.get("configurable", {}).get("session_id") or
        config.get("configurable", {}).get("thread_id") or
        state.get("session_id") or
        "default_session"
    )
    
    try:
        manager = get_client_manager()
        engine = manager.get_engine(hmlr_config, raise_on_error=True)
        
        user_message = _get_latest_user_message(state)
        if not user_message:
            return {"hmlr_error": "No user message found"}
        
        # Full HMLR chat processing
        # await_background_tasks=True ensures Scribe completes before returning
        response = await engine.process_user_message(
            user_query=user_message,
            session_id=session_id,
            await_background_tasks=True  # Wait for Scribe to finish
        )
        
        # Add response to messages
        new_message = {
            "role": "assistant",
            "content": response.response_text
        }
        
        return {
            "messages": [new_message],
            "hmlr_healthy": response.status.value != "error",
            "contexts_retrieved": response.contexts_retrieved or 0,
        }
        
    except Exception as e:
        logger.error(f"HMLR chat node error: {e}", exc_info=True)
        # Re-raise if configured to do so
        raise


def hmlr_health_check_node(state: HMLRState, config: Dict[str, Any] = None) -> dict:
    """
    LangGraph node that checks HMLR health.
    
    Use this as an entry node to verify HMLR is working before proceeding.
    
    Returns:
        State with hmlr_healthy and hmlr_error fields set
    """
    config = config or {}
    hmlr_config = _extract_config(config)
    
    manager = get_client_manager()
    healthy = manager.is_healthy(hmlr_config)
    
    if healthy:
        return {"hmlr_healthy": True, "hmlr_error": None}
    else:
        degraded = manager.get_degraded_components(hmlr_config)
        return {
            "hmlr_healthy": False,
            "hmlr_error": f"Degraded components: {degraded}"
        }


def create_hmlr_graph():
    """
    Create a simple LangGraph with HMLR memory.
    
    This is a convenience function that creates a basic agent graph
    with HMLR memory already integrated.
    
    Returns:
        Compiled LangGraph ready to use
        
    Usage:
        graph = create_hmlr_graph()
        result = await graph.ainvoke({
            "messages": [{"role": "user", "content": "Hello!"}]
        })
    """
    try:
        from langgraph.graph import StateGraph, END
    except ImportError:
        raise ImportError(
            "langgraph is required. Install with: pip install langgraph"
        )
    
    graph = StateGraph(HMLRState)
    
    # Add HMLR memory node
    graph.add_node("memory", hmlr_memory_node)
    graph.add_node("chat", hmlr_chat_node)
    
    # Connect nodes
    graph.set_entry_point("memory")
    graph.add_edge("memory", "chat")
    graph.add_edge("chat", END)
    
    return graph.compile()
