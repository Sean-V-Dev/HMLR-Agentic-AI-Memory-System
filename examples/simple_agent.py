#!/usr/bin/env python
"""
Simple HMLR + LangGraph Example

This example shows how to add HMLR long-term memory to a basic agent.

Prerequisites:
    pip install langgraph langchain-core langchain-openai

Usage:
    python simple_agent.py
"""

import asyncio
import os
from typing import Annotated
from pathlib import Path

# Load .env file
from dotenv import load_dotenv
env_path = Path(__file__).parent.parent.parent.parent / ".env"
load_dotenv(env_path)

# Check for LangGraph
try:
    from langgraph.graph import StateGraph, END
    from langgraph.graph.message import add_messages
except ImportError:
    print("Please install langgraph: pip install langgraph langchain-core")
    exit(1)

# Check for LangChain OpenAI
try:
    from langchain_openai import ChatOpenAI
except ImportError:
    print("Please install langchain-openai: pip install langchain-openai")
    exit(1)

# Import HMLR integration
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from hmlr.integrations.langgraph.nodes import hmlr_memory_node
from hmlr.integrations.langgraph.state import HMLRState

# Check for API key
if not os.environ.get("OPENAI_API_KEY"):
    print("=" * 60)
    print("ERROR: OPENAI_API_KEY environment variable not set!")
    print("=" * 60)
    print("\nSet it with PowerShell:")
    print('  $env:OPENAI_API_KEY = "your-key-here"')
    print("\nOr set it permanently in Windows environment variables.")
    exit(1)

# Create LLM
llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.7)


async def llm_node(state: HMLRState) -> dict:
    """
    LLM node that uses HMLR context.
    
    The hmlr_memory_node has already populated:
    - state["hmlr_context"]: Formatted context string
    - state["user_profile"]: User constraints/preferences
    """
    messages = state.get("messages", [])
    hmlr_context = state.get("hmlr_context")
    
    # Build system message with HMLR context
    system_parts = ["You are a helpful assistant with long-term memory."]
    
    if hmlr_context:
        system_parts.append(
            f"\nHere is relevant context from previous conversations:\n{hmlr_context}"
        )
        system_parts.append(
            "\nUse this context to provide personalized, consistent responses. "
            "Remember user preferences and past discussions."
        )
    
    system_message = {"role": "system", "content": "\n".join(system_parts)}
    
    # Call LLM
    full_messages = [system_message] + messages
    response = await llm.ainvoke(full_messages)
    
    return {
        "messages": [{"role": "assistant", "content": response.content}]
    }


def build_agent(use_hmlr_llm: bool = True):
    """
    Build the LangGraph agent with HMLR memory.
    
    Args:
        use_hmlr_llm: If True, use HMLR's full chat (includes persistence + Scribe).
                      If False, use separate memory retrieval + custom LLM.
    """
    from hmlr.integrations.langgraph.nodes import hmlr_chat_node
    
    graph = StateGraph(HMLRState)
    
    if use_hmlr_llm:
        # OPTION A: HMLR handles everything (recommended for persistence)
        # This includes: retrieval, LLM call, conversation persistence, Scribe
        graph.add_node("chat", hmlr_chat_node)
        graph.set_entry_point("chat")
        graph.add_edge("chat", END)
    else:
        # OPTION B: HMLR retrieval + custom LLM (no persistence)
        graph.add_node("memory", hmlr_memory_node)
        graph.add_node("llm", llm_node)
        graph.set_entry_point("memory")
        graph.add_edge("memory", "llm")
        graph.add_edge("llm", END)
    
    return graph.compile()


async def main():
    """Run the agent interactively."""
    
    print("=" * 60)
    print("HMLR + LangGraph Simple Agent")
    print("=" * 60)
    
    # Pre-warm HMLR (loads embedding models, ~3-5 seconds)
    print("Initializing HMLR memory system...")
    from hmlr.integrations.langgraph.client import get_client_manager
    manager = get_client_manager()
    engine = manager.get_engine({}, raise_on_error=False)
    
    if engine is None:
        print("⚠️ WARNING: HMLR engine failed to initialize. Memory disabled.")
    else:
        print("✓ HMLR ready")
    
    print("\nThis agent has long-term memory powered by HMLR.")
    print("Try telling it something about yourself, then asking about it later!")
    print("Type 'quit' to exit.\n")
    
    agent = build_agent()
    session_id = "demo-session"
    messages = []
    
    while True:
        try:
            user_input = input("You: ").strip()
            if user_input.lower() in ("quit", "exit", "q"):
                print("Goodbye!")
                break
            
            if not user_input:
                continue
            
            # Add user message
            messages.append({"role": "user", "content": user_input})
            
            # Run agent
            result = await agent.ainvoke(
                {"messages": messages, "session_id": session_id},
                config={"configurable": {"thread_id": session_id}}
            )
            
            # Get response
            response_messages = result.get("messages", [])
            if response_messages:
                last_msg = response_messages[-1]
                # Handle both dict and LangChain message objects
                if isinstance(last_msg, dict):
                    content = last_msg.get("content", str(last_msg))
                elif hasattr(last_msg, "content"):
                    content = last_msg.content
                else:
                    content = str(last_msg)
                print(f"\nAssistant: {content}\n")
                
                # Update messages for next turn (convert to list if needed)
                messages = list(response_messages)
            
            # Show HMLR stats
            contexts = result.get("contexts_retrieved", 0)
            if contexts > 0:
                print(f"[HMLR: Retrieved {contexts} relevant memories]")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
