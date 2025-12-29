# HMLR LangGraph Integration Design Document

## Overview

This document outlines the design for integrating HMLR (Hierarchical Memory for Long-Running Conversations) as a **LangGraph node**. HMLR provides long-term episodic memory, user profile management, and context retrieval - capabilities that complement LangGraph's short-term checkpointing.

---

## Architecture Position

```
┌─────────────────────────────────────────────────────────────┐
│                    LangGraph StateGraph                      │
│                                                              │
│  ┌──────────┐    ┌──────────────┐    ┌──────────┐          │
│  │ Router   │───►│ HMLR Memory  │───►│ LLM Node │          │
│  │ Node     │    │    Node      │    │          │          │
│  └──────────┘    └──────────────┘    └──────────┘          │
│                         │                                    │
│                         ▼                                    │
│              ┌──────────────────────┐                       │
│              │  HMLR SQLite DB      │  (External storage)   │
│              │  + Embeddings        │                       │
│              │  + User Profile      │                       │
│              └──────────────────────┘                       │
└─────────────────────────────────────────────────────────────┘
```

**Key Insight**: HMLR operates as **long-term memory** (cross-thread), while LangGraph's checkpointers handle **short-term memory** (within-thread). They are complementary, not competing.

---

## Integration Approach

### Option A: Thin Node Wrapper (Recommended)

HMLR is wrapped as a LangGraph node that:
1. Receives state containing user message
2. Hydrates context from HMLR
3. Returns enriched state with memory context

```python
from langgraph.graph import StateGraph
from typing import TypedDict, List, Optional

class AgentState(TypedDict):
    messages: List[dict]              # LangGraph message history
    user_id: str                      # Cross-thread identity
    session_id: str                   # Within-thread identity
    hmlr_context: Optional[str]       # Hydrated memory context
    hmlr_memories: Optional[List]     # Retrieved memory items
    user_profile: Optional[str]       # User profile/constraints

def hmlr_memory_node(state: AgentState, config: dict) -> AgentState:
    """
    LangGraph node that retrieves HMLR context.
    
    This node:
    1. Takes the latest user message
    2. Queries HMLR for relevant memories
    3. Returns enriched state with context
    """
    from hmlr import HMLRClient
    
    # Get or create HMLR client (singleton per worker)
    client = get_hmlr_client(config)
    
    # Extract latest user message
    latest_message = state["messages"][-1]["content"]
    
    # Retrieve context (read-only operation)
    result = client.retrieve_context(
        query=latest_message,
        session_id=state["session_id"]
    )
    
    return {
        **state,
        "hmlr_context": result.formatted_context,
        "hmlr_memories": result.memory_items,
        "user_profile": result.user_profile
    }
```

### Option B: Full Chat Node

HMLR handles the entire conversation turn including LLM invocation:

```python
async def hmlr_chat_node(state: AgentState, config: dict) -> AgentState:
    """Full HMLR chat processing including LLM."""
    from hmlr import HMLRClient
    
    client = get_hmlr_client(config, raise_on_error=True)
    
    latest_message = state["messages"][-1]["content"]
    
    # Full chat processing (retrieval + LLM + persistence)
    response = await client.chat_async(
        message=latest_message,
        session_id=state["session_id"]
    )
    
    # Append response to messages
    new_messages = state["messages"] + [{
        "role": "assistant",
        "content": response.response_text
    }]
    
    return {
        **state,
        "messages": new_messages,
        "hmlr_context": response.context_used,
        "contexts_retrieved": response.contexts_retrieved
    }
```

---

## Isolation Approach

### Implemented: Integration Package Inside HMLR

The LangGraph integration is included as part of the HMLR package:

```
HMLR/
├── hmlr/                          # Core HMLR library
├── integrations/                  # Integration packages
│   ├── __init__.py
│   └── langgraph/                 # LangGraph integration
│       ├── __init__.py            # Package exports
│       ├── nodes.py               # LangGraph node implementations
│       ├── state.py               # State schema definitions
│       ├── client.py              # HMLR client manager (singleton)
│       └── examples/
│           └── simple_agent.py    # Working example
├── tests/
└── docs/
```

**Why inside HMLR?**
1. **Single package**: Users `pip install hmlr[langgraph]` to get everything
2. **Version sync**: Integration always matches HMLR version
3. **Simpler testing**: Tests run alongside HMLR tests
4. **Optional deps**: LangGraph only needed if using this integration

### Usage

```python
# Import from HMLR integrations
from integrations.langgraph import hmlr_memory_node, HMLRState

# Add to any LangGraph graph
graph = StateGraph(HMLRState)
graph.add_node("memory", hmlr_memory_node)
```

### Setup for Development

```bash
cd c:\Users\seanv\HMLR

# Install LangGraph (HMLR is already installed)
pip install langgraph langchain-core langchain-openai

# Run example
python integrations/langgraph/examples/simple_agent.py
```

---

## Component Mapping

| LangGraph Concern | HMLR Component | Notes |
|-------------------|----------------|-------|
| Short-term memory | Checkpointer (SqliteSaver) | LangGraph handles this |
| Long-term memory | `Storage` + `EmbeddingStorage` | HMLR provides this |
| User profile | `UserProfileManager` + `Scribe` | Cross-thread constraints |
| Context retrieval | `TheGovernor` + `ContextHydrator` | Semantic search + Bridge Blocks |
| LLM calls | `ExternalAPIClient` | Can use LangGraph's LLM or HMLR's |

---

## State Schema Design

```python
from typing import TypedDict, List, Optional, Annotated
from langgraph.graph import add_messages

class HMLRAgentState(TypedDict):
    # Standard LangGraph fields
    messages: Annotated[List[dict], add_messages]
    
    # Identity (cross-thread)
    user_id: str
    
    # Session (within-thread)
    session_id: str
    thread_id: str
    
    # HMLR-enriched context (set by hmlr_node)
    hmlr_context: Optional[str]
    hmlr_memories: Optional[List[dict]]
    user_profile: Optional[str]
    dossiers: Optional[List[dict]]
    
    # Metadata
    contexts_retrieved: int
    hmlr_health: dict
```

---

## Configuration Injection

HMLR needs database paths and API keys. Use LangGraph's `config` mechanism:

```python
from langgraph.graph import StateGraph

graph = StateGraph(HMLRAgentState)

# Compile with configurable schema
app = graph.compile(
    checkpointer=SqliteSaver.from_conn_string(":memory:")
)

# Invoke with config
result = app.invoke(
    {"messages": [{"role": "user", "content": "Hello"}]},
    config={
        "configurable": {
            "thread_id": "conv-123",
            "hmlr_db_path": "/path/to/hmlr.db",
            "hmlr_profile_path": "/path/to/profile.json",
            "user_id": "user-456"
        }
    }
)
```

---

## Error Handling Strategy

With `raise_on_error=True` in ConversationEngine:

```python
async def hmlr_node(state, config):
    try:
        client = get_hmlr_client(config, raise_on_error=True)
        result = await client.chat_async(...)
        return {**state, "hmlr_context": result.context}
    except ApiConnectionError:
        # Return degraded state - LLM can work without memory
        return {**state, "hmlr_context": None, "hmlr_error": "API unavailable"}
    except StorageWriteError:
        # Log but don't fail the graph
        logger.error("HMLR storage error", exc_info=True)
        return {**state, "hmlr_context": None}
```

---

## Health Checking

Use `ComponentBundle.is_fully_operational()` for pre-flight:

```python
def initialize_hmlr(config: dict):
    """Initialize HMLR and verify health."""
    from hmlr.core.component_factory import ComponentFactory
    
    factory = ComponentFactory()
    components = factory.create_all_components(
        db_path=config.get("hmlr_db_path"),
        api_key=config.get("openai_api_key")
    )
    
    if not components.is_fully_operational():
        degraded = components.get_degraded_components()
        logger.warning(f"HMLR degraded: {degraded}")
    
    return ComponentFactory.create_conversation_engine(
        components, 
        raise_on_error=True
    )
```

---

## Implementation Plan

### Phase 1: Core Integration (Day 1)
- [ ] Create `langgraph-hmlr/` directory structure
- [ ] Create virtual environment with HMLR + LangGraph
- [ ] Implement basic `HMLRClient` wrapper class
- [ ] Implement `hmlr_context_node` (read-only retrieval)
- [ ] Write unit tests for node

### Phase 2: Full Node (Day 2)
- [ ] Implement `hmlr_chat_node` (full chat with persistence)
- [ ] Add client pooling for multi-worker scenarios
- [ ] Add health checking node
- [ ] Write integration test with simple graph

### Phase 3: Example Graphs (Day 3)
- [ ] Create simple agent example
- [ ] Create multi-agent handoff example
- [ ] Document usage patterns
- [ ] Validate with vegetarian test scenario

### Phase 4: Production Hardening (Future)
- [ ] Connection pooling for concurrent access
- [ ] Metrics/observability integration
- [ ] Graceful degradation patterns
- [ ] Performance benchmarks

---

## Testing Strategy

### Test 1: Context Retrieval Node
```python
def test_hmlr_context_node():
    """Test that HMLR node retrieves context correctly."""
    state = {
        "messages": [{"role": "user", "content": "What did we discuss about rowing?"}],
        "session_id": "test-session"
    }
    
    result = hmlr_context_node(state, {})
    
    assert "hmlr_context" in result
    assert result["hmlr_context"] is not None or result.get("hmlr_error")
```

### Test 2: Vegetarian Constraint (E2E)
```python
async def test_vegetarian_constraint_e2e():
    """Test that user profile constraints flow through graph."""
    graph = build_hmlr_graph()
    
    # Turn 1: Declare constraint
    await graph.ainvoke({
        "messages": [{"role": "user", "content": "I am strictly vegetarian"}]
    })
    
    # Turn 2: Test constraint (new thread)
    result = await graph.ainvoke({
        "messages": [{"role": "user", "content": "Should I eat steak?"}]
    }, config={"configurable": {"thread_id": "new-thread"}})
    
    # Verify constraint was enforced
    response = result["messages"][-1]["content"]
    assert "vegetarian" in response.lower()
```

---

## Dependencies

```
# requirements.txt for langgraph-hmlr
hmlr @ file:../HMLR    # Local editable install
langgraph>=0.0.40
langchain-core>=0.1.0
langchain-openai>=0.0.5
```

---

## Open Questions

1. **Thread ID mapping**: Should LangGraph `thread_id` map to HMLR `session_id`?
2. **Persistence timing**: Should HMLR persist on every turn or only on graph completion?
3. **Scribe integration**: Should Scribe run in the graph or stay as background task?
4. **Multi-tenant**: How to handle multiple users with different DBs?

---

## Next Steps

1. **Create directory structure** in `projects/langgraph-hmlr/`
2. **Install dependencies** in isolated venv
3. **Implement `nodes.py`** with basic context retrieval
4. **Run vegetarian test** against the graph
