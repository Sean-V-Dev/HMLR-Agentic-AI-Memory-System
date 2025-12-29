# Quick Start Guide

## Basic Usage

### 1. Import and Initialize

```python
from hmlr import HMLRClient
import asyncio

# Initialize the client
client = HMLRClient(
    api_key="your-openai-api-key",
    db_path="my_memory.db",
    model="gpt-4.1-mini"  # Only tested model!
)
```

### 2. Simple Conversation

```python
async def simple_chat():
    # Send messages
    response1 = await client.chat("My name is Alice and I love pizza")
    print(response1)
    
    # HMLR remembers context
    response2 = await client.chat("What's my favorite food?")
    print(response2)  # Will recall "pizza"

# Run it
asyncio.run(simple_chat())
```

### 3. Persistent Memory Across Sessions

```python
# Session 1 - Today
client = HMLRClient(api_key="...", db_path="persistent.db")
await client.chat("My API key is XYZ789")
# ... close session ...

# Session 2 - Tomorrow (same database!)
client = HMLRClient(api_key="...", db_path="persistent.db")
response = await client.chat("What's my API key?")
# Will correctly recall "XYZ789" from yesterday
```

## LangChain Integration

If you installed with `pip install hmlr[langchain]`:

```python
from hmlr.integrations.langchain import HMLRMemory
from langchain.chains import ConversationChain
from langchain_openai import ChatOpenAI

# Create HMLR memory
memory = HMLRMemory(
    api_key="your-openai-api-key",
    db_path="langchain_memory.db"
)

# Use with LangChain
llm = ChatOpenAI(model="gpt-4.1-mini")
chain = ConversationChain(llm=llm, memory=memory)

# Chat normally - memory is handled automatically
response = chain.invoke({"input": "I'm working on Project Apollo"})
print(response['response'])
```

## Key Features

### Temporal Reasoning
```python
# HMLR automatically resolves conflicting facts
await client.chat("My password is ABC123")
await client.chat("Actually, I changed it to XYZ789")
response = await client.chat("What's my password?")
# Correctly returns: "XYZ789" (newest fact wins)
```

### Multi-Hop Reasoning
```python
# Conversation on Day 1
await client.chat("All reports must be under 5MB")
# ... many turns later ...
# Conversation on Day 30
await client.chat("I'm designing a system that generates 10MB reports")
response = await client.chat("Is this compliant with our policy?")
# Correctly connects 30-day-old policy to new design
```

### Cross-Topic Persistence
```python
# Establish a constraint
await client.chat("I'm vegetarian")
# Switch topics
await client.chat("Let's talk about cars now")
# ... conversation about cars ...
# Back to food
response = await client.chat("What should I order for lunch?")
# Will suggest vegetarian options despite topic changes
```

## Configuration Options

```python
client = HMLRClient(
    api_key="your-key",
    db_path="memory.db",
    model="gpt-4.1-mini",           # Default and ONLY tested model
    use_llm_intent_mode=False,       # Enable LLM-based intent detection
    enable_telemetry=False,          # Enable Phoenix telemetry
    conversation_id=None,            # Optional conversation ID
)
```

## Working with Conversations

### Start a New Conversation
```python
# Automatic conversation management
client1 = HMLRClient(api_key="...", db_path="db1.db")
await client1.chat("Hello")  # Creates conversation automatically

# Manual conversation ID
client2 = HMLRClient(
    api_key="...", 
    db_path="db2.db",
    conversation_id="project-apollo-chat"
)
```

### Memory Isolation
```python
# Each database is isolated
client_work = HMLRClient(api_key="...", db_path="work_memory.db")
client_personal = HMLRClient(api_key="...", db_path="personal_memory.db")

# No cross-contamination between databases
```

## Best Practices

### 1. Use Persistent Database Paths
```python
#  Good - consistent path
client = HMLRClient(api_key="...", db_path="./data/memory.db")

#  Bad - temporary or random paths
client = HMLRClient(api_key="...", db_path=f"temp_{random()}.db")
```

### 2. Reuse Client Instances
```python
#  Good - reuse client
client = HMLRClient(api_key="...", db_path="memory.db")
for message in messages:
    await client.chat(message)

#  Bad - creating new client each time
for message in messages:
    client = HMLRClient(api_key="...", db_path="memory.db")
    await client.chat(message)
```

### 3. Handle Async Properly
```python
#  Good - proper async context
async def main():
    client = HMLRClient(api_key="...", db_path="memory.db")
    response = await client.chat("Hello")
    return response

result = asyncio.run(main())

#  Bad - mixing sync/async incorrectly
client = HMLRClient(api_key="...", db_path="memory.db")
response = client.chat("Hello")  # Missing await!
```

## Error Handling

```python
async def safe_chat():
    client = HMLRClient(api_key="...", db_path="memory.db")
    
    try:
        response = await client.chat("Hello")
        print(response)
    except Exception as e:
        print(f"Error: {e}")
        # Handle error appropriately
```

## Next Steps

- See [Model Compatibility](model_compatibility.md) for critical model warnings
- Check [examples/simple_usage.py](../examples/simple_usage.py) for full working example
- Review [API Reference](api_reference.md) for detailed documentation
- Run the test suite to see HMLR's capabilities: `pytest tests/ragas_test_8_multi_hop.py -v -s`
