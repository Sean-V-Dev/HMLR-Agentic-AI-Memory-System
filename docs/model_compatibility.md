# Model Compatibility

## ⚠️ CRITICAL WARNING

**HMLR has ONLY been tested with `gpt-4.1-mini`.**

## Why Only GPT-4.1-mini?

HMLR's architecture was designed, optimized, and benchmarked exclusively with `gpt-4.1-mini`. The system makes specific assumptions about:

- **Instruction following consistency**
- **JSON formatting reliability**
- **Prompt response patterns**
- **Token usage characteristics**
- **Reasoning capabilities at the mini-tier**

These assumptions have been validated through extensive testing with `gpt-4.1-mini` ONLY.

## Verified Benchmarks

All RAGAS benchmarks achieving **1.00 Faithfulness** and **1.00 Context Recall** were run using:

```python
model="gpt-4.1-mini"
```

| Test | Model | Faithfulness | Context Recall | Result |
|------|-------|--------------|----------------|--------|
| Multi-hop reasoning | gpt-4.1-mini | 1.00 | 1.00 | ✅ PASS |
| Temporal conflicts | gpt-4.1-mini | 1.00 | 1.00 | ✅ PASS |
| Cross-topic invariants | gpt-4.1-mini | 1.00 | 1.00 | ✅ PASS |
| Zero-keyword recall | gpt-4.1-mini | 1.00 | 1.00 | ✅ PASS |
| 50-turn long conversation | gpt-4.1-mini | 1.00 | 1.00 | ✅ PASS |
| Hydra (9-policy lethal RAG) | gpt-4.1-mini | 1.00 | 1.00 | ✅ PASS |

**No other models have been tested.**

## What Happens If You Use Another Model?

If you try to use a different model, you'll see this warning:

```python
client = HMLRClient(
    api_key="...",
    model="gpt-4o"  # Different model!
)

# ⚠️ Output:
# UserWarning: Model 'gpt-4o' has NOT been tested!
# HMLR is only validated with 'gpt-4.1-mini'.
# You may experience:
# - Incorrect memory retrieval
# - Malformed internal JSON
# - Failed fact extraction
# - Bridge block corruption
# Proceed at your own risk.
```

## Potential Failure Modes with Other Models

### 1. JSON Parsing Failures
HMLR components generate structured JSON for internal operations. Other models may:
- Use different formatting
- Include extra commentary
- Fail to follow JSON schema exactly

### 2. Fact Extraction Errors
The `FactScrubber` uses specific prompts optimized for GPT-4.1-mini. Other models may:
- Extract incorrect facts
- Miss important facts
- Hallucinate non-existent facts

### 3. Retrieval Failures
The `LatticeCrawler` and `Governor` rely on specific reasoning patterns. Other models may:
- Retrieve irrelevant context
- Miss critical memories
- Fail to resolve temporal conflicts

### 4. Bridge Block Corruption
The bridge block architecture depends on consistent LLM behavior. Other models may:
- Create malformed bridge blocks
- Fail to detect topic shifts
- Break conversation continuity

## Future Model Support

We may add support for other models in future versions, but this will require:

1. **Extensive re-testing** of all RAGAS benchmarks
2. **Prompt engineering** specific to each model
3. **Validation** that perfect scores are maintained
4. **Documentation** of any behavioral differences

Until then, **stick to `gpt-4.1-mini`**.

## Default Configuration

The default model is set to `gpt-4.1-mini` everywhere:

```python
# In HMLRClient
client = HMLRClient(
    api_key="...",
    # model="gpt-4.1-mini"  # This is the default
)

# In LangChain integration
from langchain_openai import ChatOpenAI

memory = HMLRMemory(api_key="...")
llm = ChatOpenAI(model="gpt-4.1-mini")  # Use matching model!
```

## Cost Considerations

GPT-4.1-mini is OpenAI's most cost-effective model while maintaining strong reasoning capabilities:

- **Input**: ~$0.15 per 1M tokens
- **Output**: ~$0.60 per 1M tokens

HMLR's architecture is designed to work efficiently within these economics:
- Average queries use < 4k tokens
- Bridge blocks minimize redundant context
- Fact extraction happens once per message

## Checking Your Model

To verify what model you're using:

```python
client = HMLRClient(
    api_key="...",
    db_path="memory.db",
    model="gpt-4.1-mini"
)

# The model is stored in the client
print(f"Using model: {client.conversation_engine.llama_client.model}")
```

## Summary

| Aspect | Status |
|--------|--------|
| **Supported Models** | `gpt-4.1-mini` ONLY |
| **Tested Models** | `gpt-4.1-mini` ONLY |
| **Benchmarked Models** | `gpt-4.1-mini` ONLY |
| **Other Models** | ❌ NOT SUPPORTED |
| **Future Support** | Maybe (requires extensive testing) |

**When in doubt, use `gpt-4.1-mini`.**

## Related Documentation

- [Installation Guide](installation.md)
- [Quickstart Guide](quickstart.md)
- [Benchmark Results](../README.md#-verified-benchmark-achievements-ragas)
