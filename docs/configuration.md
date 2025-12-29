# Configuration Guide

This document describes all tunable parameters in HMLR. The system has been carefully calibrated for optimal performance with GPT-4.1-mini, but you may want to experiment with these settings for your specific use case.

## ⚠️ Important Notice

**HMLR has been extensively tested and benchmarked with the default configuration using GPT-4.1-mini.** Changing these parameters may affect performance, RAGAS scores, and system behavior. Always test thoroughly after making changes.

---

## Table of Contents

1. [LLM Parameters](#llm-parameters)
2. [Sliding Window Configuration](#sliding-window-configuration)
3. [Bridge Block Settings](#bridge-block-settings)
4. [Chunking Parameters](#chunking-parameters)
5. [Retrieval Configuration](#retrieval-configuration)
6. [Token Budgets](#token-budgets)
7. [Memory Gardening](#memory-gardening)

---

## LLM Parameters

### Location: `core/external_api_client.py`

#### API Endpoint

```python
# Line ~230
def _get_base_url(self) -> str:
    """Get base URL for API provider"""
    if self.api_provider == "openai":
        return "https://api.openai.com/v1"
```

**What it controls:** The API endpoint for all LLM calls

**Default: `https://api.openai.com/v1`** (Official OpenAI API)

**When to change:**
- **Azure OpenAI:** Change to your Azure endpoint
  ```python
  return "https://your-resource.openai.azure.com/openai/deployments/your-deployment"
  ```
- **Local proxy/gateway:** Point to your proxy server
  ```python
  return "http://localhost:8000/v1"
  ```
- **OpenAI-compatible APIs:** Point to alternative providers (e.g., OpenRouter, Together.ai)
- **Other LLMs via proxy:** Use Gemini Pro, Claude, Llama 3, etc. through an OpenAI-compatible wrapper

**💡 Key Insight: HMLR is the MEMORY, not the LLM**

HMLR's job is to:
1. **Retrieve** the right memories (facts, bridge blocks, context)
2. **Build** the prompt with all relevant context
3. **Send** that prompt to whatever LLM endpoint you configure
4. **Store** the response back into memory

You can use **any LLM** you want! Examples:
- **Gemini Pro models** for stronger reasoning
- **Claude Pro models** for better writing
- **Llama** for local/private deployment
- **GPT Pro models** for production (costs more than 4.1-mini)

**⚠️ Important Caveats:**
- The endpoint must accept OpenAI's API format (request/response structure)
- You need a valid API key for that endpoint
- Model names must match what the endpoint supports
- **HMLR has ONLY been tested with GPT-4.1-mini** - other models are experimental
- The memory architecture (chunking, retrieval, fact extraction) was optimized for GPT-4.1-mini
- You may need to adjust prompts/parameters for other models

**Using Gemini Pro Example:**
```python
# In external_api_client.py, line ~230
def _get_base_url(self) -> str:
    # Use a proxy that converts OpenAI format to Gemini API
    return "https://generativelanguage.googleapis.com/v1beta/openai"
    
# Or use a local proxy like LiteLLM:
# return "http://localhost:8000/v1"  # LiteLLM proxying to Gemini
```

Then HMLR will:
- ✅ Still retrieve memories perfectly (this is what HMLR does best)
- ✅ Build context with facts, bridge blocks, user profile
- ✅ Send to Gemini Pro for the actual response generation
- ✅ Store Gemini's response back into memory

**Mix and Match:** You could even use different models for different operations:
- GPT-4.1-mini for fact extraction (cheap, fast)
- Gemini Pro for main chat responses (smart, strong reasoning)
- Claude for creative writing tasks

**Environment variable alternative:**
You could modify the code to read from an environment variable:
```python
def _get_base_url(self) -> str:
    # Custom endpoint support
    custom_endpoint = os.getenv("OPENAI_API_BASE")
    if custom_endpoint:
        return custom_endpoint
    
    # Default to official OpenAI
    if self.api_provider == "openai":
        return "https://api.openai.com/v1"
```

Then set in your `.env`:
```bash
OPENAI_API_BASE=https://your-custom-endpoint.com/v1
```

#### Temperature Settings

```python
# Line ~284: Main chat/query calls
temperature=0.7  # Default for most operations

# Line ~162: Planning operations  
temperature=0.6  # Slightly lower for structured planning
```

**What it controls:** Randomness in LLM responses
- **Lower (0.0-0.5)**: More deterministic, factual, consistent
- **Higher (0.7-1.0)**: More creative, varied responses
- **Default: 0.7** - Balanced for conversational use

**When to adjust:**
- Increase for more creative/varied responses
- Decrease for more consistent/factual outputs

#### Max Tokens

```python
# Line ~246: Default query max tokens
def query_external_api(self, query: str, max_tokens: int = 2000, model: str = "gpt-4.1-mini")

# Line ~171: Planning operations
max_tokens=4000  # For detailed plan drafts

# Line ~237: JSON plan generation
max_tokens=8000  # For 60-day structured plans
```

**What it controls:** Maximum length of LLM responses

**Defaults:**
- General queries: 2000 tokens (~1500 words)
- Planning drafts: 4000 tokens
- Large plans: 8000 tokens

**When to adjust:**
- Increase if responses are getting cut off
- Decrease to save costs and speed up responses

---

## Sliding Window Configuration

### Location: `memory/conversation_manager.py`

#### Window Size

```python
# Line ~78
self.sliding_window.max_turns = 20  # Keep last 20 turns in window
```

**What it controls:** How many recent conversation turns are kept in active memory

**Default: 20 turns**

**Memory impact:**
- Each turn ≈ 100-300 tokens
- 20 turns ≈ 2000-6000 tokens in sliding window
- Window is included in every LLM call

**When to adjust:**
- **Increase (e.g., 30-50)** for longer conversation context
  - Pro: More context, better continuity
  - Con: Higher token costs, slower responses
- **Decrease (e.g., 10-15)** for shorter context
  - Pro: Lower costs, faster responses
  - Con: May lose important context

**Location in code:**
```
hmlr/memory/conversation_manager.py:78
memory/conversation_manager.py:78 (original)
```



## Bridge Block Settings

### Location: `hmlr/memory/models.py`

#### Bridge Block Turn Limit

```python
# Line ~406
max_turns: int = 20  # Token budget management
```

**What it controls:** Maximum turns stored in a single bridge block before it's closed

**Default: 20 turns per block**

**When to adjust:**
- **Increase (e.g., 30-50)** for longer topical conversations
  - Keeps related discussion together longer
  - More context for cross-turn reasoning
  - Higher token usage when block is retrieved
- **Decrease (e.g., 10-15)** for more granular topic segmentation
  - Faster topic shifts
  - Smaller retrieval chunks
  - Less token overhead

**Impact on RAGAS scores:**
- Larger blocks → Better Context Recall (more history)
- Smaller blocks → Better Context Precision (less noise)

---

## Chunking Parameters

### Location: `hmlr/memory/chunking/chunk_engine.py`

#### Sentence Chunking

```python
# Sentence splitting regex (approximate)
SENTENCE_DELIMITERS = r'[.!?]+\s+'
```

**What it controls:** How text is split into sentence-level chunks

**Current behavior:**
- Splits on `.` `!` `?` followed by whitespace
- Preserves full sentences for embedding

**When to adjust:**
- Modify regex for different languages
- Adjust for technical content (e.g., code with periods)

#### Paragraph Chunking

**Current behavior:**
- Splits on double newlines (`\n\n`)
- Groups sentences into logical paragraphs

---

## Retrieval Configuration

### Location: `hmlr/memory/retrieval/`

#### Vector Search Similarity Threshold

```python
# Location: hmlr/memory/retrieval/hybrid_search.py
# Default similarity threshold
similarity_threshold: float = 0.4
```

**What it controls:** Minimum cosine similarity for chunk retrieval

**Default: 0.4** (relatively permissive)

**When to adjust:**
- **Increase (e.g., 0.6-0.8)** for stricter matching
  - Only highly relevant chunks retrieved
  - Better Precision, possibly lower Recall
- **Decrease (e.g., 0.2-0.3)** for broader matching
  - More chunks retrieved
  - Better Recall, possibly lower Precision

#### Top-K Results

```python
# Number of top candidates to retrieve
top_k: int = 10  # Typical default
```

**What it controls:** Maximum number of chunks to retrieve from vector search

---

## Token Budgets

### User Profile Context

```python
# Location: hmlr/core/conversation_engine.py
# Lines ~604, ~648
max_tokens=300  # User profile context budget
```

**What it controls:** Maximum tokens allocated for user profile in context

**Default: 300 tokens** (~200-250 words)

**When to adjust:**
- Increase for richer user profiles
- Decrease to save token budget for other context

### Total Context Budget

```python
# Location: hmlr/core/component_factory.py
# Line ~144
context_budget_tokens=4000  # Total context budget
```

**What it controls:** Total tokens available for hydrated context (facts + memories + profile)

**Default: 4000 tokens**

**Breakdown (approximate):**
- Sliding window: ~2000-3000 tokens
- User profile: ~300 tokens
- Facts: ~200-500 tokens
- Retrieved memories: ~500-1000 tokens

**When to adjust:**
- **Increase** if you have higher token budgets
- **Decrease** to reduce costs (may impact context quality)

---

## Memory Gardening

### Location: `hmlr/memory/gardener/manual_gardener.py`

#### Bridge Block Consolidation

```python
# Model used for bridge block generation
model="gpt-4.1-mini"  # Lines ~290, ~338
```

**What it controls:** Which model generates bridge block summaries

#### Fact Extraction

```python
# Location: hmlr/memory/fact_scrubber.py
# Line ~182
model="gpt-4.1-mini"  # Fast extraction
max_tokens=500       # Fact extraction limit
```

**What it controls:**
- Which model extracts facts from messages
- Maximum tokens for fact extraction response

---

## Configuration File Locations

### Quick Reference

| Setting | File | Line(s) |
|---------|------|---------|
| **API Endpoint** | `core/external_api_client.py` | **230** |
| Sliding window size | `hmlr/memory/conversation_manager.py` | 78 |
| Bridge block max turns | `hmlr/memory/models.py` | 406 |
| LLM temperature | `core/external_api_client.py` | 162, 284 |
| Default max tokens | `core/external_api_client.py` | 246 |
| User profile budget | `hmlr/core/conversation_engine.py` | 604, 648 |
| Context budget | `hmlr/core/component_factory.py` | 144 |
| Vector similarity threshold | `hmlr/memory/retrieval/hybrid_search.py` | Various |
| Fact extraction model | `hmlr/memory/fact_scrubber.py` | 182 |

---

## Recommended Experimentation Path

If you want to experiment with configuration, follow this order:

### 1. Start Small: Sliding Window
- **Safest change:** Adjust `max_turns` in sliding window
- **Impact:** Immediate and visible in context
- **Test:** Run a RAGAS test before/after to measure impact

### 2. Token Budgets
- Adjust `context_budget_tokens` to match your use case
- More budget = more context = higher costs
- Test with your specific conversation patterns

### 3. Bridge Block Size
- Modify `max_turns` per bridge block
- Affects long-term memory organization
- Requires multiple-session testing

### 4. Temperature (Advanced)
- Only adjust if you understand LLM behavior
- Small changes (±0.1) can have big impacts
- Always benchmark with RAGAS after changes

---

## Testing Your Changes

After modifying any configuration:

1. **Run smoke tests:**
   ```bash
   python test_package_smoke.py
   ```

2. **Run a quick RAGAS test:**
   ```bash
   cd tests
   pytest ragas_test_2a_vague_retrieval.py -v
   ```

3. **Check for regressions:**
   - Faithfulness should stay at 1.00
   - Context Recall should stay at 1.00
   - Precision may vary ±0.1

4. **Document your changes:**
   - Keep notes on what you changed and why
   - Record before/after RAGAS scores
   - Test with representative conversations

---

## Default vs Custom Configurations

### Default (Benchmarked)

These are the settings used for all published RAGAS benchmarks:

```python
# Sliding Window
max_turns = 20

# Bridge Blocks
max_turns = 20

# LLM
temperature = 0.7
max_tokens = 2000

# Context Budget
context_budget_tokens = 4000
user_profile_tokens = 300

# Model
model = "gpt-4.1-mini"
```

**Result:** 1.00 Faithfulness, 1.00 Context Recall across all tests

### When to Use Custom Configuration

- **Cost optimization:** Reduce window/budget to lower token usage
- **Specialized domains:** Adjust chunking for technical content
- **Different conversation patterns:** Longer/shorter blocks for your use case
- **Different model:** If testing with other models (unsupported, experimental)

---

## Common Configuration Mistakes

### ❌ Don't Do This

1. **Changing model without testing:**
   ```python
   model = "gpt-4o"  # NOT tested! Will likely break!
   ```

2. **Setting token budgets too low:**
   ```python
   max_tokens = 50  # Too small, responses will be cut off
   ```

3. **Extreme temperature values:**
   ```python
   temperature = 1.5  # Too high, responses will be incoherent
   temperature = 0.0  # Too deterministic, may be repetitive
   ```

4. **Forgetting to test:**
   - Changing settings without running tests
   - Assuming default behavior will hold

### ✅ Do This Instead

1. **Change one thing at a time**
2. **Test before and after**
3. **Document your changes**
4. **Keep the original values commented nearby**

```python
# Original (benchmarked): max_turns = 20
# Experiment: Trying larger window for longer conversations
self.sliding_window.max_turns = 30  # Testing: 2024-12-08
```

---

## Getting Help

If you're unsure about a configuration change:

1. Check this guide first
2. Review the RAGAS test that covers your use case
3. Start with small adjustments (±10-20% of default)
4. Always test with the full RAGAS suite before deploying

For questions or issues, see:
- [Model Compatibility](model_compatibility.md) - Model-specific warnings
- [Quickstart Guide](quickstart.md) - Basic usage
- [GitHub Issues](https://github.com/Sean-V-Dev/HMLR-Agentic-AI-Memory-System/issues)
