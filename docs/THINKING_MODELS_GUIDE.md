# Thinking Models Configuration Examples

Quick reference for configuring thinking/reasoning models.

---

## 🎯 Common Configurations

### 1. Gemini 2.0 Flash Thinking (Recommended)

**Best for:** Cost-effective thinking with good balance

```bash
# Provider & API
export API_PROVIDER="gemini"
export GEMINI_API_KEY="AIza-your-key"

# Models: Thinking model for user, fast for workers
export HMLR_DEFAULT_MODEL="gemini-1.5-flash"              # Fast workers
export HMLR_MAIN_MODEL="gemini-2.0-flash-thinking-exp"   # Thinking for users

# Thinking Budget: 1-10 scale (higher = more thinking)
export HMLR_DEFAULT_REASONING_EFFORT="3"                  # Light thinking for workers
export HMLR_MAIN_REASONING_EFFORT="7"                     # Deep thinking for users

# Temperature
export HMLR_DEFAULT_TEMPERATURE="0.1"                     # Deterministic workers
export HMLR_MAIN_TEMPERATURE="0.5"                        # Natural conversation
```

**Cost:** Low - Flash models are very cheap even with thinking
**Speed:** Fast - Flash is optimized for speed
**Quality:** High - Gemini 2.0 thinking is excellent

---

### 2. OpenAI O1/O3 (Premium)

**Best for:** Maximum reasoning quality (expensive!)

```bash
# Provider & API
export API_PROVIDER="openai"
export OPENAI_API_KEY="sk-..."

# Models: Mini for workers, O1 for complex reasoning
export HMLR_DEFAULT_MODEL="gpt-4.1-mini"                  # Cheap workers
export HMLR_MAIN_MODEL="o1"                               # O1 for users only

# Reasoning Effort: low, medium, high
export HMLR_MAIN_REASONING_EFFORT="medium"                # Balance cost/quality

# Temperature (note: o1 has fixed low temperature internally)
export HMLR_DEFAULT_TEMPERATURE="0.1"
```

**Cost:** HIGH - O1 is ~$15-60 per 1M tokens
**Speed:** Slower - O1 takes time to reason
**Quality:** Maximum - Best reasoning available

**When to use:** Complex math, coding, scientific analysis

---

### 3. Hybrid: Fast Workers + Thinking Endpoint

**Best for:** Cost optimization with thinking where it matters

```bash
# Provider
export API_PROVIDER="openai"
export OPENAI_API_KEY="sk-..."

# Models: All cheap except main conversation
export HMLR_DEFAULT_MODEL="gpt-4.1-mini"                  # $0.15/1M
export HMLR_LATTICE_MODEL="gpt-4.1-mini"                  # $0.15/1M  
export HMLR_SYNTHESIS_MODEL="gpt-4.1-mini"                # $0.15/1M
export HMLR_MAIN_MODEL="o1-mini"                          # $3/1M (only for users)

# Reasoning: Only main conversation thinks deeply
export HMLR_MAIN_REASONING_EFFORT="high"

# Temperature
export HMLR_DEFAULT_TEMPERATURE="0.1"                     # Workers stay deterministic
```

**Cost Breakdown:**
- Workers (90% of calls): $0.15/1M tokens
- User-facing (10% of calls): $3/1M tokens
- **Average:** ~$0.44/1M tokens (73% cheaper than all-o1-mini)

---

### 4. Local Development (Free!)

**Best for:** Testing without API costs

```bash
# Provider (use OpenAI-compatible API)
export API_PROVIDER="openai"
export OPENAI_API_BASE="http://localhost:11434/v1"

# Model: Local Llama
export HMLR_DEFAULT_MODEL="llama3.2:3b"

# Temperature
export HMLR_DEFAULT_TEMPERATURE="0.1"

# No reasoning parameters (local models don't support them)
```

**Cost:** $0
**Speed:** Depends on your hardware
**Quality:** Good for testing, not production-grade

---

## 🎛️ Parameter Guide

### Reasoning Effort Values

| Provider | Parameter | Values | Meaning |
|----------|-----------|--------|---------|
| **OpenAI o1/o3** | `reasoning_effort` | `low` | Fast, cheaper reasoning |
| | | `medium` | Balanced (default) |
| | | `high` | Deep reasoning, slower |
| **Google Gemini** | thinking budget | `1-3` | Light thinking |
| | | `4-6` | Moderate thinking |
| | | `7-10` | Deep thinking |

### When to Use What:

**Low/1-3:** Simple queries, fact lookups, basic Q&A
**Medium/4-6:** Standard conversations, explanations, summaries  
**High/7-10:** Complex reasoning, math, multi-step problems, code

---

## 💡 Pro Tips

### Tip 1: Don't Overthink Simple Tasks
```bash
# ❌ Wasteful: Using thinking for everything
export HMLR_DEFAULT_MODEL="o1"

# ✅ Efficient: Thinking only where needed
export HMLR_DEFAULT_MODEL="gpt-4.1-mini"
export HMLR_MAIN_MODEL="o1-mini"
```

### Tip 2: Test Reasoning Effort Incrementally
```bash
# Start low, increase only if quality improves
export HMLR_MAIN_REASONING_EFFORT="low"    # Test first
# If not good enough:
export HMLR_MAIN_REASONING_EFFORT="medium"  # Try this
# Still not enough:
export HMLR_MAIN_REASONING_EFFORT="high"   # Last resort
```

### Tip 3: Temperature vs Reasoning
- **Temperature:** Controls randomness/creativity
- **Reasoning:** Controls depth of thinking

```bash
# For precise technical answers:
export HMLR_MAIN_TEMPERATURE="0.1"          # Low randomness
export HMLR_MAIN_REASONING_EFFORT="high"    # Deep thinking

# For creative brainstorming:
export HMLR_MAIN_TEMPERATURE="0.7"          # High creativity
export HMLR_MAIN_REASONING_EFFORT="medium"  # Some thinking, not exhaustive
```

### Tip 4: Worker Models Never Need Thinking
```bash
# Workers (lattice, synthesis, fact extraction) should be:
# - Fast
# - Cheap  
# - Deterministic (low temp)
# - NO reasoning parameters

export HMLR_DEFAULT_MODEL="gpt-4.1-mini"
export HMLR_DEFAULT_TEMPERATURE="0.1"
# No HMLR_DEFAULT_REASONING_EFFORT needed!
```

---

## 🧪 Quick Test Script

Test your thinking model configuration:

```python
from hmlr.core.config import config
from hmlr.core.model_config import model_config
from hmlr.core.external_api_client import ExternalAPIClient

print("=== Configuration ===")
print(f"Provider: {config.API_PROVIDER}")
print(f"Main Model: {model_config.get_main_model()}")
print(f"Reasoning Effort: {model_config.get_reasoning_effort('main')}")
print(f"Temperature: {model_config.get_main_temperature()}")

# Test query
client = ExternalAPIClient(config.API_PROVIDER)
response = client.query_external_api(
    "Explain quantum entanglement in one sentence.",
    model=model_config.get_main_model(),
    temperature=model_config.get_main_temperature()
)

print(f"\n=== Response ===\n{response}")
```

---

## 📊 Cost Comparison

Assuming 1M tokens per month, mixed workload:

| Configuration | Monthly Cost | Use Case |
|---------------|--------------|----------|
| All gpt-4.1-mini | $0.15 | Development, simple tasks |
| Mini workers + Gemini thinking main | $2-5 | Production (recommended) |
| Mini workers + o1-mini main (low) | $5-10 | Complex reasoning, moderate budget |
| Mini workers + o1 main (high) | $15-40 | Maximum quality, no budget limit |
| All o1 (high effort) | $60+ | Research, critical applications only |

**Recommendation:** Start with Gemini thinking (cheap, fast) and upgrade to o1 only if you need maximum reasoning quality.

---

**TL;DR:** Set `HMLR_MAIN_MODEL` to a thinking model, add `HMLR_MAIN_REASONING_EFFORT`, keep workers on cheap fast models.
