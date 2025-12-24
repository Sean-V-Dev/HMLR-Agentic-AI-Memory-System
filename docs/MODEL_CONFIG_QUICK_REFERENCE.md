# Model Config Quick Reference

## 🎯 Hierarchical Configuration System

### The Key Concept

You have **ONE global knob** (`DEFAULT_MODEL`) that controls the entire stack, plus **optional per-operation overrides** for fine-tuning.

```
┌─────────────────────────────────────────────────────────────┐
│  DEFAULT_MODEL = "gpt-4.1-mini"   ← Changes everything      │
└─────────────────────────────────────────────────────────────┘
                            │
           ┌────────────────┼────────────────┬───────────────┐
           ▼                ▼                ▼               ▼
     MAIN_MODEL      LATTICE_MODEL    SYNTHESIS_MODEL   NANO_MODEL
    (if not set,    (if not set,      (if not set,    (if not set,
     uses DEFAULT)   uses DEFAULT)     uses DEFAULT)   uses DEFAULT)
```

---

## 📖 Usage Examples

### Example 1: Everything Fast & Cheap
**Goal:** Use `gpt-4.1-mini` everywhere

```bash
# Option A: Do nothing (this is the default)
python main.py

# Option B: Explicit
export HMLR_DEFAULT_MODEL="gpt-4.1-mini"
python main.py
```

**Result:**
- Main conversation: `gpt-4.1-mini`
- Lattice operations: `gpt-4.1-mini`
- Synthesis/facts: `gpt-4.1-mini`
- Metadata extraction: `gpt-4.1-mini`

---

### Example 2: Fast Workers + Premium Endpoint (Better UX)
**Goal:** Background tasks use mini, user-facing conversations use pro with thinking AND higher temperature for natural flow

```bash
export HMLR_DEFAULT_MODEL="gpt-4.1-mini"
export HMLR_MAIN_MODEL="gpt-4.1-pro"
export HMLR_DEFAULT_TEMPERATURE="0.1"      # Deterministic workers
export HMLR_MAIN_TEMPERATURE="0.6"         # Natural conversation
python main.py
```

**Result:**
- Main conversation: `gpt-4.1-pro` @ 0.6 temp ✨ (premium model, natural style)
- Lattice operations: `gpt-4.1-mini` @ 0.1 temp (fast, deterministic)
- Synthesis/facts: `gpt-4.1-mini` @ 0.1 temp (fast, deterministic)
- Metadata extraction: `gpt-4.1-mini` @ 0.1 temp (fast, deterministic)

**Why this works:**
- **Cost:** Only pay for pro where it matters (user-facing)
- **Quality:** Higher temp makes conversation feel more natural
- **Reliability:** Workers stay deterministic for accurate fact extraction

---

### Example 3: Multi-Provider Strategy
**Goal:** OpenAI for main, Google for search, Anthropic for synthesis

```bash
export HMLR_DEFAULT_MODEL="gpt-4.1-mini"
export HMLR_MAIN_MODEL="gpt-4.1-pro"
export HMLR_LATTICE_MODEL="gemini-2.0-flash-thinking"
export HMLR_SYNTHESIS_MODEL="claude-3-sonnet"
python main.py
```

**Result:**
- Main conversation: `gpt-4.1-pro` (OpenAI)
- Lattice operations: `gemini-2.0-flash-thinking` (Google)
- Synthesis/facts: `claude-3-sonnet` (Anthropic)
- Metadata extraction: `gpt-4.1-mini` (from DEFAULT)

**Use Case:** Leverage each provider's strengths (OpenAI for chat, Google for search, Anthropic for reasoning).

---

### Example 4: Testing/Development
**Goal:** Use a fast local model for development

```bash
export HMLR_DEFAULT_MODEL="ollama/llama3.2:3b"
python main.py
```

**Result:** Everything uses local model (no API costs).

---

## 🔧 In-Code Override (for tests)

```python
from hmlr.core.model_config import ModelConfig

# Override defaults directly (useful in tests)
ModelConfig.DEFAULT_MODEL = "mock-model-for-testing"
ModelConfig.SYNTHESIS_MODEL = "mock-synthesis-model"

# Now all operations use these values
```

---

## 🔍 How It Works Under the Hood

### The Old Way (Hardcoded)
```python
# ❌ Inflexible
response = self.api_client.query_external_api(
    prompt,
    model="gpt-4.1-mini"  # Can't change without editing code
)
```

### The New Way (Hierarchical)
```python
from hmlr.core.model_config import model_config

# ✅ Flexible
response = self.api_client.query_external_api(
    prompt,
    model=model_config.get_synthesis_model()  # Falls back: SYNTHESIS_MODEL → DEFAULT_MODEL
)
```

### What `get_synthesis_model()` Does:
```python
@classmethod
def get_synthesis_model(cls) -> str:
    """
    1. Check if SYNTHESIS_MODEL is explicitly set
    2. If not, fall back to DEFAULT_MODEL
    3. If that's also not set, use hardcoded "gpt-4.1-mini"
    """
    return cls.SYNTHESIS_MODEL or cls.DEFAULT_MODEL
```

---

## 🌍 Environment Variables Reference

| Variable | Default | Affects | Use Case |
|----------|---------|---------|----------|
| **MODELS** |
| `HMLR_DEFAULT_MODEL` | `gpt-4.1-mini` | All operations (global) | Change entire stack at once |
| `HMLR_MAIN_MODEL` | *(falls back to DEFAULT)* | User-facing conversations | Premium experience for users |
| `HMLR_LATTICE_MODEL` | *(falls back to DEFAULT)* | Topic classification, context filtering | Fast search operations |
| `HMLR_SYNTHESIS_MODEL` | *(falls back to DEFAULT)* | Dossier synthesis, fact extraction | Reasoning-heavy tasks |
| `HMLR_NANO_MODEL` | *(falls back to DEFAULT)* | Metadata extraction | Lightweight analysis |
| **TEMPERATURE** |
| `HMLR_DEFAULT_TEMPERATURE` | `0.1` | All operations (global) | Deterministic by default |
| `HMLR_MAIN_TEMPERATURE` | *(falls back to DEFAULT)* | User-facing conversations | Higher (0.5-0.7) for natural UX |
| `HMLR_WORKER_TEMPERATURE` | *(falls back to DEFAULT)* | All worker operations | Keep low (0.1) for accuracy |
| **TOKEN BUDGETS** |
| `CONTEXT_BUDGET_TOKENS` | `6000` | Context assembly | Total token budget |
| `MAX_RESPONSE_TOKENS` | `2000` | LLM responses | Max output length |
| `FACT_EXTRACTION_MAX_TOKENS` | `500` | Fact scrubber | Small, focused extractions |
| `USER_PROFILE_MAX_TOKENS` | `300` | User profile context | Compact summaries |
| `HYDRATOR_MAX_TOKENS` | `50000` | Context hydrator | Large retrievals |
| **EMBEDDINGS** |
| `HMLR_EMBEDDING_MODEL` | `BAAI/bge-large-en-v1.5` | Vector search | Semantic similarity |
| `HMLR_EMBEDDING_DIM` | `1024` | Database schema | Must match model |

---

## ✅ Migration Checklist

- [ ] `model_config.py` created with hierarchical defaults
- [ ] All 11 hardcoded model references updated to use `get_*_model()` methods
- [ ] Environment variable overrides tested
- [ ] Documentation updated with usage examples
- [ ] Example `.env.example` file created

---

## 🚀 Deployment Scenarios

### Production (Cost-Optimized)
```bash
HMLR_DEFAULT_MODEL=gpt-4.1-mini      # Cheap for workers
HMLR_MAIN_MODEL=gpt-4.1-pro          # Premium for users
```

### Development (Fast & Free)
```bash
HMLR_DEFAULT_MODEL=ollama/llama3.2:3b
```

### Testing (Deterministic)
```bash
HMLR_DEFAULT_MODEL=gpt-4.1-mini
HMLR_API_PROVIDER=mock  # Mock responses
```

### Research (Best Quality)
```bash
HMLR_DEFAULT_MODEL=gpt-4.1-pro
HMLR_SYNTHESIS_MODEL=claude-3-opus-20240229
```

---

**TL;DR:** 
- Set `DEFAULT_MODEL` once to control everything
- Override specific operations (`MAIN_MODEL`, `SYNTHESIS_MODEL`, etc.) when needed
- Set `DEFAULT_TEMPERATURE=0.1` for workers, override `MAIN_TEMPERATURE` for user-facing UX
- Token budgets are operation-specific (no global default needed)
