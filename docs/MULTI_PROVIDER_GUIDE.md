# Multi-Provider Configuration Guide

HMLR supports multiple LLM providers out of the box. The infrastructure is already built into `external_api_client.py`.

---

##  Supported Providers

| Provider | API_PROVIDER Value | API Key Env Var | Example Models | Status |
|----------|-------------------|-----------------|----------------|--------|
| **OpenAI** | `openai` | `OPENAI_API_KEY` | `gpt-4.1-mini`, `gpt-4.1-pro`, `gpt-4o`, `o1` | ✅ Fully Supported |
| **Google Gemini** | `gemini` | `GEMINI_API_KEY` | `gemini-3-flash-preview`, `gemini-3-pro-preview` | ✅ Fully Supported |
| **xAI Grok** | `grok` | `XAI_API_KEY` | `grok-4-1-fast-non-reasoning`, `grok-4-1-fast-reasoning` | ✅ Fully Supported |
| **Anthropic Claude** | `anthropic` | `ANTHROPIC_API_KEY` | `claude-haiku-4-5-20251001`, `claude-sonnet-4-5-20250929` | ✅ Fully Supported |
| **Local/Ollama** | `openai` | N/A | `llama3.2:3b`, `mistral:7b` |  Via OpenAI-compatible API |
| **DeepSeek** | `deepseek` | `DEEPSEEK_API_KEY` | `deepseek-chat`, `deepseek-reasoner` | ⚠️ Not Yet Implemented |

**Note:** Providers marked ⚠️ require adding support to `external_api_client.py`. PRs welcome!

---

##  Configuration Examples

### OpenAI (Default)

```bash
# .env or export
export API_PROVIDER="openai"
export OPENAI_API_KEY="sk-..."

# Use default mini for everything
export HMLR_DEFAULT_MODEL="gpt-4.1-mini"

# Or mix: mini for workers, pro for users
export HMLR_DEFAULT_MODEL="gpt-4.1-mini"
export HMLR_MAIN_MODEL="gpt-4.1-pro"
export HMLR_MAIN_TEMPERATURE="0.6"
```

**Result:** All operations use OpenAI models.

---

### Google Gemini

```bash
# .env or export
export API_PROVIDER="gemini"
export GEMINI_API_KEY="AIza..."

# Use Gemini's thinking model
export HMLR_DEFAULT_MODEL="gemini-3-flash-preview"

# Or use 1.5 Pro for main, Flash for workers
export HMLR_DEFAULT_MODEL="gemini-3-flash-preview"
export HMLR_MAIN_MODEL="gemini-3-pro-preview"
```

**Result:** All operations use Google Gemini models.

**Note:** Gemini uses the `google-genai` SDK internally.

---

### xAI Grok

```bash
# .env or export
export API_PROVIDER="grok"
export XAI_API_KEY="xai-..."

# Use Grok 4.1 with reasoning
export HMLR_DEFAULT_MODEL="grok-4-1-fast-reasoning"
```

**Result:** All operations use xAI Grok models.

**Note:** Grok uses the `xai-sdk` package internally.

---

### Anthropic Claude

```bash
# .env or export
export API_PROVIDER="anthropic"
export ANTHROPIC_API_KEY="sk-ant-..."

# Use Claude Sonnet
export HMLR_DEFAULT_MODEL="claude-sonnet-4-5-20250929"

# Or mix: Haiku for workers, Sonnet for main
export HMLR_DEFAULT_MODEL="claude-haiku-4-5-20251001"
export HMLR_MAIN_MODEL="claude-sonnet-4-5-20250929"
export HMLR_MAIN_TEMPERATURE="0.7"
```

**Result:** All operations use Anthropic Claude models.

**Note:** Claude uses the official `anthropic` SDK. Excellent for reasoning tasks and long context (200K tokens).

---

### Local Models (Ollama)

```bash
# .env or export
export API_PROVIDER="openai"  # Use OpenAI-compatible API
export OPENAI_API_KEY="ollama"  # Dummy key (Ollama doesn't need auth)

# Point to your local Ollama instance
export OPENAI_API_BASE="http://localhost:11434/v1"

# Use local model
export HMLR_DEFAULT_MODEL="llama3.2:3b"
```

**Result:** All operations use local Ollama models (no API costs!).

**Note:** Requires Ollama running locally with OpenAI-compatible API enabled.

---

##  Current Limitation: Single Provider per Session

**What Works:**
- One provider for all operations (e.g., all OpenAI or all Gemini)
- Different models from same provider (e.g., gpt-4.1-mini for workers, gpt-4.1-pro for main)

**What Doesn't Work (Yet):**
- Mixing providers within one session (e.g., OpenAI for main, Gemini for workers)

**Why:** The `API_PROVIDER` is set globally in `config.py`. The `ExternalAPIClient` is initialized once per session with one provider.

**Future Enhancement:** Could add per-operation provider overrides:
```python
# Hypothetical future syntax
MAIN_PROVIDER = "openai"
WORKER_PROVIDER = "gemini"
```

---

##  Testing Different Providers

### Quick Test Script

```python
from hmlr.core.config import config
from hmlr.core.model_config import model_config
from hmlr.core.external_api_client import ExternalAPIClient

# Initialize with your provider
client = ExternalAPIClient(api_provider=config.API_PROVIDER)

# Test a simple query
response = client.query_external_api(
    "What is 2+2?",
    model=model_config.get_main_model(),
    temperature=model_config.get_main_temperature()
)

print(f"Provider: {config.API_PROVIDER}")
print(f"Model: {model_config.get_main_model()}")
print(f"Response: {response}")
```

---

##  Advanced: Thinking/Reasoning Models

Some newer models support explicit reasoning/thinking controls. These are **optional** and only needed if using thinking models.

### Supported Thinking Models:

| Provider | Model | Reasoning Parameter | Values |
|----------|-------|---------------------|--------|
| **OpenAI** | `o1`, `o3` | `reasoning_effort` | `low`, `medium`, `high` |
| **Google** | `gemini-3-flash-preview` | thinking budget | `1-10` (integer) |
| **Anthropic** | Claude 4.5 Sonnet (extended thinking) | `thinking_enabled` | `true`/`false` |

### Configuration Examples:

#### Gemini Thinking Model
```bash
export API_PROVIDER="gemini"
export GEMINI_API_KEY="AIza..."
export HMLR_DEFAULT_MODEL="gemini-3-flash-preview"

# Control thinking budget (1-10, higher = more thinking time)
export HMLR_DEFAULT_REASONING_EFFORT="5"      # Moderate for workers
export HMLR_MAIN_REASONING_EFFORT="8"         # Deep thinking for users
```

**Use Case:** Fast workers (budget=5) for fact extraction, deep thinking (budget=8) for complex user questions.

#### OpenAI O1/O3 Models
```bash
export API_PROVIDER="openai"
export OPENAI_API_KEY="sk-..."
export HMLR_DEFAULT_MODEL="o1-mini"

# Control reasoning effort
export HMLR_DEFAULT_REASONING_EFFORT="low"    # Fast for workers
export HMLR_MAIN_REASONING_EFFORT="high"      # Deep reasoning for users
```

**Use Case:** Balance cost (o1 is expensive) by using low effort for background tasks.

#### When to Use Thinking Models:

**Use Thinking Models For:**
- ✅ Complex reasoning tasks (math, logic, planning)
- ✅ Multi-step problem solving
- ✅ Code generation and debugging
- ✅ Scientific or technical analysis

**Don't Use Thinking Models For:**
- ❌ Simple fact extraction (overkill, slower, more expensive)
- ❌ Keyword extraction or metadata
- ❌ Basic classification tasks
- ❌ Real-time chat where speed matters

**Recommended Pattern:**
```bash
# Use cheap/fast models for workers
export HMLR_DEFAULT_MODEL="gpt-4.1-mini"
export HMLR_LATTICE_MODEL="gemini-3-flash-preview"

# Use thinking model ONLY for main conversation
export HMLR_MAIN_MODEL="gemini-3-pro-preview"
export HMLR_MAIN_REASONING_EFFORT="7"
```

---

##  Advanced Model Parameters

Beyond thinking/reasoning, you can tune other model behaviors:

### Top-P (Nucleus Sampling)
Controls diversity by limiting to top probability tokens.

```bash
export HMLR_DEFAULT_TOP_P="0.95"  # 0.0-1.0, lower = more focused
```

- `0.1` = Very focused, deterministic
- `0.9` = More creative, diverse
- `1.0` = Consider all tokens (default for most models)

### Top-K Sampling
Limits sampling to top K most likely tokens (Gemini, local models).

```bash
export HMLR_DEFAULT_TOP_K="40"  # Integer, typically 1-100
```

- Lower = More focused
- Higher = More diverse

### Frequency Penalty
Reduces repetition by penalizing frequently used tokens.

```bash
export HMLR_DEFAULT_FREQUENCY_PENALTY="0.5"  # -2.0 to 2.0
```

- `0.0` = No penalty (default)
- `> 0` = Reduce repetition
- `< 0` = Encourage repetition (rare)

### Presence Penalty
Encourages new topics by penalizing tokens that have already appeared.

```bash
export HMLR_DEFAULT_PRESENCE_PENALTY="0.3"  # -2.0 to 2.0
```

- `0.0` = No penalty (default)
- `> 0` = Encourage new topics
- `< 0` = Stay on topic (rare)

### Example: Creative Writing Mode
```bash
export HMLR_MAIN_MODEL="grok-4-1-fast-reasoning"
export HMLR_MAIN_TEMPERATURE="0.8"           # High creativity
export HMLR_DEFAULT_TOP_P="0.95"             # Diverse sampling
export HMLR_DEFAULT_FREQUENCY_PENALTY="0.3"  # Reduce repetition
export HMLR_DEFAULT_PRESENCE_PENALTY="0.2"   # Encourage new topics
```

### Example: Precise Technical Mode
```bash
export HMLR_MAIN_MODEL="o1"
export HMLR_MAIN_TEMPERATURE="0.1"           # Low temperature
export HMLR_MAIN_REASONING_EFFORT="high"     # Deep thinking
export HMLR_DEFAULT_TOP_P="0.5"              # Very focused
export HMLR_DEFAULT_FREQUENCY_PENALTY="0.0"  # No penalties
```

---

##  Cost Optimization Strategies

### Strategy 1: Cheap Provider Everywhere
```bash
export API_PROVIDER="gemini"
export HMLR_DEFAULT_MODEL="gemini-3-flash-preview"  
```

### Strategy 2: Mix Cheap Workers + Premium Main
```bash
export API_PROVIDER="openai"
export HMLR_DEFAULT_MODEL="gpt-4.1-mini"      
export HMLR_MAIN_MODEL="gpt-4.1-pro"          
```

### Strategy 3: Free Local Development
```bash
export API_PROVIDER="openai"
export OPENAI_API_BASE="http://localhost:11434/v1"
export HMLR_DEFAULT_MODEL="llama3.2:3b"       # Free!
```

---

##  Provider Dependencies

Each provider requires specific packages:

### OpenAI (Default)
```bash
pip install openai requests
```

### Google Gemini
```bash
pip install google-genai
```

### xAI Grok
```bash
pip install xai-sdk
```

### Local/Ollama
```bash
# Install Ollama from https://ollama.ai
ollama pull llama3.2:3b
ollama serve
```

---

##  Environment Variables Reference

| Variable | Purpose | Example |
|----------|---------|---------|
| `API_PROVIDER` | Which provider to use | `openai`, `gemini`, `grok` |
| `OPENAI_API_KEY` | OpenAI authentication | `sk-proj-...` |
| `GEMINI_API_KEY` | Google Gemini auth | `AIza...` |
| `XAI_API_KEY` | xAI Grok auth | `xai-...` |
| `OPENAI_API_BASE` | Override OpenAI base URL (for Ollama) | `http://localhost:11434/v1` |
| `HMLR_DEFAULT_MODEL` | Global model for all operations | `gpt-4.1-mini` |
| `HMLR_MAIN_MODEL` | Override for user-facing | `gpt-4.1-pro` |
| `HMLR_LATTICE_MODEL` | Override for lattice ops | `gemini-3-flash-preview` |
| `HMLR_SYNTHESIS_MODEL` | Override for synthesis | `claude-sonnet-4-5-20250929` |

---

## ✅ Validation Checklist

After switching providers, verify:

1. **API Key is set**: Check the corresponding `*_API_KEY` env var
2. **Model name is valid**: Use provider-specific model names
3. **SDK is installed**: Install provider's Python package
4. **Test connection**: Run a simple query to verify

```bash
# Quick test
python -c "from hmlr.core.external_api_client import ExternalAPIClient; from hmlr.core.config import config; client = ExternalAPIClient(config.API_PROVIDER); print(f'✅ {config.API_PROVIDER} connected')"
```

---

##  Example: Complete Provider Switch

### From OpenAI to Google Gemini

1. **Update .env file:**
   ```bash
   # Comment out OpenAI
   # API_PROVIDER=openai
   # OPENAI_API_KEY=sk-...
   # HMLR_DEFAULT_MODEL=gpt-4.1-mini
   
   # Add Gemini
   API_PROVIDER=gemini
   GEMINI_API_KEY=AIza...
   HMLR_DEFAULT_MODEL=gemini-3-flash-preview
   ```

2. **Install Gemini SDK (if not already):**
   ```bash
   pip install google-genai
   ```

3. **Restart application:**
   ```bash
   python main.py
   ```

4. **Verify in logs:**
   ```
    API Provider: gemini
    Model: gemini-3-flash-preview
   ```

---

**TL;DR:** Just set `API_PROVIDER`, the right `*_API_KEY`, and `HMLR_DEFAULT_MODEL` to any provider's model. Everything else stays the same!

---

##  Adding New Providers (For Contributors)

Want to add Anthropic, DeepSeek, or another provider? Here's how:

### Step 1: Update `external_api_client.py`

Add your provider to the `_load_api_key()` method:

```python
def _load_api_key(self) -> str:
    """Load API key from environment"""
    # ... existing providers ...
    
    elif self.api_provider == "anthropic":
        key = os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise ConfigurationError("ANTHROPIC_API_KEY not found")
        return key
```

### Step 2: Add API Call Method

Create a `_call_anthropic_api()` method (follow pattern of `_call_gemini_api()`):

```python
def _call_anthropic_api(self, model: str, messages: List[Dict], 
                        max_tokens: int, temperature: float) -> Dict[str, Any]:
    """Make API call to Anthropic Claude"""
    try:
        # Use Anthropic's SDK or REST API
        # Return normalized response matching OpenAI format
        return {
            'choices': [{'message': {'content': response_text}}],
            'model': model,
            'usage': {...}
        }
    except Exception as e:
        print(f" Anthropic request failed: {e}")
        raise
```

### Step 3: Route in `query_external_api()`

Add routing logic:

```python
def query_external_api(self, query: str, ...):
    # ... existing code ...
    
    if self.api_provider == "anthropic":
        response_json = self._call_anthropic_api(model, messages, max_tokens, temperature)
    # ... rest of method
```

### Step 4: Update Configuration

1. Add to `.env.template` with example
2. Add to `MULTI_PROVIDER_GUIDE.md` table
3. Update `model_config.py` comments

### Step 5: Test

```bash
export API_PROVIDER="anthropic"
export ANTHROPIC_API_KEY="sk-ant-..."
export HMLR_DEFAULT_MODEL="claude-sonnet-4-5-20250929"
python main.py
```

### Example: Anthropic Implementation

```python
# In external_api_client.py

# Install: pip install anthropic
from anthropic import Anthropic

def _load_api_key(self) -> str:
    # ... existing code ...
    elif self.api_provider == "anthropic":
        key = os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise ConfigurationError("ANTHROPIC_API_KEY not found")
        self.anthropic_client = Anthropic(api_key=key)
        return key

def _call_anthropic_api(self, model: str, messages: List[Dict], 
                        max_tokens: int, temperature: float) -> Dict[str, Any]:
    """Call Anthropic Claude API"""
    try:
        # Convert messages format
        system_msg = next((m['content'] for m in messages if m['role'] == 'system'), None)
        user_msgs = [m for m in messages if m['role'] != 'system']
        
        # Make API call
        response = self.anthropic_client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_msg,
            messages=user_msgs
        )
        
        # Normalize to OpenAI format
        return {
            'choices': [{'message': {'content': response.content[0].text}}],
            'model': model,
            'usage': {
                'prompt_tokens': response.usage.input_tokens,
                'completion_tokens': response.usage.output_tokens,
                'total_tokens': response.usage.input_tokens + response.usage.output_tokens
            }
        }
    except Exception as e:
        print(f" Anthropic API error: {e}")
        raise
```

**Pull requests welcome!** See the existing Gemini and Grok implementations as reference.
