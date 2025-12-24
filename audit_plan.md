# HMLR Code Quality Audit - Actionable Issues

**Date:** December 23, 2025  
**Status:** Post-Phase 2 cleanup (multi-provider support complete, deprecated systems removed)

---

## 🎯 Philosophy
Following the "Anti-Slop" mandate: Fix real problems, delete dead code, fail loudly. No performative complexity.

---

## ⚠️ Priority 1: Silent Failures (HIGH RISK) ✅ COMPLETE

**STATUS:** ✅ Fixed across 9 files on [date]

### Problem
Multiple `except Exception` blocks that swallow errors and continue with empty defaults. Hides upstream problems and makes debugging impossible.

### Affected Files (ALL FIXED)
1. ✅ **external_api_client.py** - Added logging, fixed `_fetch_available_models()`
2. ✅ **storage.py** - All exception handlers now use `exc_info=True`
3. ✅ **lattice.py** - Added FALLBACK warnings for fail-open scenarios
4. ✅ **conversation_engine.py** - Background tasks, bridge block updates log properly
5. ✅ **fact_scrubber.py** - LLM/JSON failures log with stacktrace
6. ✅ **crawler.py** - Vector search failures log properly
7. ✅ **component_factory.py** - Init failures log with exc_info

### What Changed
- All exceptions now log with `exc_info=True` for full stack traces
- Critical operations still raise explicit exceptions (fail fast)
- Recoverable errors log and return safe defaults
- Fail-open scenarios clearly marked with "FALLBACK" warnings

### Fix Strategy
- **Critical operations**: Raise explicit exceptions (ConfigurationError, ValidationError)
- **Recoverable errors**: Log with stacktrace using `logger.error()`, add metrics/counters
- **Fail-open scenarios**: Add warning logs and flag results as "fallback mode"

### Example Refactor
```python
# BEFORE (Silent Failure)
try:
    models = api_client.get_models()
except Exception as e:
    print(f"Error: {e}")
    return []

# AFTER (Fail Loud)
try:
    models = api_client.get_models()
except APIConnectionError as e:
    logger.error(f"Failed to fetch models: {e}", exc_info=True)
    raise ConfigurationError("Cannot initialize without model list") from e
```

---

## 📝 Priority 2: Debug File Writes (MEDIUM RISK) ✅ COMPLETE

**STATUS:** ✅ Removed all debug file writes on [date]

### Problem
**lattice.py** and **conversation_engine.py**: Every memory filter wrote full candidate dumps to `debug_llm_flow.txt`. This was:
- I/O-bound on every query
- Writes potentially sensitive data to disk
- Never cleaned up (file grows indefinitely)
- Baked into runtime (not gated by debug flag)

### What Changed
1. ✅ **lattice.py lines 557-587**: Removed ~30 lines writing candidates to debug_llm_flow.txt
2. ✅ **lattice.py lines 614-631**: Removed ~20 lines writing filter results
3. ✅ **conversation_engine.py lines 376-387**: Removed duplicate debug write (~12 lines)
4. ✅ **config.py**: Removed `DEBUG_LLM_FLOW_PATH` constant
5. Replaced all with `logger.debug()` calls (gated by log level, no disk I/O)

---

## 🔊 Priority 3: Print vs Logging (MEDIUM PRIORITY) ✅ COMPLETE

**STATUS:** ✅ Migrated ~85 print() statements to logger calls across 11 files on [date]

### Problem
Inconsistent observability - mix of `print()` and `logging`. Makes it impossible to:
- Control verbosity in production
- Filter logs by level
- Capture structured logs for analysis

### Files Changed (ALL COMPLETE)
1. ✅ **lattice.py** - 29 prints → logger.info/debug
2. ✅ **conversation_engine.py** - 41 prints → logger.info/debug/warning
3. ✅ **component_factory.py** - 29 prints → logger.info/warning
4. ✅ **external_api_client.py** - 13 prints → logger.debug/error
5. ✅ **synthesis_engine.py** - 5 prints → logger.info/debug
6. ✅ **scribe.py** - 2 prints → logger.info/error
7. ✅ **storage.py** - 4 prints → logger.info
8. ✅ **sliding_window_persistence.py** - 5 prints → logger.info/debug/warning
9. ✅ **metadata_extractor.py** - 1 print → logger.warning (test block kept)
10. ✅ **model_config.py** - 2 prints → logger.warning
11. ✅ **fact_scrubber.py** - Already done in Priority 1
12. ✅ **crawler.py** - Already done in Priority 1

**REMAINING PRINTS (ACCEPTABLE):**
- `__init__.py` line 23: Docstring example (not runtime code)
- `run_gardener.py`: CLI script (print() correct for CLI output)
- `metadata_extractor.py` lines 360-404: Test block (acceptable for testing)

### Logging Levels Used
- User-facing status: `logger.info()`
- Debug/verbose info: `logger.debug()` (hidden by default)
- Warnings: `logger.warning()`
- Errors: `logger.error()` with `exc_info=True`

---

## 🔧 Priority 4: API Extensibility (LOW PRIORITY) ✅ COMPLETE

**STATUS:** ✅ Refactored `ExternalAPIClient` to support `**options` passthrough on December 23, 2025

### Problem
`ExternalAPIClient.query_external_api()` had fixed signature - could not accept custom headers, timeouts, or provider-specific metadata.

### What Changed
- Added `**options` to `query_external_api` and all provider-specific `_call_*` methods
- Supports `headers` override in OpenAI/responses calls
- Supports `timeout` and `temperature` overrides across all providers
- Automatically passes unknown parameters as provider-specific payload items (e.g., `top_p`, `safety_settings` for Gemini, etc.)
- Improved error handling and normalization for all providers

---

## 💾 Priority 5: Atomic File Writes (LOW PRIORITY) ✅ COMPLETE

**STATUS:** ✅ Implemented atomic write pattern in `UserProfileManager` on December 23, 2025

### Problem
`user_profile_lite.json` (written by Scribe) used direct `json.dump()`. If system crashed mid-write, the file could become corrupted.

### What Changed
- Updated `UserProfileManager._create_default_profile` and `update_profile_db` to use atomic write pattern
- Data is written to a `.tmp` file first, then moved with `os.replace()`
- Ensures data integrity even if the system loses power during a write

### Affected Files
- `hmlr/memory/synthesis/scribe.py` - profile write operations

### Fix Strategy
Use atomic write pattern:
1. Write to temporary file (`user_profile_lite.json.tmp`)
2. Use `os.replace()` to atomically swap with real file
3. Ensures file is either 100% old or 100% new, never corrupted

### Example Implementation
```python
import os
import json

def save_profile_atomic(profile_data: dict, path: str):
    """Save profile with atomic write to prevent corruption."""
    # Write to temp file first
    temp_path = f"{path}.tmp"
    with open(temp_path, 'w', encoding='utf-8') as f:
        json.dump(profile_data, f, indent=2)
    
    # Atomic replace (either succeeds or fails, never partial)
    os.replace(temp_path, path)
```

---

## ✅ Already Fixed (No Action Needed)

- ~~Hardcoded model names~~ → Centralized in model_config.py
- ~~Hardcoded OpenAI base URL~~ → Multi-provider support added (Gemini, Grok, Claude, Anthropic)
- ~~Deprecated keyword/synthesis systems~~ → Removed (~1,300 lines deleted)
- ~~Telemetry references~~ → Cleaned up

---

## 📋 Implementation Plan

### Phase A: Exception Handling (1-2 hours)
1. Grep for `except Exception` patterns
2. Categorize as critical (raise) vs recoverable (log + metrics)
3. Replace silent failures with explicit error handling
4. Add ConfigurationError for setup failures

### Phase B: Debug File Writes (30 mins)
1. Remove `debug_llm_flow.txt` writes from lattice.py
2. Keep console prints but convert to logger.debug()
3. Add config.DEBUG_MODE flag for future use

### Phase C: Print → Logging Migration (2-3 hours)
1. Add logging config to main.py (set level from env variable)
2. Replace prints in lattice.py
3. Replace prints in conversation_engine.py
4. Replace prints in external_api_client.py
5. Search for remaining `print(` and migrate

### Phase D: API Extensibility (Optional, 1-2 hours) ✅ COMPLETE
1. Add `**options` to query_external_api
2. Pass through to _call_* methods
3. Update documentation with examples

### Phase E: Native Async Client Support (2-3 hours)
1.  Introduce an asynchronous LLM interface.
2.  Switch `ExternalAPIClient` to use native async SDKs (AsyncOpenAI, AsyncAnthropic, Gemini `.aio`).
3.  Replace `requests` with `httpx` for generic API calls.
4.  Remove `run_in_executor` from call sites in `scribe.py`, `lattice.py`, etc.
5.  Wait/await results directly in async flows.

---

## 🧪 Validation

After fixes:
1. Run full test suite (`pytest`)
2. Grep for remaining `except: pass` patterns
3. Grep for remaining bare `print(` (should only be in CLI output)
4. Test with `DEBUG_MODE=false` to ensure no debug file writes

---

## 🚫 Explicitly Rejected (Not Problems)

1. **"Huge LLM prompts"** - FALSE. Governor prompts are ~2-4K tokens, not "massive"
2. **Shadow state** - MISUNDERSTANDING. Bridge blocks ARE persisted; sliding window is just a cache
3. **Architectural leakage** - PREMATURE. DI/ABC patterns add complexity without clear benefit

---

## ⚡ Priority 6: Native Async Client Support (MEDIUM PRIORITY)

### Problem
The entire HMLR conversation flow is `async` (from `HMLRClient.chat()` down through `ConversationEngine.process_user_message()`), but the LLM client layer (`ExternalAPIClient`) uses **synchronous** blocking calls:
- Uses `requests` library (blocking I/O)
- All SDK calls are synchronous (OpenAI, Gemini, Anthropic, Grok)
- Wrapped with `loop.run_in_executor(None, ...)` to avoid blocking the event loop

This "thread-pool workaround" creates unnecessary overhead:
- Thread pool spawning/teardown on every LLM call
- Can't leverage native async connection pooling in modern SDKs
- Makes parallel LLM calls (Governor's 3 parallel tasks) less efficient
- Adds complexity to call sites (every LLM call needs `run_in_executor` wrapper)

### Affected Files
1. **hmlr/core/external_api_client.py** - All methods are synchronous
2. **hmlr/memory/synthesis/scribe.py** - Lines 113-117: `run_in_executor` wrapper
3. **hmlr/memory/retrieval/lattice.py** - Lines 181-182, 190: `run_in_executor` for fact lookups
4. **Any file calling `query_external_api()`** - 11 call sites that would benefit from native async

### Current Architecture
```python
# Current (inefficient):
async def run_scribe_agent(self, user_input: str):
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(
        None,                          # Thread pool
        self._query_llm,               # Synchronous blocking call
        user_input
    )
```

### Desired Architecture
```python
# Native async (efficient):
async def run_scribe_agent(self, user_input: str):
    response = await self.api_client.query_external_api_async(
        user_input
    )
```

### Fix Strategy

**Phase 1: Update Dependencies**
- Add `httpx` for async HTTP requests (replaces `requests`)
- Update to async SDK versions:
  - `openai` → already has `AsyncOpenAI`
  - `anthropic` → already has `AsyncAnthropic`
  - `google-genai` → has `.aio` module
  - `xai-sdk` → check for async support or use httpx directly

**Phase 2: Refactor ExternalAPIClient**
1. Create async versions of all methods:
   - `query_external_api()` → `query_external_api_async()`
   - `_call_openai_api()` → `_call_openai_api_async()`
   - `_call_gemini_api()` → `_call_gemini_api_async()`
   - `_call_anthropic_api()` → `_call_anthropic_api_async()`
   - `_call_grok_api()` → `_call_grok_api_async()`

2. Replace `requests` with `httpx.AsyncClient`:
   ```python
   async with httpx.AsyncClient(timeout=timeout) as client:
       response = await client.post(url, headers=headers, json=payload)
   ```

3. Use native async SDKs:
   - OpenAI: `AsyncOpenAI().chat.completions.create()`
   - Anthropic: `AsyncAnthropic().messages.create()`
   - Gemini: `genai.Client(...).aio.models.generate_content()`

**Phase 3: Update Call Sites**
Remove `run_in_executor` wrappers from:
1. **scribe.py** (line 113): Direct `await self.api_client.query_external_api_async()`
2. **lattice.py** (lines 181-182, 190): Keep parallel execution but remove executor
3. **dossier_governor.py** (4 call sites): Direct async calls
4. **fact_scrubber.py** (1 call site): Direct async call
5. **manual_gardener.py** (2 call sites): Direct async calls
6. **conversation_engine.py** (1 call site): Direct async call

**Phase 4: Maintain Backward Compatibility (Optional)**
Keep synchronous methods for any CLI tools or non-async entry points:
```python
def query_external_api(self, query: str, **options) -> str:
    """Synchronous wrapper for CLI/scripts"""
    return asyncio.run(self.query_external_api_async(query, **options))
```

### Benefits
- **Performance**: Native async connection pooling, no thread overhead
- **Simplicity**: Remove `run_in_executor` wrapper boilerplate (saves ~5 lines per call site)
- **Scalability**: True parallel LLM calls without thread pool limits
- **Modern**: Aligns with Python async best practices
- **Debugging**: Easier to trace async call stacks

### Risks
- **Breaking Change**: All call sites need updates (but caught by type checker)
- **SDK Version Bumps**: May require newer SDK versions
- **Testing**: Need to verify all 4 providers work with async clients

### Estimated Effort
- **Phase 1 (Dependencies)**: 15 minutes
- **Phase 2 (Refactor Client)**: 2-3 hours
- **Phase 3 (Update Call Sites)**: 1-2 hours
- **Phase 4 (Testing)**: 1 hour
- **Total**: 4-6 hours