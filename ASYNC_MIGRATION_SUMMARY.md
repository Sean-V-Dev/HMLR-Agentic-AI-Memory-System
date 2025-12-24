# Priority 6: Native Async Client Support - COMPLETED

## Overview
Migrated all 11 LLM call sites from `run_in_executor` thread pool pattern to native async SDK calls, eliminating thread overhead and enabling true async concurrency.

## Changes Summary

### Dependencies Added
- **httpx>=0.24.0**: Async HTTP client for providers without native async SDKs

### Files Modified

#### 1. requirements.txt
- Added `httpx>=0.24.0` for async HTTP support

#### 2. hmlr/core/external_api_client.py (~190 lines added)
**New Async Methods:**
- `query_external_api_async()`: Main async entry point
- `_call_openai_api_async()`: Uses `AsyncOpenAI.chat.completions.create()`
- `_call_gemini_api_async()`: Uses `genai.Client.aio.models.generate_content()`
- `_call_grok_api_async()`: Uses `httpx.AsyncClient` (xai-sdk lacks async)
- `_call_anthropic_api_async()`: Uses `AsyncAnthropic.messages.create()`

**Features:**
- All responses normalized to OpenAI format
- Proper timeout handling via AsyncClient
- Comprehensive error logging with exc_info
- Temperature/options passthrough support

#### 3. hmlr/memory/synthesis/scribe.py (1 call site)
- **Before:** `await loop.run_in_executor(None, self._query_llm, user_input)`
- **After:** `await self._query_llm_async(user_input)`
- Added `_query_llm_async()` method for native async
- Kept `_query_llm()` for backward compatibility

#### 4. hmlr/memory/retrieval/lattice.py (2 call sites)
- Line 368: Routing decision LLM call
- Line 584: Memory filtering LLM call
- **Changed:** `query_external_api()` → `await query_external_api_async()`
- **Note:** `_lookup_facts()` and `_retrieve_dossiers()` still use run_in_executor (database I/O, not LLM)

#### 5. hmlr/memory/synthesis/dossier_governor.py (4 call sites)
- Line 293: Routing decision for fact classification
- Line 494: Dossier summary update
- Line 554: Literal fact restatement
- Line 616: Search summary generation
- All converted to `await query_external_api_async()`

#### 6. hmlr/memory/fact_scrubber.py (1 call site)
- Line 200: Fact extraction from messages
- **Changed:** `query_external_api()` → `await query_external_api_async()`

#### 7. hmlr/memory/gardener/manual_gardener.py (2 call sites)
- Line 349: Chunking classification
- Line 478: Fact grouping
- Both converted to `await query_external_api_async()`

#### 8. hmlr/core/conversation_engine.py (1 call site)
- Line 371: Main chat response generation
- **Changed:** `query_external_api()` → `await query_external_api_async()`

## Performance Benefits

### Before (Thread Pool)
```python
loop = asyncio.get_event_loop()
response = await loop.run_in_executor(None, sync_llm_call, prompt)
```
- Thread pool overhead
- Limited by thread pool size
- Synchronous blocking inside threads
- Extra call stack layers

### After (Native Async)
```python
response = await api_client.query_external_api_async(prompt)
```
- No thread overhead
- True async concurrency
- Native SDK connection pooling
- Cleaner call stack

### Real-World Impact
- **lattice.py govern()**: Runs 3-4 parallel LLM/DB tasks via `asyncio.gather()` - now truly parallel
- **dossier_governor**: Multiple sequential LLM calls now have better async flow
- **scribe**: Background profile updates no longer block on thread pool

## Validation

### Syntax Validation
✅ All 7 modified files pass linting with no errors

### Integration Test
✅ Ran `python main.py chat "Test async: What's 2+2?"` successfully
- System initialized correctly
- LLM call completed
- Response generated
- No errors or warnings

### Pattern Verification
✅ No remaining `run_in_executor` wrappers around LLM calls
✅ Database operations correctly retain `run_in_executor` (blocking I/O)

## Architecture Notes

### Why Some run_in_executor Remain
Not all `run_in_executor` calls were removed. The following correctly remain:
- **Database queries** (`_lookup_facts`, `_retrieve_dossiers`): SQLite is blocking I/O
- **File operations**: Writing to disk is blocking
- **Embedding generation**: Sentence transformers use blocking CPU operations

**Rule:** Only LLM API calls (HTTP requests) were converted to native async.

### Provider-Specific Implementations

| Provider   | Async SDK              | Implementation               |
|------------|------------------------|------------------------------|
| OpenAI     | AsyncOpenAI            | Native async SDK             |
| Anthropic  | AsyncAnthropic         | Native async SDK             |
| Gemini     | genai.Client.aio       | Native async via .aio        |
| Grok (xAI) | httpx.AsyncClient      | Direct REST API (no SDK)     |

## Backward Compatibility

All sync methods remain functional:
- `query_external_api()`: Original sync entry point (still works)
- `_query_llm()` in scribe.py: Sync helper for any legacy code
- All async methods coexist with sync versions

## Next Steps

### Optional Enhancements (Future)
1. **Connection Pooling Tuning**: Configure httpx/SDK connection limits for high-volume scenarios
2. **Retry Logic**: Add exponential backoff for transient failures
3. **Metrics**: Track async call latencies vs old thread pool times
4. **Testing**: Add unit tests with mocked async clients

### Priority 5 (Deferred)
- Atomic file writes remain LOW priority
- Current implementation is stable

## Completion Checklist
- ✅ Dependencies added (httpx)
- ✅ Async methods implemented (5 new methods)
- ✅ Call sites migrated (11/11 = 100%)
- ✅ No syntax errors
- ✅ Integration test passed
- ✅ ROADMAP.md updated
- ✅ No remaining run_in_executor for LLM calls
- ✅ Documentation created (this file)

**Phase 14 (Priority 6) Status:** ✅ COMPLETE
