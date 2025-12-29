# Embedding Model Comparison Tests

This directory contains tests for evaluating embedding models in HMLR's RAG retrieval pipeline.

## Purpose

These tests help determine which embedding model provides the best balance of **accuracy, speed, and resource usage** for conversational memory retrieval. While larger models might seem better on paper, real-world RAG systems rarely rely on perfect rank-1 precision — they retrieve top-k results and let the LLM filter.

## Test Suite

### 1. **Basic Similarity Test** (`test_bge_model_comparison.py`)

Tests straightforward semantic matching with clear query-to-content relationships.

**What it tests:**
- Direct keyword matching (e.g., "Where does he work?" → "I work at Tesla")
- Simple conversational queries with obvious correct answers
- Baseline retrieval accuracy

**Use case:** Validates the model works for basic RAG scenarios.

---

### 2. **Multi-Hit Recall Test** (`test_bge_small_vs_base_rigorous.py` - Test 1)

Tests if models can retrieve **ALL relevant items** when multiple correct answers exist.

**Example scenario:**
- Corpus: 5 different cars (Honda, Toyota, Tesla, Porsche, Ford)
- Query: "Which cars do I own?" → Should return all 5

**What it tests:**
- Recall: Does it find all relevant items?
- Precision: Does it avoid irrelevant distractors (Python, guitar, dog, etc.)?
- Threshold behavior: Can it distinguish between strong matches (cars) and weak matches (distractors)?

**Use case:** Real-world queries often have multiple relevant memories, not just one.

---

### 3. **Conceptual Matching / Nuance Test** (`test_bge_small_vs_base_rigorous.py` - Test 2)

Tests if models understand **abstract concepts vs surface-level keyword matching**.

**Example scenario:**
- **Correct answer:** Company gift policy (formal, no keyword overlap with query)
- **Keyword trap:** Holiday gift exchange memo (shares "gift", "supplier", "value")
- **Query:** "Can I keep a Rolex a supplier sent me?"

The trap shares many keywords with the query, but the policy is the conceptually correct answer.

**What it tests:**
- Does the model fall for keyword traps?
- Can it understand abstract intent vs lexical overlap?
- Rank-1 precision: Is the correct answer at the top?

**Critical nuance:** Even when models "fail" this test (correct answer is #2 or #3 instead of #1), **they still succeed in production** because:
- RAG systems retrieve top-k (typically k=3-5, never k=1)
- The correct answer is always in the top-3 for all models tested
- The LLM downstream filters the final results

**Use case:** Validates the model doesn't over-rely on keyword matching for vague queries.

---

## Findings

### Winner: **bge-small (BAAI/bge-small-en-v1.5)**

| Metric | bge-small | bge-base | bge-large |
|--------|-----------|----------|-----------|
| **Dimensions** | 384 | 768 | 1024 |
| **Multi-hit recall** | 100% | 100% | 100% |
| **False positives** | Lowest | Medium | Medium |
| **Conceptual (Rank-1)** | 50% | 50% | 100% |
| **Conceptual (Recall@3)** | **100%** | **100%** | **100%** |
| **Speed** | **Fastest** | Medium | Slower |
| **GPU Memory** | **~500MB** | ~900MB | ~1.3GB |

### Why bge-small wins:

1. **Functional tie on conceptual matching:** All three models include the correct policy in top-3 results. Since HMLR (and all RAG systems) retrieve multiple hits and let the LLM filter, the rank-1 precision difference is irrelevant in production.

2. **Speed advantage:** bge-small embeds ~6x faster than bge-large after GPU warm-up.

3. **Similarity scores:** bge-small consistently produces higher confidence scores for correct matches in conversational queries.

4. **Resource efficiency:** Uses 60% less GPU memory, allowing larger batch sizes or concurrent operations.

5. **Vague query handling:** Smaller embedding space means fewer spurious correlations. For conversational AI, "simpler is better."


---

## Running the Tests

### Test Your Own Model

All tests are model-agnostic. To test a different model:

1. Edit the `MODELS` array in either test file:

```python
MODELS = [
    {
        "name": "your-model-name",
        "model_id": "your-org/your-model-id",  # Hugging Face model ID
        "dims": 768  # Embedding dimensions
    }
]
```

2. Adjust the threshold if needed (rigorous test only):

```python
threshold = 0.5  # Lower = more permissive, higher = more strict
```

3. Run the test:

```powershell
python test_bge_model_comparison.py          # Basic similarity
python test_bge_small_vs_base_rigorous.py   # Multi-hit + conceptual
```

### Requirements

- `sentence-transformers`
- `torch` with CUDA support (optional but recommended)
- Target model downloaded from Hugging Face (auto-downloads on first run)

### What to Look For

1. **Recall@3 or Recall@5:** Is the correct answer in top-k? (Most important)
2. **False positives:** Does it match irrelevant distractors above threshold?
3. **Speed:** Embedding time for your corpus size
4. **GPU memory:** Will it fit alongside your LLM?

---

## Interpreting Results

### Good signs:
- ✅ Recall@3 ≥ 95% (correct answer almost always in top-3)
- ✅ Few false positives (distractors don't exceed threshold)
- ✅ Reasonable speed (<0.5s for 10-20 items)

### Red flags:
- ❌ Recall@3 < 90% (missing correct answers frequently)
- ❌ Many false positives (retrieves irrelevant content)
- ❌ Correct answer consistently ranks below #5

### Common misconceptions:
- **"Larger model = better"**: Not for conversational RAG. Smaller models often outperform on vague, colloquial queries.
- **"Rank-1 precision matters"**: Only if k=1 in production (extremely rare). Focus on Recall@k instead.
- **"Higher dimensions = more accurate"**: Sometimes yes, but also more prone to spurious correlations and keyword traps.

---

## Contributing

If you test a different model and find it outperforms bge-small for HMLR's use case:

1. Document the model ID, dimensions, and key metrics
2. Run all three test types
3. Note any special installation requirements
4. Share findings in a GitHub issue or PR

The tests are designed to be reproducible and model-agnostic.
