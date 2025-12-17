# Hydra of 9 Heads Test - Test Plan

**Test Name:** test_13_hydra_dossier_e2e  
**Purpose:** Validate dossier system's ability to track evolving, contradictory policy information across time  
**Complexity:** Extreme - Multiple aliases, policy reversions, temporal context, trap question  
**Date Created:** December 16, 2025

---

## Overview

This test simulates 90 days of conversation about Project Cerberus encryption policies, where:
- A single cipher has 9+ different names (the "hydra heads")
- Policies are issued, superseded, revoked, and reinstated
- Information contradicts itself across time
- Final trap question requires tracking the CURRENT valid policy through all the noise

**The Trap:** User asks if Cerberus can use Tartarus-v3 for 4.85M records/day. The correct answer is **NON-COMPLIANT** because Policy v6 (the current valid policy) limits to 400k records/day.

---

## Test Objectives

### Primary Goals
1. **Multi-Vector Voting Validation:** Do related facts about the same entity (cipher/policy) merge into unified dossiers?
2. **Temporal Reasoning:** Can the system track "this policy was revoked 3 days later"?
3. **Alias Resolution:** Does the system understand that Phoenix = Aether = K-12 = Styx = River-9 = Charon = Tartarus-v3?
4. **Policy Supersession:** Can it track that v8 reverts to v6, making v7 invalid?
5. **Trap Question:** Does retrieval surface the RIGHT policy (v6: 400k/day) to answer the compliance question?

### Secondary Goals
- Test dossier retrieval across long temporal spans (90 days)
- Validate that atomic turn processing doesn't lose context
- Ensure DossierGovernor merges related facts intelligently
- Verify Read Governor can reason through contradictions

---

## Test Data Structure

### 22 Turns Across 90 Days (Sept 1 - Nov 30, 2025)

**Distribution Strategy:**
- Early September: Cipher naming history (Turns 1-8)
- Mid-September to October: Policy evolution (Turns 9-17)
- Late October: Long email with hidden context (Turn 18)
- November: Final capacity updates (Turns 19-21)
- End of test: Trap question (Turn 22)

**Temporal Gaps:**
- Some turns same day (rapid updates)
- Some turns weeks apart (slow evolution)
- Realistic conversation cadence

---

## Turn-by-Turn Breakdown

### Phase 1: Cipher Identity Chain (Sept 1-5, 2025)

**Turn 1 - Sept 1, 2025 @ 10:00 AM**
```
User: "Project Cerberus will use the Legacy-Phi encryption scheme (internal codename LΦ-88)."
Expected: New dossier created for Project Cerberus encryption details
```

**Turn 2 - Sept 1, 2025 @ 10:15 AM** (same day, 15 min later)
```
User: "LΦ-88 is the same as the old 'Phoenix' cipher from 2019."
Expected: Fact added to same dossier (Phoenix = LΦ-88)
```

**Turn 3 - Sept 2, 2025 @ 2:30 PM** (next day)
```
User: "Phoenix was renamed to 'Aether' in 2021."
Expected: Continues alias chain (Phoenix → Aether)
```

**Turn 4 - Sept 2, 2025 @ 3:00 PM** (30 min later)
```
User: "Aether is identical to the current 'K-12' production cipher."
Expected: Extends chain (Aether = K-12)
```

**Turn 5 - Sept 3, 2025 @ 9:00 AM** (next day)
```
User: "K-12 is now called 'Styx' in all new docs."
Expected: Continues naming evolution (K-12 → Styx)
```

**Turn 6 - Sept 3, 2025 @ 9:20 AM** (20 min later)
```
User: "Styx is the official name for what legal calls 'River-9'."
Expected: Adds legal vs official name distinction
```

**Turn 7 - Sept 4, 2025 @ 11:00 AM** (next day)
```
User: "River-9 is the new marketing name for 'Charon'."
Expected: Marketing name layer added
```

**Turn 8 - Sept 5, 2025 @ 4:00 PM** (next day)
```
User: "Charon is the final production name for 'Tartarus-v3'."
Expected: Final name in chain revealed
```

### Phase 2: Policy Evolution & Supersession (Sept 8 - Oct 20, 2025)

**Turn 9 - Sept 8, 2025 @ 10:00 AM** (3 days later)
```
User: "Tartarus-v3 is the only supported name after March 1st 2025."
Expected: New dossier OR fact about naming policy
```

**Turn 10 - Sept 15, 2025 @ 2:00 PM** (1 week later)
```
User: "Policy v1: Styx may be used for up to 500 TiB/day."
Expected: First usage policy fact
```

**Turn 11 - Sept 22, 2025 @ 3:30 PM** (1 week later)
```
User: "Policy v2: River-9 is limited to 10,000 objects per second."
Expected: Second policy version (different metric)
```

**Turn 12 - Sept 29, 2025 @ 11:00 AM** (1 week later)
```
User: "Policy v3: Charon is forbidden entirely. Update: this was revoked 3 days later in a footnote."
Expected: Policy with immediate revocation (temporal complexity)
```

**Turn 13 - Oct 6, 2025 @ 9:00 AM** (1 week later)
```
User: "Policy v4: Tartarus-v3 is approved without limit for EU regions only."
Expected: Geographic restriction introduced
```

**Turn 14 - Oct 13, 2025 @ 10:30 AM** (1 week later)
```
User: "Policy v5: All previous policies superseded. Tartarus-v3 limited to 2.5 GB/day."
Expected: Global supersession with strict limit
```

**Turn 15 - Oct 15, 2025 @ 2:00 PM** (2 days later)
```
User: "Ignore Policy v5, it was a draft. New limit: 400,000 records/day. This is Policy v6."
Expected: v5 invalidated, v6 established (400k records/day = KEY FACT)
```

**Turn 16 - Oct 18, 2025 @ 4:00 PM** (3 days later)
```
User: "Policy v7: Global ban on Tartarus-v3 for workloads exceeding 1 GiB/day (supersedes v1-v6)."
Expected: New supersession (but will be revoked later)
```

**Turn 17 - Oct 20, 2025 @ 9:00 AM** (2 days later)
```
User: "Policy v8: Policy v7 was issued by rogue employee; revert to v6. The 400,000 records/day limit from v6 is reinstated."
Expected: v7 invalidated, v6 is CURRENT POLICY (critical for trap question)
```

### Phase 3: Context Overload (Oct 25, 2025)

**Turn 18 - Oct 25, 2025 @ 3:00 PM** (5 days later)
```
User: [Full multi-page email from Marcus T., VP of Engineering Operations -3k tokens]
- Subject: URGENT: Q4 Strategic Alignment / Merger Integration
- 7 sections of corporate fluff
- Buried in Section 3 (Risk Management): Confirms Policy v6 is enforceable, v7 retracted
- Section 5: Mentions 4.85M records/day capacity for Cerberus
Expected: Multiple dossiers (merger, compliance, Cerberus capacity)
Challenge: Can system extract the buried policy confirmation?
```

### Phase 4: Capacity Finalization (Nov 10-15, 2025)

**Turn 19 - Nov 10, 2025 @ 10:00 AM** (16 days later)
```
User: "Cerberus Phase-2 will encrypt approximately 4.2 million client records daily."
Expected: Cerberus capacity fact (contradicts v6 limit)
```

**Turn 20 - Nov 12, 2025 @ 2:00 PM** (2 days later)
```
User: "Updated planning: Cerberus expects 4.7 million records/day peak."
Expected: Capacity update (higher than 4.2M)
```

**Turn 21 - Nov 15, 2025 @ 11:00 AM** (3 days later)
```
User: "Final capacity sign-off: Cerberus steady-state = 4,850,000 encrypted records per day."
Expected: Final capacity number (4.85M = 12x over policy limit)
```

### Phase 5: The Trap Question (Nov 30, 2025)

**Turn 22 - Nov 30, 2025 @ 4:00 PM** (15 days later, end of 90-day period)
```
Query: "Is it compliant for Project Cerberus to use Tartarus-v3 (a.k.a. Styx/Charon/River-9/etc.) 
for its production encryption workload of approximately 4.85 million records per day? 
Explain your reasoning."

Expected Answer: NON-COMPLIANT

Required Reasoning:
1. Identify that all cipher names refer to same algorithm
2. Current enforceable policy is v6 (v7 was revoked by v8)
3. Policy v6 limits to 400,000 records/day
4. Cerberus capacity is 4,850,000 records/day
5. 4.85M > 400k = NON-COMPLIANT
```

---

## Success Criteria

### Must Pass (Critical)
1. ✅ **Correct Answer:** System outputs "NON-COMPLIANT"
2. ✅ **Policy Tracking:** Reasoning cites Policy v6 (400k/day limit) as current
3. ✅ **Alias Resolution:** Understands Tartarus-v3 = Styx = Charon = River-9 = etc.
4. ✅ **Supersession Logic:** Knows v8 reverted to v6, making v7 invalid
5. ✅ **Capacity Retrieval:** Retrieves 4.85M records/day capacity fact

### Should Pass (Important)
6. ✅ **Dossier Merging:** Related cipher/policy facts in unified dossier(s)
7. ✅ **Temporal Context:** Correctly handles "revoked 3 days later" type statements
8. ✅ **Buried Context:** Extracts policy confirmation from long email (Turn 18)
9. ✅ **Multi-Vector Voting:** Multiple fact hits bubble up relevant dossiers

### Nice to Have (Bonus)
10. ✅ **Explanation Quality:** Clear, logical reasoning with evidence
11. ✅ **No Hallucination:** Doesn't cite invalid policies (v5, v7)
12. ✅ **Quantitative Comparison:** Shows 4.85M vs 400k calculation

---

## Expected Dossier Structure

### Dossier 1: Cipher Identity & Naming
- **Title:** "Tartarus-v3 Cipher Naming History" (or similar)
- **Facts:** 8-10 facts about name aliases
- **Key relationships:** Phoenix → Aether → K-12 → Styx ↔ River-9 ↔ Charon ↔ Tartarus-v3

### Dossier 2: Encryption Policy Evolution
- **Title:** "Tartarus-v3 Usage Policies" (or similar)
- **Facts:** 8-12 facts about policy versions
- **Key status:** Policy v6 (400k/day) is current, v7 revoked, v8 confirms reversion

### Dossier 3: Project Cerberus Capacity
- **Title:** "Project Cerberus Encryption Capacity"
- **Facts:** 3-5 facts about capacity planning
- **Key number:** 4.85M records/day steady-state

### Possible Dossier 4: Corporate Integration Context
- **Title:** "Q4 Merger Integration" (if DossierGovernor creates separate)
- **Facts:** From Turn 18 email about merger, audits, etc.
- **Note:** May merge into Dossier 2 if Governor sees compliance overlap

---

## Test Implementation Strategy

### Setup Phase
1. Delete test database (fresh start)
2. Initialize all components (Storage, FactScrubber, Gardener, DossierGovernor, Retriever)
3. Use snowflake-arctic-embed-l model (1024D)

### Execution Phase (22 iterations)
```python
for turn_num, turn_data in enumerate(turns, 1):
    # 1. Store turn in metadata_staging with actual date
    store_turn(
        turn_id=f"turn_{turn_num:03d}",
        date=turn_data['date'],  # Actual date from 90-day span
        user_message=turn_data['user_message'],
        ai_response=turn_data['ai_response']  # Simple acknowledgment
    )
    
    # 2. FactScrubber extracts facts (LLM call)
    facts = fact_scrubber.extract_facts(turn_data['user_message'])
    
    # 3. Create bridge block
    block_id = create_bridge_block(turn_id, facts)
    
    # 4. Garden immediately (Phase 2 flow)
    gardener.process_bridge_block(block_id)
    # This:
    # - Classifies facts (tags vs dossiers)
    # - Groups dossier facts semantically
    # - Routes to DossierGovernor
    # - DossierGovernor does Multi-Vector Voting
    # - Merges or creates dossiers
    # - Embeds facts with snowflake-l
    # - Deletes bridge block
```

### Query Phase
```python
# Turn 22: The trap question
query = """Is it compliant for Project Cerberus to use Tartarus-v3 
(a.k.a. Styx/Charon/River-9/etc.) for its production encryption workload 
of approximately 4.85 million records per day? Explain your reasoning."""

# Retrieve relevant dossiers
dossiers = retriever.retrieve_relevant_dossiers(query, top_k=10)

# Format context
context = context_assembler.format_dossiers_for_llm(dossiers)

# Ask LLM with context
response = llm_client.query(
    system_prompt="You are a compliance analyst...",
    context=context,
    query=query
)

# Validate response
assert "NON-COMPLIANT" in response
assert "400,000" in response or "400k" in response
assert "4.85" in response or "4,850,000" in response
```

### Validation Phase
1. **Response Validation:** Check for correct answer
2. **Reasoning Validation:** Parse explanation for key facts
3. **Dossier Inspection:** Verify structure and merging
4. **Embedding Quality:** Check similarity scores
5. **Performance Metrics:** Query time, retrieval accuracy

---

## Potential Failure Modes

### Architecture Failures
1. **Over-segmentation:** Creates 20+ tiny dossiers instead of 3-4 unified ones
2. **Under-merging:** Fails to recognize cipher aliases as same entity
3. **Policy confusion:** Retrieves v5 or v7 instead of v6
4. **Temporal loss:** Doesn't track "revoked 3 days later"

### Retrieval Failures
5. **Missing capacity fact:** Doesn't retrieve 4.85M number
6. **Missing policy limit:** Doesn't retrieve 400k limit
7. **Wrong dossiers:** Returns merger info but not policy info
8. **Threshold too high:** Filters out relevant dossiers

### Reasoning Failures
9. **Hallucination:** Cites policies that don't exist
10. **Alias confusion:** Treats Styx and Tartarus-v3 as different ciphers
11. **Wrong compliance:** Says COMPLIANT despite 12x overage
12. **No explanation:** Answers without showing reasoning

---

## RAGAS Evaluation (Future)

### Metrics to Track
- **Context Precision:** Are retrieved dossiers relevant?
- **Context Recall:** Are all necessary facts retrieved?
- **Faithfulness:** Does answer match retrieved context?
- **Answer Relevancy:** Does explanation address the question?

### Ground Truth Dataset
```json
{
  "question": "Is it compliant for Project Cerberus to use Tartarus-v3...",
  "ground_truth": "NON-COMPLIANT",
  "contexts": [
    "Policy v6 limits Tartarus-v3 to 400,000 records/day",
    "Cerberus capacity is 4,850,000 records/day",
    "Policy v8 reverted to v6, invalidating v7"
  ],
  "required_reasoning": [
    "Current policy is v6 (400k/day)",
    "Cerberus needs 4.85M/day",
    "4.85M > 400k = non-compliant"
  ]
}
```

---

## Test Output Format

### Console Output
```
================================================================================
🧪 HYDRA OF 9 HEADS TEST (90-Day Policy Evolution)
================================================================================

Phase 1: Cipher Identity Chain (Sept 1-5, 2025)
   Turn 1  [Sept 1, 10:00] Legacy-Phi (LΦ-88) ✅
   Turn 2  [Sept 1, 10:15] Phoenix = LΦ-88 ✅
   ...
   
Phase 2: Policy Evolution (Sept 8 - Oct 20, 2025)
   Turn 10 [Sept 15] Policy v1: 500 TiB/day ✅
   ...
   
Phase 3: Context Overload (Oct 25, 2025)
   Turn 18 [Oct 25] Marcus email (7 sections) ✅
   
Phase 4: Capacity Finalization (Nov 10-15, 2025)
   Turn 19 [Nov 10] 4.2M records/day ✅
   ...
   
Phase 5: THE TRAP QUESTION (Nov 30, 2025)
   Query: Is Cerberus compliant?
   
   Retrieved: 4 dossiers
   • Tartarus-v3 Cipher Naming History (8 facts, score: 0.78)
   • Encryption Policy Evolution (11 facts, score: 0.82)
   • Project Cerberus Capacity (3 facts, score: 0.85)
   • Corporate Compliance Context (5 facts, score: 0.71)
   
   LLM Response:
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   NON-COMPLIANT
   
   Reasoning:
   1. Policy v6 limits Tartarus-v3 to 400,000 records/day
   2. Policy v8 reverted to v6 (v7 was issued by rogue employee)
   3. Cerberus requires 4,850,000 records/day
   4. 4.85M ÷ 400k = 12.1x over limit
   5. Therefore: NON-COMPLIANT
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   
✅ TEST PASSED
   ✅ Correct answer: NON-COMPLIANT
   ✅ Cited Policy v6 (400k/day)
   ✅ Retrieved capacity (4.85M/day)
   ✅ Quantitative comparison shown
   ✅ No invalid policies cited
   
📊 Performance:
   • Query time: 68ms
   • Dossiers created: 4
   • Facts extracted: 27
   • Avg similarity: 0.79
```

---

## Next Steps

1. **Create test harness:** `test_13_hydra_dossier_e2e.py`
2. **Generate turn data:** JSON file with 22 turns + dates
3. **Run test:** Execute with snowflake-l model
4. **Analyze results:** Inspect dossiers, validate answer
5. **RAGAS evaluation:** If test passes, run full metrics
6. **Document findings:** Update Phase 2 completion report

---

## Notes

- This test is the "boss fight" for the dossier system
- If it passes, the system can handle production complexity
- Focus on Multi-Vector Voting working correctly
- Watch for policy supersession tracking
- The long email (Turn 18) tests chunking tolerance
- 90-day span tests temporal context retention
