"""
Rigorous BGE Comparison: small vs base (Multi-Hit Test)

Tests if models can:
1. Retrieve ALL relevant items (recall)
2. NOT retrieve irrelevant items (precision)
3. Rank relevant items appropriately
"""

import time
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

MODELS = [
    {
        "name": "bge-small",
        "model_id": "BAAI/bge-small-en-v1.5",
        "dims": 384
    },
    {
        "name": "bge-base",
        "model_id": "BAAI/bge-base-en-v1.5",
        "dims": 768
    },
    {
        "name": "bge-large",
        "model_id": "BAAI/bge-large-en-v1.5",
        "dims": 1024
    }
]

# Car sentences: All are relevant to car-related queries
CAR_SENTENCES = [
    ("car_honda_civic", "I drive a Honda Civic as my daily commuter. It's reliable, fuel-efficient, and perfect for city driving."),
    ("car_toyota_sienna", "I have a Toyota Sienna minivan that I use for family road trips. It has tons of cargo space and is very comfortable."),
    ("car_tesla_model_y", "My wife drives a Tesla Model Y. It's incredibly efficient and we love the instant acceleration and tech features."),
    ("car_porsche_911", "I have a Porsche 911 that I take out on weekends. It's pure driving pleasure and absolutely thrilling on winding roads."),
    ("car_ford_f150", "I own a Ford F-150 pickup truck for towing my boat and hauling construction materials. It's a workhorse."),
]

# Distractor sentences: These should NOT match car queries
DISTRACTOR_SENTENCES = [
    ("python_learning", "I'm learning Python for data analysis and working through pandas tutorials."),
    ("guitar_band", "I play guitar in a local indie rock band called Cerberus on weekends."),
    ("dog_max", "My dog Max is a 3-year-old golden retriever who loves the dog park."),
    ("kitchen_reno", "I'm renovating my kitchen with quartz countertops and a gas range."),
    ("hiking_mountains", "I went hiking in the Rockies last summer and saw incredible wildlife."),
    ("coffee_morning", "I drink three cups of black coffee every morning before work."),
    ("book_scifi", "Currently reading a sci-fi novel about Mars colonization."),
    ("gym_routine", "I go to the gym four times a week for strength training."),
]

# Test queries with expected matches
# Format: (query, [list of expected car IDs], threshold for "should match")
threshold = 0.5
TEST_QUERIES = [
    ("Which of my cars should I take to the track?", ["car_porsche_911"], threshold),  # Porsche should be top
    ("Which car is best for a family road trip?", ["car_toyota_sienna", "car_ford_f150"], threshold),  # Minivan primary, truck backup
    ("Which of my cars is most fuel efficient?", ["car_tesla_model_y", "car_honda_civic"], threshold),  # Tesla and Civic
    ("Which car should I use for towing heavy equipment?", ["car_ford_f150"], threshold),  # F-150 only
    ("What cars do I own?", ["car_honda_civic", "car_toyota_sienna", "car_tesla_model_y", "car_porsche_911", "car_ford_f150"], threshold),  # ALL cars
    ("Which car is best for daily commuting in the city?", ["car_honda_civic", "car_tesla_model_y"], threshold),  # Civic and Tesla
]

# ============================================================================
# CONCEPTUAL MATCHING TEST DATA
# Tests if models understand CONCEPTS vs just matching keywords
# ============================================================================

# Target policies: Correct conceptual answers (minimal keyword overlap with queries)
POLICY_TARGETS = [
    ("policy_gifts", "Employees must strictly refrain from accepting any gratuities, favors, or items of material value from current or prospective business partners. Any such offers must be reported to the Compliance Officer immediately to ensure objective decision-making."),
    ("policy_remote_work", "Staff members may request flexible work arrangements including telecommuting options. Managers will evaluate requests based on role requirements and team needs. Approval requires documented agreement on availability hours and performance metrics."),
    ("policy_compensation", "Non-exempt employees working beyond standard hours are entitled to additional pay calculated at one and a half times their regular rate. All extended work periods must be pre-approved by department heads and properly logged in the timekeeping system."),
    ("policy_travel_expenses", "When conducting business-related travel, employees should retain all receipts for costs incurred including transportation, lodging, and meals. The Finance Department will process reimbursement requests within 14 business days of submission with proper documentation."),
]

# Keyword traps: Wrong answers that share many keywords with queries
POLICY_DISTRACTORS = [
    ("event_holiday_party", "The annual office Holiday Gift Exchange is coming up! We encourage all staff to bring a small item of value (under $25) to trade with colleagues. Note: Our office suppliers have kindly donated wrapping paper for the event."),
    ("newsletter_remote_teams", "Shoutout to our remote sales team in Austin! Their recent client pitch went viral on LinkedIn. If you're working remotely this week, join our virtual coffee chat on Friday at 3pm."),
    ("schedule_overtime_volunteers", "We need overtime volunteers for the weekend charity event! Pizza and drinks provided. Sign up at the front desk. This is outside regular work hours, so it's completely optional."),
    ("memo_travel_tips", "Planning your summer vacation? Check out our employee travel discount program! Save on flights, hotels, and rental cars. Pro tip: Book early for the best deals and always keep your personal expense receipts organized."),
]

# Conceptual queries: Should match policies, NOT distractors (despite keyword overlap)
CONCEPTUAL_QUERIES = [
    ("Can I keep a Rolex that a supplier sent me as a thank you?", ["policy_gifts"]),
    ("Is it okay to work from home on Tuesdays?", ["policy_remote_work"]),
    ("Do I get paid extra if I stay late to finish a project?", ["policy_compensation"]),
    ("How do I get reimbursed for my hotel during the conference?", ["policy_travel_expenses"]),
]

def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors."""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def test_model_rigorous(model_info):
    """Test model with multi-hit queries and distractors."""
    print(f"\n{'=' * 80}")
    print(f"Testing: {model_info['name']} ({model_info['dims']}D)")
    print("=" * 80)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Load model
    print(f"\nLoading {model_info['model_id']}...")
    start = time.time()
    model = SentenceTransformer(model_info['model_id'], device=device)
    load_time = time.time() - start
    print(f"✓ Loaded in {load_time:.2f}s")
    
    # Combine car + distractor sentences
    all_sentences = CAR_SENTENCES + DISTRACTOR_SENTENCES
    sentence_ids = [s[0] for s in all_sentences]
    sentence_texts = [s[1] for s in all_sentences]
    
    print(f"\nEmbedding corpus:")
    print(f"  - {len(CAR_SENTENCES)} car sentences")
    print(f"  - {len(DISTRACTOR_SENTENCES)} distractor sentences")
    print(f"  - {len(all_sentences)} total")
    
    start = time.time()
    sentence_embeddings = model.encode(
        sentence_texts,
        convert_to_numpy=True,
        show_progress_bar=False,
        batch_size=8
    )
    embed_time = time.time() - start
    print(f"✓ Embedded in {embed_time:.3f}s")
    
    # Test queries
    print(f"\nTesting {len(TEST_QUERIES)} queries...")
    results = []
    total_expected = 0
    total_recalled = 0
    total_false_positives = 0
    
    for query_text, expected_ids, threshold in TEST_QUERIES:
        # Embed query
        query_embedding = model.encode(
            query_text,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        
        # Calculate similarities
        similarities = [cosine_similarity(query_embedding, sent_emb)
                       for sent_emb in sentence_embeddings]
        
        # Get all matches above threshold
        matches = []
        for i, (sent_id, score) in enumerate(zip(sentence_ids, similarities)):
            if score >= threshold:
                matches.append({
                    "id": sent_id,
                    "score": score,
                    "text": sentence_texts[i],
                    "is_car": sent_id.startswith("car_"),
                    "is_expected": sent_id in expected_ids
                })
        
        # Sort by score
        matches.sort(key=lambda x: x['score'], reverse=True)
        
        # Calculate recall and precision
        recalled = [m for m in matches if m['is_expected']]
        false_positives = [m for m in matches if not m['is_car']]  # Non-car matches
        
        recall = len(recalled) / len(expected_ids) if expected_ids else 1.0
        precision = len(recalled) / len(matches) if matches else 0.0
        
        total_expected += len(expected_ids)
        total_recalled += len(recalled)
        total_false_positives += len(false_positives)
        
        results.append({
            "query": query_text,
            "expected_ids": expected_ids,
            "threshold": threshold,
            "matches": matches,
            "recalled": recalled,
            "false_positives": false_positives,
            "recall": recall,
            "precision": precision,
            "all_similarities": dict(zip(sentence_ids, similarities))
        })
    
    overall_recall = total_recalled / total_expected if total_expected > 0 else 0
    
    return {
        "model": model_info,
        "load_time": load_time,
        "embed_time": embed_time,
        "results": results,
        "total_expected": total_expected,
        "total_recalled": total_recalled,
        "total_false_positives": total_false_positives,
        "overall_recall": overall_recall
    }

def print_results(test_result):
    """Print detailed results."""
    model = test_result['model']
    results = test_result['results']
    
    print(f"\n{'=' * 80}")
    print(f"RESULTS: {model['name']}")
    print("=" * 80)
    
    print(f"\n Overall Metrics:")
    print(f"   Total expected items: {test_result['total_expected']}")
    print(f"   Total recalled: {test_result['total_recalled']}")
    print(f"   Overall recall: {test_result['overall_recall']:.1%}")
    print(f"   False positives (distractors): {test_result['total_false_positives']}")
    
    print(f"\n Query Details:")
    for i, r in enumerate(results, 1):
        recall_pct = r['recall'] * 100
        precision_pct = r['precision'] * 100 if r['matches'] else 0
        
        status = "✅" if r['recall'] == 1.0 and not r['false_positives'] else "⚠️" if r['recall'] >= 0.5 else "❌"
        
        print(f"\n{i}. {status} {r['query']}")
        print(f"   Expected: {len(r['expected_ids'])} items | Recalled: {len(r['recalled'])} | Recall: {recall_pct:.0f}%")
        print(f"   Total matches: {len(r['matches'])} | Precision: {precision_pct:.0f}%")
        
        if r['false_positives']:
            print(f"    False positives: {len(r['false_positives'])} distractors matched!")
        
        # Show all matches above threshold
        print(f"   Matches above threshold ({r['threshold']}):")
        for rank, match in enumerate(r['matches'][:8], 1):  # Show top 8
            if match['is_expected']:
                marker = "✅"
            elif match['is_car']:
                marker = "⚠️"  # Car but not expected
            else:
                marker = "❌"  # Distractor
            
            print(f"      {marker} {rank}. {match['id']:30} {match['score']:.4f}")
        
        # Show missed items
        missed = [exp_id for exp_id in r['expected_ids'] if exp_id not in [m['id'] for m in r['recalled']]]
        if missed:
            print(f"   ❌ MISSED {len(missed)} expected items:")
            for miss_id in missed:
                score = r['all_similarities'][miss_id]
                rank = sorted(r['all_similarities'].values(), reverse=True).index(score) + 1
                print(f"      → {miss_id}: {score:.4f} (rank {rank}, below threshold {r['threshold']})")

def compare_models(all_results):
    """Side-by-side comparison."""
    print(f"\n\n{'=' * 80}")
    print("SIDE-BY-SIDE COMPARISON")
    print("=" * 80)
    
    print(f"\n{'Model':<15} {'Dims':<8} {'Recall':<12} {'False Pos':<12} {'Speed':<10}")
    print("-" * 80)
    
    for result in all_results:
        model = result['model']
        
        print(f"{model['name']:<15} "
              f"{model['dims']:<8} "
              f"{result['total_recalled']}/{result['total_expected']} ({result['overall_recall']:.0%}){'':<3} "
              f"{result['total_false_positives']}{'':<11} "
              f"{result['embed_time']:.3f}s")
    
    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)
    
    # Winner by recall
    best_recall = max(r['overall_recall'] for r in all_results)
    best_model = next(r for r in all_results if r['overall_recall'] == best_recall)
    
    print(f"\n Best Recall: {best_model['model']['name']} ({best_model['overall_recall']:.0%})")
    
    # Check false positives
    fewest_fps = min(r['total_false_positives'] for r in all_results)
    cleanest_model = next(r for r in all_results if r['total_false_positives'] == fewest_fps)
    
    print(f" Fewest False Positives: {cleanest_model['model']['name']} ({cleanest_model['total_false_positives']} distractors)")
    
    # Overall recommendation
    if len(all_results) == 2:
        small_result = next(r for r in all_results if r['model']['name'] == 'bge-small')
        other_result = next(r for r in all_results if r['model']['name'] != 'bge-small')
        
        if small_result['overall_recall'] >= other_result['overall_recall'] and small_result['total_false_positives'] <= other_result['total_false_positives']:
            print("\n✅ bge-small WINS on both recall and precision")
            print("   → Better at finding all relevant items")
            print("   → Better at avoiding false positives")
            print(f"   → Faster ({small_result['embed_time']:.3f}s vs {other_result['embed_time']:.3f}s)")
        elif small_result['overall_recall'] > other_result['overall_recall']:
            print(f"\n✅ bge-small has BETTER recall ({small_result['overall_recall']:.0%} vs {other_result['overall_recall']:.0%})")
            print(f"   ⚠️  But more false positives ({small_result['total_false_positives']} vs {other_result['total_false_positives']})")
            print("   → Consider if recall is more important than precision")
        else:
            print(f"\n⚠️  {other_result['model']['name']} has better recall")
            print(f"   → But bge-small is faster ({small_result['embed_time']:.3f}s vs {other_result['embed_time']:.3f}s)")

def test_conceptual_matching(model_info):
    """Test model's ability to match concepts vs keywords."""
    print(f"\n{'=' * 80}")
    print(f"CONCEPTUAL TEST: {model_info['name']} ({model_info['dims']}D)")
    print("=" * 80)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load model (may already be loaded)
    model = SentenceTransformer(model_info['model_id'], device=device)
    
    # Combine policies + distractors
    all_docs = POLICY_TARGETS + POLICY_DISTRACTORS
    doc_ids = [d[0] for d in all_docs]
    doc_texts = [d[1] for d in all_docs]
    
    print(f"\nEmbedding corpus:")
    print(f"  - {len(POLICY_TARGETS)} policy targets (correct answers)")
    print(f"  - {len(POLICY_DISTRACTORS)} keyword traps (wrong answers)")
    
    # Embed documents
    doc_embeddings = model.encode(
        doc_texts,
        convert_to_numpy=True,
        show_progress_bar=False,
        batch_size=8
    )
    
    print(f"\nTesting {len(CONCEPTUAL_QUERIES)} conceptual queries...")
    results = []
    correct = 0
    keyword_trapped = 0
    
    for query_text, expected_ids in CONCEPTUAL_QUERIES:
        # Embed query
        query_embedding = model.encode(query_text, convert_to_numpy=True, show_progress_bar=False)
        
        # Calculate similarities
        similarities = [cosine_similarity(query_embedding, doc_emb) for doc_emb in doc_embeddings]
        
        # Get top match
        top_idx = np.argmax(similarities)
        top_id = doc_ids[top_idx]
        top_score = similarities[top_idx]
        
        # Check if correct
        is_correct = top_id in expected_ids
        is_policy = top_id.startswith("policy_")
        
        if is_correct:
            correct += 1
        elif not is_policy:
            keyword_trapped += 1  # Fell for the keyword trap
        
        results.append({
            "query": query_text,
            "expected": expected_ids[0],
            "matched": top_id,
            "score": top_score,
            "correct": is_correct,
            "keyword_trap": not is_policy,
            "similarities": dict(zip(doc_ids, similarities))
        })
    
    return {
        "model": model_info,
        "results": results,
        "correct": correct,
        "keyword_trapped": keyword_trapped,
        "accuracy": correct / len(CONCEPTUAL_QUERIES)
    }

def print_conceptual_results(test_result):
    """Print conceptual test results."""
    model = test_result['model']
    results = test_result['results']
    
    print(f"\n{'=' * 80}")
    print(f"CONCEPTUAL RESULTS: {model['name']}")
    print("=" * 80)
    
    print(f"\n Accuracy: {test_result['correct']}/{len(results)} ({test_result['accuracy']:.0%})")
    print(f"   ✅ Correct (concept match): {test_result['correct']}")
    print(f"   ❌ Keyword trapped: {test_result['keyword_trapped']}")
    
    print(f"\n Query Details:")
    for i, r in enumerate(results, 1):
        status = "✅" if r['correct'] else "❌"
        trap_warning = "  KEYWORD TRAP!" if r['keyword_trap'] else ""
        
        print(f"\n{i}. {status} {r['query']}{trap_warning}")
        print(f"   Expected: {r['expected']} (policy)")
        print(f"   Matched:  {r['matched']} (score: {r['score']:.4f})")
        
        # Show top 3 to see if keyword trap scored higher
        sorted_matches = sorted(r['similarities'].items(), key=lambda x: x[1], reverse=True)
        print(f"   Top 3 scores:")
        for rank, (doc_id, score) in enumerate(sorted_matches[:3], 1):
            marker = "👉" if doc_id == r['expected'] else "  "
            label = "POLICY" if doc_id.startswith("policy_") else "TRAP"
            print(f"      {marker} {rank}. {doc_id:30} {score:.4f} ({label})")

def compare_conceptual(all_results):
    """Compare conceptual matching performance."""
    print(f"\n\n{'=' * 80}")
    print("CONCEPTUAL COMPARISON")
    print("=" * 80)
    
    print(f"\n{'Model':<15} {'Dims':<8} {'Accuracy':<15} {'Keyword Traps':<15}")
    print("-" * 80)
    
    for result in all_results:
        model = result['model']
        print(f"{model['name']:<15} "
              f"{model['dims']:<8} "
              f"{result['correct']}/{len(result['results'])} ({result['accuracy']:.0%}){'':<6} "
              f"{result['keyword_trapped']}/{len(result['results'])}")
    
    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)
    
    best_accuracy = max(r['accuracy'] for r in all_results)
    best_model = next(r for r in all_results if r['accuracy'] == best_accuracy)
    
    fewest_traps = min(r['keyword_trapped'] for r in all_results)
    smartest_model = next(r for r in all_results if r['keyword_trapped'] == fewest_traps)
    
    print(f"\n Best Conceptual Accuracy: {best_model['model']['name']} ({best_model['accuracy']:.0%})")
    print(f" Fewest Keyword Traps: {smartest_model['model']['name']} ({smartest_model['keyword_trapped']} traps)")
    
    if len(all_results) == 2:
        small_result = next(r for r in all_results if r['model']['name'] == 'bge-small')
        other_result = next(r for r in all_results if r['model']['name'] != 'bge-small')
        
        if small_result['accuracy'] > other_result['accuracy']:
            print(f"\n✅ bge-small BETTER at conceptual matching ({small_result['accuracy']:.0%} vs {other_result['accuracy']:.0%})")
            print("   → Understands abstract concepts despite smaller size")
        elif small_result['accuracy'] == other_result['accuracy']:
            print(f"\n  Both models equal at conceptual matching ({small_result['accuracy']:.0%})")
            print("   → Larger model provides no advantage for concept understanding")
        else:
            print(f"\n {other_result['model']['name']} BETTER at conceptual matching ({other_result['accuracy']:.0%} vs {small_result['accuracy']:.0%})")
            print("   → Larger dimensions help with abstract reasoning")
            print("   → This is where model size matters!")

def main():
    print("=" * 80)
    print("RIGOROUS BGE COMPARISON: Multi-Test Suite")
    print("=" * 80)
    
    # Test 1: Multi-hit recall test
    print("\n" + "=" * 80)
    print("TEST 1: MULTI-HIT RECALL (Car Queries)")
    print("=" * 80)
    print("\nThis test checks if models can:")
    print("  1. Retrieve ALL relevant items (recall)")
    print("  2. NOT retrieve irrelevant items (precision)")
    print("  3. Handle queries with multiple correct answers")
    
    car_results = []
    for model_info in MODELS:
        result = test_model_rigorous(model_info)
        print_results(result)
        car_results.append(result)
    compare_models(car_results)
    
    # Test 2: Conceptual matching test
    print("\n\n" + "=" * 80)
    print("TEST 2: CONCEPTUAL MATCHING (Policies vs Keyword Traps)")
    print("=" * 80)
    print("\nThis test checks if models can:")
    print("  1. Match concepts vs just keywords")
    print("  2. Avoid keyword traps (distractors with shared words)")
    print("  3. Understand abstract queries")
    
    conceptual_results = []
    for model_info in MODELS:
        result = test_conceptual_matching(model_info)
        print_conceptual_results(result)
        conceptual_results.append(result)
    compare_conceptual(conceptual_results)
    
    # Final summary
    print("\n\n" + "=" * 80)
    print("FINAL VERDICT")
    print("=" * 80)
    
    if len(car_results) == 2 and len(conceptual_results) == 2:
        small_car = next(r for r in car_results if r['model']['name'] == 'bge-small')
        small_concept = next(r for r in conceptual_results if r['model']['name'] == 'bge-small')
        other_car = next(r for r in car_results if r['model']['name'] != 'bge-small')
        other_concept = next(r for r in conceptual_results if r['model']['name'] != 'bge-small')
        
        print(f"\n{'Metric':<30} {'bge-small':<15} {other_car['model']['name']:<15}")
        print("-" * 60)
        print(f"{'Multi-hit recall':<30} {small_car['overall_recall']:.0%}{'':<12} {other_car['overall_recall']:.0%}")
        print(f"{'False positives':<30} {small_car['total_false_positives']}{'':<14} {other_car['total_false_positives']}")
        print(f"{'Conceptual accuracy':<30} {small_concept['accuracy']:.0%}{'':<12} {other_concept['accuracy']:.0%}")
        print(f"{'Keyword traps':<30} {small_concept['keyword_trapped']}{'':<14} {other_concept['keyword_trapped']}")
        print(f"{'Speed (embed time)':<30} {small_car['embed_time']:.3f}s{'':<9} {other_car['embed_time']:.3f}s")
        
        # Recommendation
        small_wins = 0
        other_wins = 0
        
        if small_car['overall_recall'] >= other_car['overall_recall']:
            small_wins += 1
        else:
            other_wins += 1
            
        if small_car['total_false_positives'] <= other_car['total_false_positives']:
            small_wins += 1
        else:
            other_wins += 1
            
        if small_concept['accuracy'] >= other_concept['accuracy']:
            small_wins += 1
        else:
            other_wins += 1
        
        print(f"\n{'=' * 60}")
        if small_wins > other_wins:
            print(" WINNER: bge-small")
            print("   ✅ Faster, smaller, and just as accurate (or better)")
            print("   ✅ Recommend switching from bge-large to bge-small")
        elif other_wins > small_wins:
            print(f" WINNER: {other_car['model']['name']}")
            print("   ✅ Better accuracy on key metrics")
            print("   ✅ Worth the extra size/speed cost")
        else:
            print("  TIE: Both models perform similarly")
            print("   → Recommend bge-small for speed advantage")

if __name__ == "__main__":
    main()

