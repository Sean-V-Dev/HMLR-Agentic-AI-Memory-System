"""
Test embedding granularity: Full sentences vs keywords-only.

Goal: Determine if stopwords provide beneficial context for fuzzy semantic matching.
"""
import re
from sentence_transformers import SentenceTransformer
import numpy as np

# Common English stopwords
STOPWORDS = {
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he',
    'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 'to', 'was', 'will', 'with',
    'i', 'you', 'we', 'they', 'this', 'what', 'when', 'where', 'who', 'which', 'how',
    'do', 'does', 'did', 'have', 'had', 'can', 'could', 'would', 'should', 'may', 'might'
}

def extract_keywords(text):
    """Remove stopwords and punctuation, keep meaningful words."""
    # Lowercase and split
    words = re.findall(r'\b[a-zA-Z0-9]+\b', text.lower())
    # Filter stopwords
    keywords = [w for w in words if w not in STOPWORDS and len(w) > 2]
    return ' '.join(keywords)

def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors."""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# Test sentences - designed to test fuzzy matching scenarios
test_corpus = [
    # Encryption context
    "The encryption algorithm we use is AES-256 for data at rest.",
    "We implemented RSA for the key exchange protocol.",
    "I can't remember what that encryption method was called.",
    
    # Project context
    "Project Cerberus is the codename for the security initiative.",
    "The Cerberus project focuses on authentication improvements.",
    "What was the name of that security project again?",
    
    # Technical details
    "The API endpoint is located at /api/v2/users/authenticate.",
    "You need to call the authentication endpoint to get a token.",
    "Where do I send the login request in the API?",
    
    # Unrelated noise (should NOT match encryption queries)
    "I had lunch with Sarah at the new restaurant downtown.",
    "The weather forecast predicts rain for the weekend.",
    "We need to order more coffee beans for the office.",
]

# Test queries - intentionally vague or use different wording
test_queries = [
    "What encryption algorithm do we use?",  # Should match sentences 0, 1 (maybe 2)
    "Tell me about the Cerberus project",    # Should match sentences 3, 4 (maybe 5)
    "How do I authenticate with the API?",   # Should match sentences 6, 7 (maybe 8)
    "What's for lunch today?",               # Should match sentence 9, NOT others
]

def run_experiment():
    print("=" * 80)
    print("EMBEDDING GRANULARITY EXPERIMENT")
    print("=" * 80)
    print("\nLoading embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("✓ Model loaded\n")
    
    # Prepare both versions
    corpus_full = test_corpus
    corpus_keywords = [extract_keywords(s) for s in test_corpus]
    
    print("CORPUS (showing both versions):")
    print("-" * 80)
    for i, (full, kw) in enumerate(zip(corpus_full, corpus_keywords)):
        print(f"[{i}] FULL: {full}")
        print(f"    KEYS: {kw}")
        print()
    
    # Embed corpus
    print("Embedding corpus...")
    embeddings_full = model.encode(corpus_full)
    embeddings_keywords = model.encode(corpus_keywords)
    print("✓ Corpus embedded\n")
    
    # Test each query
    for query_idx, query in enumerate(test_queries):
        query_keywords = extract_keywords(query)
        
        print("=" * 80)
        print(f"QUERY {query_idx + 1}: {query}")
        print(f"KEYWORDS: {query_keywords}")
        print("=" * 80)
        
        # Embed query
        query_emb_full = model.encode([query])[0]
        query_emb_keywords = model.encode([query_keywords])[0]
        
        # Calculate similarities
        scores_full = [cosine_similarity(query_emb_full, emb) for emb in embeddings_full]
        scores_keywords = [cosine_similarity(query_emb_keywords, emb) for emb in embeddings_keywords]
        
        # Get top 3 for each approach
        top3_full_idx = sorted(range(len(scores_full)), key=lambda i: scores_full[i], reverse=True)[:3]
        top3_keywords_idx = sorted(range(len(scores_keywords)), key=lambda i: scores_keywords[i], reverse=True)[:3]
        
        print("\nRESULTS - FULL SENTENCE EMBEDDINGS:")
        for rank, idx in enumerate(top3_full_idx, 1):
            print(f"  {rank}. [{idx}] Score: {scores_full[idx]:.4f}")
            print(f"      {corpus_full[idx][:80]}...")
        
        print("\nRESULTS - KEYWORDS-ONLY EMBEDDINGS:")
        for rank, idx in enumerate(top3_keywords_idx, 1):
            print(f"  {rank}. [{idx}] Score: {scores_keywords[idx]:.4f}")
            print(f"      {corpus_full[idx][:80]}...")
        
        print("\nANALYSIS:")
        # Check if top results differ
        if top3_full_idx[0] != top3_keywords_idx[0]:
            print(f"  ⚠️  Different top matches!")
            print(f"      Full: [{top3_full_idx[0]}] vs Keywords: [{top3_keywords_idx[0]}]")
        else:
            print(f"  ✓ Both approaches agree on top match: [{top3_full_idx[0]}]")
        
        # Score difference analysis
        avg_score_full = np.mean([scores_full[i] for i in top3_full_idx])
        avg_score_keywords = np.mean([scores_keywords[i] for i in top3_keywords_idx])
        print(f"  Avg top-3 score (full): {avg_score_full:.4f}")
        print(f"  Avg top-3 score (keywords): {avg_score_keywords:.4f}")
        
        # Check noise filtering
        noise_indices = [9, 10, 11]  # Lunch, weather, coffee
        max_noise_full = max([scores_full[i] for i in noise_indices])
        max_noise_keywords = max([scores_keywords[i] for i in noise_indices])
        print(f"  Max noise score (full): {max_noise_full:.4f}")
        print(f"  Max noise score (keywords): {max_noise_keywords:.4f}")
        
        print()

if __name__ == "__main__":
    run_experiment()
