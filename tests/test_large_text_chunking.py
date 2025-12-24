"""
Test large text chunking strategy for FactScrubber.

Verifies that frankenstein.txt (15k tokens) splits into 2 chunks
with proper overlap and boundary handling.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def estimate_tokens(text: str) -> int:
    """Quick token estimation (4 chars ≈ 1 token)."""
    return len(text) // 4


def chunk_large_text_for_extraction(
    text: str,
    chunk_size_tokens: int = 10000,
    overlap_tokens: int = 500,
    heuristic_tokens: int = 400
) -> list[dict]:
    """
    Split large text into chunks for fact extraction.
    
    Strategy: Each chunk = 10k content tokens + 400 heuristic tokens = 10,400 total to LLM
    Overlap: 500 tokens between chunks for context preservation
    Tax: ~7% for 10,001 token input (500 overlap / 10,001 = 5%, plus heuristics)
    
    Args:
        text: Full text to chunk
        chunk_size_tokens: Target content size per chunk (10k tokens)
        overlap_tokens: Overlap between chunks (500 for context)
        heuristic_tokens: Prompt overhead per chunk (400 tokens)
    
    Returns:
        List of chunk dicts with 'text', 'start_char', 'end_char', 'chunk_index'
    """
    estimated_tokens = estimate_tokens(text)
    
    # Only chunk if >10k tokens (keeps overhead low)
    if estimated_tokens <= 10000:
        return [{
            'text': text,
            'start_char': 0,
            'end_char': len(text),
            'chunk_index': 0,
            'total_chunks': 1
        }]
    
    # Convert token counts to character counts (rough estimate)
    chunk_size_chars = chunk_size_tokens * 4
    overlap_chars = overlap_tokens * 4
    
    chunks = []
    start_char = 0
    chunk_index = 0
    
    while start_char < len(text):
        end_char = min(start_char + chunk_size_chars, len(text))
        
        # Try to break at sentence boundary (. ! ?) if not at end
        if end_char < len(text):
            # Look back up to 500 chars for a sentence boundary
            search_start = max(end_char - 500, start_char)
            last_period = text.rfind('.', search_start, end_char)
            last_exclaim = text.rfind('!', search_start, end_char)
            last_question = text.rfind('?', search_start, end_char)
            
            boundary = max(last_period, last_exclaim, last_question)
            if boundary > search_start:
                end_char = boundary + 1  # Include the punctuation
        
        chunk_text = text[start_char:end_char]
        
        chunks.append({
            'text': chunk_text,
            'start_char': start_char,
            'end_char': end_char,
            'chunk_index': chunk_index,
            'total_chunks': None  # Will update after loop
        })
        
        # Move to next chunk with overlap
        # For last chunk, we're done
        if end_char >= len(text):
            break
            
        start_char = end_char - overlap_chars
        chunk_index += 1
    
    # Update total_chunks count
    total = len(chunks)
    for chunk in chunks:
        chunk['total_chunks'] = total
    
    return chunks


def test_frankenstein_chunking():
    """Test chunking on frankenstein.txt."""
    print("=" * 80)
    print("LARGE TEXT CHUNKING TEST - frankenstein.txt")
    print("=" * 80)
    print()
    
    # Load frankenstein.txt
    frank_path = Path(__file__).parent.parent / "frankenstein.txt"
    
    if not frank_path.exists():
        print(f"[FAIL] File not found: {frank_path}")
        return
    
    text = frank_path.read_text(encoding='utf-8')
    print(f"Loaded frankenstein.txt:")
    print(f"  Characters: {len(text):,}")
    print(f"  Estimated tokens: {estimate_tokens(text):,}")
    print()
    
    # Chunk the text
    print("Chunking text (target: 10k content tokens/chunk, 500 overlap)...")
    print(f"  Each chunk to LLM: 10,000 content + 400 heuristics = 10,400 tokens")
    print()
    chunks = chunk_large_text_for_extraction(text)
    print(f"[OK] Created {len(chunks)} chunks")
    print()
    
    # Verify chunks
    print("=" * 80)
    print("CHUNK BREAKDOWN")
    print("=" * 80)
    print()
    
    for i, chunk in enumerate(chunks):
        print(f"CHUNK {i + 1} of {chunk['total_chunks']}:")
        print(f"  Character range: {chunk['start_char']:,} - {chunk['end_char']:,}")
        print(f"  Length: {len(chunk['text']):,} chars")
        print(f"  Estimated tokens: {estimate_tokens(chunk['text']):,}")
        print(f"  First 100 chars: {chunk['text'][:100].strip()}...")
        print(f"  Last 100 chars: ...{chunk['text'][-100:].strip()}")
        print()
    
    # Verify overlap
    if len(chunks) > 1:
        print("=" * 80)
        print("OVERLAP VERIFICATION")
        print("=" * 80)
        print()
        
        for i in range(len(chunks) - 1):
            chunk1 = chunks[i]
            chunk2 = chunks[i + 1]
            
            # Check if chunk2 starts before chunk1 ends (overlap)
            if chunk2['start_char'] < chunk1['end_char']:
                overlap_chars = chunk1['end_char'] - chunk2['start_char']
                overlap_tokens = estimate_tokens(text[chunk2['start_char']:chunk1['end_char']])
                print(f"Chunks {i+1} → {i+2}:")
                print(f"  Overlap: {overlap_chars} chars (~{overlap_tokens} tokens)")
                print(f"  Overlap text: ...{text[chunk2['start_char']:chunk1['end_char']][:100]}...")
                print()
            else:
                print(f"[WARNING] No overlap between chunks {i+1} and {i+2}")
                print()
    
    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    print(f"Total text: {estimate_tokens(text):,} tokens")
    print(f"Number of chunks: {len(chunks)}")
    print(f"Avg chunk size: {sum(estimate_tokens(c['text']) for c in chunks) / len(chunks):.0f} tokens")
    
    # Calculate heuristic tax
    heuristic_tokens_per_chunk = 400
    total_content_tokens = sum(estimate_tokens(c['text']) for c in chunks)
    total_with_heuristics = total_content_tokens + (len(chunks) * heuristic_tokens_per_chunk)
    
    original_tokens = estimate_tokens(text)
    original_with_heuristics = original_tokens + heuristic_tokens_per_chunk
    
    overhead = total_with_heuristics - original_with_heuristics
    overhead_pct = (overhead / original_with_heuristics) * 100
    
    print(f"Original: {original_tokens:,} tokens + {heuristic_tokens_per_chunk} heuristics = {original_with_heuristics:,} total")
    print(f"Chunked: {total_content_tokens:,} content + {len(chunks) * heuristic_tokens_per_chunk} heuristics = {total_with_heuristics:,} total")
    print(f"Heuristic tax: {overhead:,} tokens ({overhead_pct:.1f}%)")
    
    if overhead_pct <= 10:
        print(f"[OK] Heuristic tax within 10% target")
    else:
        print(f"[WARNING] Heuristic tax exceeds 10% target")
    print()
    
    # Verification
    print("=" * 80)
    print("VERIFICATION")
    print("=" * 80)
    print()
    
    if len(chunks) == 2:
        print("[OK] Expected 2 chunks for 15k token document")
    else:
        print(f"[INFO] Got {len(chunks)} chunks (expected 2)")
    
    if overhead_pct <= 10:
        print("[OK] Heuristic tax ≤10%")
    else:
        print(f"[FAIL] Heuristic tax {overhead_pct:.1f}% exceeds 10%")
    
    if len(chunks) > 1:
        has_overlap = all(
            chunks[i+1]['start_char'] < chunks[i]['end_char'] 
            for i in range(len(chunks) - 1)
        )
        if has_overlap:
            print("[OK] All chunks have overlap")
        else:
            print("[FAIL] Missing overlap between chunks")


if __name__ == "__main__":
    test_frankenstein_chunking()
