"""
Semantic Chunker - Splits conversation turns into semantic chunks

Chunks responses by paragraphs/sentences to avoid topic dilution.
Always includes user context in each chunk for better retrieval.
"""

import re
from typing import List, Tuple



# Convenience function
def chunk_conversation(user_message: str, assistant_response: str, 
                      max_chunk_size: int = 400) -> List[str]:
    """
    Quick function to chunk a conversation turn.
    
    Args:
        user_message: User's query
        assistant_response: Assistant's response
        max_chunk_size: Max characters per chunk
        
    Returns:
        List of chunks
    """
    chunker = SemanticChunker(max_chunk_size=max_chunk_size)
    return chunker.chunk_turn(user_message, assistant_response)


if __name__ == "__main__":
    # Test the chunker
    print("🧪 Testing SemanticChunker...\n")
    
    chunker = SemanticChunker(max_chunk_size=200)
    
    # Test 1: Short response (no chunking)
    user_msg = "What's a bicycle?"
    assistant_msg = "A bicycle is a two-wheeled vehicle powered by pedaling."
    
    chunks = chunker.chunk_turn(user_msg, assistant_msg)
    print(f"Test 1: Short response")
    print(f"  Chunks: {len(chunks)}")
    print(f"  Content: {chunks[0][:100]}...\n")
    
    # Test 2: Long response (paragraph chunking)
    user_msg = "Tell me about Python and JavaScript"
    assistant_msg = """Python is a high-level programming language known for its readability. It's widely used in data science, machine learning, and web development. The syntax is clean and emphasizes code readability with significant whitespace.

JavaScript, on the other hand, is primarily used for web development. It runs in browsers and enables interactive web pages. Node.js allows JavaScript to run on servers as well.

Both languages have their strengths and are popular in modern software development."""
    
    chunks = chunker.chunk_turn(user_msg, assistant_msg)
    print(f"Test 2: Long response")
    print(f"  Chunks: {len(chunks)}")
    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i+1}: {len(chunk)} chars")
        print(f"    Preview: {chunk[:100]}...")
