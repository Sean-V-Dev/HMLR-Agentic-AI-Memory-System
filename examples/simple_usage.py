"""
HMLR Simple Usage Example

This demonstrates the basic usage of HMLR with the new public API.
"""

import asyncio
import os
from hmlr import HMLRClient


async def main():
    """Run a simple conversation with memory."""
    
    # Get API key from environment or use your key
    api_key = os.getenv("OPENAI_API_KEY", "your-api-key-here")
    
    # Initialize HMLR client
    print("=" * 60)
    print("HMLR Simple Chat Example")
    print("=" * 60)
    
    client = HMLRClient(
        api_key=api_key,
        db_path="example_memory.db",
        model="gpt-4.1-mini"  # ONLY tested model!
    )
    
    try:
        # First conversation - introduce yourself
        print("\n--- Turn 1: Introduction ---")
        response = await client.chat(
            "Hello! My name is Alice and I'm a Python developer. "
            "I love building AI applications."
        )
        print(f"AI: {response['content']}\n")
        
        # Second conversation - HMLR remembers your name
        print("--- Turn 2: Memory Test ---")
        response = await client.chat("What's my name?")
        print(f"AI: {response['content']}\n")
        
        # Third conversation - HMLR remembers your interests
        print("--- Turn 3: Interest Recall ---")
        response = await client.chat("What do I like to build?")
        print(f"AI: {response['content']}\n")
        
        # Fourth conversation - Add more context
        print("--- Turn 4: Add More Context ---")
        response = await client.chat(
            "I'm currently working on a memory system for AI agents. "
            "It uses bridge blocks to organize information."
        )
        print(f"AI: {response['content']}\n")
        
        # Fifth conversation - Multi-hop retrieval
        print("--- Turn 5: Multi-hop Memory ---")
        response = await client.chat(
            "Can you remind me what I'm working on and what I like to do?"
        )
        print(f"AI: {response['content']}\n")
        
        # Show memory statistics
        print("=" * 60)
        stats = client.get_memory_stats()
        print("Memory Statistics:")
        print(f"  Total turns: {stats['total_turns']}")
        print(f"  Sliding window: {stats['sliding_window_size']} turns")
        print(f"  Database: {stats['db_path']}")
        print(f"  Model: {stats['model']}")
        print("=" * 60)
        
    finally:
        # Always close the client
        client.close()


if __name__ == "__main__":
    asyncio.run(main())
