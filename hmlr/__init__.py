"""
HMLR - Hierarchical Memory with Lattice Retrieval

A memory system for AI agents with:
- Bridge block architecture for topic segmentation
- Fact extraction and storage  
- Hierarchical retrieval (sentence -> paragraph -> block)
- GPT-4.1-mini optimized

WARNING: This package has ONLY been tested with GPT-4.1-mini.
Other models may produce incorrect results or fail completely.

Example:
    ```python
    from hmlr import HMLRClient
    
    client = HMLRClient(
        api_key="your-openai-key",
        db_path="memory.db"
    )
    
    response = await client.chat("Hello! My name is Alice.")
    print(response["content"])
    ```
"""

__version__ = "0.1.0"

from .client import HMLRClient

__all__ = ["HMLRClient"]
