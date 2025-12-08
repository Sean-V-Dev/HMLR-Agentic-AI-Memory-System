"""
LangChain integration for HMLR.

Install with: pip install hmlr[langchain]

Example:
    ```python
    from langchain.chat_models import ChatOpenAI
    from langchain.chains import ConversationChain
    from hmlr.integrations.langchain import HMLRMemory
    
    memory = HMLRMemory(
        api_key="your-key",
        db_path="memory.db"
    )
    
    llm = ChatOpenAI(model="gpt-4.1-mini")
    chain = ConversationChain(llm=llm, memory=memory)
    
    response = chain.run("Hello!")
    ```
"""

try:
    from langchain.memory import BaseMemory
    from langchain.schema import BaseMessage, HumanMessage, AIMessage
except ImportError:
    raise ImportError(
        "LangChain integration requires langchain to be installed.\n"
        "Install with: pip install hmlr[langchain]"
    )

from typing import Dict, List, Any
import asyncio
from ..client import HMLRClient


class HMLRMemory(BaseMemory):
    """
    LangChain memory adapter for HMLR.
    
    This adapter allows HMLR to be used as a LangChain memory backend.
    Note that HMLR handles context retrieval internally, so this adapter
    primarily saves conversations to HMLR's storage.
    """
    
    hmlr_client: HMLRClient
    memory_key: str = "history"
    
    def __init__(
        self,
        api_key: str,
        db_path: str = "hmlr_memory.db",
        model: str = "gpt-4.1-mini",
        **kwargs
    ):
        """
        Initialize LangChain memory with HMLR backend.
        
        Args:
            api_key: OpenAI API key
            db_path: Path to HMLR database
            model: Model to use (default: gpt-4.1-mini)
            **kwargs: Additional HMLR client options
        """
        super().__init__()
        self.hmlr_client = HMLRClient(
            api_key=api_key,
            db_path=db_path,
            model=model,
            **kwargs
        )
    
    @property
    def memory_variables(self) -> List[str]:
        """Return memory variables."""
        return [self.memory_key]
    
    def load_memory_variables(
        self,
        inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Load memory context for the conversation.
        
        Returns empty dict - HMLR handles context internally during chat.
        The retrieved context is automatically included when processing
        the message through HMLR.
        """
        # HMLR retrieves and includes context internally
        return {self.memory_key: ""}
    
    def save_context(
        self,
        inputs: Dict[str, Any],
        outputs: Dict[str, str]
    ) -> None:
        """
        Save conversation turn to HMLR memory.
        
        Args:
            inputs: Dictionary containing user input
            outputs: Dictionary containing AI response
        """
        user_message = inputs.get("input", "")
        
        if not user_message:
            return
        
        # Process through HMLR (automatically stores the turn)
        try:
            asyncio.run(
                self.hmlr_client.chat(user_message)
            )
        except Exception as e:
            # Log error but don't crash the chain
            print(f"⚠️  HMLR memory save failed: {e}")
    
    def clear(self) -> None:
        """
        Clear memory.
        
        Note: HMLR doesn't support clearing the entire database.
        This only clears the sliding window. To fully reset,
        create a new database file.
        """
        import warnings
        warnings.warn(
            "HMLR does not support clearing the entire memory database.\n"
            "This only clears the sliding window (recent context).\n"
            "To fully reset, create a new database file.",
            UserWarning
        )
        self.hmlr_client.clear_sliding_window()
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        if hasattr(self, 'hmlr_client'):
            self.hmlr_client.close()
