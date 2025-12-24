"""
Standardized response format for ConversationEngine.

This module provides a unified response structure that works across
different interfaces (CLI, Flask API, Discord, etc.).
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum
from datetime import datetime


class ResponseStatus(Enum):
    """Response status indicators."""
    SUCCESS = "success"
    ERROR = "error"
    PARTIAL = "partial"
    PENDING = "pending"


@dataclass
class ConversationResponse:
    """
    Standardized response format for ConversationEngine.
    
    This dataclass provides a consistent response structure that can be
    rendered differently for different interfaces:
    - Console: Use to_console_display()
    - Flask API: Use to_dict()
    - Discord/Slack: Use response_text directly
    
    Attributes:
        response_text: The main response message
        status: Response status (success/error/partial/pending)
        detected_intent: The intent that was detected (chat/query/task/planning/web_automation)
        detected_action: The action within that intent
        contexts_retrieved: Number of context turns retrieved
        sliding_window_turns: Number of turns in sliding window
        citations_found: Number of citations found in response
        context_efficiency: Percentage of provided context that was used
        tools_used: List of tool names that were invoked
        tool_results: Dictionary of tool results
        planning_session_id: ID of active planning session (if any)
        planning_phase: Current phase of planning (gathering/draft/verification/approved/cancelled)
        task_id: ID of active task (if any)
        task_step_number: Current step number in task
        task_completed: Whether task is completed
        error_message: Error message if status is ERROR
        error_traceback: Full traceback if status is ERROR
        timestamp: ISO timestamp of response generation
        processing_time_ms: Processing time in milliseconds
    """
    
    # Core response
    response_text: str
    status: ResponseStatus
    
    # Intent information
    detected_intent: str
    detected_action: str
    
    # Context metadata
    contexts_retrieved: int = 0
    sliding_window_turns: int = 0
    citations_found: int = 0
    context_efficiency: Optional[float] = None
    
    # Error information
    error_message: Optional[str] = None
    error_traceback: Optional[str] = None
    
    # Additional metadata
    timestamp: Optional[str] = None
    processing_time_ms: int = 0
    
    def __post_init__(self):
        """Set timestamp if not provided."""
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> dict:
        """
        Convert to dictionary for JSON serialization (Flask API).
        
        Returns:
            Dictionary with all response data
        """
        result = {
            "response": self.response_text,
            "status": self.status.value,
            "intent": self.detected_intent,
            "action": self.detected_action,
            "metadata": {
                "contexts_retrieved": self.contexts_retrieved,
                "sliding_window_turns": self.sliding_window_turns,
                "citations_found": self.citations_found,
                "context_efficiency": self.context_efficiency,
            },
            "timestamp": self.timestamp
        }
        
        # Add error info if present
        if self.error_message:
            result["error"] = {
                "message": self.error_message,
                "traceback": self.error_traceback
            }
        
        return result
    
    def to_console_display(self) -> str:
        """
        Format for console display (main.py).
        
        Returns:
            Formatted string for console output
        """
        output = []
        
        if self.status == ResponseStatus.SUCCESS:
            output.append(f"Response: {self.response_text}")
        elif self.status == ResponseStatus.ERROR:
            output.append(f"Error: {self.error_message}")
            if self.error_traceback:
                output.append(f"\nTraceback:\n{self.error_traceback}")
        elif self.status == ResponseStatus.PARTIAL:
            output.append(f"Partial Response: {self.response_text}")
        elif self.status == ResponseStatus.PENDING:
            output.append(f"Pending: {self.response_text}")
        
        # Add context efficiency if available
        if self.context_efficiency is not None:
            output.append(f"Context efficiency: {self.context_efficiency:.1f}%")
        
        return "\n".join(output)
