"""
Bridge Block Data Models

A Bridge Block is a multi-dimensional save state that preserves:
1. The "What" (Semantic Context): Topic label, summary
2. The "Who" (Affect & Persona): User emotional state, bot persona  
3. The "State" (Continuity Hooks): Open loops, decisions, active variables
4. The "Keys" (Sparse Indexing): Keywords for fast retrieval

Bridge Blocks enable "Forever Chat" by preserving conversation state
across topic shifts and session boundaries.
"""
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional
from datetime import datetime
from enum import Enum
import json


class BlockStatus(Enum):
    """Bridge Block lifecycle states."""
    ACTIVE = "ACTIVE"           # Currently being updated (current topic)
    PAUSED = "PAUSED"           # Topic closed, waiting in daily_ledger
    ARCHIVED = "ARCHIVED"       # Consolidated by Gardener, in long-term memory
    PARTIAL = "PARTIAL"         # Checkpoint block (volume threshold)


class ExitReason(Enum):
    """Why a Bridge Block was closed."""
    TOPIC_SHIFT = "topic_shift"             # User changed topics
    VOLUME_THRESHOLD = "volume_threshold"   # Token limit safety valve
    USER_QUIT = "user_quit"                 # Session ended
    MANUAL = "manual"                       # Explicitly closed


class EmbeddingStatus(Enum):
    """Gardener processing status."""
    PENDING = "PENDING"         # Not yet processed by Gardener
    DONE = "DONE"               # Embedded and archived


@dataclass
class BridgeBlock:
    """
    Multi-dimensional save state for conversation context.
    
    Design Philosophy:
    - NOT just a summary - preserves semantic, affective, and state information
    - Links to raw chunks for full fidelity retrieval
    - Staged in daily_ledger before consolidation
    - Forms linked list (prev_block_id) for context chains
    
    Attributes:
        block_id: Unique identifier (bb_YYYYMMDD_HHMM_uuid)
        span_id: Reference to conversation span
        topic_label: Human-readable topic name
        summary: High-level summary of conversation
        user_affect: User emotional state/tone (e.g., "[T2] Focused, Technical")
        bot_persona: Assistant persona used (e.g., "Senior Architect")
        open_loops: Tasks/questions still pending
        decisions_made: Key decisions from this topic
        active_variables: State variables (e.g., {"project": "HMLR"})
        keywords: Keywords for sparse indexing
        created_at: When block was created
        updated_at: Last update (for PARTIAL blocks)
        status: Lifecycle state (ACTIVE, PAUSED, ARCHIVED, PARTIAL)
        exit_reason: Why block was closed
        prev_block_id: Previous block in chain (for context)
        embedding_status: Whether Gardener processed it
        embedded_at: When Gardener embedded it
    """
    block_id: str
    span_id: str
    topic_label: str
    summary: str
    
    # The "Who" (Affect & Persona)
    user_affect: str = ""
    bot_persona: str = "Helpful Assistant"
    
    # The "State" (Continuity Hooks)
    open_loops: List[str] = field(default_factory=list)
    decisions_made: List[str] = field(default_factory=list)
    active_variables: Dict[str, str] = field(default_factory=dict)
    
    # The "Keys" (Sparse Indexing)
    keywords: List[str] = field(default_factory=list)
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None
    status: BlockStatus = BlockStatus.PAUSED
    exit_reason: Optional[ExitReason] = None
    prev_block_id: Optional[str] = None
    
    # Gardener Status
    embedding_status: EmbeddingStatus = EmbeddingStatus.PENDING
    embedded_at: Optional[datetime] = None
    
    def to_json(self) -> str:
        """
        Serialize to JSON for storage in daily_ledger.content_json.
        
        Returns:
            JSON string representation
        """
        data = {
            "block_id": self.block_id,
            "prev_block_id": self.prev_block_id,
            "span_id": self.span_id,
            "timestamp": self.created_at.isoformat(),
            "status": self.status.value,
            "exit_reason": self.exit_reason.value if self.exit_reason else None,
            "topic_label": self.topic_label,
            "summary": self.summary,
            "user_affect": self.user_affect,
            "bot_persona": self.bot_persona,
            "open_loops": self.open_loops,
            "decisions_made": self.decisions_made,
            "active_variables": self.active_variables,
            "keywords": self.keywords
        }
        return json.dumps(data, indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'BridgeBlock':
        """
        Deserialize from JSON stored in daily_ledger.content_json.
        
        Args:
            json_str: JSON string representation
            
        Returns:
            BridgeBlock instance
        """
        data = json.loads(json_str)
        
        return cls(
            block_id=data["block_id"],
            span_id=data["span_id"],
            topic_label=data["topic_label"],
            summary=data["summary"],
            user_affect=data.get("user_affect", ""),
            bot_persona=data.get("bot_persona", "Helpful Assistant"),
            open_loops=data.get("open_loops", []),
            decisions_made=data.get("decisions_made", []),
            active_variables=data.get("active_variables", {}),
            keywords=data.get("keywords", []),
            created_at=datetime.fromisoformat(data["timestamp"]),
            status=BlockStatus(data["status"]),
            exit_reason=ExitReason(data["exit_reason"]) if data.get("exit_reason") else None,
            prev_block_id=data.get("prev_block_id")
        )