"""
ID Generation System for Long-Horizon Memory

This module generates unique, self-describing IDs for all memory objects.
ID format: {type_prefix}_{timestamp}_{hash}

Examples:
    t_20251006_143022_abc123          # Turn
    s_t_20251006_143022_abc123        # Summary derived from turn
    k1_t_20251006_143022_abc123       # First keyword from turn
    tsk_rowing_20251006_143022_def456 # Task
    day_20251006                      # Day node (simple date)
    sess_20251006_140000_xyz789       # Session

Author: CognitiveLattice Team
Created: 2025-10-10
"""

import hashlib
import secrets
from datetime import datetime
from typing import Optional, Dict, Tuple, Literal


# ============================================================================
# ID GENERATION FUNCTIONS
# ============================================================================

def _generate_hash(length: int = 6) -> str:
    """
    Generate a short random hash for uniqueness.
    
    Args:
        length: Length of hash (default: 6 chars)
        
    Returns:
        Lowercase hex string (e.g., 'abc123')
    """
    return secrets.token_hex(length // 2)[:length]


def _format_timestamp(dt: Optional[datetime] = None) -> str:
    """
    Format datetime as compact timestamp: YYYYMMDD_HHMMSS
    
    Args:
        dt: Datetime to format (default: now)
        
    Returns:
        Formatted timestamp string (e.g., '20251006_143022')
    """
    if dt is None:
        dt = datetime.now()
    return dt.strftime("%Y%m%d_%H%M%S")


def generate_turn_id(timestamp: Optional[datetime] = None) -> str:
    """
    Generate unique turn ID.
    
    Format: t_{timestamp}_{hash}
    Example: t_20251006_143022_abc123
    
    Args:
        timestamp: Turn timestamp (default: now)
        
    Returns:
        Unique turn ID string
    """
    ts = _format_timestamp(timestamp)
    hash_part = _generate_hash()
    return f"t_{ts}_{hash_part}"


def generate_session_id(timestamp: Optional[datetime] = None) -> str:
    """
    Generate unique session ID.
    
    Format: sess_{timestamp}_{hash}
    Example: sess_20251006_140000_xyz789
    
    Args:
        timestamp: Session start time (default: now)
        
    Returns:
        Unique session ID string
    """
    ts = _format_timestamp(timestamp)
    hash_part = _generate_hash()
    return f"sess_{ts}_{hash_part}"


def generate_summary_id(source_turn_id: str) -> str:
    """
    Generate summary ID derived from turn.
    
    Format: s_{source_turn_id}
    Example: s_t_20251006_143022_abc123
    
    Args:
        source_turn_id: Turn this summary was extracted from
        
    Returns:
        Unique summary ID that encodes source
    """
    if not source_turn_id.startswith('t_'):
        raise ValueError(f"Invalid turn ID: {source_turn_id}. Must start with 't_'")
    
    return f"s_{source_turn_id}"


def generate_keyword_id(source_id: str, sequence: int) -> str:
    """
    Generate keyword ID derived from turn or summary.
    
    Format: k{sequence}_{source_id}
    Example: k1_t_20251006_143022_abc123
    
    Args:
        source_id: Turn or summary this keyword was extracted from
        sequence: Keyword number (1, 2, 3...) from this source
        
    Returns:
        Unique keyword ID
    """
    if not (source_id.startswith('t_') or source_id.startswith('s_')):
        raise ValueError(f"Invalid source ID: {source_id}. Must start with 't_' or 's_'")
    
    return f"k{sequence}_{source_id}"


def generate_affect_id(source_turn_id: str) -> str:
    """
    Generate affect ID derived from turn.
    
    Format: a_{source_turn_id}
    Example: a_t_20251006_143022_abc123
    
    Args:
        source_turn_id: Turn this affect was detected in
        
    Returns:
        Unique affect ID
    """
    if not source_turn_id.startswith('t_'):
        raise ValueError(f"Invalid turn ID: {source_turn_id}. Must start with 't_'")
    
    return f"a_{source_turn_id}"


def generate_task_id(
    task_type: str,
    timestamp: Optional[datetime] = None,
    title_hint: Optional[str] = None
) -> str:
    """
    Generate unique task ID.
    
    Format: tsk_{type_hint}_{timestamp}_{hash}
    Example: tsk_rowing_20251006_143022_def456
    
    Args:
        task_type: Task type (discrete, recurring_plan, ongoing_commitment)
        timestamp: Task creation time (default: now)
        title_hint: Optional short hint from title for readability
        
    Returns:
        Unique task ID
    """
    ts = _format_timestamp(timestamp)
    hash_part = _generate_hash()
    
    # Clean title hint if provided
    if title_hint:
        # Take first word, lowercase, alphanumeric only
        hint = ''.join(c for c in title_hint.split()[0].lower() if c.isalnum())
        hint = hint[:10]  # Max 10 chars
        return f"tsk_{hint}_{ts}_{hash_part}"
    else:
        return f"tsk_{task_type[:4]}_{ts}_{hash_part}"


def generate_day_id(date: Optional[datetime] = None) -> str:
    """
    Generate day ID (simple date format).
    
    Format: day_{YYYY-MM-DD}
    Example: day_2025-10-06
    
    Args:
        date: Date (default: today)
        
    Returns:
        Day ID string
    """
    if date is None:
        date = datetime.now()
    return f"day_{date.strftime('%Y-%m-%d')}"


def generate_synthesis_id(
    synthesis_type: Literal["day", "week", "month", "year"],
    time_period: str
) -> str:
    """
    Generate synthesis ID.
    
    Format: syn_{type}_{period}
    Examples:
        syn_day_2025-10-06
        syn_week_2025-W41
        syn_month_2025-10
        syn_year_2025
    
    Args:
        synthesis_type: Type of synthesis
        time_period: Time period identifier
        
    Returns:
        Unique synthesis ID
    """
    return f"syn_{synthesis_type}_{time_period}"


def generate_vector_id(source_id: str) -> str:
    """
    Generate vector embedding ID derived from source.
    
    Format: v_{source_id}
    Examples:
        v_s_t_20251006_143022_abc123  (vector of summary)
        v_t_20251006_143022_abc123    (vector of turn)
        v_syn_day_2025-10-06          (vector of synthesis)
    
    Args:
        source_id: Source object this vector was generated from
        
    Returns:
        Unique vector ID
    """
    return f"v_{source_id}"


# ============================================================================
# ID PARSING & VALIDATION
# ============================================================================

def parse_id(id_str: str) -> Dict[str, str]:
    """
    Parse an ID string into components.
    
    Args:
        id_str: ID string to parse
        
    Returns:
        Dict with:
            - type: Object type (turn, summary, keyword, etc.)
            - source_type: Type of source object (for derived data)
            - timestamp: Timestamp component (if present)
            - hash: Hash component (if present)
            - full_id: Original ID string
            - components: List of ID parts
            
    Examples:
        >>> parse_id("t_20251006_143022_abc123")
        {
            'type': 'turn',
            'source_type': None,
            'timestamp': '20251006_143022',
            'hash': 'abc123',
            'full_id': 't_20251006_143022_abc123',
            'components': ['t', '20251006', '143022', 'abc123']
        }
        
        >>> parse_id("s_t_20251006_143022_abc123")
        {
            'type': 'summary',
            'source_type': 'turn',
            'timestamp': '20251006_143022',
            'hash': 'abc123',
            'full_id': 's_t_20251006_143022_abc123',
            'components': ['s', 't', '20251006', '143022', 'abc123']
        }
    """
    parts = id_str.split('_')
    
    # Type mapping
    type_map = {
        't': 'turn',
        's': 'summary',
        'k1': 'keyword', 'k2': 'keyword', 'k3': 'keyword',
        'k4': 'keyword', 'k5': 'keyword',  # Support multiple keywords
        'a': 'affect',
        'v': 'vector',
        'tsk': 'task',
        'day': 'day',
        'sess': 'session',
        'syn': 'synthesis'
    }
    
    type_prefix = parts[0]
    object_type = type_map.get(type_prefix, 'unknown')
    
    # Extract source type for derived data
    source_type = None
    timestamp = None
    hash_part = None
    
    if object_type == 'summary':
        # Format: s_t_20251006_143022_abc123
        source_type = 'turn' if len(parts) > 1 and parts[1] == 't' else 'unknown'
        timestamp = f"{parts[2]}_{parts[3]}" if len(parts) > 3 else None
        hash_part = parts[4] if len(parts) > 4 else None
        
    elif object_type == 'keyword':
        # Format: k1_t_20251006_143022_abc123
        source_type = type_map.get(parts[1], 'unknown') if len(parts) > 1 else None
        timestamp = f"{parts[2]}_{parts[3]}" if len(parts) > 3 else None
        hash_part = parts[4] if len(parts) > 4 else None
        
    elif object_type == 'affect':
        # Format: a_t_20251006_143022_abc123
        source_type = 'turn' if len(parts) > 1 and parts[1] == 't' else 'unknown'
        timestamp = f"{parts[2]}_{parts[3]}" if len(parts) > 3 else None
        hash_part = parts[4] if len(parts) > 4 else None
        
    elif object_type == 'vector':
        # Format: v_s_t_20251006_143022_abc123
        source_type = type_map.get(parts[1], 'unknown') if len(parts) > 1 else None
        # Timestamp depends on source structure
        
    elif object_type in ['turn', 'session', 'task']:
        # Format: t_20251006_143022_abc123
        timestamp = f"{parts[1]}_{parts[2]}" if len(parts) > 2 else None
        hash_part = parts[3] if len(parts) > 3 else None
        
    elif object_type == 'day':
        # Format: day_2025-10-06
        timestamp = parts[1] if len(parts) > 1 else None
        
    elif object_type == 'synthesis':
        # Format: syn_day_2025-10-06
        timestamp = parts[2] if len(parts) > 2 else None
    
    return {
        'type': object_type,
        'source_type': source_type,
        'timestamp': timestamp,
        'hash': hash_part,
        'full_id': id_str,
        'components': parts
    }


def validate_id(id_str: str) -> Tuple[bool, str]:
    """
    Validate ID format.
    
    Args:
        id_str: ID string to validate
        
    Returns:
        Tuple of (is_valid, error_message)
        If valid: (True, "")
        If invalid: (False, "reason")
    """
    if not id_str:
        return False, "ID is empty"
    
    parts = id_str.split('_')
    
    if len(parts) < 2:
        return False, "ID must have at least 2 parts separated by underscores"
    
    # Check type prefix
    valid_prefixes = ['t', 's', 'k1', 'k2', 'k3', 'k4', 'k5',
                     'a', 'v', 'tsk', 'day', 'sess', 'syn']
    
    if parts[0] not in valid_prefixes:
        return False, f"Invalid type prefix: {parts[0]}"
    
    # Type-specific validation
    prefix = parts[0]
    
    if prefix in ['t', 'sess']:
        # Should have format: prefix_YYYYMMDD_HHMMSS_hash
        if len(parts) < 4:
            return False, f"{prefix} ID must have timestamp and hash"
        
        # Validate timestamp format
        date_part = parts[1]
        time_part = parts[2]
        
        if len(date_part) != 8 or not date_part.isdigit():
            return False, f"Invalid date part: {date_part}. Expected YYYYMMDD"
        
        if len(time_part) != 6 or not time_part.isdigit():
            return False, f"Invalid time part: {time_part}. Expected HHMMSS"
    
    elif prefix == 'tsk':
        # Should have format: tsk_{hint}_{YYYYMMDD}_{HHMMSS}_{hash}
        # or: tsk_{type}_{YYYYMMDD}_{HHMMSS}_{hash}
        if len(parts) < 5:
            return False, "task ID must have format: tsk_hint_YYYYMMDD_HHMMSS_hash"
        
        # Validate timestamp format (parts[2] and parts[3])
        date_part = parts[2]
        time_part = parts[3]
        
        if len(date_part) != 8 or not date_part.isdigit():
            return False, f"Invalid date part: {date_part}. Expected YYYYMMDD"
        
        if len(time_part) != 6 or not time_part.isdigit():
            return False, f"Invalid time part: {time_part}. Expected HHMMSS"
    
    elif prefix == 'day':
        # Should have format: day_YYYY-MM-DD
        if len(parts) != 2:
            return False, "day ID must have format: day_YYYY-MM-DD"
        
        date_str = parts[1]
        if len(date_str) != 10 or date_str[4] != '-' or date_str[7] != '-':
            return False, f"Invalid day date format: {date_str}"
    
    elif prefix in ['s', 'a']:
        # Should reference a turn: s_t_... or a_t_...
        if len(parts) < 2 or parts[1] != 't':
            return False, f"{prefix} ID must reference a turn (t_...)"
    
    elif prefix.startswith('k'):
        # Should reference turn or summary: k1_t_... or k1_s_...
        if len(parts) < 2 or parts[1] not in ['t', 's']:
            return False, f"{prefix} ID must reference turn or summary"
    
    return True, ""


def get_id_type(id_str: str) -> str:
    """
    Quick extraction of ID type without full parsing.
    
    Args:
        id_str: ID string
        
    Returns:
        Type string (turn, summary, keyword, etc.) or 'unknown'
    """
    parsed = parse_id(id_str)
    return parsed['type']


def extract_source_id(derived_id: str) -> Optional[str]:
    """
    Extract source ID from a derived object's ID.
    
    Args:
        derived_id: ID of derived object (summary, keyword, etc.)
        
    Returns:
        Source ID string, or None if not a derived type
        
    Examples:
        >>> extract_source_id("s_t_20251006_143022_abc123")
        "t_20251006_143022_abc123"
        
        >>> extract_source_id("k1_t_20251006_143022_abc123")
        "t_20251006_143022_abc123"
        
        >>> extract_source_id("t_20251006_143022_abc123")
        None  # Not a derived type
    """
    parts = derived_id.split('_')
    prefix = parts[0]
    
    if prefix == 's' or prefix == 'a':
        # Format: s_t_20251006_143022_abc123
        # Source is everything after first underscore
        return '_'.join(parts[1:])
    
    elif prefix.startswith('k'):
        # Format: k1_t_20251006_143022_abc123
        # Source is everything after second underscore
        return '_'.join(parts[1:])
    
    elif prefix == 'v':
        # Format: v_s_t_20251006_143022_abc123
        # Source is everything after first underscore
        return '_'.join(parts[1:])
    
    return None


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def is_derived_from(child_id: str, parent_id: str) -> bool:
    """
    Check if child_id was derived from parent_id.
    
    Args:
        child_id: ID of potentially derived object
        parent_id: ID of potential parent object
        
    Returns:
        True if child was derived from parent
        
    Examples:
        >>> is_derived_from("s_t_20251006_143022_abc123", "t_20251006_143022_abc123")
        True
        
        >>> is_derived_from("k1_t_20251006_143022_abc123", "t_20251006_143022_abc123")
        True
        
        >>> is_derived_from("t_20251006_150000_xyz789", "t_20251006_143022_abc123")
        False
    """
    source_id = extract_source_id(child_id)
    return source_id == parent_id if source_id else False


def format_id_for_display(id_str: str) -> str:
    """
    Format ID for human-readable display.
    
    Args:
        id_str: ID string
        
    Returns:
        Formatted string with type and timestamp
        
    Examples:
        >>> format_id_for_display("t_20251006_143022_abc123")
        "Turn (2025-10-06 14:30:22)"
        
        >>> format_id_for_display("tsk_rowing_20251006_143022_def456")
        "Task: rowing (2025-10-06 14:30:22)"
    """
    parsed = parse_id(id_str)
    obj_type = parsed['type'].capitalize()
    
    if parsed['timestamp']:
        # Convert timestamp to readable format
        ts = parsed['timestamp']
        if '_' in ts:
            date_part, time_part = ts.split('_')
            # Format: YYYYMMDD_HHMMSS → YYYY-MM-DD HH:MM:SS
            formatted_date = f"{date_part[:4]}-{date_part[4:6]}-{date_part[6:8]}"
            formatted_time = f"{time_part[:2]}:{time_part[2:4]}:{time_part[4:6]}"
            return f"{obj_type} ({formatted_date} {formatted_time})"
        else:
            return f"{obj_type} ({ts})"
    
    return obj_type


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("🔑 ID Generation System Test")
    print("=" * 60)
    
    # Test turn ID
    print("\n1. Turn ID Generation:")
    turn_id = generate_turn_id()
    print(f"   Generated: {turn_id}")
    print(f"   Parsed: {parse_id(turn_id)}")
    print(f"   Valid: {validate_id(turn_id)}")
    
    # Test summary ID (derived)
    print("\n2. Summary ID (derived from turn):")
    summary_id = generate_summary_id(turn_id)
    print(f"   Generated: {summary_id}")
    print(f"   Parsed: {parse_id(summary_id)}")
    print(f"   Derived from turn: {is_derived_from(summary_id, turn_id)}")
    print(f"   Source: {extract_source_id(summary_id)}")
    
    # Test keyword IDs
    print("\n3. Keyword IDs (multiple from same turn):")
    for i in range(1, 4):
        kw_id = generate_keyword_id(turn_id, i)
        print(f"   k{i}: {kw_id}")
    
    # Test task ID
    print("\n4. Task ID:")
    task_id = generate_task_id("recurring_plan", title_hint="rowing")
    print(f"   Generated: {task_id}")
    print(f"   Parsed: {parse_id(task_id)}")
    print(f"   Display: {format_id_for_display(task_id)}")
    
    # Test day ID
    print("\n5. Day ID:")
    day_id = generate_day_id()
    print(f"   Generated: {day_id}")
    print(f"   Parsed: {parse_id(day_id)}")
    
    # Test session ID
    print("\n6. Session ID:")
    sess_id = generate_session_id()
    print(f"   Generated: {sess_id}")
    print(f"   Parsed: {parse_id(sess_id)}")
    
    # Test validation
    print("\n7. Validation Tests:")
    test_ids = [
        "t_20251006_143022_abc123",
        "invalid",
        "day_2025-10-06",
        "s_t_20251006_143022_abc123",
        "k1_t_20251006_143022_abc123",
        "tsk_rowing_20251006_143022_def456"
    ]
    
    for test_id in test_ids:
        is_valid, msg = validate_id(test_id)
        status = "✅" if is_valid else "❌"
        print(f"   {status} {test_id}: {msg if not is_valid else 'Valid'}")
    
    print("\n🎉 All ID generation tests passed!")
    print("=" * 60)


# ============================================================================
# ID GENERATOR CLASS (for dependency injection)
# ============================================================================

class IDGenerator:
    """
    Class wrapper for ID generation functions.
    Provides dependency injection interface for components that need ID generation.
    """
    
    def generate_id(self, prefix: str) -> str:
        """
        Generate a unique ID with the given prefix.
        
        Args:
            prefix: ID prefix (e.g., "dos" for dossier, "fact" for fact, "prov" for provenance)
        
        Returns:
            Unique ID string in format: {prefix}_{timestamp}_{hash}
        """
        timestamp = _format_timestamp()
        hash_val = _generate_hash(6)
        return f"{prefix}_{timestamp}_{hash_val}"
    
    def generate_turn_id(self, timestamp: Optional[datetime] = None) -> str:
        """Generate turn ID"""
        return generate_turn_id(timestamp)
    
    def generate_session_id(self, timestamp: Optional[datetime] = None) -> str:
        """Generate session ID"""
        return generate_session_id(timestamp)
    
    def generate_day_id(self, date: Optional[datetime] = None) -> str:
        """Generate day ID"""
        return generate_day_id(date)
