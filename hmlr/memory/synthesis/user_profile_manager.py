import json
import logging
import threading
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class UserProfileManager:
    """
    Manages the User Profile (Lite) JSON file.
    Handles thread-safe reading and writing of the glossary.
    """
    
    def __init__(self, profile_path: str = None):
        # Default to package-relative path
        if profile_path is None:
            # Resolve from hmlr/config/ relative to this file
            default_path = Path(__file__).parent.parent.parent / "config" / "user_profile_lite.json"
            profile_path = str(default_path)
        
        # Use absolute path to avoid CWD issues
        import os
        
        # Check if environment variable overrides the default path
        env_profile_path = os.environ.get('USER_PROFILE_PATH')
        if env_profile_path:
            profile_path = env_profile_path
        
        if not os.path.isabs(profile_path):
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            profile_path = os.path.join(base_dir, profile_path)
            
        self.profile_path = Path(profile_path)
        self._lock = threading.Lock()
        
        # Ensure directory exists
        self.profile_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create default if not exists
        if not self.profile_path.exists():
            self._create_default_profile()

    def _create_default_profile(self):
        default_data = {
            "version": "1.0",
            "last_updated": datetime.now().isoformat(),
            "glossary": {
                "projects": [],
                "entities": [],
                "constraints": []
            }
        }
        # Use atomic write pattern
        import os
        temp_path = f"{self.profile_path}.tmp"
        try:
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(default_data, f, indent=2)
            os.replace(temp_path, self.profile_path)
        except Exception as e:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            logger.error(f"Failed to create default profile: {e}", exc_info=True)
            raise

    def get_user_profile_context(self, max_tokens: int = 300) -> str:
        """
        Reads the profile and flattens it into a string for the Context Window.
        
        Args:
            max_tokens: Maximum estimated tokens (approx 4 chars/token) to return.
        """
        with self._lock:
            try:
                with open(self.profile_path, "r", encoding='utf-8') as f:
                    data = json.load(f)
                
                glossary = data.get('glossary', {})
                context_str = "<user_glossary>\n"
                
                # CRITICAL: Constraints FIRST - they must never be truncated
                if glossary.get('constraints'):
                    context_str += "  [Constraints - IMMUTABLE]\n"
                    for c in glossary['constraints']:
                        # Constraints have 'description' not 'value'
                        desc = c.get('description', c.get('value', ''))
                        constraint_type = c.get('type', '')
                        severity = c.get('severity', '')
                        updated = c.get('updated', '')
                        
                        # Build metadata string
                        metadata_parts = []
                        if constraint_type:
                            metadata_parts.append(f"Type: {constraint_type}")
                        if severity:
                            metadata_parts.append(f"Severity: {severity}")
                        if updated:
                            metadata_parts.append(f"Updated: {updated}")
                        
                        # Format with metadata if available
                        if metadata_parts:
                            metadata_str = ", ".join(metadata_parts)
                            context_str += f"  - {c['key']}: {desc} [{metadata_str}]\n"
                        else:
                            context_str += f"  - {c['key']}: {desc}\n"
                
                # Projects (can be truncated if needed)
                if glossary.get('projects'):
                    context_str += "  [Projects]\n"
                    for p in glossary['projects']:
                        desc = p.get('description', '')
                        domain = p.get('domain', '')
                        status = p.get('status', '')
                        updated = p.get('updated', '')
                        
                        # Build metadata string
                        metadata_parts = []
                        if status:
                            metadata_parts.append(status)
                        if updated:
                            metadata_parts.append(f"Updated: {updated}")
                        
                        metadata_str = ", ".join(metadata_parts) if metadata_parts else status
                        if domain and metadata_str:
                            context_str += f"  - {p['key']}: {desc} ({domain}) [{metadata_str}]\n"
                        elif domain:
                            context_str += f"  - {p['key']}: {desc} ({domain})\n"
                        elif metadata_str:
                            context_str += f"  - {p['key']}: {desc} [{metadata_str}]\n"
                        else:
                            context_str += f"  - {p['key']}: {desc}\n"
                
                # Entities (can be truncated if needed)
                if glossary.get('entities'):
                    context_str += "  [Entities]\n"
                    for e in glossary['entities']:
                        desc = e.get('description', '')
                        etype = e.get('type', '')
                        updated = e.get('updated', '')
                        
                        # Build metadata string
                        metadata_parts = []
                        if etype:
                            metadata_parts.append(etype)
                        if updated:
                            metadata_parts.append(f"Updated: {updated}")
                        
                        if metadata_parts:
                            metadata_str = ", ".join(metadata_parts)
                            context_str += f"  - {e['key']}: {desc} ({metadata_str})\n"
                        else:
                            context_str += f"  - {e['key']}: {desc}\n"
                
                context_str += "</user_glossary>"
                
                # Simple truncation
                max_chars = max_tokens * 4
                if len(context_str) > max_chars:
                    context_str = context_str[:max_chars] + "... (truncated)"
                    
                return context_str
                
            except Exception as e:
                logger.error(f"Error reading user profile: {e}")
                return ""

    def update_profile_db(self, updates: List[Dict[str, Any]]) -> None:
        """
        Updates the profile database with a list of changes.
        Handles UPSERT logic (Append vs Edit).
        """
        if not updates:
            return

        with self._lock:
            try:
                # Read current state
                with open(self.profile_path, "r", encoding='utf-8') as f:
                    data = json.load(f)
                
                glossary = data.get('glossary', {})
                changes_made = False
                
                for update in updates:
                    category = update.get('category')
                    key = update.get('key')
                    action = update.get('action', 'UPSERT')
                    attributes = update.get('attributes', {})
                    
                    if not category or not key:
                        continue
                        
                    # Ensure category exists
                    if category not in glossary:
                        glossary[category] = []
                    
                    # Find existing item
                    existing_item = None
                    existing_index = -1
                    
                    for i, item in enumerate(glossary[category]):
                        if item.get('key', '').lower() == key.lower():
                            existing_item = item
                            existing_index = i
                            break
                    
                    if existing_item:
                        # Update existing
                        # If action is OVERWRITE, we might want to replace entirely, 
                        # but usually we just merge attributes.
                        # For now, let's merge attributes.
                        for k, v in attributes.items():
                            existing_item[k] = v
                        # Always update timestamp when modifying an entry
                        existing_item['updated'] = attributes.get('updated', datetime.now().strftime("%m/%d/%Y %H:%M:%S"))
                        logger.info(f"Updated {category} '{key}' in user profile.")
                        changes_made = True
                    else:
                        # Create new
                        new_item = {"key": key}
                        new_item.update(attributes)
                        # Ensure updated timestamp is set
                        if 'updated' not in new_item:
                            new_item['updated'] = datetime.now().strftime("%m/%d/%Y %H:%M:%S")
                        glossary[category].append(new_item)
                        logger.info(f"Created new {category} '{key}' in user profile.")
                        changes_made = True
                
                if changes_made:
                    data['last_updated'] = datetime.now().isoformat()
                    # Write back using atomic write pattern
                    import os
                    temp_path = f"{self.profile_path}.tmp"
                    try:
                        with open(temp_path, 'w', encoding='utf-8') as f:
                            json.dump(data, f, indent=2)
                        os.replace(temp_path, self.profile_path)
                    except Exception as e:
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                        raise e
                        
            except Exception as e:
                logger.error(f"Error updating user profile: {e}", exc_info=True)
