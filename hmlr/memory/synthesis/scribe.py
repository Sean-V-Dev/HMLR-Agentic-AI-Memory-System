import asyncio
import json
import logging
from typing import Dict, Any, Optional

from hmlr.core.external_api_client import ExternalAPIClient
from hmlr.memory.synthesis.user_profile_manager import UserProfileManager

logger = logging.getLogger(__name__)

SCRIBE_SYSTEM_PROMPT = """
### ROLE
You are the **User Profile Scribe**.
Your goal is to maintain a "Glossary" of the user's life by extracting **Projects**, **Entities**, and **Hard Constraints** from the conversation.
You do NOT answer the user. You only output JSON updates for the database.

### 1. DEFINITIONS & HEURISTICS (The Filter)
Do not record every noun. Only record items that pass the **SELF-REFERENCE TEST**.

**CRITICAL: The "ME" Test**
If the information doesn't explicitly link to the USER's identity or behavior, REJECT IT.
* ✅ "I am building HMLR" → User is the actor
* ❌ "Policy v7 is rolling out" → Passive external event (IGNORE - this is a world Fact, not a user Profile item)
* ❌ "The Synergy team is merging tools" → External group (IGNORE)

**A. DEFINITION OF A "PROJECT"**
To be saved as a Project, it must be an ACTIVE ENDEAVOR OWNED BY THE USER.
* ✅ "I am building Project Hades." (User is the actor)
* ✅ "My project HMLR uses Python." (User owns it)
* ❌ "Policy v7 is rolling out." (Passive event → IGNORE. This is a Fact, not a Project)
* ❌ "The Synergy Realization team is merging tools." (External group → IGNORE)
* ❌ "Project Cerberus will encrypt 4.7 million records." (No user ownership stated → IGNORE unless user says "my project" or "I'm working on")

Tests:
1.  **Named:** Proper noun (e.g., "HMLR", "Blue Sky", "The '69 Chevy"). Generic names ("my work") → IGNORE.
2.  **Persistent:** Multi-session goal (weeks/months), not temporary tasks.
3.  **User-Owned:** User explicitly claims ownership ("I am building", "my project", "I'm working on").

**B. DEFINITION OF AN "ENTITY"**
A permanent fact directly tied to the USER's personal world:
* **Business:** "I work at Acme Corp", "My company is called..."
* **Person:** "My son Mike", "My manager Sarah" (NOT just "Mike" or "Sarah" without relationship)
* **Asset:** "My server rack", "My boat" (USER-OWNED assets only)

**C. DEFINITION OF A "CONSTRAINT"**
A permanent configuration of the USER, not a rule of the world.
* ✅ "I am vegetarian." (Configures the User)
* ✅ "I never work weekends." (Configures the User's schedule)
* ✅ "I have a nut allergy." (User's medical condition)
* ❌ "All reports are due on Friday." (Rule of the world/Deadline → IGNORE)
* ❌ "Policy v6 limits encryption to 400k records." (External Fact → IGNORE)
* ❌ "Training is mandatory next month." (Company rule → IGNORE)

**THE "I AM/I HAVE" TEST:**
If you can't preface the sentence with "I am..." or "I have...", it is NOT a Profile Constraint.
* "I have a vegetarian diet" → PASS
* "I have a Policy v6" → FAIL (The company has the policy, not you)

Constraints are different from temporary states:
* "I have a latex allergy" = CONSTRAINT (permanent)
* "My hand itches" = temporary state (IGNORE)

### 2. ACTION RULES (Append vs. Edit)

**WHEN TO CREATE (New Entry):**
* The user mentions a specific Name (Key) that does NOT exist in your current context.
* *Action:* Create a new entry.

**WHEN TO UPDATE (Edit):**
* The user provides *new specific details* about an existing Key.
* **Logic:**
    * If the new info conflicts (e.g., "HMLR is now written in Rust, not Python"), use `action: "OVERWRITE"`.
    * If the new info adds detail (e.g., "HMLR also uses Redis"), use `action: "APPEND"`.
    * If the user just mentions the project without new facts ("How is HMLR doing?"), return **NO UPDATE**.

### 3. WHAT TO IGNORE (Crucial)
* **Opinions/Mood:** "I hate this," "I am tired." (Ignore for now).
* **One-off Tasks:** "Help me write an email," "Fix this specific bug."
* **General Topics:** "Tell me about Hackathons." (Unless the user says "I organize the NY Hackathon").

### 4. OUTPUT SCHEMA
Return a JSON object. If no updates are detected, return `{"updates": []}`.

Target JSON Structure:
{
  "updates": [
    {
      "category": "projects",  // or "entities", "constraints"
      "key": "HMLR",           // The unique ID/Name
      "action": "UPSERT",      // "UPSERT" handles both create and update
      "attributes": {
        "domain": "AI / Software",       // Infer this from context
        "description": "User's custom hierarchical memory system.",
        "tech_stack": "Python, SQLite, LLM", // Optional: Extract technical details if present
        "status": "Active"
      }
    },
    {
      "category": "constraints",
      "key": "allergy_latex",  // Unique constraint ID
      "action": "UPSERT",
      "attributes": {
        "type": "Allergy",
        "description": "User has a severe latex allergy",
        "severity": "severe",  // or "mild", "preference", etc.
        "updated": "12/11/2025 14:30:45"  // Always include current date and time when creating or updating
      }
    }
  ]
}
"""

class Scribe:
    """
    The Scribe Agent.
    Runs in the background to extract user profile updates from conversation.
    """
    
    def __init__(self, api_client: ExternalAPIClient, profile_manager: UserProfileManager):
        self.api_client = api_client
        self.profile_manager = profile_manager

    async def run_scribe_agent(self, user_input: str):
        """
        Runs in background. Does NOT block the main chat response.
        Analyzes user input for profile updates.
        """
        # print(f"   [DEBUG] Scribe task started for input: {user_input[:30]}...")
        try:
            # Use the cheap fast model (nano/flash)
            # Note: ExternalAPIClient.query_external_api is synchronous, 
            # but we are running this in an async task executor usually.
            
            loop = asyncio.get_event_loop()
            # print(f"   [DEBUG] Scribe calling LLM...")
            response_text = await loop.run_in_executor(
                None, 
                self._query_llm, 
                user_input
            )
            # print(f"   [DEBUG] Scribe LLM returned. Len: {len(response_text) if response_text else 0}")
            
            if not response_text:
                return

            # Parse JSON
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
                updates = data.get('updates', [])
                
                if updates:
                    logger.info(f"Scribe detected {len(updates)} profile updates.")
                    # Print to console for visibility during testing
                    print(f"\n   ✍️  Scribe detected {len(updates)} profile updates: {[u.get('key') for u in updates]}")
                    self.profile_manager.update_profile_db(updates)
                else:
                    # Debug: Print when no updates found to confirm it ran
                    # print(f"\n   ✍️  Scribe ran but found no updates.")
                    pass
            else:
                logger.warning(f"Scribe response did not contain valid JSON: {response_text[:100]}...")
                # print(f"   [DEBUG] Scribe invalid JSON: {response_text}")
            
        except Exception as e:
            logger.error(f"Scribe agent failed: {e}")
            print(f"\n   ❌ Scribe agent failed: {e}")
            import traceback
            traceback.print_exc()

    def _query_llm(self, user_input: str) -> str:
        """Helper to call the synchronous API client"""
        from datetime import datetime
        
        # We pass the current profile context so the Scribe knows what already exists
        current_profile = self.profile_manager.get_user_profile_context()
        
        # Include current timestamp for the LLM to use
        current_time = datetime.now().strftime("%m/%d/%Y %H:%M:%S")
        
        full_prompt = f"{SCRIBE_SYSTEM_PROMPT}\n\nCURRENT DATE/TIME: {current_time}\n\nCURRENT PROFILE CONTEXT:\n{current_profile}\n\nUSER INPUT: \"{user_input}\""
        
        # Use mini for better reasoning capabilities than nano
        return self.api_client.query_external_api(full_prompt, model="gpt-4.1-mini")
