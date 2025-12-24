import asyncio
import json
import logging
from typing import Dict, Any, Optional
from hmlr.core.model_config import model_config

from hmlr.core.external_api_client import ExternalAPIClient
from hmlr.memory.synthesis.user_profile_manager import UserProfileManager

logger = logging.getLogger(__name__)

SCRIBE_SYSTEM_PROMPT = """
You are the User Profile Scribe. Record only facts that permanently configure how the system should treat the user.

SAVE:
- Hard constraints (medical, ethical, physical, scheduling)
  ✓ "I'm a strict vegan" → constraint: dietary_vegan
  ✓ "I'm allergic to latex" → constraint: allergy_latex
  ✓ "I'm afraid of heights" → constraint: phobia_heights
  ✓ "I never work weekends" → constraint: schedule_no_weekends

- Stable identity facts (education, role, language, credentials)
  ✓ "I have a PhD in Physics" → identity: education_phd_physics
  ✓ "I speak fluent Mandarin" → identity: language_mandarin
  ✓ "I'm a senior engineer at Google" → identity: role_engineer_google
  ✓ "I'm AWS certified" → identity: credential_aws

- Explicit long-term preferences stated by the user
  ✓ "I prefer dark mode always" → preference: ui_dark_mode
  ✓ "I use metric units" → preference: units_metric

IGNORE:
- Projects, plans, trips, documents
  ✗ "I'm building HMLR"
  ✗ "I'm going to China next month"
  ✗ "I'm working on the Q4 report"

- Temporary states, moods, tasks
  ✗ "I'm tired today"
  ✗ "My hand itches"
  ✗ "Help me write an email"

- World rules, policies, external events
  ✗ "Policy X limits records to 100k"
  ✗ "Training is mandatory next month"
  ✗ "The deadline is Friday"

IF UNSURE: Do not record.

OUTPUT FORMAT:
Return JSON: {"updates": []} if nothing to save.

{
  "updates": [
    {
      "category": "constraints",
      "key": "dietary_vegan",
      "action": "UPSERT",
      "attributes": {
        "type": "Dietary",
        "description": "User is a strict vegan",
        "updated": "12/22/2025 10:30:00"
      }
    },
    {
      "category": "identity",
      "key": "education_phd_physics",
      "action": "UPSERT",
      "attributes": {
        "type": "Education",
        "description": "User has PhD in Physics from MIT",
        "institution": "MIT",
        "year": "2018",
        "updated": "12/22/2025 10:30:00"
      }
    },
    {
      "category": "preferences",
      "key": "ui_dark_mode",
      "action": "UPSERT",
      "attributes": {
        "description": "User prefers dark mode UI",
        "updated": "12/22/2025 10:30:00"
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
        try:
            # Use the cheap fast model (nano/flash) - Now with native async!
            response_text = await self._query_llm_async(user_input)
            
            if not response_text:
                return

            # Parse JSON
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
                updates = data.get('updates', [])
                
                if updates:
                    logger.info(f"Scribe detected {len(updates)} profile updates: {[u.get('key') for u in updates]}")
                    self.profile_manager.update_profile_db(updates)
            else:
                logger.warning(f"Scribe response did not contain valid JSON: {response_text[:100]}...")
            
        except Exception as e:
            logger.error(f"Scribe agent failed: {e}", exc_info=True)

    async def _query_llm_async(self, user_input: str) -> str:
        """Async helper to call the API client"""
        from datetime import datetime
        
        # We pass the current profile context so the Scribe knows what already exists
        current_profile = self.profile_manager.get_user_profile_context()
        
        # Include current timestamp for the LLM to use
        current_time = datetime.now().strftime("%m/%d/%Y %H:%M:%S")
        
        full_prompt = f"{SCRIBE_SYSTEM_PROMPT}\n\nCURRENT DATE/TIME: {current_time}\n\nCURRENT PROFILE CONTEXT:\n{current_profile}\n\nUSER INPUT: \"{user_input}\""
        
        # Use async method - no more run_in_executor!
        return await self.api_client.query_external_api_async(
            full_prompt, 
            model=model_config.get_synthesis_model()
        )

    def _query_llm(self, user_input: str) -> str:
        """Helper to call the synchronous API client (kept for backward compatibility)"""
        from datetime import datetime
        
        # We pass the current profile context so the Scribe knows what already exists
        current_profile = self.profile_manager.get_user_profile_context()
        
        # Include current timestamp for the LLM to use
        current_time = datetime.now().strftime("%m/%d/%Y %H:%M:%S")
        
        full_prompt = f"{SCRIBE_SYSTEM_PROMPT}\n\nCURRENT DATE/TIME: {current_time}\n\nCURRENT PROFILE CONTEXT:\n{current_profile}\n\nUSER INPUT: \"{user_input}\""
        
        # Use synthesis model for profile updates
        return self.api_client.query_external_api(
            full_prompt, 
            model=model_config.get_synthesis_model()
        )
