"""
Synthesis module for CognitiveLattice.

Provides user profile management, scribe, and dossier governance.
"""

from .user_profile_manager import UserProfileManager
from .scribe import Scribe
from .dossier_governor import DossierGovernor

__all__ = [
    'UserProfileManager',
    'Scribe',
    'DossierGovernor'
]
