from hmlr.memory.conversation_manager import ConversationManager
from hmlr.memory.models import create_day_id
import unittest
from unittest.mock import MagicMock

class TestConversationManagerBug(unittest.TestCase):
    def test_init_bug(self):
        # This should succeed
        cm = ConversationManager(storage=MagicMock())
        print("Init successful")
        
        # This simulates a day rollover triggering the bug
        # We manually force the condition that triggers line 117
        cm.current_day = "old_day"
        try:
            # We need to mock generate_session_id or pass one
            # and mock storage methods to avoid SQL errors
            cm.storage.get_day.return_value = MagicMock()
            cm.storage.conn.cursor.return_value.fetchone.return_value = [0]
            
            cm.log_turn(
                session_id="test_session",
                user_message="hi",
                assistant_response="hello"
            )
        except AttributeError as e:
            print(f"Caught expected error: {e}")
            if "'ConversationManager' object has no attribute 'turn_sequence_by_session'" in str(e):
                print("CONFIRMED: Bug exists.")
            raise

if __name__ == "__main__":
    unittest.main()
