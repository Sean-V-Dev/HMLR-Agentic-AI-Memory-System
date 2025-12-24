
import os
import sys
import io

# Force UTF-8 encoding for stdout/stderr on Windows to handle emojis
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import asyncio
from typing import Optional


from hmlr.core.component_factory import ComponentFactory

# Memory import for type hints
from hmlr.memory import Storage


async def main():
    """Main console interface for CognitiveLattice."""
    
    
    print("Initializing CognitiveLattice...")
    components = ComponentFactory.create_all_components()
    
    
    
    conversation_engine = ComponentFactory.create_conversation_engine(components)
    
    # === Welcome Message === #
    print("\nCognitiveLattice Interactive Agent")
    print("=" * 50)
    print("Starting Interactive Analysis Engine")
    print("=" * 50)
    print("NOTE: External API calls will ONLY be made when you explicitly request them!")
    print("Enter your request (e.g., 'Help me plan a trip'), or type 'exit' to quit.")
    
    # === Main Loop === #
    loop = asyncio.get_running_loop()
    while True:
        try:
            # Use run_in_executor for input to avoid blocking the event loop
            # This allows background tasks (like Scribe) to complete while waiting for user input
            user_query = await loop.run_in_executor(None, input, "\nYour request: ")
            
            if user_query.lower() in ['exit', 'quit']:
                print("\nExiting interactive session.")
                
                # Display usage metrics if available
                if hasattr(components, 'usage_tracker'):
                    print("\n" + "="*70)
                    print("Session Summary - Context Usage Metrics")
                    print("="*70)
                    
                    overall_eff = components.usage_tracker.get_overall_efficiency()
                    summary = components.usage_tracker.get_summary()
                    query_count = summary.get('total_queries', 0)
                    total_turns = summary.get('total_turns_tracked', 0)
                    
                    print(f"\nOverall Context Efficiency:")
                    print(f"   Queries processed: {query_count}")
                    print(f"   Avg efficiency: {overall_eff:.1f}%")
                    print(f"   Total turns tracked: {total_turns}")
                    
                    # Most used turns
                    most_used = components.usage_tracker.get_most_used_turns(limit=5)
                    if most_used:
                        print(f"\nMost Referenced Turns:")
                        for turn_usage in most_used[:5]:
                            print(f"   {turn_usage.turn_id}: used {turn_usage.usage_count} times")
                
                print(f"\nSession complete. Memory state saved.")
                break
            

            # === Delegate ALL conversation logic to ConversationEngine === #
            response = await conversation_engine.process_user_message(user_query)
            
            # Display the response
            print(response.to_console_display())

        except KeyboardInterrupt:
            print("\nProcess interrupted by user. Exiting.")
            break
        except Exception as e:
            print(f"\nAn error occurred during interactive analysis: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
    except Exception as e:
        print(f"\nError in main execution: {e}")
        import traceback
        traceback.print_exc()
