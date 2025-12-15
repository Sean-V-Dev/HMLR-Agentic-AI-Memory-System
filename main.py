
import os
# Force Phoenix to use the same storage folder every time
# Must be set BEFORE importing phoenix (which happens in core.telemetry)
os.environ["PHOENIX_WORKING_DIR"] = os.path.join(os.getcwd(), "phoenix_storage")

import asyncio
from typing import Optional

# Phase 3 Refactor: Component Factory
from hmlr.core.component_factory import ComponentFactory

# Memory import for type hints
from hmlr.memory import Storage
from hmlr.core.telemetry import init_telemetry


async def main():
    """Main console interface for CognitiveLattice."""
    
    # === Phase 10: Initialize Observability === #
    init_telemetry()
    
    # === Phase 3: Initialize all components via factory === #
    print("🏗️  Initializing CognitiveLattice...")
    components = ComponentFactory.create_all_components()
    
    # Extract commonly used components for convenience
    storage = components.storage
    
    # === Phase 1 & 3: Create ConversationEngine === #
    conversation_engine = ComponentFactory.create_conversation_engine(components)
    
    # === Welcome Message === #
    print("\n📋 CognitiveLattice Interactive Agent")
    print("=" * 50)
    print("💬 Starting Interactive Analysis Engine")
    print("=" * 50)
    print("🔔 NOTE: External API calls will ONLY be made when you explicitly request them!")
    print("Enter your request (e.g., 'Help me plan a trip'), or type 'exit' to quit.")
    
    # === Main Loop === #
    loop = asyncio.get_running_loop()
    while True:
        try:
            # Use run_in_executor for input to avoid blocking the event loop
            # This allows background tasks (like Scribe) to complete while waiting for user input
            user_query = await loop.run_in_executor(None, input, "\nYour request: ")
            
            # === Special Commands (kept in main.py) === #
            
            if user_query.lower() == 'synthesize':
                print("🔄 Manually triggering daily synthesis...")
                today = components.conversation_mgr.current_day
                synthesis_result = components.synthesis_manager.trigger_daily_synthesis(today)
                if synthesis_result:
                    print(f"✅ Synthesis completed for {today}")
                    stats = components.synthesis_manager.get_synthesis_stats()
                    print(f"   User profile: {stats['user_profile_topics']} topics, {stats['day_emotions_tracked']} day patterns")
                else:
                    print(f"ℹ️ No data available for synthesis on {today}")
                continue
            
            if user_query.lower() in ['exit', 'quit']:
                print("\n✅ Exiting interactive session.")
                
                # Display usage metrics if available
                if hasattr(components, 'usage_tracker'):
                    print("\n" + "="*70)
                    print("📊 Session Summary - Context Usage Metrics")
                    print("="*70)
                    
                    overall_eff = components.usage_tracker.get_overall_efficiency()
                    summary = components.usage_tracker.get_summary()
                    query_count = summary.get('total_queries', 0)
                    total_turns = summary.get('total_turns_tracked', 0)
                    
                    print(f"\n🎯 Overall Context Efficiency:")
                    print(f"   Queries processed: {query_count}")
                    print(f"   Avg efficiency: {overall_eff:.1f}%")
                    print(f"   Total turns tracked: {total_turns}")
                    
                    # Most used turns
                    most_used = components.usage_tracker.get_most_used_turns(limit=5)
                    if most_used:
                        print(f"\n🔥 Most Referenced Turns:")
                        for turn_usage in most_used[:5]:
                            print(f"   {turn_usage.turn_id}: used {turn_usage.usage_count} times")
                
                print(f"\n✅ Session complete. Memory state saved.")
                break
            
            if user_query.lower() in ['show plans', 'list plans', 'my plans']:
                display_user_plans(storage)
                continue
                
            if user_query.lower() in ['today', 'today\'s tasks', 'tasks today']:
                display_todays_tasks(storage)
                continue
                
            if user_query.lower().startswith('show plan '):
                plan_id = user_query.lower().replace('show plan ', '').strip()
                display_plan_details(storage, plan_id)
                continue
                
            if user_query.lower().startswith('complete task ') or user_query.lower().startswith('mark done '):
                # Parse task completion request
                task_desc = user_query.lower().replace('complete task ', '').replace('mark done ', '').strip()
                print(f"🔍 Looking for task: '{task_desc}'")
                
                # Find matching tasks in today's plans
                todays_tasks = get_todays_tasks(storage)
                matching_tasks = []
                
                for task in todays_tasks:
                    if task_desc.lower() in task.task.lower():
                        matching_tasks.append(task)
                
                if not matching_tasks:
                    print(f"❌ No matching tasks found for today.")
                    continue
                    
                if len(matching_tasks) == 1:
                    task = matching_tasks[0]
                    # Find which plan this task belongs to
                    plans = storage.get_active_plans()
                    plan_id = None
                    for plan in plans:
                        if any(item.task == task.task and item.date == task.date for item in plan.items):
                            plan_id = plan.plan_id
                            break
                    
                    if plan_id:
                        # Mark as completed
                        storage.update_plan_item_completion(
                            plan_id=plan_id,
                            date=task.date,
                            task=task.task,
                            completed=True
                        )
                        print(f"✅ Marked complete: {task.task}")
                    else:
                        print(f"❌ Could not find plan for this task.")
                else:
                    print(f"🤔 Found {len(matching_tasks)} matching tasks:")
                    for i, task in enumerate(matching_tasks, 1):
                        print(f"   {i}. {task.task}")
                    print(f"   Please be more specific with your task description.")
                
                continue

            # === Phase 4 Refactor: Delegate ALL conversation logic to ConversationEngine === #
            response = await conversation_engine.process_user_message(user_query)
            
            # Display the response
            print(response.to_console_display())

        except KeyboardInterrupt:
            print("\n⚠️ Process interrupted by user. Exiting.")
            break
        except Exception as e:
            print(f"\n❌ An error occurred during interactive analysis: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⚠️ Process interrupted by user")
    except Exception as e:
        print(f"\n❌ Error in main execution: {e}")
        import traceback
        traceback.print_exc()
