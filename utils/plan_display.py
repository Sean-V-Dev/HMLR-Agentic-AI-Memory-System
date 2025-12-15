"""
Plan Display Utilities

Functions for displaying user plans and tasks in the console interface.
"""

from datetime import datetime
from typing import List
from hmlr.memory.models import UserPlan, PlanItem


def display_user_plans(storage):
    """
    Display all active user plans.
    
    Args:
        storage: Storage instance
    """
    plans = storage.get_active_plans()
    
    if not plans:
        print("📋 No active plans found.")
        return
    
    print("📋 Active Plans:")
    print("-" * 50)
    for plan in plans:
        progress = f"{plan.progress_percentage:.1f}%" if plan.progress_percentage else "0%"
        print(f"🆔 {plan.plan_id}")
        print(f"   📝 {plan.title}")
        print(f"   📊 Progress: {progress} | Status: {plan.status}")
        print(f"   📅 Created: {plan.created_date}")
        if plan.notes:
            print(f"   📌 Notes: {plan.notes}")
        print()


def display_plan_details(storage, plan_id: str):
    """
    Display detailed information about a specific plan.
    
    Args:
        storage: Storage instance
        plan_id: Plan identifier
    """
    plan = storage.get_user_plan(plan_id)
    
    if not plan:
        print(f"❌ Plan '{plan_id}' not found.")
        return
    
    print(f"📋 Plan Details: {plan.title}")
    print("=" * 60)
    print(f"🆔 ID: {plan.plan_id}")
    print(f"📝 Topic: {plan.topic}")
    print(f"📅 Created: {plan.created_date}")
    print(f"⏱️  Duration: {plan.duration_weeks} weeks")
    print(f"📊 Status: {plan.status}")
    print(f"📈 Progress: {plan.progress_percentage:.1f}%" if plan.progress_percentage else "📈 Progress: 0%")
    if plan.notes:
        print(f"📌 Notes: {plan.notes}")
    
    print("\n📝 Plan Items:")
    print("-" * 60)
    
    if not plan.items:
        print("   No items in this plan.")
        return
    
    for item in plan.items:
        status = "✅" if item.completed else "⏳"
        duration = f" ({item.duration_minutes} min)" if item.duration_minutes else ""
        notes = f" - {item.notes}" if item.notes else ""
        print(f"   {status} {item.date}: {item.task}{duration}{notes}")


def get_todays_tasks(storage) -> List[PlanItem]:
    """
    Get all incomplete plan items for today.
    
    Args:
        storage: Storage instance
        
    Returns:
        List of PlanItem objects for today
    """
    today = datetime.now().strftime("%Y-%m-%d")
    return storage.get_plans_for_date(today)


def display_todays_tasks(tasks: List[PlanItem]):
    """
    Display today's tasks.
    
    Args:
        tasks: List of PlanItem objects
    """
    if not tasks:
        print("📋 No tasks scheduled for today.")
        return
    
    print("📋 Today's Tasks:")
    print("-" * 50)
    
    total_duration = sum(task.duration_minutes for task in tasks if task.duration_minutes)
    
    for task in tasks:
        duration = f" ({task.duration_minutes} min)" if task.duration_minutes else ""
        notes = f" - {task.notes}" if task.notes else ""
        print(f"   ⏳ {task.task}{duration}{notes}")
    
    if total_duration > 0:
        print(f"\n⏱️  Total estimated time: {total_duration} minutes")