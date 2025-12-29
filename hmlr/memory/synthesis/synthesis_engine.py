"""
Synthesis System for CognitiveLattice.

Provides hierarchical synthesis of user behavior patterns:
- Daily synthesis from conversation metadata
- Weekly/monthly aggregation and pattern analysis
- User profiling for longitudinal behavior tracking
- Integration into LLM prompts for personalized responses
"""

from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import json
import logging

from hmlr.memory.models import DayNode, DaySynthesis, Keyword, Summary, Affect, ConversationTurn
from hmlr.memory.storage import Storage

logger = logging.getLogger(__name__)


@dataclass
class UserProfile:
    """
    Longitudinal user profile built from synthesis data.
    Tracks patterns across days, weeks, months.
    """
    user_id: str = "default"

    # Emotional patterns
    day_of_week_emotions: Dict[str, str] = field(default_factory=dict)  # Monday: "curious", etc.
    emotional_tendencies: Dict[str, float] = field(default_factory=dict)  # emotion: frequency
    emotional_triggers: Dict[str, List[str]] = field(default_factory=dict)  # emotion: [topics]

    # Topic preferences
    favorite_topics: List[Tuple[str, float]] = field(default_factory=list)  # [(topic, engagement_score)]
    topic_evolution: Dict[str, List[datetime]] = field(default_factory=dict)  # topic: [dates_discussed]

    # Communication patterns
    communication_style: str = ""  # "analytical", "conversational", "technical", etc.
    preferred_discussion_depth: str = ""  # "surface", "detailed", "exploratory"
    response_patterns: Dict[str, int] = field(default_factory=dict)  # pattern: frequency

    # Support needs
    support_indicators: List[str] = field(default_factory=list)  # ["stress_management", "decision_support"]
    vulnerability_patterns: Dict[str, str] = field(default_factory=dict)  # day_of_week: vulnerability_type

    # Behavioral insights
    productivity_patterns: Dict[str, str] = field(default_factory=dict)  # time_of_day: productivity_level
    learning_style: str = ""  # "hands_on", "theoretical", "visual", etc.
    
    # === Phase 6.2: Planning Patterns === #
    planning_frequency: str = ""  # "daily", "weekly", "occasional", "rare"
    plan_completion_rate: float = 0.0  # Average completion rate across all plans
    preferred_plan_types: List[str] = field(default_factory=list)  # ["work", "personal", "health", "learning"]
    planning_horizon: str = ""  # "short_term", "medium_term", "long_term"
    plan_adherence_patterns: Dict[str, float] = field(default_factory=dict)  # day_of_week: adherence_rate

    def to_prompt_context(self, max_tokens: int = 300) -> str:
        """Convert profile to concise prompt context."""
        sections = []

        # Emotional patterns
        if self.day_of_week_emotions:
            today = datetime.now().strftime("%A")
            if today in self.day_of_week_emotions:
                sections.append(f"Today ({today}) you often feel {self.day_of_week_emotions[today]}.")

        # Favorite topics
        if self.favorite_topics:
            top_topics = [topic for topic, _ in self.favorite_topics[:3]]
            sections.append(f"Your favorite topics include: {', '.join(top_topics)}.")

        # Communication style
        if self.communication_style:
            sections.append(f"You prefer {self.communication_style} communication.")

        # Support needs
        if self.support_indicators:
            sections.append(f"You've shown interest in: {', '.join(self.support_indicators[:2])}.")

        # Behavioral insights
        if self.learning_style:
            sections.append(f"You learn best through {self.learning_style} approaches.")

        # === Phase 6.2: Planning Context === #
        if self.planning_frequency and self.planning_frequency != "rare":
            plan_context = f"You {self.planning_frequency} create plans"
            if self.plan_completion_rate > 0:
                completion_pct = int(self.plan_completion_rate * 100)
                plan_context += f" and complete about {completion_pct}% of your planned tasks"
            if self.preferred_plan_types:
                plan_context += f", focusing on {', '.join(self.preferred_plan_types[:2])}"
            sections.append(plan_context + ".")

        result = " ".join(sections)
        # Rough token estimation (4 chars per token)
        if len(result) > max_tokens * 4:
            result = result[:max_tokens * 4 - 3] + "..."

        return result


class DaySynthesizer:
    """
    Analyzes daily conversation metadata to create comprehensive day synthesis.
    """

    def __init__(self, storage: Storage):
        self.storage = storage

    def synthesize_day(self, day_id: str) -> Optional[DaySynthesis]:
        """
        Create comprehensive synthesis for a given day.

        Args:
            day_id: Date string in YYYY-MM-DD format

        Returns:
            DaySynthesis object or None if insufficient data
        """
        # Get all metadata for the day
        day_data = self._gather_day_metadata(day_id)
        if not day_data['turns']:
            return None

        # Analyze emotional arc
        emotional_arc = self._analyze_emotional_arc(day_data['affect'])

        # Identify key patterns
        key_patterns = self._identify_patterns(day_data)

        # Map topics to emotions
        topic_affect_mapping = self._map_topics_to_affect(day_data)

        # Generate behavioral notes
        behavioral_notes = self._generate_behavioral_notes(day_data)

        # Create synthesis
        synthesis = DaySynthesis(
            day_id=day_id,
            created_at=datetime.now(),
            emotional_arc=emotional_arc,
            key_patterns=key_patterns,
            topic_affect_mapping=topic_affect_mapping,
            behavioral_notes=behavioral_notes
        )

        return synthesis

    def _gather_day_metadata(self, day_id: str) -> Dict:
        """Gather all metadata for a day."""
        # Get turns for the day (using get_recent_turns with day_id filter)
        turns = self.storage.get_recent_turns(day_id=day_id, limit=1000)  # Get all turns for the day

        # Get derived metadata
        keywords = self.storage.get_day_keywords(day_id)
        summaries = self.storage.get_day_summaries(day_id)
        affect_patterns = self.storage.get_day_affect(day_id)

        return {
            'turns': turns,
            'keywords': keywords,
            'summaries': summaries,
            'affect': affect_patterns
        }

    def _analyze_emotional_arc(self, affect_patterns: List[Affect]) -> str:
        """Analyze the emotional journey throughout the day."""
        if not affect_patterns:
            return "Neutral emotional tone throughout the day."

        # Sort by time
        sorted_affect = sorted(affect_patterns, key=lambda x: x.first_detected)

        # Group by time periods
        morning_emotions = []
        afternoon_emotions = []
        evening_emotions = []

        for affect in sorted_affect:
            hour = affect.first_detected.hour
            if hour < 12:
                morning_emotions.append(affect.affect_label)
            elif hour < 18:
                afternoon_emotions.append(affect.affect_label)
            else:
                evening_emotions.append(affect.affect_label)

        # Build arc description
        arc_parts = []
        if morning_emotions:
            most_common = Counter(morning_emotions).most_common(1)[0][0]
            arc_parts.append(f"Started {most_common} in the morning")

        if afternoon_emotions:
            most_common = Counter(afternoon_emotions).most_common(1)[0][0]
            arc_parts.append(f"felt {most_common} in the afternoon")

        if evening_emotions:
            most_common = Counter(evening_emotions).most_common(1)[0][0]
            arc_parts.append(f"ended {most_common} in the evening")

        if arc_parts:
            return ", ".join(arc_parts) + "."
        else:
            return "Maintained consistent emotional state throughout the day."

    def _identify_patterns(self, day_data: Dict) -> List[str]:
        """Identify notable patterns in the day's conversations."""
        patterns = []

        turns = day_data['turns']
        keywords = day_data['keywords']
        affect = day_data['affect']

        # Conversation frequency
        if len(turns) > 10:
            patterns.append("Highly active conversation day")
        elif len(turns) < 3:
            patterns.append("Quiet, reflective day")

        # Topic diversity
        unique_topics = set()
        for kw in keywords:
            unique_topics.add(kw.keyword)
        if len(unique_topics) > 15:
            patterns.append("Exploratory day with diverse topics")
        elif len(unique_topics) < 5:
            patterns.append("Focused on specific topics")

        # Emotional patterns
        emotion_counts = Counter(a.emotion for a in affect)
        dominant_emotion = emotion_counts.most_common(1)[0][0] if emotion_counts else None
        if dominant_emotion:
            patterns.append(f"Predominantly {dominant_emotion} emotional tone")

        # Question patterns
        question_count = sum(1 for turn in turns if '?' in turn.user_message)
        if question_count > len(turns) * 0.3:
            patterns.append("Inquisitive and exploratory")
        elif question_count < len(turns) * 0.1:
            patterns.append("More declarative than questioning")

        return patterns

    def _map_topics_to_affect(self, day_data: Dict) -> Dict[str, str]:
        """Map topics to associated emotions."""
        mapping = {}

        # Group affect by associated topics
        topic_emotions = defaultdict(list)
        for affect in day_data['affect']:
            for topic in affect.associated_topics:
                topic_emotions[topic].append(affect.affect_label)

        # Find dominant emotion per topic
        for topic, emotions in topic_emotions.items():
            if emotions:
                dominant = Counter(emotions).most_common(1)[0][0]
                mapping[topic] = dominant

        return mapping

    def _generate_behavioral_notes(self, day_data: Dict) -> str:
        """Generate notes about user behavior patterns."""
        notes = []

        turns = day_data['turns']

        # Response length patterns
        avg_response_length = sum(len(turn.user_message) for turn in turns) / len(turns)
        if avg_response_length > 200:
            notes.append("Detailed, thoughtful responses")
        elif avg_response_length < 50:
            notes.append("Concise, direct communication")

        # Conversation flow
        if len(turns) > 1:
            time_gaps = []
            sorted_turns = sorted(turns, key=lambda t: t.timestamp)
            for i in range(1, len(sorted_turns)):
                gap = (sorted_turns[i].timestamp - sorted_turns[i-1].timestamp).total_seconds() / 60
                time_gaps.append(gap)

            avg_gap = sum(time_gaps) / len(time_gaps)
            if avg_gap < 5:
                notes.append("Fast-paced, continuous conversation")
            elif avg_gap > 30:
                notes.append("Spaced-out, reflective conversation style")

        return "; ".join(notes) if notes else "Standard conversational patterns."


class HierarchicalSynthesizer:
    """
    Creates higher-level synthesis from daily data.
    Weekly, monthly, and yearly patterns.
    """

    def __init__(self, storage: Storage, day_synthesizer: DaySynthesizer):
        self.storage = storage
        self.day_synthesizer = day_synthesizer

    def synthesize_week(self, week_start_date: datetime) -> Dict:
        """Synthesize weekly patterns."""
        week_data = self._gather_week_data(week_start_date)

        return {
            'week_of': week_start_date.strftime("%Y-%m-%d"),
            'emotional_patterns': self._analyze_weekly_emotions(week_data),
            'topic_evolution': self._analyze_topic_evolution(week_data),
            'productivity_patterns': self._analyze_productivity_patterns(week_data),
            'key_insights': self._generate_weekly_insights(week_data)
        }

    def synthesize_month(self, month_start_date: datetime) -> Dict:
        """Synthesize monthly patterns."""
        month_data = self._gather_month_data(month_start_date)

        return {
            'month_of': month_start_date.strftime("%Y-%m"),
            'monthly_themes': self._identify_monthly_themes(month_data),
            'emotional_trends': self._analyze_monthly_emotions(month_data),
            'behavioral_changes': self._identify_behavioral_changes(month_data),
            'growth_indicators': self._assess_growth_indicators(month_data)
        }

    def _gather_week_data(self, week_start: datetime) -> List[DaySynthesis]:
        """Gather synthesis data for a week."""
        syntheses = []
        for i in range(7):
            day_date = week_start + timedelta(days=i)
            day_id = day_date.strftime("%Y-%m-%d")
            synthesis = self.day_synthesizer.synthesize_day(day_id)
            if synthesis:
                syntheses.append(synthesis)
        return syntheses

    def _gather_month_data(self, month_start: datetime) -> List[Dict]:
        """Gather weekly syntheses for a month."""
        weekly_syntheses = []
        for i in range(0, 28, 7):  # 4 weeks
            week_start = month_start + timedelta(days=i)
            week_synthesis = self.synthesize_week(week_start)
            weekly_syntheses.append(week_synthesis)
        return weekly_syntheses

    def _analyze_weekly_emotions(self, week_data: List[DaySynthesis]) -> Dict:
        """Analyze emotional patterns across the week."""
        day_emotions = {}
        for synthesis in week_data:
            # Extract dominant emotion from emotional_arc
            arc = synthesis.emotional_arc.lower()
            if 'curious' in arc:
                day_emotions[synthesis.day_id] = 'curious'
            elif 'frustrated' in arc:
                day_emotions[synthesis.day_id] = 'frustrated'
            elif 'excited' in arc:
                day_emotions[synthesis.day_id] = 'excited'
            elif 'satisfied' in arc:
                day_emotions[synthesis.day_id] = 'satisfied'
            else:
                day_emotions[synthesis.day_id] = 'neutral'

        return day_emotions

    def _analyze_topic_evolution(self, week_data: List[DaySynthesis]) -> Dict:
        """Track how topics evolved throughout the week."""
        topic_progression = defaultdict(list)

        for synthesis in week_data:
            for topic, emotion in synthesis.topic_affect_mapping.items():
                topic_progression[topic].append({
                    'day': synthesis.day_id,
                    'emotion': emotion
                })

        return dict(topic_progression)

    def _analyze_productivity_patterns(self, week_data: List[DaySynthesis]) -> Dict:
        """Analyze productivity patterns by day."""
        productivity = {}

        for synthesis in week_data:
            day_name = datetime.strptime(synthesis.day_id, "%Y-%m-%d").strftime("%A")

            # Infer productivity from patterns
            patterns = [p.lower() for p in synthesis.key_patterns]
            if any('active' in p or 'exploratory' in p for p in patterns):
                productivity[day_name] = 'high'
            elif any('quiet' in p or 'reflective' in p for p in patterns):
                productivity[day_name] = 'moderate'
            else:
                productivity[day_name] = 'standard'

        return productivity

    def _generate_weekly_insights(self, week_data: List[DaySynthesis]) -> List[str]:
        """Generate key insights from weekly data."""
        insights = []

        if len(week_data) >= 3:
            # Compare first half vs second half
            mid_point = len(week_data) // 2
            first_half = week_data[:mid_point]
            second_half = week_data[mid_point:]

            first_emotions = [self._extract_emotion(s.emotional_arc) for s in first_half]
            second_emotions = [self._extract_emotion(s.emotional_arc) for s in second_half]

            if first_emotions and second_emotions:
                first_avg = Counter(first_emotions).most_common(1)[0][0]
                second_avg = Counter(second_emotions).most_common(1)[0][0]

                if first_avg != second_avg:
                    insights.append(f"Emotional shift from {first_avg} to {second_avg} mid-week")

        return insights

    def _extract_emotion(self, emotional_arc: str) -> str:
        """Extract primary emotion from emotional arc text."""
        arc_lower = emotional_arc.lower()
        emotions = ['curious', 'frustrated', 'excited', 'satisfied', 'neutral']
        for emotion in emotions:
            if emotion in arc_lower:
                return emotion
        return 'neutral'

    def _identify_monthly_themes(self, month_data: List[Dict]) -> List[str]:
        """Identify overarching themes for the month."""
        all_topics = set()
        for week in month_data:
            for topic in week.get('topic_evolution', {}):
                all_topics.add(topic)

        # Group related topics
        themes = []
        if any(t in ['programming', 'coding', 'algorithms'] for t in all_topics):
            themes.append('Technical Learning')
        if any(t in ['personal', 'reflection', 'goals'] for t in all_topics):
            themes.append('Personal Development')

        return themes

    def _analyze_monthly_emotions(self, month_data: List[Dict]) -> Dict:
        """Analyze emotional trends across the month."""
        monthly_emotions = defaultdict(int)

        for week in month_data:
            emotions = week.get('emotional_patterns', {})
            for day, emotion in emotions.items():
                monthly_emotions[emotion] += 1

        return dict(monthly_emotions)

    def _identify_behavioral_changes(self, month_data: List[Dict]) -> List[str]:
        """Identify changes in behavior patterns."""
        changes = []

        # Compare first week vs last week
        if len(month_data) >= 2:
            first_week = month_data[0]
            last_week = month_data[-1]

            first_productivity = first_week.get('productivity_patterns', {})
            last_productivity = last_week.get('productivity_patterns', {})

            # Simple comparison
            first_avg = Counter(first_productivity.values()).most_common(1)[0][0]
            last_avg = Counter(last_productivity.values()).most_common(1)[0][0]

            if first_avg != last_avg:
                changes.append(f"Productivity shifted from {first_avg} to {last_avg}")

        return changes

    def _assess_growth_indicators(self, month_data: List[Dict]) -> List[str]:
        """Assess indicators of personal growth."""
        indicators = []

        # Look for increasing topic diversity or consistent engagement
        topic_counts = [len(week.get('topic_evolution', {})) for week in month_data]
        if len(set(topic_counts)) > 1 and topic_counts[-1] > topic_counts[0]:
            indicators.append("Increasing topic exploration")

        return indicators


class SynthesisManager:
    """
    Main orchestrator for synthesis operations.
    Integrates with conversation manager for automatic synthesis triggers.
    """

    def __init__(self, storage: Storage):
        self.storage = storage
        self.day_synthesizer = DaySynthesizer(storage)
        self.hierarchical_synthesizer = HierarchicalSynthesizer(storage, self.day_synthesizer)
        self.user_profile = UserProfile()

    def trigger_daily_synthesis(self, day_id: str) -> bool:
        """
        Trigger end-of-day synthesis.
        Called automatically when day changes or on demand.

        Returns True if synthesis was created.
        """
        synthesis = self.day_synthesizer.synthesize_day(day_id)
        if synthesis:
            # Store synthesis in day node
            self.storage.save_day_synthesis(synthesis)

            # Update user profile
            self._update_user_profile_from_day(synthesis)

            logger.info(f"Daily synthesis completed for {day_id}")
            return True

        return False

    def trigger_weekly_synthesis(self, week_start: datetime) -> bool:
        """Trigger end-of-week synthesis."""
        week_synthesis = self.hierarchical_synthesizer.synthesize_week(week_start)

        # Store weekly synthesis (could extend storage schema)
        self._store_weekly_synthesis(week_synthesis)

        # Update user profile with weekly patterns
        self._update_user_profile_from_week(week_synthesis)

        logger.info(f"Weekly synthesis completed for week of {week_start.strftime('%Y-%m-%d')}")
        return True

    def trigger_monthly_synthesis(self, month_start: datetime) -> bool:
        """Trigger end-of-month synthesis."""
        month_synthesis = self.hierarchical_synthesizer.synthesize_month(month_start)

        # Store monthly synthesis
        self._store_monthly_synthesis(month_synthesis)

        # Update user profile with monthly insights
        self._update_user_profile_from_month(month_synthesis)

        logger.info(f"Monthly synthesis completed for {month_start.strftime('%Y-%m')}")
        return True

    def get_user_profile_context(self, max_tokens: int = 300) -> str:
        """Get user profile context for LLM prompts."""
        return self.user_profile.to_prompt_context(max_tokens)

    def _update_user_profile_from_day(self, synthesis: DaySynthesis):
        """Update user profile from daily synthesis."""
        day_name = datetime.strptime(synthesis.day_id, "%Y-%m-%d").strftime("%A")

        # Extract dominant emotion for day of week
        emotion = self.hierarchical_synthesizer._extract_emotion(synthesis.emotional_arc)
        self.user_profile.day_of_week_emotions[day_name] = emotion

        # Update emotional tendencies
        if emotion not in self.user_profile.emotional_tendencies:
            self.user_profile.emotional_tendencies[emotion] = 0
        self.user_profile.emotional_tendencies[emotion] += 1

        # Update topic preferences
        for topic, emotion in synthesis.topic_affect_mapping.items():
            # Simple engagement scoring based on emotion
            engagement_score = 1.0
            if emotion in ['excited', 'curious']:
                engagement_score = 1.5
            elif emotion in ['frustrated']:
                engagement_score = 0.8

            # Update or add topic
            existing = next((t for t in self.user_profile.favorite_topics if t[0] == topic), None)
            if existing:
                # Update score (running average)
                current_score = existing[1]
                new_score = (current_score + engagement_score) / 2
                existing = (topic, new_score)
            else:
                self.user_profile.favorite_topics.append((topic, engagement_score))

            # Sort by engagement score
            self.user_profile.favorite_topics.sort(key=lambda x: x[1], reverse=True)

        # === Phase 6.2: Update Planning Profile === #
        self._update_planning_profile_from_day(synthesis.day_id)

    def _update_user_profile_from_week(self, week_synthesis: Dict):
        """Update user profile from weekly synthesis."""
        # Update productivity patterns
        productivity = week_synthesis.get('productivity_patterns', {})
        for day, level in productivity.items():
            self.user_profile.productivity_patterns[day] = level

    def _update_user_profile_from_month(self, month_synthesis: Dict):
        """Update user profile from monthly synthesis."""
        # Update communication style based on monthly themes
        themes = month_synthesis.get('monthly_themes', [])
        if 'Technical Learning' in themes:
            self.user_profile.communication_style = 'analytical'
            self.user_profile.learning_style = 'hands_on'
        elif 'Personal Development' in themes:
            self.user_profile.communication_style = 'reflective'

    # === Phase 6.2: Planning Profile Updates === #

    def _update_planning_profile_from_day(self, day_id: str):
        """Update planning profile from daily plan activity."""
        # Planning system removed - this method is now a no-op
        return
        plans_by_id = {}
        for item in day_plan_items:
            plan_id = getattr(item, 'plan_id', None)
            if plan_id and plan_id not in plans_by_id:
                # Get the full plan details
                full_plan = self.storage.get_user_plan(plan_id)
                if full_plan:
                    plans_by_id[plan_id] = full_plan

        day_plans = list(plans_by_id.values())
        if not day_plans:
            return

        # Analyze planning frequency
        plan_count = len(day_plans)
        if plan_count >= 3:
            self.user_profile.planning_frequency = "daily"
        elif plan_count >= 1:
            self.user_profile.planning_frequency = "occasional"
        else:
            self.user_profile.planning_frequency = "rare"

        # Analyze plan types and completion
        plan_types = set()
        total_completion = 0.0
        completed_plans = 0

        for plan in day_plans:
            # Track plan types
            if hasattr(plan, 'category') and plan.category:
                plan_types.add(plan.category.lower())

            # Calculate completion rate
            if hasattr(plan, 'items') and plan.items:
                completed = sum(1 for item in plan.items if getattr(item, 'completed', False))
                completion_rate = completed / len(plan.items) if plan.items else 0.0
                total_completion += completion_rate
                if completion_rate >= 1.0:
                    completed_plans += 1

        # Update plan completion rate (running average)
        if day_plans:
            day_completion = total_completion / len(day_plans)
            if self.user_profile.plan_completion_rate == 0.0:
                self.user_profile.plan_completion_rate = day_completion
            else:
                self.user_profile.plan_completion_rate = (
                    self.user_profile.plan_completion_rate + day_completion
                ) / 2

        # Update preferred plan types
        self.user_profile.preferred_plan_types = list(plan_types)[:3]  # Keep top 3

        # Update plan adherence patterns for day of week
        day_name = datetime.strptime(day_id, "%Y-%m-%d").strftime("%A")
        adherence_rate = completed_plans / plan_count if plan_count > 0 else 0.0
        self.user_profile.plan_adherence_patterns[day_name] = adherence_rate

    def _store_weekly_synthesis(self, synthesis: Dict):
        """Store weekly synthesis (placeholder - would need schema extension)."""
        # For now, just log it
        logger.debug(f"Weekly synthesis stored: {synthesis['week_of']}")

    def _store_monthly_synthesis(self, synthesis: Dict):
        """Store monthly synthesis (placeholder - would need schema extension)."""
        # For now, just log it
        logger.debug(f"Monthly synthesis stored: {synthesis['month_of']}")

    def get_synthesis_stats(self) -> Dict:
        """Get statistics about synthesis operations."""
        return {
            'user_profile_topics': len(self.user_profile.favorite_topics),
            'day_emotions_tracked': len(self.user_profile.day_of_week_emotions),
            'emotional_tendencies': dict(self.user_profile.emotional_tendencies),
            'communication_style': self.user_profile.communication_style,
            'learning_style': self.user_profile.learning_style
        }