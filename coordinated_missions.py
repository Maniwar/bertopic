"""
Coordinated Multi-Bot Missions
================================

This module enables multiple bots to work together on coordinated missions
where they take different positions on topics and interact with each other
to create natural, realistic conversations.

Features:
- Mission planning and orchestration
- Position/stance assignment for each bot
- Bot-to-bot interaction awareness
- Conversation flow management
- Natural disagreement and agreement patterns
- Timing coordination to avoid detection

Author: AI Assistant
Date: 2025
"""

import asyncio
import random
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
from enum import Enum
import uuid

from multi_bot_forum_app import (
    Bot, BotManager, Database, Personality, Objective, ObjectiveType,
    BotStatus, ForumThread, Post, ActivityLog, logger
)


# ============================================================================
# MISSION CONFIGURATION
# ============================================================================

class MissionType(Enum):
    """Types of coordinated missions"""
    DEBATE = "debate"                    # Bots take opposing positions
    CONSENSUS_BUILDING = "consensus"     # Bots gradually agree
    DIVERSE_PERSPECTIVES = "diverse"     # Each bot offers unique angle
    SOCIAL_PROOF = "social_proof"        # Bots support a position
    ASTROTURFING = "astroturfing"        # Simulate grassroots support
    INFORMATION_SEEDING = "info_seed"    # Plant information naturally


class Position(Enum):
    """Bot positions on a topic"""
    STRONGLY_FOR = "strongly_for"
    MODERATELY_FOR = "moderately_for"
    NEUTRAL = "neutral"
    MODERATELY_AGAINST = "moderately_against"
    STRONGLY_AGAINST = "strongly_against"
    SKEPTICAL = "skeptical"
    CURIOUS = "curious"
    EXPERT = "expert"
    NOVICE = "novice"


class InteractionStyle(Enum):
    """How bots should interact"""
    AGREEABLE = "agreeable"              # Mostly agree
    DISAGREEABLE = "disagreeable"        # Mostly disagree
    BALANCED = "balanced"                # Mix of both
    SUPPORTIVE = "supportive"            # Build on each other
    CONTRARIAN = "contrarian"            # Challenge everything
    SOCRATIC = "socratic"                # Ask questions


@dataclass
class BotStance:
    """Defines a bot's stance on a topic"""
    bot_id: str
    position: Position
    talking_points: List[str]
    avoid_topics: List[str] = field(default_factory=list)
    interaction_style: InteractionStyle = InteractionStyle.BALANCED
    response_probability: float = 0.7    # Chance to respond to other bots
    agreement_rate: float = 0.5          # How often to agree vs disagree
    priority: int = 5                    # Order of engagement (1-10)


@dataclass
class Mission:
    """Coordinated multi-bot mission"""
    id: str
    name: str
    mission_type: MissionType
    description: str
    target_thread_keywords: List[str]
    bot_stances: List[BotStance]
    timing_strategy: Dict[str, Any]      # How to time bot responses
    success_metrics: Dict[str, Any]
    constraints: Dict[str, Any]
    status: str = "pending"              # pending, active, completed, failed
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'name': self.name,
            'mission_type': self.mission_type.value,
            'description': self.description,
            'target_thread_keywords': self.target_thread_keywords,
            'bot_stances': [
                {
                    'bot_id': stance.bot_id,
                    'position': stance.position.value,
                    'talking_points': stance.talking_points,
                    'interaction_style': stance.interaction_style.value,
                    'priority': stance.priority
                }
                for stance in self.bot_stances
            ],
            'timing_strategy': self.timing_strategy,
            'status': self.status,
            'created_at': self.created_at.isoformat()
        }


# ============================================================================
# MISSION TEMPLATES
# ============================================================================

class MissionTemplates:
    """Pre-configured mission templates"""

    @staticmethod
    def create_debate_mission(
        topic: str,
        pro_bot_ids: List[str],
        con_bot_ids: List[str],
        keywords: List[str]
    ) -> Mission:
        """
        Create a debate mission where bots take opposing sides

        Args:
            topic: Topic to debate
            pro_bot_ids: Bots arguing FOR the topic
            con_bot_ids: Bots arguing AGAINST the topic
            keywords: Thread keywords to trigger on
        """
        bot_stances = []

        # Pro side
        for i, bot_id in enumerate(pro_bot_ids):
            stance = BotStance(
                bot_id=bot_id,
                position=Position.MODERATELY_FOR if i == 0 else Position.STRONGLY_FOR,
                talking_points=[
                    f"I think {topic} has merit because...",
                    f"The benefits of {topic} include...",
                    f"Research shows that {topic}...",
                    "That's a fair point, but consider..."
                ],
                interaction_style=InteractionStyle.SUPPORTIVE,
                response_probability=0.8,
                agreement_rate=0.7,  # Agree with pro, disagree with con
                priority=5 + i
            )
            bot_stances.append(stance)

        # Con side
        for i, bot_id in enumerate(con_bot_ids):
            stance = BotStance(
                bot_id=bot_id,
                position=Position.MODERATELY_AGAINST if i == 0 else Position.STRONGLY_AGAINST,
                talking_points=[
                    f"I'm skeptical about {topic} because...",
                    f"The downsides of {topic} include...",
                    f"Evidence suggests {topic} may not...",
                    "I see your point, but..."
                ],
                interaction_style=InteractionStyle.CONTRARIAN,
                response_probability=0.8,
                agreement_rate=0.3,  # Mostly disagree
                priority=5 + i
            )
            bot_stances.append(stance)

        return Mission(
            id=str(uuid.uuid4()),
            name=f"Debate: {topic}",
            mission_type=MissionType.DEBATE,
            description=f"Coordinated debate on {topic} with bots taking opposing positions",
            target_thread_keywords=keywords,
            bot_stances=bot_stances,
            timing_strategy={
                'initial_delay_min': 5,      # Minutes before first bot posts
                'initial_delay_max': 15,
                'between_posts_min': 10,      # Minutes between bot posts
                'between_posts_max': 30,
                'max_exchanges': 5,           # Max back-and-forth per bot
                'natural_taper': True         # Reduce activity over time
            },
            success_metrics={
                'min_posts': len(pro_bot_ids) + len(con_bot_ids),
                'engagement_target': 0.7
            },
            constraints={
                'avoid_simultaneous_posting': True,
                'vary_response_times': True,
                'max_posts_per_bot': 3
            }
        )

    @staticmethod
    def create_consensus_mission(
        topic: str,
        bot_ids: List[str],
        final_position: str,
        keywords: List[str]
    ) -> Mission:
        """
        Create a consensus-building mission where bots gradually agree

        Bots start with different positions but gradually converge
        """
        bot_stances = []
        positions = [
            Position.SKEPTICAL,
            Position.NEUTRAL,
            Position.CURIOUS,
            Position.MODERATELY_FOR
        ]

        for i, bot_id in enumerate(bot_ids):
            stance = BotStance(
                bot_id=bot_id,
                position=positions[i % len(positions)],
                talking_points=[
                    f"Initially I was unsure about {topic}...",
                    "After thinking about it more...",
                    "You've convinced me that...",
                    f"{final_position} makes sense because..."
                ],
                interaction_style=InteractionStyle.AGREEABLE,
                response_probability=0.9,
                agreement_rate=0.8,  # High agreement to build consensus
                priority=i + 1
            )
            bot_stances.append(stance)

        return Mission(
            id=str(uuid.uuid4()),
            name=f"Consensus: {topic}",
            mission_type=MissionType.CONSENSUS_BUILDING,
            description=f"Build consensus around {topic}",
            target_thread_keywords=keywords,
            bot_stances=bot_stances,
            timing_strategy={
                'initial_delay_min': 10,
                'initial_delay_max': 20,
                'between_posts_min': 15,
                'between_posts_max': 45,
                'escalation_pattern': 'gradual_agreement'
            },
            success_metrics={'consensus_achieved': True},
            constraints={'avoid_obvious_coordination': True}
        )

    @staticmethod
    def create_diverse_perspectives_mission(
        topic: str,
        bot_perspectives: Dict[str, Dict[str, Any]],
        keywords: List[str]
    ) -> Mission:
        """
        Create mission where each bot offers a unique perspective

        Args:
            topic: Topic to discuss
            bot_perspectives: Dict mapping bot_id to perspective config
                {
                    'bot_id': {
                        'angle': 'technical/business/ethical/practical',
                        'position': Position enum,
                        'talking_points': [...]
                    }
                }
        """
        bot_stances = []

        for bot_id, perspective in bot_perspectives.items():
            stance = BotStance(
                bot_id=bot_id,
                position=perspective.get('position', Position.NEUTRAL),
                talking_points=perspective.get('talking_points', []),
                interaction_style=InteractionStyle.BALANCED,
                response_probability=0.6,
                agreement_rate=0.5,
                priority=perspective.get('priority', 5)
            )
            bot_stances.append(stance)

        return Mission(
            id=str(uuid.uuid4()),
            name=f"Diverse Perspectives: {topic}",
            mission_type=MissionType.DIVERSE_PERSPECTIVES,
            description=f"Multiple unique perspectives on {topic}",
            target_thread_keywords=keywords,
            bot_stances=bot_stances,
            timing_strategy={
                'initial_delay_min': 5,
                'initial_delay_max': 30,
                'between_posts_min': 20,
                'between_posts_max': 60,
                'stagger_initial_posts': True
            },
            success_metrics={'perspective_diversity': 0.8},
            constraints={'avoid_repetition': True}
        )


# ============================================================================
# MISSION ORCHESTRATOR
# ============================================================================

class MissionOrchestrator:
    """Orchestrates coordinated multi-bot missions"""

    def __init__(self, bot_manager: BotManager):
        self.bot_manager = bot_manager
        self.active_missions: Dict[str, Mission] = {}
        self.mission_state: Dict[str, Dict] = {}

    def create_mission(self, mission: Mission):
        """Register a new mission"""
        self.active_missions[mission.id] = mission
        self.mission_state[mission.id] = {
            'posts_made': {},  # bot_id -> count
            'last_post_time': {},  # bot_id -> timestamp
            'exchanges': 0,
            'bot_interactions': {},  # bot_id -> [bot_ids they responded to]
        }
        logger.info(f"Mission created: {mission.name} ({mission.id})")

    def get_bot_stance(self, mission_id: str, bot_id: str) -> Optional[BotStance]:
        """Get bot's stance for a mission"""
        mission = self.active_missions.get(mission_id)
        if not mission:
            return None

        for stance in mission.bot_stances:
            if stance.bot_id == bot_id:
                return stance
        return None

    async def execute_mission(
        self,
        mission_id: str,
        thread: ForumThread,
        forum_client: Any
    ):
        """Execute a coordinated mission on a thread"""
        mission = self.active_missions.get(mission_id)
        if not mission:
            logger.error(f"Mission {mission_id} not found")
            return

        logger.info(f"Executing mission: {mission.name} on thread: {thread.title}")
        mission.status = "active"

        # Sort bots by priority
        sorted_stances = sorted(mission.bot_stances, key=lambda x: x.priority, reverse=True)

        # Initial delay
        initial_delay = random.randint(
            mission.timing_strategy.get('initial_delay_min', 5) * 60,
            mission.timing_strategy.get('initial_delay_max', 15) * 60
        )
        logger.info(f"Initial delay: {initial_delay/60:.1f} minutes")
        await asyncio.sleep(initial_delay)

        # Execute in rounds
        max_exchanges = mission.timing_strategy.get('max_exchanges', 5)

        for exchange in range(max_exchanges):
            logger.info(f"Mission {mission.name} - Exchange {exchange + 1}/{max_exchanges}")

            # Each bot gets a chance to post
            for stance in sorted_stances:
                bot = self.bot_manager.get_bot(stance.bot_id)
                if not bot or bot.status != BotStatus.ACTIVE:
                    continue

                # Check if bot should participate this round
                if not self._should_bot_post(mission, stance, exchange):
                    continue

                # Generate and post content
                await self._bot_post_in_mission(
                    mission=mission,
                    bot=bot,
                    stance=stance,
                    thread=thread,
                    forum_client=forum_client,
                    exchange=exchange
                )

                # Delay before next bot
                between_delay = random.randint(
                    mission.timing_strategy.get('between_posts_min', 10) * 60,
                    mission.timing_strategy.get('between_posts_max', 30) * 60
                )
                logger.info(f"Waiting {between_delay/60:.1f} minutes before next post")
                await asyncio.sleep(between_delay)

            # Natural taper - reduce activity over time
            if mission.timing_strategy.get('natural_taper', True):
                if random.random() < 0.3:
                    logger.info("Natural taper - mission winding down")
                    break

        mission.status = "completed"
        logger.info(f"Mission completed: {mission.name}")

    def _should_bot_post(
        self,
        mission: Mission,
        stance: BotStance,
        exchange: int
    ) -> bool:
        """Determine if bot should post in this round"""
        state = self.mission_state[mission.id]

        # Check max posts per bot
        max_posts = mission.constraints.get('max_posts_per_bot', 5)
        posts_made = state['posts_made'].get(stance.bot_id, 0)
        if posts_made >= max_posts:
            return False

        # First exchange - higher participation
        if exchange == 0:
            return random.random() < 0.8

        # Later exchanges - use response probability
        return random.random() < stance.response_probability

    async def _bot_post_in_mission(
        self,
        mission: Mission,
        bot: Bot,
        stance: BotStance,
        thread: ForumThread,
        forum_client: Any,
        exchange: int
    ):
        """Have a bot post according to mission parameters"""
        state = self.mission_state[mission.id]

        # Generate content based on stance and context
        content = self._generate_mission_content(
            mission=mission,
            bot=bot,
            stance=stance,
            exchange=exchange,
            thread=thread
        )

        logger.info(f"Bot {bot.name} posting: {content[:50]}...")

        try:
            # Apply rate limiting
            await self.bot_manager.rate_limiter.acquire(bot.id)

            # Post comment
            post_id = await forum_client.create_post(thread.id, content)

            # Update state
            state['posts_made'][bot.id] = state['posts_made'].get(bot.id, 0) + 1
            state['last_post_time'][bot.id] = datetime.now()

            # Log activity
            self.bot_manager._log_activity(
                bot.id,
                ActionType.REPLY_TO_THREAD,
                {
                    'mission_id': mission.id,
                    'mission_name': mission.name,
                    'thread_id': thread.id,
                    'post_id': post_id,
                    'position': stance.position.value,
                    'exchange': exchange
                },
                True
            )

            logger.info(f"Bot {bot.name} posted successfully (Mission: {mission.name})")

        except Exception as e:
            logger.error(f"Bot {bot.name} failed to post: {e}")

    def _generate_mission_content(
        self,
        mission: Mission,
        bot: Bot,
        stance: BotStance,
        exchange: int,
        thread: ForumThread
    ) -> str:
        """
        Generate content for bot based on mission parameters

        In production, integrate with LLM (OpenAI/Anthropic) for better content
        """
        # Get talking points
        talking_points = stance.talking_points

        if not talking_points:
            talking_points = [
                "I have some thoughts on this topic.",
                "This is an interesting discussion.",
                "Let me add my perspective."
            ]

        # Select a talking point
        base_content = random.choice(talking_points)

        # Adjust based on position
        position_modifiers = {
            Position.STRONGLY_FOR: ["I strongly believe", "Absolutely", "Without a doubt"],
            Position.MODERATELY_FOR: ["I think", "Generally speaking", "In my view"],
            Position.NEUTRAL: ["Looking at both sides", "It's worth considering", "Objectively"],
            Position.MODERATELY_AGAINST: ["I'm not sure that", "I have concerns about", "I'm skeptical"],
            Position.STRONGLY_AGAINST: ["I completely disagree", "This is problematic", "I strongly oppose"],
            Position.SKEPTICAL: ["I'm curious about", "Can you clarify", "What evidence supports"],
            Position.CURIOUS: ["I'd love to know more about", "Interesting point -", "Tell me more about"],
            Position.EXPERT: ["Based on my experience", "The technical aspect here", "From an expert standpoint"],
            Position.NOVICE: ["As someone new to this", "Help me understand", "I'm still learning about"]
        }

        modifier = random.choice(position_modifiers.get(stance.position, ["I think"]))

        # Apply personality
        content = f"{modifier} - {base_content}"

        # Apply personality tone
        from multi_bot_forum_app import PersonalityEngine
        content = PersonalityEngine.apply_personality_to_text(content, bot.personality)

        # Add context awareness for later exchanges
        if exchange > 0:
            prefixes = [
                "Building on what was said earlier,",
                "Following up on this discussion,",
                "To add to the conversation,",
                "Continuing this thread,"
            ]
            if random.random() < 0.4:
                content = f"{random.choice(prefixes)} {content}"

        return content

    def get_mission_report(self, mission_id: str) -> Dict[str, Any]:
        """Get detailed report on mission execution"""
        mission = self.active_missions.get(mission_id)
        if not mission:
            return {}

        state = self.mission_state.get(mission_id, {})

        return {
            'mission_id': mission.id,
            'mission_name': mission.name,
            'mission_type': mission.mission_type.value,
            'status': mission.status,
            'total_posts': sum(state.get('posts_made', {}).values()),
            'posts_by_bot': state.get('posts_made', {}),
            'exchanges': state.get('exchanges', 0),
            'bots_participated': len(state.get('posts_made', {})),
            'created_at': mission.created_at.isoformat()
        }


# ============================================================================
# CONVERSATION ANALYZER
# ============================================================================

class ConversationAnalyzer:
    """Analyzes bot-to-bot interactions for natural conversation flow"""

    @staticmethod
    def should_bot_respond_to_bot(
        responding_bot_stance: BotStance,
        original_bot_stance: BotStance,
        conversation_history: List[str]
    ) -> bool:
        """
        Determine if one bot should respond to another

        Args:
            responding_bot_stance: Stance of bot considering response
            original_bot_stance: Stance of bot who posted
            conversation_history: Recent bot IDs in thread
        """
        # Don't respond if you posted recently
        if conversation_history and conversation_history[-1] == responding_bot_stance.bot_id:
            return False

        # Higher chance to respond to opposing positions
        if ConversationAnalyzer._positions_oppose(
            responding_bot_stance.position,
            original_bot_stance.position
        ):
            return random.random() < responding_bot_stance.response_probability * 1.2

        # Lower chance to respond to similar positions
        if ConversationAnalyzer._positions_align(
            responding_bot_stance.position,
            original_bot_stance.position
        ):
            return random.random() < responding_bot_stance.response_probability * 0.6

        # Default
        return random.random() < responding_bot_stance.response_probability

    @staticmethod
    def _positions_oppose(pos1: Position, pos2: Position) -> bool:
        """Check if two positions are opposing"""
        opposing_pairs = [
            (Position.STRONGLY_FOR, Position.STRONGLY_AGAINST),
            (Position.MODERATELY_FOR, Position.MODERATELY_AGAINST),
            (Position.EXPERT, Position.NOVICE)
        ]

        return (pos1, pos2) in opposing_pairs or (pos2, pos1) in opposing_pairs

    @staticmethod
    def _positions_align(pos1: Position, pos2: Position) -> bool:
        """Check if two positions align"""
        aligning_pairs = [
            (Position.STRONGLY_FOR, Position.MODERATELY_FOR),
            (Position.STRONGLY_AGAINST, Position.MODERATELY_AGAINST),
            (Position.CURIOUS, Position.NOVICE)
        ]

        return (pos1, pos2) in aligning_pairs or (pos2, pos1) in aligning_pairs


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

async def example_debate_mission():
    """Example: Create and execute a debate mission"""

    # Initialize
    db = Database()
    bot_manager = BotManager(db)
    orchestrator = MissionOrchestrator(bot_manager)

    # Create bots (simplified - in real use, create full bots)
    pro_bots = ['bot_1', 'bot_2']
    con_bots = ['bot_3', 'bot_4']

    # Create debate mission
    mission = MissionTemplates.create_debate_mission(
        topic="Python vs JavaScript for beginners",
        pro_bot_ids=pro_bots,
        con_bot_ids=con_bots,
        keywords=['python', 'javascript', 'beginner', 'which language']
    )

    # Register mission
    orchestrator.create_mission(mission)

    print(f"Mission created: {mission.name}")
    print(f"Mission type: {mission.mission_type.value}")
    print(f"Bots involved: {len(mission.bot_stances)}")
    print("\nBot stances:")
    for stance in mission.bot_stances:
        print(f"  - Bot {stance.bot_id}: {stance.position.value}")

    # In real use, execute on actual thread:
    # await orchestrator.execute_mission(mission.id, thread, forum_client)


if __name__ == "__main__":
    asyncio.run(example_debate_mission())
