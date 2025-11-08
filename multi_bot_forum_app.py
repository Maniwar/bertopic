"""
Multi-Bot Forum Interaction Application
==========================================

A scalable, production-ready application for managing multiple bot personalities
that can post on forums and interact with threads based on configurable objectives.

Features:
- Multiple bot personality management
- Objective-driven behavior
- Forum API integration with rate limiting
- Comprehensive activity logging
- Error handling and retry logic
- Database persistence
- Production-ready monitoring

Author: AI Assistant
Date: 2025
"""

import asyncio
import json
import logging
import sqlite3
import time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from contextlib import contextmanager
import hashlib
import random
import re


# ============================================================================
# CONFIGURATION MANAGEMENT
# ============================================================================

class Config:
    """Centralized configuration management"""

    # Database
    DB_PATH = "multi_bot_app.db"

    # Rate Limiting
    GLOBAL_RATE_LIMIT = 100  # requests per minute
    PER_BOT_RATE_LIMIT = 20  # requests per minute

    # Retry Logic
    MAX_RETRIES = 3
    RETRY_DELAY = 2  # seconds
    RETRY_BACKOFF = 2  # exponential backoff multiplier

    # Circuit Breaker
    CIRCUIT_BREAKER_THRESHOLD = 5  # failures before opening
    CIRCUIT_BREAKER_TIMEOUT = 60  # seconds before retry

    # Logging
    LOG_LEVEL = logging.INFO
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    LOG_FILE = 'multi_bot_app.log'

    # Bot Behavior
    MIN_POST_INTERVAL = 60  # seconds between posts
    MAX_POST_LENGTH = 5000
    CONTEXT_WINDOW = 10  # previous messages to consider


# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging():
    """Configure application-wide logging"""
    logging.basicConfig(
        level=Config.LOG_LEVEL,
        format=Config.LOG_FORMAT,
        handlers=[
            logging.FileHandler(Config.LOG_FILE),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()


# ============================================================================
# DATA MODELS
# ============================================================================

class ActionType(Enum):
    """Types of bot actions"""
    CREATE_POST = "create_post"
    REPLY_TO_THREAD = "reply_to_thread"
    LIKE_POST = "like_post"
    MONITOR_THREAD = "monitor_thread"
    ERROR = "error"
    STATUS_CHANGE = "status_change"


class BotStatus(Enum):
    """Bot operational status"""
    ACTIVE = "active"
    PAUSED = "paused"
    ERROR = "error"
    DISABLED = "disabled"


class ObjectiveType(Enum):
    """Types of bot objectives"""
    ENGAGEMENT = "engagement"  # Maximize engagement
    PROMOTION = "promotion"    # Promote specific topics/products
    INFORMATION = "information"  # Share information/education
    MONITORING = "monitoring"  # Monitor and respond to keywords
    SUPPORT = "support"        # Provide support and help


@dataclass
class PersonalityTrait:
    """Defines a personality trait"""
    name: str
    value: float  # 0.0 to 1.0


@dataclass
class Personality:
    """Bot personality configuration"""
    id: str
    name: str
    description: str
    tone: str  # formal, casual, friendly, professional, etc.
    formality: float  # 0.0 (very casual) to 1.0 (very formal)
    enthusiasm: float  # 0.0 (reserved) to 1.0 (enthusiastic)
    technical_level: float  # 0.0 (simple) to 1.0 (technical)
    verbosity: float  # 0.0 (concise) to 1.0 (verbose)
    empathy: float  # 0.0 (factual) to 1.0 (empathetic)
    vocabulary_style: str  # simple, professional, technical, casual
    response_patterns: List[str]  # Common phrases/patterns

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'Personality':
        return cls(**data)


@dataclass
class Objective:
    """Bot objective configuration"""
    id: str
    type: ObjectiveType
    description: str
    target_keywords: List[str]
    success_metrics: Dict[str, Any]
    constraints: Dict[str, Any]
    priority: int  # 1-10

    def to_dict(self) -> Dict:
        data = asdict(self)
        data['type'] = self.type.value
        return data

    @classmethod
    def from_dict(cls, data: Dict) -> 'Objective':
        data['type'] = ObjectiveType(data['type'])
        return cls(**data)


@dataclass
class Bot:
    """Bot instance"""
    id: str
    name: str
    personality: Personality
    objectives: List[Objective]
    status: BotStatus
    credentials: Dict[str, str]
    metadata: Dict[str, Any]
    created_at: datetime
    last_active: datetime

    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'name': self.name,
            'personality': self.personality.to_dict(),
            'objectives': [obj.to_dict() for obj in self.objectives],
            'status': self.status.value,
            'credentials': self.credentials,
            'metadata': self.metadata,
            'created_at': self.created_at.isoformat(),
            'last_active': self.last_active.isoformat()
        }


@dataclass
class ForumThread:
    """Forum thread information"""
    id: str
    forum_id: str
    url: str
    title: str
    content: str
    author: str
    created_at: datetime
    last_checked: datetime
    metadata: Dict[str, Any]


@dataclass
class Post:
    """Bot post"""
    id: str
    bot_id: str
    thread_id: str
    content: str
    created_at: datetime
    status: str  # pending, posted, failed
    metadata: Dict[str, Any]


@dataclass
class ActivityLog:
    """Activity log entry"""
    id: str
    bot_id: str
    action_type: ActionType
    details: Dict[str, Any]
    timestamp: datetime
    success: bool
    error_message: Optional[str] = None


# ============================================================================
# RATE LIMITING
# ============================================================================

class TokenBucket:
    """Token bucket rate limiter"""

    def __init__(self, capacity: int, refill_rate: float):
        """
        Args:
            capacity: Maximum tokens in bucket
            refill_rate: Tokens added per second
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = capacity
        self.last_refill = time.time()
        self._lock = asyncio.Lock()

    async def consume(self, tokens: int = 1) -> bool:
        """
        Attempt to consume tokens

        Returns:
            True if tokens were consumed, False if insufficient tokens
        """
        async with self._lock:
            self._refill()

            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False

    def _refill(self):
        """Refill tokens based on elapsed time"""
        now = time.time()
        elapsed = now - self.last_refill
        tokens_to_add = elapsed * self.refill_rate

        self.tokens = min(self.capacity, self.tokens + tokens_to_add)
        self.last_refill = now

    async def wait_for_token(self):
        """Wait until a token is available"""
        while not await self.consume():
            await asyncio.sleep(0.1)


class RateLimiter:
    """Multi-level rate limiting"""

    def __init__(self):
        self.global_limiter = TokenBucket(
            capacity=Config.GLOBAL_RATE_LIMIT,
            refill_rate=Config.GLOBAL_RATE_LIMIT / 60
        )
        self.bot_limiters: Dict[str, TokenBucket] = {}

    def get_bot_limiter(self, bot_id: str) -> TokenBucket:
        """Get or create rate limiter for specific bot"""
        if bot_id not in self.bot_limiters:
            self.bot_limiters[bot_id] = TokenBucket(
                capacity=Config.PER_BOT_RATE_LIMIT,
                refill_rate=Config.PER_BOT_RATE_LIMIT / 60
            )
        return self.bot_limiters[bot_id]

    async def acquire(self, bot_id: str):
        """Acquire rate limit token for bot"""
        # Wait for both global and bot-specific tokens
        await self.global_limiter.wait_for_token()
        bot_limiter = self.get_bot_limiter(bot_id)
        await bot_limiter.wait_for_token()


# ============================================================================
# CIRCUIT BREAKER
# ============================================================================

class CircuitBreakerState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Blocking requests
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreaker:
    """Circuit breaker pattern for fault tolerance"""

    def __init__(self, threshold: int = Config.CIRCUIT_BREAKER_THRESHOLD,
                 timeout: int = Config.CIRCUIT_BREAKER_TIMEOUT):
        self.threshold = threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.state = CircuitBreakerState.CLOSED
        self._lock = asyncio.Lock()

    async def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        async with self._lock:
            if self.state == CircuitBreakerState.OPEN:
                if time.time() - self.last_failure_time >= self.timeout:
                    self.state = CircuitBreakerState.HALF_OPEN
                    logger.info("Circuit breaker entering HALF_OPEN state")
                else:
                    raise Exception("Circuit breaker is OPEN")

        try:
            result = await func(*args, **kwargs)
            await self._on_success()
            return result
        except Exception as e:
            await self._on_failure()
            raise e

    async def _on_success(self):
        """Handle successful call"""
        async with self._lock:
            self.failure_count = 0
            if self.state == CircuitBreakerState.HALF_OPEN:
                self.state = CircuitBreakerState.CLOSED
                logger.info("Circuit breaker CLOSED after successful call")

    async def _on_failure(self):
        """Handle failed call"""
        async with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()

            if self.failure_count >= self.threshold:
                self.state = CircuitBreakerState.OPEN
                logger.warning(f"Circuit breaker OPEN after {self.failure_count} failures")


# ============================================================================
# DATABASE LAYER
# ============================================================================

class Database:
    """Database management with connection pooling"""

    def __init__(self, db_path: str = Config.DB_PATH):
        self.db_path = db_path
        self.init_database()

    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            conn.close()

    def init_database(self):
        """Initialize database schema"""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Bots table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS bots (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    personality TEXT NOT NULL,
                    objectives TEXT NOT NULL,
                    status TEXT NOT NULL,
                    credentials TEXT NOT NULL,
                    metadata TEXT,
                    created_at TEXT NOT NULL,
                    last_active TEXT NOT NULL
                )
            """)

            # Threads table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS threads (
                    id TEXT PRIMARY KEY,
                    forum_id TEXT NOT NULL,
                    url TEXT NOT NULL,
                    title TEXT NOT NULL,
                    content TEXT,
                    author TEXT,
                    created_at TEXT NOT NULL,
                    last_checked TEXT NOT NULL,
                    metadata TEXT
                )
            """)

            # Posts table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS posts (
                    id TEXT PRIMARY KEY,
                    bot_id TEXT NOT NULL,
                    thread_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    status TEXT NOT NULL,
                    metadata TEXT,
                    FOREIGN KEY (bot_id) REFERENCES bots(id),
                    FOREIGN KEY (thread_id) REFERENCES threads(id)
                )
            """)

            # Activity logs table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS activity_logs (
                    id TEXT PRIMARY KEY,
                    bot_id TEXT NOT NULL,
                    action_type TEXT NOT NULL,
                    details TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    success INTEGER NOT NULL,
                    error_message TEXT,
                    FOREIGN KEY (bot_id) REFERENCES bots(id)
                )
            """)

            # Create indices
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_posts_bot ON posts(bot_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_posts_thread ON posts(thread_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_logs_bot ON activity_logs(bot_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_logs_timestamp ON activity_logs(timestamp)")

            logger.info("Database initialized successfully")

    def save_bot(self, bot: Bot):
        """Save bot to database"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO bots
                (id, name, personality, objectives, status, credentials, metadata, created_at, last_active)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                bot.id,
                bot.name,
                json.dumps(bot.personality.to_dict()),
                json.dumps([obj.to_dict() for obj in bot.objectives]),
                bot.status.value,
                json.dumps(bot.credentials),
                json.dumps(bot.metadata),
                bot.created_at.isoformat(),
                bot.last_active.isoformat()
            ))

    def load_bot(self, bot_id: str) -> Optional[Bot]:
        """Load bot from database"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM bots WHERE id = ?", (bot_id,))
            row = cursor.fetchone()

            if not row:
                return None

            return Bot(
                id=row['id'],
                name=row['name'],
                personality=Personality.from_dict(json.loads(row['personality'])),
                objectives=[Objective.from_dict(obj) for obj in json.loads(row['objectives'])],
                status=BotStatus(row['status']),
                credentials=json.loads(row['credentials']),
                metadata=json.loads(row['metadata']),
                created_at=datetime.fromisoformat(row['created_at']),
                last_active=datetime.fromisoformat(row['last_active'])
            )

    def get_all_bots(self) -> List[Bot]:
        """Get all bots"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM bots")
            bot_ids = [row['id'] for row in cursor.fetchall()]
            return [self.load_bot(bot_id) for bot_id in bot_ids]

    def log_activity(self, log: ActivityLog):
        """Log bot activity"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO activity_logs
                (id, bot_id, action_type, details, timestamp, success, error_message)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                log.id,
                log.bot_id,
                log.action_type.value,
                json.dumps(log.details),
                log.timestamp.isoformat(),
                1 if log.success else 0,
                log.error_message
            ))

    def get_bot_activities(self, bot_id: str, limit: int = 100) -> List[ActivityLog]:
        """Get recent activities for a bot"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM activity_logs
                WHERE bot_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (bot_id, limit))

            logs = []
            for row in cursor.fetchall():
                logs.append(ActivityLog(
                    id=row['id'],
                    bot_id=row['bot_id'],
                    action_type=ActionType(row['action_type']),
                    details=json.loads(row['details']),
                    timestamp=datetime.fromisoformat(row['timestamp']),
                    success=bool(row['success']),
                    error_message=row['error_message']
                ))
            return logs

    def save_thread(self, thread: ForumThread):
        """Save forum thread"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO threads
                (id, forum_id, url, title, content, author, created_at, last_checked, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                thread.id,
                thread.forum_id,
                thread.url,
                thread.title,
                thread.content,
                thread.author,
                thread.created_at.isoformat(),
                thread.last_checked.isoformat(),
                json.dumps(thread.metadata)
            ))

    def save_post(self, post: Post):
        """Save bot post"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO posts
                (id, bot_id, thread_id, content, created_at, status, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                post.id,
                post.bot_id,
                post.thread_id,
                post.content,
                post.created_at.isoformat(),
                post.status,
                json.dumps(post.metadata)
            ))


# ============================================================================
# PERSONALITY ENGINE
# ============================================================================

class PersonalityEngine:
    """Manages bot personalities and generates personality-driven content"""

    @staticmethod
    def create_personality(name: str, config: Dict[str, Any]) -> Personality:
        """Create a personality from configuration"""
        return Personality(
            id=str(uuid.uuid4()),
            name=name,
            description=config.get('description', ''),
            tone=config.get('tone', 'neutral'),
            formality=config.get('formality', 0.5),
            enthusiasm=config.get('enthusiasm', 0.5),
            technical_level=config.get('technical_level', 0.5),
            verbosity=config.get('verbosity', 0.5),
            empathy=config.get('empathy', 0.5),
            vocabulary_style=config.get('vocabulary_style', 'professional'),
            response_patterns=config.get('response_patterns', [])
        )

    @staticmethod
    def apply_personality_to_text(text: str, personality: Personality) -> str:
        """
        Apply personality traits to generated text

        This is a simplified version. In production, you'd integrate with
        an LLM API (OpenAI, Anthropic, etc.) to generate personality-driven content.
        """
        modified_text = text

        # Adjust formality
        if personality.formality < 0.3:
            # Make more casual
            modified_text = modified_text.replace("Hello", "Hey")
            modified_text = modified_text.replace("Thank you", "Thanks")
        elif personality.formality > 0.7:
            # Make more formal
            modified_text = modified_text.replace("Hey", "Hello")
            modified_text = modified_text.replace("Thanks", "Thank you")

        # Adjust enthusiasm
        if personality.enthusiasm > 0.7:
            # Add enthusiasm markers (in moderation)
            if not modified_text.endswith('!'):
                modified_text = modified_text.rstrip('.') + '!'

        # Add personality-specific patterns
        if personality.response_patterns and random.random() < 0.3:
            pattern = random.choice(personality.response_patterns)
            modified_text = f"{pattern} {modified_text}"

        return modified_text

    @staticmethod
    def get_preset_personalities() -> Dict[str, Personality]:
        """Get preset personality templates"""
        return {
            'professional': PersonalityEngine.create_personality(
                'Professional',
                {
                    'description': 'Professional and knowledgeable expert',
                    'tone': 'professional',
                    'formality': 0.8,
                    'enthusiasm': 0.4,
                    'technical_level': 0.7,
                    'verbosity': 0.6,
                    'empathy': 0.5,
                    'vocabulary_style': 'professional',
                    'response_patterns': [
                        'In my experience,',
                        'From a professional standpoint,',
                        'I would recommend'
                    ]
                }
            ),
            'friendly': PersonalityEngine.create_personality(
                'Friendly',
                {
                    'description': 'Friendly and approachable helper',
                    'tone': 'friendly',
                    'formality': 0.3,
                    'enthusiasm': 0.8,
                    'technical_level': 0.4,
                    'verbosity': 0.5,
                    'empathy': 0.9,
                    'vocabulary_style': 'casual',
                    'response_patterns': [
                        'Hey there!',
                        'Great question!',
                        'I totally get what you mean!'
                    ]
                }
            ),
            'technical': PersonalityEngine.create_personality(
                'Technical',
                {
                    'description': 'Highly technical expert',
                    'tone': 'technical',
                    'formality': 0.7,
                    'enthusiasm': 0.5,
                    'technical_level': 0.9,
                    'verbosity': 0.7,
                    'empathy': 0.4,
                    'vocabulary_style': 'technical',
                    'response_patterns': [
                        'Technically speaking,',
                        'From an implementation perspective,',
                        'The optimal approach would be'
                    ]
                }
            ),
            'casual': PersonalityEngine.create_personality(
                'Casual',
                {
                    'description': 'Casual and laid-back contributor',
                    'tone': 'casual',
                    'formality': 0.2,
                    'enthusiasm': 0.6,
                    'technical_level': 0.3,
                    'verbosity': 0.4,
                    'empathy': 0.7,
                    'vocabulary_style': 'simple',
                    'response_patterns': [
                        'Yeah,',
                        'Cool!',
                        'Nice!'
                    ]
                }
            )
        }


# ============================================================================
# OBJECTIVE ENGINE
# ============================================================================

class ObjectiveEngine:
    """Manages bot objectives and decision-making"""

    @staticmethod
    def create_objective(obj_type: ObjectiveType, config: Dict[str, Any]) -> Objective:
        """Create an objective from configuration"""
        return Objective(
            id=str(uuid.uuid4()),
            type=obj_type,
            description=config.get('description', ''),
            target_keywords=config.get('target_keywords', []),
            success_metrics=config.get('success_metrics', {}),
            constraints=config.get('constraints', {}),
            priority=config.get('priority', 5)
        )

    @staticmethod
    def should_engage_with_thread(thread: ForumThread, objectives: List[Objective]) -> tuple[bool, Optional[Objective]]:
        """
        Determine if bot should engage with a thread based on objectives

        Returns:
            (should_engage, matched_objective)
        """
        # Sort objectives by priority
        sorted_objectives = sorted(objectives, key=lambda x: x.priority, reverse=True)

        for objective in sorted_objectives:
            # Check if thread matches objective keywords
            thread_text = f"{thread.title} {thread.content}".lower()

            matches = sum(1 for keyword in objective.target_keywords
                         if keyword.lower() in thread_text)

            # If enough keywords match, engage
            threshold = objective.constraints.get('keyword_match_threshold', 1)
            if matches >= threshold:
                return True, objective

        return False, None

    @staticmethod
    def generate_response_strategy(objective: Objective, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate response strategy based on objective type

        Returns:
            Strategy configuration for response generation
        """
        strategy = {
            'objective_type': objective.type.value,
            'tone': 'helpful',
            'include_cta': False,
            'max_length': Config.MAX_POST_LENGTH
        }

        if objective.type == ObjectiveType.ENGAGEMENT:
            strategy.update({
                'tone': 'friendly',
                'ask_questions': True,
                'encourage_discussion': True
            })

        elif objective.type == ObjectiveType.PROMOTION:
            strategy.update({
                'include_cta': True,
                'highlight_benefits': True,
                'subtle': objective.constraints.get('subtle', True)
            })

        elif objective.type == ObjectiveType.INFORMATION:
            strategy.update({
                'tone': 'educational',
                'provide_sources': True,
                'detailed': True
            })

        elif objective.type == ObjectiveType.MONITORING:
            strategy.update({
                'tone': 'observant',
                'gather_data': True,
                'minimal_engagement': True
            })

        elif objective.type == ObjectiveType.SUPPORT:
            strategy.update({
                'tone': 'helpful',
                'empathetic': True,
                'solution_focused': True
            })

        return strategy


# ============================================================================
# FORUM API CLIENT
# ============================================================================

class ForumAPIClient(ABC):
    """Abstract base class for forum API clients"""

    def __init__(self, credentials: Dict[str, str], rate_limiter: RateLimiter):
        self.credentials = credentials
        self.rate_limiter = rate_limiter
        self.circuit_breaker = CircuitBreaker()

    @abstractmethod
    async def authenticate(self) -> bool:
        """Authenticate with the forum"""
        pass

    @abstractmethod
    async def get_threads(self, forum_id: str, limit: int = 10) -> List[ForumThread]:
        """Fetch recent threads from forum"""
        pass

    @abstractmethod
    async def get_thread_details(self, thread_id: str) -> ForumThread:
        """Get detailed information about a thread"""
        pass

    @abstractmethod
    async def create_post(self, thread_id: str, content: str) -> str:
        """Create a post in a thread, returns post ID"""
        pass

    @abstractmethod
    async def like_post(self, post_id: str) -> bool:
        """Like/upvote a post"""
        pass


class MockForumAPIClient(ForumAPIClient):
    """Mock implementation for testing and demonstration"""

    def __init__(self, credentials: Dict[str, str], rate_limiter: RateLimiter):
        super().__init__(credentials, rate_limiter)
        self.authenticated = False
        self.mock_threads: Dict[str, ForumThread] = {}
        self.mock_posts: Dict[str, List[str]] = defaultdict(list)

    async def authenticate(self) -> bool:
        """Mock authentication"""
        await asyncio.sleep(0.1)  # Simulate API call
        self.authenticated = True
        logger.info("Mock forum authentication successful")
        return True

    async def get_threads(self, forum_id: str, limit: int = 10) -> List[ForumThread]:
        """Get mock threads"""
        await asyncio.sleep(0.2)

        # Generate some mock threads if none exist
        if not self.mock_threads:
            topics = [
                ("How to get started with Python?", "I'm new to programming and want to learn Python. Any tips?"),
                ("Best practices for async programming", "What are the best practices when working with asyncio?"),
                ("Database design question", "Should I use SQL or NoSQL for my project?"),
                ("Performance optimization help needed", "My application is running slow, need help optimizing"),
                ("Looking for project ideas", "What are some good beginner-friendly projects?")
            ]

            for i, (title, content) in enumerate(topics[:limit]):
                thread_id = f"thread_{i}"
                self.mock_threads[thread_id] = ForumThread(
                    id=thread_id,
                    forum_id=forum_id,
                    url=f"https://example-forum.com/thread/{thread_id}",
                    title=title,
                    content=content,
                    author=f"user_{i}",
                    created_at=datetime.now() - timedelta(hours=random.randint(1, 24)),
                    last_checked=datetime.now(),
                    metadata={}
                )

        return list(self.mock_threads.values())[:limit]

    async def get_thread_details(self, thread_id: str) -> ForumThread:
        """Get mock thread details"""
        await asyncio.sleep(0.1)

        if thread_id in self.mock_threads:
            return self.mock_threads[thread_id]

        raise ValueError(f"Thread {thread_id} not found")

    async def create_post(self, thread_id: str, content: str) -> str:
        """Create mock post"""
        await asyncio.sleep(0.3)

        post_id = str(uuid.uuid4())
        self.mock_posts[thread_id].append(post_id)
        logger.info(f"Mock post created: {post_id} in thread {thread_id}")
        return post_id

    async def like_post(self, post_id: str) -> bool:
        """Mock like post"""
        await asyncio.sleep(0.1)
        logger.info(f"Mock post liked: {post_id}")
        return True


# ============================================================================
# CONTENT GENERATOR
# ============================================================================

class ContentGenerator:
    """Generates contextual content for posts"""

    def __init__(self, personality_engine: PersonalityEngine):
        self.personality_engine = personality_engine

    def generate_post(
        self,
        thread: ForumThread,
        personality: Personality,
        strategy: Dict[str, Any],
        context: Optional[List[str]] = None
    ) -> str:
        """
        Generate a post based on thread, personality, and strategy

        In production, this would integrate with an LLM API.
        This is a simplified template-based version.
        """
        # Template-based generation (simplified)
        templates = self._get_templates_for_strategy(strategy)
        template = random.choice(templates)

        # Generate base content
        content = template.format(
            topic=thread.title,
            author=thread.author
        )

        # Apply personality
        content = self.personality_engine.apply_personality_to_text(content, personality)

        # Apply length constraints
        max_length = strategy.get('max_length', Config.MAX_POST_LENGTH)
        if len(content) > max_length:
            content = content[:max_length-3] + "..."

        return content

    def _get_templates_for_strategy(self, strategy: Dict[str, Any]) -> List[str]:
        """Get appropriate templates based on strategy"""
        obj_type = strategy.get('objective_type', 'engagement')

        templates = {
            'engagement': [
                "Great question! I think this is really interesting because it touches on important aspects of {topic}.",
                "I've been thinking about this too! Have you considered looking into some alternative approaches?",
                "This is a fascinating topic. What specific aspect are you most curious about?"
            ],
            'promotion': [
                "This reminds me of a solution I've found helpful. Have you looked into [relevant product/service]?",
                "Great question! One approach that works well is using [solution]. It helped me solve similar challenges.",
            ],
            'information': [
                "Based on current best practices, here's what I'd recommend for {topic}.",
                "There are several approaches to this. Let me break down the most effective ones.",
            ],
            'support': [
                "I understand this can be frustrating. Let's work through this step by step.",
                "Happy to help! First, let's clarify what exactly you're trying to achieve.",
            ],
            'monitoring': [
                "Interesting perspective on {topic}.",
                "Thanks for sharing your thoughts on this.",
            ]
        }

        return templates.get(obj_type, templates['engagement'])


# ============================================================================
# BOT MANAGER
# ============================================================================

class BotManager:
    """Manages multiple bot instances and their operations"""

    def __init__(self, database: Database):
        self.database = database
        self.rate_limiter = RateLimiter()
        self.personality_engine = PersonalityEngine()
        self.objective_engine = ObjectiveEngine()
        self.content_generator = ContentGenerator(self.personality_engine)
        self.active_bots: Dict[str, Bot] = {}
        self.forum_clients: Dict[str, ForumAPIClient] = {}

    def create_bot(
        self,
        name: str,
        personality_config: Dict[str, Any],
        objectives_config: List[Dict[str, Any]],
        credentials: Dict[str, str]
    ) -> Bot:
        """Create a new bot"""
        personality = self.personality_engine.create_personality(
            name=f"{name}_personality",
            config=personality_config
        )

        objectives = [
            self.objective_engine.create_objective(
                obj_type=ObjectiveType(obj_config['type']),
                config=obj_config
            )
            for obj_config in objectives_config
        ]

        bot = Bot(
            id=str(uuid.uuid4()),
            name=name,
            personality=personality,
            objectives=objectives,
            status=BotStatus.ACTIVE,
            credentials=credentials,
            metadata={},
            created_at=datetime.now(),
            last_active=datetime.now()
        )

        self.database.save_bot(bot)
        self.active_bots[bot.id] = bot
        logger.info(f"Bot created: {bot.name} ({bot.id})")

        return bot

    def get_bot(self, bot_id: str) -> Optional[Bot]:
        """Get bot by ID"""
        if bot_id in self.active_bots:
            return self.active_bots[bot_id]

        bot = self.database.load_bot(bot_id)
        if bot:
            self.active_bots[bot_id] = bot
        return bot

    def list_bots(self) -> List[Bot]:
        """List all bots"""
        return self.database.get_all_bots()

    def update_bot_status(self, bot_id: str, status: BotStatus):
        """Update bot status"""
        bot = self.get_bot(bot_id)
        if bot:
            bot.status = status
            bot.last_active = datetime.now()
            self.database.save_bot(bot)
            self._log_activity(bot_id, ActionType.STATUS_CHANGE, {'new_status': status.value}, True)

    def get_forum_client(self, bot: Bot) -> ForumAPIClient:
        """Get or create forum client for bot"""
        if bot.id not in self.forum_clients:
            # In production, you'd select the appropriate client based on credentials
            self.forum_clients[bot.id] = MockForumAPIClient(
                credentials=bot.credentials,
                rate_limiter=self.rate_limiter
            )
        return self.forum_clients[bot.id]

    async def process_bot_objectives(self, bot_id: str, forum_id: str):
        """Process a bot's objectives for a given forum"""
        bot = self.get_bot(bot_id)
        if not bot or bot.status != BotStatus.ACTIVE:
            logger.warning(f"Bot {bot_id} is not active")
            return

        try:
            # Get forum client and authenticate
            client = self.get_forum_client(bot)
            await client.authenticate()

            # Apply rate limiting
            await self.rate_limiter.acquire(bot_id)

            # Get recent threads
            threads = await client.get_threads(forum_id, limit=20)
            logger.info(f"Bot {bot.name} found {len(threads)} threads")

            # Evaluate each thread against objectives
            for thread in threads:
                should_engage, objective = self.objective_engine.should_engage_with_thread(
                    thread, bot.objectives
                )

                if should_engage and objective:
                    logger.info(f"Bot {bot.name} engaging with thread {thread.id} for objective {objective.type.value}")
                    await self._engage_with_thread(bot, thread, objective, client)

                    # Add delay between posts
                    await asyncio.sleep(Config.MIN_POST_INTERVAL)

            # Update bot last active
            bot.last_active = datetime.now()
            self.database.save_bot(bot)

        except Exception as e:
            logger.error(f"Error processing objectives for bot {bot_id}: {e}")
            self._log_activity(bot_id, ActionType.ERROR, {'error': str(e)}, False, str(e))
            self.update_bot_status(bot_id, BotStatus.ERROR)

    async def _engage_with_thread(
        self,
        bot: Bot,
        thread: ForumThread,
        objective: Objective,
        client: ForumAPIClient
    ):
        """Engage with a specific thread"""
        try:
            # Generate response strategy
            strategy = self.objective_engine.generate_response_strategy(
                objective,
                {'thread': thread}
            )

            # Generate post content
            content = self.content_generator.generate_post(
                thread=thread,
                personality=bot.personality,
                strategy=strategy
            )

            # Apply rate limiting
            await self.rate_limiter.acquire(bot.id)

            # Create post
            post_id = await client.create_post(thread.id, content)

            # Save post to database
            post = Post(
                id=post_id,
                bot_id=bot.id,
                thread_id=thread.id,
                content=content,
                created_at=datetime.now(),
                status='posted',
                metadata={'objective_id': objective.id}
            )
            self.database.save_post(post)

            # Log activity
            self._log_activity(
                bot.id,
                ActionType.REPLY_TO_THREAD,
                {
                    'thread_id': thread.id,
                    'post_id': post_id,
                    'objective_type': objective.type.value,
                    'content_length': len(content)
                },
                True
            )

            logger.info(f"Bot {bot.name} posted to thread {thread.id}")

        except Exception as e:
            logger.error(f"Error engaging with thread {thread.id}: {e}")
            self._log_activity(
                bot.id,
                ActionType.ERROR,
                {'thread_id': thread.id, 'error': str(e)},
                False,
                str(e)
            )

    def _log_activity(
        self,
        bot_id: str,
        action_type: ActionType,
        details: Dict[str, Any],
        success: bool,
        error_message: Optional[str] = None
    ):
        """Log bot activity"""
        log = ActivityLog(
            id=str(uuid.uuid4()),
            bot_id=bot_id,
            action_type=action_type,
            details=details,
            timestamp=datetime.now(),
            success=success,
            error_message=error_message
        )
        self.database.log_activity(log)

    def get_bot_statistics(self, bot_id: str) -> Dict[str, Any]:
        """Get statistics for a bot"""
        activities = self.database.get_bot_activities(bot_id, limit=1000)

        total_actions = len(activities)
        successful_actions = sum(1 for a in activities if a.success)
        failed_actions = total_actions - successful_actions

        action_counts = defaultdict(int)
        for activity in activities:
            action_counts[activity.action_type.value] += 1

        return {
            'bot_id': bot_id,
            'total_actions': total_actions,
            'successful_actions': successful_actions,
            'failed_actions': failed_actions,
            'success_rate': successful_actions / total_actions if total_actions > 0 else 0,
            'action_breakdown': dict(action_counts),
            'last_activity': activities[0].timestamp.isoformat() if activities else None
        }


# ============================================================================
# APPLICATION ORCHESTRATOR
# ============================================================================

class MultiBot Application:
    """Main application orchestrator"""

    def __init__(self):
        self.database = Database()
        self.bot_manager = BotManager(self.database)
        self.running = False

    def initialize_demo_bots(self):
        """Initialize demo bots for testing"""
        logger.info("Initializing demo bots...")

        # Get preset personalities
        personalities = PersonalityEngine.get_preset_personalities()

        # Bot 1: Friendly engagement bot
        bot1 = self.bot_manager.create_bot(
            name="FriendlyHelper",
            personality_config=personalities['friendly'].to_dict(),
            objectives_config=[
                {
                    'type': 'engagement',
                    'description': 'Engage with beginners and help them feel welcome',
                    'target_keywords': ['beginner', 'help', 'new', 'started', 'learn'],
                    'success_metrics': {'engagement_rate': 0.7},
                    'constraints': {'keyword_match_threshold': 1},
                    'priority': 8
                }
            ],
            credentials={'username': 'friendly_bot', 'api_key': 'demo_key_1'}
        )

        # Bot 2: Technical expert bot
        bot2 = self.bot_manager.create_bot(
            name="TechnicalExpert",
            personality_config=personalities['technical'].to_dict(),
            objectives_config=[
                {
                    'type': 'information',
                    'description': 'Provide technical information and best practices',
                    'target_keywords': ['best practice', 'performance', 'optimization', 'architecture', 'async'],
                    'success_metrics': {'helpfulness_score': 0.8},
                    'constraints': {'keyword_match_threshold': 1},
                    'priority': 9
                }
            ],
            credentials={'username': 'tech_expert', 'api_key': 'demo_key_2'}
        )

        # Bot 3: Professional support bot
        bot3 = self.bot_manager.create_bot(
            name="ProfessionalSupport",
            personality_config=personalities['professional'].to_dict(),
            objectives_config=[
                {
                    'type': 'support',
                    'description': 'Provide professional support and guidance',
                    'target_keywords': ['problem', 'issue', 'error', 'bug', 'help needed'],
                    'success_metrics': {'resolution_rate': 0.75},
                    'constraints': {'keyword_match_threshold': 1},
                    'priority': 10
                }
            ],
            credentials={'username': 'pro_support', 'api_key': 'demo_key_3'}
        )

        logger.info(f"Initialized {len([bot1, bot2, bot3])} demo bots")
        return [bot1, bot2, bot3]

    async def run_bot_cycle(self, bot_id: str, forum_id: str = "demo_forum"):
        """Run a single cycle for a bot"""
        try:
            await self.bot_manager.process_bot_objectives(bot_id, forum_id)
        except Exception as e:
            logger.error(f"Error in bot cycle for {bot_id}: {e}")

    async def run_all_bots(self, forum_id: str = "demo_forum", cycles: int = 1):
        """Run all active bots for specified number of cycles"""
        logger.info(f"Starting multi-bot run for {cycles} cycle(s)")

        bots = self.bot_manager.list_bots()
        active_bots = [b for b in bots if b.status == BotStatus.ACTIVE]

        for cycle in range(cycles):
            logger.info(f"Starting cycle {cycle + 1}/{cycles}")

            # Run all bots concurrently
            tasks = [
                self.run_bot_cycle(bot.id, forum_id)
                for bot in active_bots
            ]

            await asyncio.gather(*tasks, return_exceptions=True)

            logger.info(f"Completed cycle {cycle + 1}/{cycles}")

            # Wait before next cycle
            if cycle < cycles - 1:
                await asyncio.sleep(30)

    def print_statistics(self):
        """Print statistics for all bots"""
        logger.info("=" * 80)
        logger.info("BOT STATISTICS")
        logger.info("=" * 80)

        bots = self.bot_manager.list_bots()

        for bot in bots:
            stats = self.bot_manager.get_bot_statistics(bot.id)

            print(f"\n Bot: {bot.name} ({bot.id})")
            print(f"  Status: {bot.status.value}")
            print(f"  Personality: {bot.personality.name}")
            print(f"  Objectives: {len(bot.objectives)}")
            print(f"  Total Actions: {stats['total_actions']}")
            print(f"  Success Rate: {stats['success_rate']:.2%}")
            print(f"  Action Breakdown:")
            for action, count in stats['action_breakdown'].items():
                print(f"    - {action}: {count}")
            print(f"  Last Activity: {stats['last_activity']}")

        logger.info("=" * 80)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

async def main():
    """Main application entry point"""
    logger.info("Starting Multi-Bot Forum Application")
    logger.info("=" * 80)

    # Initialize application
    app = MultiBotApplication()

    # Initialize demo bots
    demo_bots = app.initialize_demo_bots()

    # Display bot information
    print("\n Initialized Bots:")
    for bot in demo_bots:
        print(f"  - {bot.name}: {bot.personality.description}")
        print(f"    Objectives: {[obj.type.value for obj in bot.objectives]}")

    # Run bots for 2 cycles
    print("\n Running bot cycles...")
    await app.run_all_bots(cycles=2)

    # Print statistics
    print("\n")
    app.print_statistics()

    logger.info("=" * 80)
    logger.info("Application completed successfully")


if __name__ == "__main__":
    # Run the application
    asyncio.run(main())
