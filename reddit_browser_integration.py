"""
Reddit and Browser Automation Integration
==========================================

Extends the multi-bot application with Reddit API integration
and browser automation for human-like interactions.

This module provides:
- Reddit API integration using PRAW
- Browser automation using Selenium
- Human-like behavior simulation
- Thread reading and posting
- Upvoting/downvoting
- Comment interactions
"""

import asyncio
import random
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
import logging
import os
from pathlib import Path

# Reddit API
import praw
from praw.models import Submission, Comment, Subreddit
from prawcore.exceptions import PrawcoreException

# Browser automation
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.common.exceptions import TimeoutException, NoSuchElementException

# Import from main application
from multi_bot_forum_app import (
    ForumAPIClient, ForumThread, Bot, logger,
    Config, RateLimiter, CircuitBreaker
)

# Configuration
import yaml
from dotenv import load_dotenv

load_dotenv()


# ============================================================================
# REDDIT API CLIENT
# ============================================================================

class RedditAPIClient(ForumAPIClient):
    """Reddit API client using PRAW"""

    def __init__(self, credentials: Dict[str, str], rate_limiter: RateLimiter):
        super().__init__(credentials, rate_limiter)
        self.reddit: Optional[praw.Reddit] = None
        self.authenticated = False

    async def authenticate(self) -> bool:
        """Authenticate with Reddit API"""
        try:
            self.reddit = praw.Reddit(
                client_id=self.credentials['client_id'],
                client_secret=self.credentials['client_secret'],
                username=self.credentials['username'],
                password=self.credentials['password'],
                user_agent=self.credentials.get('user_agent', 'Multi-Bot Forum App v1.0')
            )

            # Test authentication
            _ = self.reddit.user.me()
            self.authenticated = True
            logger.info(f"Reddit authentication successful for u/{self.credentials['username']}")
            return True

        except Exception as e:
            logger.error(f"Reddit authentication failed: {e}")
            self.authenticated = False
            return False

    async def get_threads(self, forum_id: str, limit: int = 10) -> List[ForumThread]:
        """
        Fetch recent threads from a subreddit

        Args:
            forum_id: Subreddit name (without r/)
            limit: Maximum number of threads to fetch
        """
        if not self.authenticated:
            await self.authenticate()

        try:
            # Apply rate limiting
            await asyncio.sleep(1)  # Reddit rate limit

            subreddit = self.reddit.subreddit(forum_id)
            submissions = list(subreddit.hot(limit=limit))

            threads = []
            for submission in submissions:
                thread = ForumThread(
                    id=submission.id,
                    forum_id=forum_id,
                    url=f"https://reddit.com{submission.permalink}",
                    title=submission.title,
                    content=submission.selftext,
                    author=str(submission.author),
                    created_at=datetime.fromtimestamp(submission.created_utc),
                    last_checked=datetime.now(),
                    metadata={
                        'score': submission.score,
                        'num_comments': submission.num_comments,
                        'upvote_ratio': submission.upvote_ratio,
                        'is_self': submission.is_self,
                        'link_flair_text': submission.link_flair_text
                    }
                )
                threads.append(thread)

            logger.info(f"Fetched {len(threads)} threads from r/{forum_id}")
            return threads

        except PrawcoreException as e:
            logger.error(f"Reddit API error: {e}")
            raise

    async def get_thread_details(self, thread_id: str) -> ForumThread:
        """Get detailed information about a Reddit submission"""
        if not self.authenticated:
            await self.authenticate()

        try:
            submission = self.reddit.submission(id=thread_id)

            # Load all comments
            submission.comments.replace_more(limit=0)

            thread = ForumThread(
                id=submission.id,
                forum_id=submission.subreddit.display_name,
                url=f"https://reddit.com{submission.permalink}",
                title=submission.title,
                content=submission.selftext,
                author=str(submission.author),
                created_at=datetime.fromtimestamp(submission.created_utc),
                last_checked=datetime.now(),
                metadata={
                    'score': submission.score,
                    'num_comments': submission.num_comments,
                    'upvote_ratio': submission.upvote_ratio,
                    'comments': [
                        {
                            'id': comment.id,
                            'author': str(comment.author),
                            'body': comment.body,
                            'score': comment.score
                        }
                        for comment in submission.comments.list()[:20]  # Limit comments
                    ]
                }
            )

            return thread

        except PrawcoreException as e:
            logger.error(f"Reddit API error fetching thread {thread_id}: {e}")
            raise

    async def create_post(self, thread_id: str, content: str) -> str:
        """Create a comment on a Reddit submission"""
        if not self.authenticated:
            await self.authenticate()

        try:
            # Apply rate limiting
            await asyncio.sleep(2)  # Be conservative with Reddit rate limits

            submission = self.reddit.submission(id=thread_id)
            comment = submission.reply(content)

            logger.info(f"Posted comment {comment.id} to submission {thread_id}")
            return comment.id

        except PrawcoreException as e:
            logger.error(f"Reddit API error posting comment: {e}")
            raise

    async def like_post(self, post_id: str) -> bool:
        """Upvote a Reddit post or comment"""
        if not self.authenticated:
            await self.authenticate()

        try:
            # Try as submission first
            try:
                item = self.reddit.submission(id=post_id)
                item.upvote()
                logger.info(f"Upvoted submission {post_id}")
                return True
            except:
                # Try as comment
                item = self.reddit.comment(id=post_id)
                item.upvote()
                logger.info(f"Upvoted comment {post_id}")
                return True

        except PrawcoreException as e:
            logger.error(f"Reddit API error upvoting {post_id}: {e}")
            return False

    async def search_subreddit(self, subreddit_name: str, query: str, limit: int = 10) -> List[ForumThread]:
        """Search for threads in a subreddit"""
        if not self.authenticated:
            await self.authenticate()

        try:
            subreddit = self.reddit.subreddit(subreddit_name)
            submissions = list(subreddit.search(query, limit=limit, sort='relevance'))

            threads = []
            for submission in submissions:
                thread = ForumThread(
                    id=submission.id,
                    forum_id=subreddit_name,
                    url=f"https://reddit.com{submission.permalink}",
                    title=submission.title,
                    content=submission.selftext,
                    author=str(submission.author),
                    created_at=datetime.fromtimestamp(submission.created_utc),
                    last_checked=datetime.now(),
                    metadata={
                        'score': submission.score,
                        'num_comments': submission.num_comments,
                        'search_query': query
                    }
                )
                threads.append(thread)

            logger.info(f"Found {len(threads)} threads for query '{query}' in r/{subreddit_name}")
            return threads

        except PrawcoreException as e:
            logger.error(f"Reddit search error: {e}")
            raise


# ============================================================================
# BROWSER AUTOMATION CLIENT
# ============================================================================

class BrowserAutomationClient:
    """
    Browser automation for human-like interactions on Reddit
    Uses Selenium WebDriver to simulate real user behavior
    """

    def __init__(self, bot: Bot, config: Dict[str, Any]):
        self.bot = bot
        self.config = config
        self.driver: Optional[webdriver.Chrome] = None
        self.profile_dir = Path(config.get('user_data_dir', './browser_profiles')) / bot.id

    def initialize_driver(self):
        """Initialize Chrome WebDriver with options"""
        if self.driver:
            return

        # Setup Chrome options
        options = Options()

        if self.config.get('headless', True):
            options.add_argument('--headless')

        options.add_argument(f'--window-size={self.config.get("window_size", "1920,1080")}')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-blink-features=AutomationControlled')
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option('useAutomationExtension', False)

        # Use persistent profile for cookies/session
        self.profile_dir.mkdir(parents=True, exist_ok=True)
        options.add_argument(f'--user-data-dir={self.profile_dir}')

        if self.config.get('disable_images', False):
            prefs = {"profile.managed_default_content_settings.images": 2}
            options.add_experimental_option("prefs", prefs)

        # Initialize driver
        service = Service(ChromeDriverManager().install())
        self.driver = webdriver.Chrome(service=service, options=options)

        logger.info(f"Browser initialized for bot {self.bot.name}")

    def close_driver(self):
        """Close browser driver"""
        if self.driver:
            self.driver.quit()
            self.driver = None
            logger.info(f"Browser closed for bot {self.bot.name}")

    async def human_like_delay(self, min_seconds: float = 1.0, max_seconds: float = 3.0):
        """Add random delay to simulate human behavior"""
        delay = random.uniform(min_seconds, max_seconds)
        await asyncio.sleep(delay)

    async def login_to_reddit(self) -> bool:
        """Login to Reddit via browser"""
        try:
            self.initialize_driver()

            # Check if already logged in
            self.driver.get("https://www.reddit.com/")
            await self.human_like_delay(2, 4)

            # Check for login button
            try:
                login_button = self.driver.find_element(By.XPATH, "//a[contains(@href, '/login')]")
                # Need to login
            except NoSuchElementException:
                # Already logged in
                logger.info(f"Bot {self.bot.name} already logged in to Reddit")
                return True

            # Navigate to login page
            self.driver.get("https://www.reddit.com/login/")
            await self.human_like_delay(2, 3)

            # Find and fill username
            username_field = WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.ID, "loginUsername"))
            )
            username_field.send_keys(self.bot.credentials['username'])
            await self.human_like_delay(0.5, 1.5)

            # Find and fill password
            password_field = self.driver.find_element(By.ID, "loginPassword")
            password_field.send_keys(self.bot.credentials['password'])
            await self.human_like_delay(0.5, 1.5)

            # Click login button
            login_button = self.driver.find_element(By.XPATH, "//button[contains(text(), 'Log In')]")
            login_button.click()

            # Wait for redirect
            await self.human_like_delay(3, 5)

            # Verify login
            if "reddit.com" in self.driver.current_url and "/login" not in self.driver.current_url:
                logger.info(f"Bot {self.bot.name} successfully logged in to Reddit via browser")
                return True
            else:
                logger.error(f"Bot {self.bot.name} failed to login to Reddit")
                return False

        except Exception as e:
            logger.error(f"Browser login error for bot {self.bot.name}: {e}")
            return False

    async def browse_subreddit(self, subreddit_name: str, scroll_times: int = 3):
        """Browse a subreddit like a human"""
        try:
            self.driver.get(f"https://www.reddit.com/r/{subreddit_name}/")
            await self.human_like_delay(2, 4)

            # Scroll through posts
            for _ in range(scroll_times):
                # Scroll down
                self.driver.execute_script("window.scrollBy(0, window.innerHeight * 0.7);")
                await self.human_like_delay(1, 3)

                # Occasionally click on a post (read-only)
                if random.random() < 0.3:
                    await self._click_random_post()

            logger.info(f"Bot {self.bot.name} browsed r/{subreddit_name}")

        except Exception as e:
            logger.error(f"Browser browsing error: {e}")

    async def _click_random_post(self):
        """Click on a random post to read it"""
        try:
            posts = self.driver.find_elements(By.CSS_SELECTOR, "a[data-click-id='body']")
            if posts:
                post = random.choice(posts[:10])  # Choose from first 10 visible
                original_window = self.driver.current_window_handle

                # Open in new tab
                post.send_keys(Keys.CONTROL + Keys.RETURN)
                await self.human_like_delay(1, 2)

                # Switch to new tab
                new_window = [w for w in self.driver.window_handles if w != original_window][0]
                self.driver.switch_to.window(new_window)

                # Read for a bit
                await self.human_like_delay(3, 7)

                # Scroll through comments
                for _ in range(random.randint(1, 3)):
                    self.driver.execute_script("window.scrollBy(0, window.innerHeight * 0.5);")
                    await self.human_like_delay(1, 2)

                # Close tab and return
                self.driver.close()
                self.driver.switch_to.window(original_window)

        except Exception as e:
            logger.debug(f"Error clicking random post: {e}")

    async def post_comment_browser(self, thread_url: str, comment_text: str) -> bool:
        """Post a comment using browser automation"""
        try:
            # Navigate to thread
            self.driver.get(thread_url)
            await self.human_like_delay(2, 4)

            # Scroll a bit (simulate reading)
            scroll_times = random.randint(1, 3)
            for _ in range(scroll_times):
                self.driver.execute_script("window.scrollBy(0, window.innerHeight * 0.4);")
                await self.human_like_delay(1, 2)

            # Find comment box
            comment_box = WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "textarea[placeholder*='Comment']"))
            )

            # Click to focus
            comment_box.click()
            await self.human_like_delay(0.5, 1)

            # Type comment with human-like speed
            for char in comment_text:
                comment_box.send_keys(char)
                if random.random() < 0.1:  # Occasional pause
                    await asyncio.sleep(random.uniform(0.1, 0.3))

            await self.human_like_delay(1, 2)

            # Find and click submit button
            submit_button = self.driver.find_element(By.XPATH, "//button[contains(text(), 'Comment')]")
            submit_button.click()

            # Wait for submission
            await self.human_like_delay(2, 4)

            logger.info(f"Bot {self.bot.name} posted comment via browser to {thread_url}")
            return True

        except Exception as e:
            logger.error(f"Browser comment posting error: {e}")
            return False

    async def upvote_post(self, thread_url: str) -> bool:
        """Upvote a post using browser automation"""
        try:
            self.driver.get(thread_url)
            await self.human_like_delay(2, 3)

            # Find upvote button
            upvote_button = WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "button[aria-label*='upvote']"))
            )

            upvote_button.click()
            await self.human_like_delay(0.5, 1)

            logger.info(f"Bot {self.bot.name} upvoted post via browser")
            return True

        except Exception as e:
            logger.error(f"Browser upvote error: {e}")
            return False


# ============================================================================
# HYBRID INTERACTION MANAGER
# ============================================================================

class HybridRedditManager:
    """
    Manages both API and browser-based interactions with Reddit
    Intelligently chooses the best method for each action
    """

    def __init__(self, bot: Bot, config: Dict[str, Any]):
        self.bot = bot
        self.config = config
        self.rate_limiter = RateLimiter()

        # Initialize both clients
        self.api_client = RedditAPIClient(bot.credentials, self.rate_limiter)
        self.browser_client = BrowserAutomationClient(bot, config.get('browser', {}))

        # Decide strategy
        self.use_browser_for_posting = config.get('browser', {}).get('enabled', True)
        self.simulate_human_behavior = config.get('bot_behavior', {}).get('human_like_delays', {}).get('enabled', True)

    async def initialize(self):
        """Initialize both clients"""
        await self.api_client.authenticate()

        if self.use_browser_for_posting:
            success = await self.browser_client.login_to_reddit()
            if not success:
                logger.warning(f"Browser login failed for {self.bot.name}, falling back to API only")
                self.use_browser_for_posting = False

    async def get_threads(self, subreddit: str, limit: int = 10) -> List[ForumThread]:
        """Get threads using API (faster and more reliable)"""
        return await self.api_client.get_threads(subreddit, limit)

    async def post_comment(self, thread_url: str, thread_id: str, content: str, use_browser: bool = None) -> str:
        """
        Post comment using either API or browser

        Args:
            thread_url: Full URL to the thread
            thread_id: Reddit thread ID
            content: Comment content
            use_browser: Force browser usage (None = auto-decide)
        """
        if use_browser is None:
            use_browser = self.use_browser_for_posting

        if use_browser and self.browser_client.driver:
            # Occasionally browse before posting (more human-like)
            if self.simulate_human_behavior and random.random() < 0.3:
                subreddit = thread_url.split('/r/')[1].split('/')[0]
                await self.browser_client.browse_subreddit(subreddit, scroll_times=2)

            success = await self.browser_client.post_comment_browser(thread_url, content)
            if success:
                return f"browser_comment_{int(time.time())}"
            else:
                logger.warning("Browser posting failed, falling back to API")

        # Fallback to API
        return await self.api_client.create_post(thread_id, content)

    async def upvote(self, thread_url: str, post_id: str, use_browser: bool = None) -> bool:
        """Upvote using either API or browser"""
        if use_browser is None:
            use_browser = self.use_browser_for_posting and random.random() < 0.5

        if use_browser and self.browser_client.driver:
            return await self.browser_client.upvote_post(thread_url)
        else:
            return await self.api_client.like_post(post_id)

    async def search(self, subreddit: str, query: str, limit: int = 10) -> List[ForumThread]:
        """Search for threads (using API)"""
        return await self.api_client.search_subreddit(subreddit, query, limit)

    def cleanup(self):
        """Cleanup resources"""
        self.browser_client.close_driver()


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

async def example_usage():
    """Example of how to use the Reddit integration"""

    # Load configuration
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Create example bot
    from multi_bot_forum_app import Personality, Objective, ObjectiveType

    bot = Bot(
        id="example_bot_1",
        name="ExampleRedditBot",
        personality=Personality(
            id="pers_1",
            name="Friendly Helper",
            description="Friendly and helpful",
            tone="friendly",
            formality=0.4,
            enthusiasm=0.7,
            technical_level=0.5,
            verbosity=0.5,
            empathy=0.8,
            vocabulary_style="casual",
            response_patterns=["Hey!", "Great question!"]
        ),
        objectives=[
            Objective(
                id="obj_1",
                type=ObjectiveType.ENGAGEMENT,
                description="Engage with Python learners",
                target_keywords=["python", "help", "beginner"],
                success_metrics={},
                constraints={},
                priority=8
            )
        ],
        status="active",
        credentials={
            'client_id': os.getenv('REDDIT_BOT1_CLIENT_ID'),
            'client_secret': os.getenv('REDDIT_BOT1_CLIENT_SECRET'),
            'username': os.getenv('REDDIT_BOT1_USERNAME'),
            'password': os.getenv('REDDIT_BOT1_PASSWORD'),
            'user_agent': 'Multi-Bot Forum App v1.0'
        },
        metadata={},
        created_at=datetime.now(),
        last_active=datetime.now()
    )

    # Initialize hybrid manager
    manager = HybridRedditManager(bot, config)
    await manager.initialize()

    # Get threads from subreddit
    threads = await manager.get_threads('learnpython', limit=5)
    print(f"Found {len(threads)} threads")

    for thread in threads[:2]:
        print(f"\nThread: {thread.title}")
        print(f"URL: {thread.url}")

        # Example: Post a comment
        comment = "This is a helpful comment generated by the bot!"
        # comment_id = await manager.post_comment(thread.url, thread.id, comment)
        # print(f"Posted comment: {comment_id}")

        # Example: Upvote
        # await manager.upvote(thread.url, thread.id)

    # Cleanup
    manager.cleanup()


if __name__ == "__main__":
    asyncio.run(example_usage())
