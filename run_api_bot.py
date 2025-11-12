"""
Simple Reddit API Bot Example
==============================

This script demonstrates using the Reddit API (PRAW) to:
- Authenticate with Reddit
- Fetch threads from a subreddit
- Identify threads matching bot objectives
- Optionally post comments

Usage:
    python run_api_bot.py
"""

import asyncio
import os
from dotenv import load_dotenv
from reddit_browser_integration import RedditAPIClient, HybridRedditManager
from multi_bot_forum_app import (
    Bot, BotStatus, Personality, Objective, ObjectiveType,
    RateLimiter
)
from datetime import datetime
import yaml

load_dotenv()


async def main():
    print("=" * 60)
    print("🤖 Reddit API Bot - Starting...")
    print("=" * 60)

    # Check if credentials are set
    if not os.getenv('REDDIT_BOT1_CLIENT_ID'):
        print("\n❌ Error: Reddit credentials not found!")
        print("   Please set up your .env file with Reddit API credentials.")
        print("   See QUICKSTART.md for instructions.\n")
        return

    # Load config
    try:
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print("\n⚠️  config.yaml not found, using defaults...")
        config = {}

    # Create bot personality
    personality = Personality(
        id="pers_1",
        name="Helpful Assistant",
        description="Friendly and helpful Reddit user who loves helping Python learners",
        tone="friendly",
        formality=0.5,
        enthusiasm=0.7,
        technical_level=0.6,
        verbosity=0.5,
        empathy=0.8,
        vocabulary_style="casual",
        response_patterns=[
            "Hey!",
            "Great question!",
            "Happy to help!",
            "I've been there!",
            "Welcome to the community!"
        ]
    )

    # Create objective
    objective = Objective(
        id="obj_1",
        type=ObjectiveType.SUPPORT,
        description="Help Python learners with their questions and problems",
        target_keywords=[
            "python", "help", "beginner", "learn", "error", "problem",
            "question", "how do i", "how to", "stuck", "confused"
        ],
        success_metrics={},
        constraints={"keyword_match_threshold": 1},
        priority=9
    )

    # Create bot
    bot = Bot(
        id="api_bot_1",
        name="PythonHelper",
        personality=personality,
        objectives=[objective],
        status=BotStatus.ACTIVE,
        credentials={
            'client_id': os.getenv('REDDIT_BOT1_CLIENT_ID'),
            'client_secret': os.getenv('REDDIT_BOT1_CLIENT_SECRET'),
            'username': os.getenv('REDDIT_BOT1_USERNAME'),
            'password': os.getenv('REDDIT_BOT1_PASSWORD'),
            'user_agent': 'Multi-Bot Forum App v1.0 (Educational)'
        },
        metadata={},
        created_at=datetime.now(),
        last_active=datetime.now()
    )

    print(f"\n📝 Bot Configuration:")
    print(f"   Name: {bot.name}")
    print(f"   Personality: {personality.name} ({personality.tone})")
    print(f"   Objective: {objective.description}")
    print(f"   Target Keywords: {', '.join(objective.target_keywords[:5])}...")

    # Initialize manager
    print(f"\n🔐 Authenticating with Reddit...")
    print(f"   Username: {bot.credentials['username']}")

    try:
        manager = HybridRedditManager(bot, config)
        await manager.initialize()
        print("   ✅ Authentication successful!")
    except Exception as e:
        print(f"   ❌ Authentication failed: {e}")
        print("\n   Check your credentials in .env file")
        return

    # Get threads from subreddit
    subreddit = "learnpython"
    print(f"\n📖 Fetching threads from r/{subreddit}...")

    try:
        threads = await manager.get_threads(subreddit, limit=10)
        print(f"   ✅ Found {len(threads)} threads")
    except Exception as e:
        print(f"   ❌ Error fetching threads: {e}")
        return

    # Display threads
    print(f"\n{'='*60}")
    print(f"📋 Recent Threads from r/{subreddit}:")
    print(f"{'='*60}\n")

    matching_threads = []

    for i, thread in enumerate(threads, 1):
        print(f"{i}. {thread.title}")
        print(f"   Author: u/{thread.author}")
        print(f"   Score: {thread.metadata.get('score', 0)} ⬆ | "
              f"Comments: {thread.metadata.get('num_comments', 0)} 💬")
        print(f"   URL: {thread.url}")

        # Check if thread matches objective
        thread_text = f"{thread.title} {thread.content}".lower()
        matched_keywords = [
            kw for kw in objective.target_keywords
            if kw.lower() in thread_text
        ]

        if matched_keywords:
            print(f"   🎯 MATCH! Keywords: {', '.join(matched_keywords[:3])}")
            matching_threads.append((thread, matched_keywords))
        else:
            print(f"   ⚪ No keyword match")

        print()

    # Summary
    print(f"{'='*60}")
    print(f"📊 Summary:")
    print(f"   Total threads analyzed: {len(threads)}")
    print(f"   Matching threads: {len(matching_threads)}")
    print(f"   Match rate: {len(matching_threads)/len(threads)*100:.1f}%")
    print(f"{'='*60}\n")

    # Example: Show what bot would comment
    if matching_threads:
        print(f"💬 Example Engagement (Simulation):\n")

        for i, (thread, keywords) in enumerate(matching_threads[:3], 1):
            print(f"{i}. Thread: {thread.title[:60]}...")
            print(f"   Matched: {', '.join(keywords[:3])}")
            print(f"   Bot would comment with {personality.tone} tone")
            print(f"   Strategy: {objective.type.value}")

            # Generate example comment
            example_comments = [
                f"Hey! {thread.author}, welcome! I'd be happy to help you with this.",
                f"Great question! This is a common challenge when learning Python.",
                f"I've been there! Here's what helped me understand this concept...",
            ]

            print(f"   Example: \"{example_comments[i % len(example_comments)]}\"")
            print()

        print("\n⚠️  To actually post comments, uncomment the posting code in this script.")
        print("   For now, this is a READ-ONLY demonstration.\n")

        # Uncomment below to actually post comments
        """
        print("\n🚀 Posting comments? (Uncomment code to enable)")

        for thread, keywords in matching_threads[:1]:  # Just first match
            comment_text = f"Great question! I'd be happy to help with this. "
            comment_text += f"This is a common area of confusion for Python beginners."

            print(f"\n   Posting to: {thread.title[:50]}...")

            try:
                post_id = await manager.post_comment(
                    thread.url,
                    thread.id,
                    comment_text,
                    use_browser=False  # Use API
                )
                print(f"   ✅ Comment posted! ID: {post_id}")
            except Exception as e:
                print(f"   ❌ Failed to post: {e}")
        """

    else:
        print("ℹ️  No matching threads found in this batch.")
        print("   Try a different subreddit or adjust target keywords.\n")

    # Cleanup
    print("\n🧹 Cleaning up...")
    manager.cleanup()

    print("\n" + "="*60)
    print("✅ Bot session complete!")
    print("="*60)
    print("\nNext steps:")
    print("  1. Uncomment posting code to enable actual comments")
    print("  2. Adjust target keywords in the script")
    print("  3. Try different subreddits")
    print("  4. Create multiple bots with different personalities")
    print("\nSee QUICKSTART.md and README_MULTIBOT.md for more info.\n")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Bot stopped by user (Ctrl+C)")
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
