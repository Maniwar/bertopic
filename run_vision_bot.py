"""
Vision-Based Reddit Bot Example
================================

This script demonstrates using computer vision and AI to interact with
Reddit through the browser UI like a real human user.

⚠️  WARNING: This will control your mouse and keyboard!
    - Save all work before running
    - Move mouse to top-left corner to abort (failsafe)
    - Don't touch mouse/keyboard during automation

Usage:
    python run_vision_bot.py
"""

import asyncio
import os
from dotenv import load_dotenv
from vision_desktop_automation import VisionBasedRedditBot
from multi_bot_forum_app import (
    Bot, BotStatus, Personality, Objective, ObjectiveType
)
from datetime import datetime
import yaml

load_dotenv()


async def main():
    print("=" * 70)
    print("👁️  Vision-Based Reddit Bot - Starting...")
    print("=" * 70)

    # Safety warnings
    print("\n⚠️  WARNING: This script will control your mouse and keyboard!")
    print("   - Save all your work before continuing")
    print("   - Don't touch mouse/keyboard during automation")
    print("   - Move mouse to top-left corner to abort (failsafe)")
    print("   - Make sure you have Chrome/Chromium installed")
    print()

    # Check credentials
    if not os.getenv('REDDIT_BOT1_USERNAME'):
        print("❌ Error: Reddit credentials not found in .env file")
        print("   Set REDDIT_BOT1_USERNAME and REDDIT_BOT1_PASSWORD\n")
        return

    # Countdown
    print("Starting in:")
    for i in range(5, 0, -1):
        print(f"   {i}...")
        await asyncio.sleep(1)
    print("   🚀 GO!\n")

    # Load config
    try:
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print("⚠️  config.yaml not found, using defaults...\n")
        config = {
            'openai': {'enabled': False},
            'anthropic': {'enabled': False}
        }

    # Check if AI vision is available
    has_ai_vision = False
    if os.getenv('OPENAI_API_KEY'):
        config['openai'] = {'enabled': True}
        has_ai_vision = True
        print("✅ OpenAI API key found - AI vision enabled")
    elif os.getenv('ANTHROPIC_API_KEY'):
        config['anthropic'] = {'enabled': True}
        has_ai_vision = True
        print("✅ Anthropic API key found - AI vision enabled")
    else:
        print("ℹ️  No AI vision API keys found - using OCR only")

    print()

    # Create bot personality
    personality = Personality(
        id="pers_1",
        name="Friendly Helper",
        description="Friendly and helpful community member",
        tone="friendly",
        formality=0.4,
        enthusiasm=0.7,
        technical_level=0.5,
        verbosity=0.5,
        empathy=0.8,
        vocabulary_style="casual",
        response_patterns=[
            "Hey!",
            "Great question!",
            "Happy to help!",
            "I've been there!"
        ]
    )

    # Create bot
    bot = Bot(
        id="vision_bot_1",
        name="VisionRedditBot",
        personality=personality,
        objectives=[],
        status=BotStatus.ACTIVE,
        credentials={
            'username': os.getenv('REDDIT_BOT1_USERNAME'),
            'password': os.getenv('REDDIT_BOT1_PASSWORD')
        },
        metadata={},
        created_at=datetime.now(),
        last_active=datetime.now()
    )

    print(f"📝 Bot Configuration:")
    print(f"   Name: {bot.name}")
    print(f"   Username: {bot.credentials['username']}")
    print(f"   Personality: {personality.tone}")
    print(f"   AI Vision: {'Enabled' if has_ai_vision else 'Disabled (OCR only)'}")
    print()

    # Initialize vision bot
    try:
        print("🖥️  Step 1: Opening browser...")
        vision_bot = VisionBasedRedditBot(bot, config)

        await vision_bot.initialize_browser()
        print("   ✅ Browser opened")
        await asyncio.sleep(2)

    except Exception as e:
        print(f"   ❌ Failed to open browser: {e}")
        print("\n   Make sure Chrome/Chromium is installed")
        return

    # Login
    try:
        print("\n🔐 Step 2: Logging in to Reddit...")
        print("   (Watch the screen - bot is typing like a human!)")

        success = await vision_bot.login_to_reddit()

        if success:
            print("   ✅ Login successful!")
        else:
            print("   ❌ Login failed")
            print("   Note: Vision-based login is experimental")
            print("   You may need to login manually first time")
            return

    except Exception as e:
        print(f"   ❌ Login error: {e}")
        return

    # Navigate to subreddit
    try:
        print("\n🧭 Step 3: Navigating to r/learnpython...")

        await vision_bot.navigate_to_subreddit('learnpython')
        print("   ✅ Navigation complete")

    except Exception as e:
        print(f"   ❌ Navigation error: {e}")

    # Browse naturally
    try:
        browse_minutes = 2
        print(f"\n📖 Step 4: Browsing naturally for {browse_minutes} minutes...")
        print("   The bot will:")
        print("   - Scroll through posts (human-like speed)")
        print("   - Pause to 'read' content")
        print("   - Occasionally click on posts")
        print("   - Move mouse naturally (Bezier curves)")
        print()
        print("   Watch carefully - this is how it avoids detection!")
        print()

        await vision_bot.browse_like_human(duration_minutes=browse_minutes)
        print("   ✅ Browsing session complete")

    except Exception as e:
        print(f"   ❌ Browsing error: {e}")

    # Optional: Post a comment (disabled by default for safety)
    print("\n💬 Step 5: Posting comments (DISABLED)")
    print("   To enable comment posting:")
    print("   1. Uncomment the code below in run_vision_bot.py")
    print("   2. Adjust the thread title and comment text")
    print("   3. Re-run the script")
    print()

    """
    # Uncomment to enable comment posting
    try:
        print("\n💬 Step 5: Finding and commenting on a thread...")

        # Find a thread by title (adjust this!)
        thread_title = "How do I learn Python"

        comment_text = (
            "Great question! Python is an excellent choice for beginners. "
            "I'd recommend starting with the official Python tutorial and "
            "then working on small projects that interest you."
        )

        success = await vision_bot.read_thread_and_post_comment(
            thread_title=thread_title,
            comment_text=comment_text
        )

        if success:
            print("   ✅ Comment posted successfully!")
        else:
            print("   ⚠️  Could not find thread or post comment")

    except Exception as e:
        print(f"   ❌ Comment error: {e}")
    """

    # Summary
    print("\n" + "="*70)
    print("✅ Vision Bot Demo Complete!")
    print("="*70)
    print("\n📊 What happened:")
    print("   ✓ Opened browser naturally")
    print("   ✓ Logged in with human-like typing")
    print("   ✓ Navigated using address bar")
    print("   ✓ Browsed with realistic behavior")
    print("   ✓ Mouse moved along Bezier curves")
    print("   ✓ Scrolling had natural rhythm")
    print()
    print("🎯 Key Features Demonstrated:")
    print("   • Computer vision (OCR) for UI detection")
    if has_ai_vision:
        print("   • AI vision for complex UI understanding")
    print("   • Human-like mouse movements")
    print("   • Realistic typing with variations")
    print("   • Natural browsing patterns")
    print()
    print("🔒 Anti-Detection Measures:")
    print("   • No Selenium markers")
    print("   • No webdriver flags")
    print("   • Realistic timing and delays")
    print("   • Natural human behavior simulation")
    print()
    print("Next steps:")
    print("  1. Review the code to understand how it works")
    print("  2. Enable comment posting (uncomment code)")
    print("  3. Adjust browsing duration and behavior")
    print("  4. See VISION_AUTOMATION_GUIDE.md for advanced features")
    print()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Bot stopped by user (Ctrl+C)")
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        print("\nTroubleshooting:")
        print("  • Make sure Chrome/Chromium is installed")
        print("  • Check that Tesseract OCR is installed")
        print("  • Verify credentials in .env file")
        print("  • See QUICKSTART.md for setup instructions")
