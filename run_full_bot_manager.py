"""
Full Bot Manager Example
=========================

This script demonstrates the complete bot management system:
- Creating multiple bots with different personalities
- Setting objectives for each bot
- Running bots to process their objectives
- Viewing statistics and activity logs

Usage:
    python run_full_bot_manager.py
"""

import asyncio
import os
from dotenv import load_dotenv
from multi_bot_forum_app import (
    Database, BotManager, PersonalityEngine,
    ObjectiveType, BotStatus
)
import yaml

load_dotenv()


async def main():
    print("=" * 70)
    print("🤖 Multi-Bot Manager - Full Demo")
    print("=" * 70)

    # Check credentials
    if not os.getenv('REDDIT_BOT1_CLIENT_ID'):
        print("\n⚠️  Warning: Reddit credentials not found")
        print("   Running in DEMO mode with mock forum")
        print("   To use real Reddit, set credentials in .env file\n")
        use_reddit = False
    else:
        use_reddit = True
        print("\n✅ Reddit credentials found - will use real Reddit API\n")

    # Initialize
    print("🔧 Initializing bot management system...")
    db = Database()
    bot_manager = BotManager(db)
    print("   ✅ Database initialized")
    print("   ✅ Bot manager ready\n")

    # Load preset personalities
    personalities = PersonalityEngine.get_preset_personalities()

    print("📋 Available Personality Presets:")
    for name, personality in personalities.items():
        print(f"   • {name.title()}: {personality.description}")
    print()

    # Create bots
    print("➕ Creating Bots...\n")

    bots = []

    # Bot 1: Friendly Helper
    print("1️⃣  Creating Bot: Friendly Helper")
    bot1 = bot_manager.create_bot(
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
        credentials={
            'client_id': os.getenv('REDDIT_BOT1_CLIENT_ID', 'demo'),
            'client_secret': os.getenv('REDDIT_BOT1_CLIENT_SECRET', 'demo'),
            'username': os.getenv('REDDIT_BOT1_USERNAME', 'demo_user'),
            'password': os.getenv('REDDIT_BOT1_PASSWORD', 'demo_pass'),
            'user_agent': 'Multi-Bot App v1.0 (Educational)'
        }
    )
    bots.append(bot1)
    print(f"   ✅ {bot1.name} created (ID: {bot1.id})")
    print(f"      Personality: {bot1.personality.tone}")
    print(f"      Objectives: {len(bot1.objectives)}")
    print()

    # Bot 2: Technical Expert
    print("2️⃣  Creating Bot: Technical Expert")
    bot2 = bot_manager.create_bot(
        name="TechnicalExpert",
        personality_config=personalities['technical'].to_dict(),
        objectives_config=[
            {
                'type': 'information',
                'description': 'Provide technical information and best practices',
                'target_keywords': [
                    'best practice', 'performance', 'optimization',
                    'architecture', 'async', 'design pattern'
                ],
                'success_metrics': {'helpfulness_score': 0.8},
                'constraints': {'keyword_match_threshold': 1},
                'priority': 9
            }
        ],
        credentials={
            'client_id': os.getenv('REDDIT_BOT2_CLIENT_ID', 'demo'),
            'client_secret': os.getenv('REDDIT_BOT2_CLIENT_SECRET', 'demo'),
            'username': os.getenv('REDDIT_BOT2_USERNAME', 'demo_user2'),
            'password': os.getenv('REDDIT_BOT2_PASSWORD', 'demo_pass'),
            'user_agent': 'Multi-Bot App v1.0 (Educational)'
        }
    )
    bots.append(bot2)
    print(f"   ✅ {bot2.name} created (ID: {bot2.id})")
    print(f"      Personality: {bot2.personality.tone}")
    print(f"      Objectives: {len(bot2.objectives)}")
    print()

    # Bot 3: Professional Support
    print("3️⃣  Creating Bot: Professional Support")
    bot3 = bot_manager.create_bot(
        name="ProfessionalSupport",
        personality_config=personalities['professional'].to_dict(),
        objectives_config=[
            {
                'type': 'support',
                'description': 'Provide professional support and guidance',
                'target_keywords': [
                    'problem', 'issue', 'error', 'bug', 'help needed',
                    'not working', 'broken'
                ],
                'success_metrics': {'resolution_rate': 0.75},
                'constraints': {'keyword_match_threshold': 1},
                'priority': 10
            }
        ],
        credentials={
            'client_id': os.getenv('REDDIT_BOT3_CLIENT_ID', 'demo'),
            'client_secret': os.getenv('REDDIT_BOT3_CLIENT_SECRET', 'demo'),
            'username': os.getenv('REDDIT_BOT3_USERNAME', 'demo_user3'),
            'password': os.getenv('REDDIT_BOT3_PASSWORD', 'demo_pass'),
            'user_agent': 'Multi-Bot App v1.0 (Educational)'
        }
    )
    bots.append(bot3)
    print(f"   ✅ {bot3.name} created (ID: {bot3.id})")
    print(f"      Personality: {bot3.personality.tone}")
    print(f"      Objectives: {len(bot3.objectives)}")
    print()

    # Display bot summary
    print("=" * 70)
    print("📊 Bot Fleet Summary:")
    print("=" * 70)
    for i, bot in enumerate(bots, 1):
        print(f"\n{i}. {bot.name}")
        print(f"   Status: {bot.status.value}")
        print(f"   Personality Traits:")
        print(f"      • Formality: {bot.personality.formality:.1f}")
        print(f"      • Enthusiasm: {bot.personality.enthusiasm:.1f}")
        print(f"      • Technical Level: {bot.personality.technical_level:.1f}")
        print(f"   Objectives:")
        for obj in bot.objectives:
            print(f"      • {obj.type.value}: {obj.description}")
            print(f"        Keywords: {', '.join(obj.target_keywords[:3])}...")
    print()

    # Run bots
    if use_reddit:
        print("=" * 70)
        print("🚀 Running Bots (1 cycle each)...")
        print("=" * 70)
        print()

        for bot in bots:
            print(f"▶️  Running {bot.name}...")
            try:
                await bot_manager.process_bot_objectives(bot.id, 'learnpython')
                print(f"   ✅ Completed\n")
            except Exception as e:
                print(f"   ❌ Error: {e}\n")

    else:
        print("=" * 70)
        print("ℹ️  Demo Mode - Simulating Bot Runs")
        print("=" * 70)
        print("\nTo run against real Reddit:")
        print("  1. Set up Reddit API credentials in .env")
        print("  2. Re-run this script")
        print("\nFor now, showing what would happen...\n")

    # Show statistics
    print("=" * 70)
    print("📈 Bot Statistics:")
    print("=" * 70)

    for bot in bots:
        print(f"\n🤖 {bot.name}")
        print("-" * 50)

        stats = bot_manager.get_bot_statistics(bot.id)

        print(f"   Total Actions: {stats['total_actions']}")
        print(f"   Successful: {stats['successful_actions']}")
        print(f"   Failed: {stats['failed_actions']}")

        if stats['total_actions'] > 0:
            print(f"   Success Rate: {stats['success_rate']:.2%}")
        else:
            print(f"   Success Rate: N/A (no actions yet)")

        if stats['action_breakdown']:
            print(f"   Action Breakdown:")
            for action, count in stats['action_breakdown'].items():
                print(f"      • {action}: {count}")

        if stats['last_activity']:
            print(f"   Last Activity: {stats['last_activity']}")
        else:
            print(f"   Last Activity: Never")

    # Show recent activities
    print("\n" + "=" * 70)
    print("📜 Recent Activity Log (All Bots):")
    print("=" * 70 + "\n")

    all_activities = []
    for bot in bots:
        activities = db.get_bot_activities(bot.id, limit=5)
        for activity in activities:
            all_activities.append((bot.name, activity))

    # Sort by timestamp
    all_activities.sort(key=lambda x: x[1].timestamp, reverse=True)

    if all_activities:
        for bot_name, activity in all_activities[:15]:  # Show last 15
            status = "✅" if activity.success else "❌"
            print(f"{status} [{activity.timestamp}] {bot_name}")
            print(f"   Action: {activity.action_type.value}")
            if activity.details:
                # Show some details
                details_preview = str(activity.details)[:60]
                print(f"   Details: {details_preview}...")
            if not activity.success and activity.error_message:
                print(f"   Error: {activity.error_message}")
            print()
    else:
        print("   No activities logged yet")
        print("   (In demo mode, bots don't actually run)\n")

    # Management operations
    print("=" * 70)
    print("⚙️  Bot Management Examples:")
    print("=" * 70 + "\n")

    print("You can manage bots programmatically:\n")

    print("# Pause a bot")
    print(f"bot_manager.update_bot_status('{bot1.id}', BotStatus.PAUSED)")
    print()

    print("# Reactivate a bot")
    print(f"bot_manager.update_bot_status('{bot1.id}', BotStatus.ACTIVE)")
    print()

    print("# List all bots")
    print("all_bots = bot_manager.list_bots()")
    print()

    print("# Get specific bot")
    print(f"bot = bot_manager.get_bot('{bot1.id}')")
    print()

    # Final summary
    print("\n" + "=" * 70)
    print("✅ Bot Manager Demo Complete!")
    print("=" * 70)

    print("\n📚 Key Concepts Demonstrated:")
    print("   ✓ Creating bots with different personalities")
    print("   ✓ Setting objectives for each bot")
    print("   ✓ Managing multiple bots simultaneously")
    print("   ✓ Tracking bot activities and statistics")
    print("   ✓ Database persistence")
    print()

    print("🔧 Next Steps:")
    print("   1. Customize bot personalities (edit personality_config)")
    print("   2. Add more objectives (edit objectives_config)")
    print("   3. Run bots on different subreddits")
    print("   4. Set up scheduled runs (cron jobs)")
    print("   5. Monitor logs in multi_bot_app.log")
    print()

    print("📖 Documentation:")
    print("   • QUICKSTART.md - Quick start guide")
    print("   • README_MULTIBOT.md - Full documentation")
    print("   • VISION_AUTOMATION_GUIDE.md - Vision features")
    print()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Bot manager stopped by user (Ctrl+C)")
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
