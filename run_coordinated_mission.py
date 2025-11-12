"""
Coordinated Mission Example
============================

This script demonstrates how to set up and run coordinated multi-bot missions
where bots take different positions and interact with each other naturally.

Examples:
- Debate mission (bots argue different sides)
- Consensus mission (bots gradually agree)
- Diverse perspectives (each bot offers unique angle)

Usage:
    python run_coordinated_mission.py
"""

import asyncio
import os
from dotenv import load_dotenv
from datetime import datetime
import yaml

from coordinated_missions import (
    Mission, MissionTemplates, MissionOrchestrator, BotStance,
    Position, InteractionStyle, MissionType
)
from multi_bot_forum_app import (
    Database, BotManager, PersonalityEngine, BotStatus
)
from reddit_browser_integration import HybridRedditManager

load_dotenv()


async def main():
    print("=" * 80)
    print("🎯 Coordinated Multi-Bot Mission")
    print("=" * 80)
    print()

    # Initialize
    print("🔧 Initializing bot management system...")
    db = Database()
    bot_manager = BotManager(db)
    orchestrator = MissionOrchestrator(bot_manager)
    print("   ✅ Ready\n")

    # Load config
    try:
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
    except:
        config = {}

    # Get preset personalities
    personalities = PersonalityEngine.get_preset_personalities()

    # ========================================================================
    # EXAMPLE 1: DEBATE MISSION
    # ========================================================================

    print("=" * 80)
    print("📋 EXAMPLE 1: Debate Mission")
    print("=" * 80)
    print()
    print("Scenario: Python vs JavaScript for beginners")
    print("Setup: 2 bots argue PRO (Python), 2 bots argue CON (JavaScript)")
    print()

    # Create bots for debate
    print("➕ Creating debate bots...\n")

    # PRO Python bots
    pro_bot1 = bot_manager.create_bot(
        name="PythonAdvocate",
        personality_config=personalities['friendly'].to_dict(),
        objectives_config=[{
            'type': 'engagement',
            'description': 'Advocate for Python',
            'target_keywords': ['python', 'javascript', 'beginner'],
            'priority': 8
        }],
        credentials={
            'client_id': os.getenv('REDDIT_BOT1_CLIENT_ID', 'demo'),
            'client_secret': os.getenv('REDDIT_BOT1_CLIENT_SECRET', 'demo'),
            'username': os.getenv('REDDIT_BOT1_USERNAME', 'python_fan'),
            'password': os.getenv('REDDIT_BOT1_PASSWORD', 'demo'),
            'user_agent': 'Multi-Bot App v1.0'
        }
    )

    pro_bot2 = bot_manager.create_bot(
        name="PythonEnthusiast",
        personality_config=personalities['technical'].to_dict(),
        objectives_config=[{
            'type': 'information',
            'description': 'Share Python benefits',
            'target_keywords': ['python', 'javascript', 'beginner'],
            'priority': 7
        }],
        credentials={
            'client_id': os.getenv('REDDIT_BOT2_CLIENT_ID', 'demo'),
            'client_secret': os.getenv('REDDIT_BOT2_CLIENT_SECRET', 'demo'),
            'username': os.getenv('REDDIT_BOT2_USERNAME', 'py_expert'),
            'password': os.getenv('REDDIT_BOT2_PASSWORD', 'demo'),
            'user_agent': 'Multi-Bot App v1.0'
        }
    )

    # CON (JavaScript) bots
    con_bot1 = bot_manager.create_bot(
        name="JavaScriptAdvocate",
        personality_config=personalities['professional'].to_dict(),
        objectives_config=[{
            'type': 'engagement',
            'description': 'Advocate for JavaScript',
            'target_keywords': ['python', 'javascript', 'beginner'],
            'priority': 8
        }],
        credentials={
            'client_id': os.getenv('REDDIT_BOT3_CLIENT_ID', 'demo'),
            'client_secret': os.getenv('REDDIT_BOT3_CLIENT_SECRET', 'demo'),
            'username': os.getenv('REDDIT_BOT3_USERNAME', 'js_advocate'),
            'password': os.getenv('REDDIT_BOT3_PASSWORD', 'demo'),
            'user_agent': 'Multi-Bot App v1.0'
        }
    )

    con_bot2 = bot_manager.create_bot(
        name="WebDevPro",
        personality_config=personalities['casual'].to_dict(),
        objectives_config=[{
            'type': 'support',
            'description': 'Recommend JavaScript',
            'target_keywords': ['python', 'javascript', 'beginner'],
            'priority': 7
        }],
        credentials={
            'client_id': os.getenv('REDDIT_BOT1_CLIENT_ID', 'demo'),  # Reuse for demo
            'client_secret': os.getenv('REDDIT_BOT1_CLIENT_SECRET', 'demo'),
            'username': os.getenv('REDDIT_BOT1_USERNAME', 'webdev_pro'),
            'password': os.getenv('REDDIT_BOT1_PASSWORD', 'demo'),
            'user_agent': 'Multi-Bot App v1.0'
        }
    )

    print(f"   ✅ Created 4 bots for debate\n")

    # Create debate mission
    debate_mission = MissionTemplates.create_debate_mission(
        topic="Python for beginners",
        pro_bot_ids=[pro_bot1.id, pro_bot2.id],
        con_bot_ids=[con_bot1.id, con_bot2.id],
        keywords=['python', 'javascript', 'beginner', 'which language', 'learn first']
    )

    # Register mission
    orchestrator.create_mission(debate_mission)

    print(f"📌 Mission Created: {debate_mission.name}")
    print(f"   Type: {debate_mission.mission_type.value}")
    print(f"   Target Keywords: {', '.join(debate_mission.target_thread_keywords)}")
    print()

    print("🎭 Bot Positions:")
    for stance in debate_mission.bot_stances:
        bot = bot_manager.get_bot(stance.bot_id)
        print(f"   • {bot.name}")
        print(f"     Position: {stance.position.value}")
        print(f"     Style: {stance.interaction_style.value}")
        print(f"     Talking Points:")
        for point in stance.talking_points[:2]:
            print(f"       - {point[:60]}...")
        print()

    print("⏱️  Timing Strategy:")
    timing = debate_mission.timing_strategy
    print(f"   • Initial delay: {timing['initial_delay_min']}-{timing['initial_delay_max']} minutes")
    print(f"   • Between posts: {timing['between_posts_min']}-{timing['between_posts_max']} minutes")
    print(f"   • Max exchanges: {timing['max_exchanges']}")
    print(f"   • Natural taper: {timing['natural_taper']}")
    print()

    # ========================================================================
    # EXAMPLE 2: CONSENSUS BUILDING MISSION
    # ========================================================================

    print("\n" + "=" * 80)
    print("📋 EXAMPLE 2: Consensus Building Mission")
    print("=" * 80)
    print()
    print("Scenario: Gradually build consensus around using type hints in Python")
    print("Setup: 3 bots start skeptical/neutral, gradually agree on benefits")
    print()

    # Create consensus bots
    consensus_bots = []
    for i in range(3):
        bot = bot_manager.create_bot(
            name=f"TypeHintDiscusser{i+1}",
            personality_config=personalities['professional'].to_dict(),
            objectives_config=[{
                'type': 'engagement',
                'description': 'Discuss type hints',
                'target_keywords': ['type hints', 'typing', 'python types'],
                'priority': 6
            }],
            credentials={
                'client_id': os.getenv('REDDIT_BOT1_CLIENT_ID', 'demo'),
                'client_secret': os.getenv('REDDIT_BOT1_CLIENT_SECRET', 'demo'),
                'username': f'type_discusser_{i+1}',
                'password': 'demo',
                'user_agent': 'Multi-Bot App v1.0'
            }
        )
        consensus_bots.append(bot)

    print(f"   ✅ Created {len(consensus_bots)} bots for consensus building\n")

    consensus_mission = MissionTemplates.create_consensus_mission(
        topic="using type hints in Python",
        bot_ids=[b.id for b in consensus_bots],
        final_position="Type hints improve code quality and maintainability",
        keywords=['type hints', 'typing', 'python', 'annotations']
    )

    orchestrator.create_mission(consensus_mission)

    print(f"📌 Mission Created: {consensus_mission.name}")
    print(f"   Type: {consensus_mission.mission_type.value}")
    print()

    print("🎭 Bot Evolution:")
    for i, stance in enumerate(consensus_mission.bot_stances):
        bot = bot_manager.get_bot(stance.bot_id)
        print(f"   {i+1}. {bot.name}")
        print(f"      Starts: {stance.position.value}")
        print(f"      Evolves to: Agreement through discussion")
        print(f"      Agreement rate: {stance.agreement_rate:.0%}")
        print()

    # ========================================================================
    # EXAMPLE 3: DIVERSE PERSPECTIVES MISSION
    # ========================================================================

    print("\n" + "=" * 80)
    print("📋 EXAMPLE 3: Diverse Perspectives Mission")
    print("=" * 80)
    print()
    print("Scenario: Different angles on using async/await in Python")
    print("Setup: Each bot represents different perspective (technical, business, beginner)")
    print()

    # Define perspectives
    perspectives = {
        'technical_bot': {
            'angle': 'technical',
            'position': Position.EXPERT,
            'talking_points': [
                "From a performance standpoint, async/await is excellent for I/O-bound operations.",
                "The event loop architecture allows for efficient concurrency.",
                "Understanding asyncio internals helps optimize async code."
            ],
            'priority': 9
        },
        'business_bot': {
            'angle': 'business',
            'position': Position.MODERATELY_FOR,
            'talking_points': [
                "Async/await can reduce infrastructure costs by handling more concurrent requests.",
                "Development time may increase initially but pays off in scalability.",
                "Consider the learning curve for your team."
            ],
            'priority': 7
        },
        'beginner_bot': {
            'angle': 'beginner',
            'position': Position.CURIOUS,
            'talking_points': [
                "Can someone explain when I should use async/await?",
                "I'm confused about the difference between threading and async.",
                "Are there good beginner-friendly tutorials?"
            ],
            'priority': 5
        }
    }

    # Create bots with different perspectives
    perspective_bots = {}
    for bot_key, perspective in perspectives.items():
        bot = bot_manager.create_bot(
            name=f"{perspective['angle'].title()}Perspective",
            personality_config=personalities.get(
                'technical' if perspective['angle'] == 'technical' else 'professional'
            ).to_dict(),
            objectives_config=[{
                'type': 'information',
                'description': f"Provide {perspective['angle']} perspective",
                'target_keywords': ['async', 'await', 'asyncio'],
                'priority': perspective['priority']
            }],
            credentials={
                'client_id': os.getenv('REDDIT_BOT1_CLIENT_ID', 'demo'),
                'client_secret': os.getenv('REDDIT_BOT1_CLIENT_SECRET', 'demo'),
                'username': f"{perspective['angle']}_view",
                'password': 'demo',
                'user_agent': 'Multi-Bot App v1.0'
            }
        )
        perspective_bots[bot.id] = perspective

    print(f"   ✅ Created {len(perspective_bots)} bots with diverse perspectives\n")

    diverse_mission = MissionTemplates.create_diverse_perspectives_mission(
        topic="async/await in Python",
        bot_perspectives=perspective_bots,
        keywords=['async', 'await', 'asyncio', 'asynchronous', 'concurrency']
    )

    orchestrator.create_mission(diverse_mission)

    print(f"📌 Mission Created: {diverse_mission.name}")
    print(f"   Type: {diverse_mission.mission_type.value}")
    print()

    print("🎭 Perspectives:")
    for stance in diverse_mission.bot_stances:
        bot = bot_manager.get_bot(stance.bot_id)
        perspective_data = perspective_bots[stance.bot_id]
        print(f"   • {bot.name} ({perspective_data['angle'].upper()})")
        print(f"     Position: {stance.position.value}")
        print(f"     Sample: \"{stance.talking_points[0][:60]}...\"")
        print()

    # ========================================================================
    # MISSION EXECUTION SIMULATION
    # ========================================================================

    print("\n" + "=" * 80)
    print("🚀 Mission Execution")
    print("=" * 80)
    print()

    # Check if we have Reddit credentials
    if os.getenv('REDDIT_BOT1_CLIENT_ID') and os.getenv('REDDIT_BOT1_CLIENT_ID') != 'your_client_id_here':
        print("✅ Reddit credentials found")
        print()
        print("To execute a mission on a real thread:")
        print()
        print("  # Find a matching thread")
        print("  threads = await manager.get_threads('learnpython', limit=10)")
        print()
        print("  # Execute mission")
        print("  await orchestrator.execute_mission(")
        print("      mission_id=debate_mission.id,")
        print("      thread=threads[0],")
        print("      forum_client=manager.api_client")
        print("  )")
        print()
        print("⚠️  For this demo, we're not actually executing to avoid posting")
    else:
        print("ℹ️  No Reddit credentials - simulation mode")
        print("   Set up credentials in .env to execute real missions")

    print()

    # ========================================================================
    # MISSION REPORTS
    # ========================================================================

    print("\n" + "=" * 80)
    print("📊 Mission Configuration Summary")
    print("=" * 80)
    print()

    all_missions = [debate_mission, consensus_mission, diverse_mission]

    for mission in all_missions:
        report = orchestrator.get_mission_report(mission.id)

        print(f"\n{'─' * 60}")
        print(f"Mission: {report['mission_name']}")
        print(f"{'─' * 60}")
        print(f"Type: {report['mission_type']}")
        print(f"Status: {report['status']}")
        print(f"Bots: {len(mission.bot_stances)}")
        print(f"Target Keywords: {', '.join(mission.target_thread_keywords[:3])}...")

    # ========================================================================
    # BEST PRACTICES
    # ========================================================================

    print("\n\n" + "=" * 80)
    print("💡 Best Practices for Coordinated Missions")
    print("=" * 80)
    print()

    print("1. TIMING:")
    print("   • Stagger bot responses (10-30 min between posts)")
    print("   • Add random delays to avoid patterns")
    print("   • Use natural taper (reduce activity over time)")
    print()

    print("2. POSITIONS:")
    print("   • Vary positions for realistic debate")
    print("   • Don't make all bots extreme - use moderate positions")
    print("   • Include curious/skeptical bots for natural flow")
    print()

    print("3. INTERACTION:")
    print("   • Don't have bots respond to each other too quickly")
    print("   • Vary response probability (50-80%)")
    print("   • Avoid obvious coordination patterns")
    print()

    print("4. CONTENT:")
    print("   • Use distinct talking points for each bot")
    print("   • Integrate LLM (GPT-4/Claude) for better content")
    print("   • Match personality to position")
    print()

    print("5. DETECTION AVOIDANCE:")
    print("   • Don't post simultaneously")
    print("   • Vary writing styles (use different personalities)")
    print("   • Include natural mistakes/typos")
    print("   • Don't always agree within same stance")
    print()

    # ========================================================================
    # CUSTOM MISSION EXAMPLE
    # ========================================================================

    print("\n" + "=" * 80)
    print("🛠️  Creating Custom Missions")
    print("=" * 80)
    print()

    print("Custom mission template:")
    print()
    print("""
from coordinated_missions import Mission, BotStance, Position, InteractionStyle

# Define custom stances
custom_stances = [
    BotStance(
        bot_id='bot_1',
        position=Position.STRONGLY_FOR,
        talking_points=[
            'Your custom talking point 1',
            'Your custom talking point 2'
        ],
        interaction_style=InteractionStyle.SUPPORTIVE,
        response_probability=0.7,
        agreement_rate=0.6,
        priority=10
    ),
    # ... more stances
]

# Create custom mission
custom_mission = Mission(
    id=str(uuid.uuid4()),
    name='My Custom Mission',
    mission_type=MissionType.DEBATE,
    description='Description of what this mission does',
    target_thread_keywords=['keyword1', 'keyword2'],
    bot_stances=custom_stances,
    timing_strategy={
        'initial_delay_min': 5,
        'initial_delay_max': 15,
        'between_posts_min': 10,
        'between_posts_max': 30
    },
    success_metrics={'min_posts': 5},
    constraints={'max_posts_per_bot': 3}
)

# Execute
orchestrator.create_mission(custom_mission)
await orchestrator.execute_mission(custom_mission.id, thread, client)
    """)

    print("\n" + "=" * 80)
    print("✅ Coordinated Mission Demo Complete!")
    print("=" * 80)
    print()

    print("📚 Next Steps:")
    print("   1. Review the mission configurations above")
    print("   2. Customize bot stances and talking points")
    print("   3. Set up Reddit credentials in .env")
    print("   4. Execute missions on real threads")
    print("   5. Monitor mission reports and adjust strategy")
    print()

    print("⚠️  Remember:")
    print("   • Use coordinated missions ethically")
    print("   • Follow platform Terms of Service")
    print("   • Avoid manipulation and spam")
    print("   • Provide genuine value to discussions")
    print()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Mission demo stopped by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
