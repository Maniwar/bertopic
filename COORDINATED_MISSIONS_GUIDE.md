# Coordinated Multi-Bot Missions Guide

## Overview

This guide explains how to configure and execute coordinated missions where multiple bots work together, take different positions on topics, and interact with each other to create natural, realistic conversations.

---

## 🎯 What Are Coordinated Missions?

Coordinated missions allow you to orchestrate multiple bots to:

✅ **Take Different Positions** - For, against, neutral, skeptical, etc.
✅ **Interact Naturally** - Bots respond to each other with realistic timing
✅ **Create Debates** - Bots argue different sides of an issue
✅ **Build Consensus** - Bots gradually agree on a topic
✅ **Provide Perspectives** - Each bot offers a unique angle (technical, business, beginner)
✅ **Coordinate Timing** - Staggered responses to avoid detection

---

## 📋 Mission Types

### 1. DEBATE Mission

Bots take opposing positions and argue different sides.

**Use Case:** Create realistic debate on controversial topics

```python
from coordinated_missions import MissionTemplates

mission = MissionTemplates.create_debate_mission(
    topic="Python vs JavaScript for beginners",
    pro_bot_ids=['bot_1', 'bot_2'],      # Bots arguing FOR
    con_bot_ids=['bot_3', 'bot_4'],      # Bots arguing AGAINST
    keywords=['python', 'javascript', 'beginner']
)
```

**Result:**
```
Bot1 (PRO): I think Python is better for beginners because...
Bot3 (CON): I respectfully disagree. JavaScript has advantages...
Bot2 (PRO): Building on what Bot1 said, Python's syntax...
Bot4 (CON): That's a fair point, but consider that...
```

---

### 2. CONSENSUS BUILDING Mission

Bots start with different views but gradually agree.

**Use Case:** Build support for a position organically

```python
mission = MissionTemplates.create_consensus_mission(
    topic="using type hints in Python",
    bot_ids=['bot_1', 'bot_2', 'bot_3'],
    final_position="Type hints improve code quality",
    keywords=['type hints', 'typing', 'python']
)
```

**Result:**
```
Bot1: I'm not sure about type hints, seems like extra work...
Bot2: I was skeptical too, but after trying them...
Bot3: That's interesting, maybe I should give them a try
Bot1: You've convinced me, the benefits do outweigh...
```

---

### 3. DIVERSE PERSPECTIVES Mission

Each bot offers a unique perspective on the same topic.

**Use Case:** Provide comprehensive coverage of a topic

```python
perspectives = {
    'tech_bot': {
        'angle': 'technical',
        'position': Position.EXPERT,
        'talking_points': [
            "From a performance standpoint...",
            "The architecture allows for..."
        ]
    },
    'business_bot': {
        'angle': 'business',
        'position': Position.MODERATELY_FOR,
        'talking_points': [
            "This can reduce costs by...",
            "Consider the ROI..."
        ]
    },
    'beginner_bot': {
        'angle': 'beginner',
        'position': Position.CURIOUS,
        'talking_points': [
            "Can someone explain...",
            "I'm still learning about..."
        ]
    }
}

mission = MissionTemplates.create_diverse_perspectives_mission(
    topic="async/await in Python",
    bot_perspectives=perspectives,
    keywords=['async', 'await', 'asyncio']
)
```

**Result:**
```
TechBot: From a performance standpoint, async/await is excellent...
BusinessBot: This can reduce infrastructure costs significantly...
BeginnerBot: Can someone explain when I should use this?
TechBot: @BeginnerBot Great question! It's best for I/O-bound...
```

---

## 🎭 Bot Positions

Each bot can take different positions on a topic:

| Position | Description | Use When |
|----------|-------------|----------|
| `STRONGLY_FOR` | Passionate advocate | Need strong support |
| `MODERATELY_FOR` | Generally supportive | Balanced pro position |
| `NEUTRAL` | No strong opinion | Observer/moderator |
| `MODERATELY_AGAINST` | Has concerns | Balanced con position |
| `STRONGLY_AGAINST` | Passionate opponent | Need strong opposition |
| `SKEPTICAL` | Questions claims | Critical thinking |
| `CURIOUS` | Wants to learn | Beginner perspective |
| `EXPERT` | Authoritative | Technical depth |
| `NOVICE` | New to topic | Asking questions |

---

## 🎬 Complete Example: Setting Up a Debate

### Step 1: Create Bots with Different Personalities

```python
from multi_bot_forum_app import BotManager, Database, PersonalityEngine

db = Database()
bot_manager = BotManager(db)
personalities = PersonalityEngine.get_preset_personalities()

# Create PRO bot (friendly personality)
pro_bot = bot_manager.create_bot(
    name="PythonAdvocate",
    personality_config=personalities['friendly'].to_dict(),
    objectives_config=[{
        'type': 'engagement',
        'description': 'Advocate for Python',
        'target_keywords': ['python', 'javascript'],
        'priority': 8
    }],
    credentials={...}
)

# Create CON bot (professional personality)
con_bot = bot_manager.create_bot(
    name="JavaScriptAdvocate",
    personality_config=personalities['professional'].to_dict(),
    objectives_config=[{
        'type': 'engagement',
        'description': 'Advocate for JavaScript',
        'target_keywords': ['python', 'javascript'],
        'priority': 8
    }],
    credentials={...}
)
```

### Step 2: Configure Mission

```python
from coordinated_missions import MissionTemplates, MissionOrchestrator

orchestrator = MissionOrchestrator(bot_manager)

# Create debate mission
mission = MissionTemplates.create_debate_mission(
    topic="Python for beginners",
    pro_bot_ids=[pro_bot.id],
    con_bot_ids=[con_bot.id],
    keywords=['python', 'javascript', 'beginner', 'which language']
)

# Register mission
orchestrator.create_mission(mission)
```

### Step 3: Execute Mission on a Thread

```python
from reddit_browser_integration import HybridRedditManager

# Initialize Reddit manager
manager = HybridRedditManager(pro_bot, config)
await manager.initialize()

# Find matching thread
threads = await manager.get_threads('learnpython', limit=10)

# Find thread matching mission keywords
for thread in threads:
    thread_text = f"{thread.title} {thread.content}".lower()
    if any(kw in thread_text for kw in mission.target_thread_keywords):
        # Execute mission on this thread
        await orchestrator.execute_mission(
            mission_id=mission.id,
            thread=thread,
            forum_client=manager.api_client
        )
        break
```

### Step 4: Monitor Mission

```python
# Get mission report
report = orchestrator.get_mission_report(mission.id)

print(f"Mission: {report['mission_name']}")
print(f"Status: {report['status']}")
print(f"Total Posts: {report['total_posts']}")
print(f"Posts by Bot:")
for bot_id, count in report['posts_by_bot'].items():
    bot = bot_manager.get_bot(bot_id)
    print(f"  {bot.name}: {count} posts")
```

---

## ⚙️ Advanced Configuration

### Custom Bot Stance

Create fully custom bot stances:

```python
from coordinated_missions import BotStance, Position, InteractionStyle

custom_stance = BotStance(
    bot_id='my_bot_id',

    # Position on topic
    position=Position.MODERATELY_FOR,

    # What bot will say
    talking_points=[
        "I think this approach has merit because...",
        "Research shows that...",
        "From my experience...",
        "That's a fair point, but consider..."
    ],

    # Topics to avoid
    avoid_topics=['politics', 'religion'],

    # How bot interacts
    interaction_style=InteractionStyle.BALANCED,

    # Response behavior
    response_probability=0.7,    # 70% chance to respond
    agreement_rate=0.5,          # 50% agree, 50% disagree

    # Priority (1-10, higher posts first)
    priority=8
)
```

### Custom Mission

Create completely custom missions:

```python
from coordinated_missions import Mission, MissionType
import uuid

custom_mission = Mission(
    id=str(uuid.uuid4()),
    name="Custom Tech Discussion",
    mission_type=MissionType.DIVERSE_PERSPECTIVES,
    description="Discuss microservices from multiple angles",

    # Trigger keywords
    target_thread_keywords=[
        'microservices', 'architecture', 'monolith'
    ],

    # Bot configurations
    bot_stances=[
        stance1, stance2, stance3  # Your custom stances
    ],

    # Timing configuration
    timing_strategy={
        'initial_delay_min': 10,        # Wait 10-20 min before first post
        'initial_delay_max': 20,
        'between_posts_min': 15,         # 15-45 min between bot posts
        'between_posts_max': 45,
        'max_exchanges': 5,              # Max rounds of discussion
        'natural_taper': True            # Reduce activity over time
    },

    # Success criteria
    success_metrics={
        'min_posts': 6,
        'engagement_target': 0.7
    },

    # Constraints
    constraints={
        'avoid_simultaneous_posting': True,
        'vary_response_times': True,
        'max_posts_per_bot': 3
    }
)
```

---

## ⏱️ Timing Strategies

### Natural Conversation Flow

```python
timing_strategy={
    'initial_delay_min': 5,          # 5-15 min before anyone responds
    'initial_delay_max': 15,
    'between_posts_min': 10,         # 10-30 min between responses
    'between_posts_max': 30,
    'max_exchanges': 5,
    'natural_taper': True,           # Activity decreases over time
    'stagger_initial_posts': True    # Don't all respond at once
}
```

### Rapid Response (Higher Risk)

```python
timing_strategy={
    'initial_delay_min': 1,
    'initial_delay_max': 5,
    'between_posts_min': 2,
    'between_posts_max': 10,
    'max_exchanges': 3
}
```

### Slow Burn (More Natural)

```python
timing_strategy={
    'initial_delay_min': 30,
    'initial_delay_max': 60,
    'between_posts_min': 60,
    'between_posts_max': 180,
    'max_exchanges': 3,
    'natural_taper': True
}
```

---

## 🎨 Interaction Styles

Configure how bots interact with each other:

```python
class InteractionStyle:
    AGREEABLE      # Mostly agree with others
    DISAGREEABLE   # Mostly disagree with others
    BALANCED       # Mix of agreement and disagreement
    SUPPORTIVE     # Build on others' points
    CONTRARIAN     # Challenge everything
    SOCRATIC       # Ask questions
```

**Example:**

```python
# Supportive bot
BotStance(
    bot_id='supporter',
    interaction_style=InteractionStyle.SUPPORTIVE,
    response_probability=0.8,
    agreement_rate=0.8  # Agrees 80% of time
)

# Contrarian bot
BotStance(
    bot_id='devil_advocate',
    interaction_style=InteractionStyle.CONTRARIAN,
    response_probability=0.7,
    agreement_rate=0.2  # Disagrees 80% of time
)
```

---

## 📊 Execution Patterns

### Pattern 1: Timed Sequence

Bots post in specific order:

```python
# Set priorities (higher = posts first)
bot1_stance.priority = 10  # Posts first
bot2_stance.priority = 8   # Posts second
bot3_stance.priority = 6   # Posts third
```

### Pattern 2: Response-Based

Bots respond to each other:

```python
# High response probability
stance.response_probability = 0.9  # 90% chance to respond to others
```

### Pattern 3: Random Participation

Not all bots participate every round:

```python
# Lower response probability
stance.response_probability = 0.5  # 50% chance to participate
```

---

## 🛡️ Anti-Detection Best Practices

### 1. Vary Timing

```python
# DON'T: Consistent timing
between_posts_min = 10
between_posts_max = 10  # ❌ Too predictable

# DO: Variable timing
between_posts_min = 10
between_posts_max = 30  # ✅ Random 10-30 minutes
```

### 2. Different Personalities

```python
# DON'T: All bots same personality
all_bots_personality = 'professional'  # ❌ Obvious

# DO: Mix personalities
bot1.personality = 'friendly'   # ✅ Diverse
bot2.personality = 'technical'
bot3.personality = 'casual'
```

### 3. Natural Disagreement

```python
# DON'T: Bots on same side always agree
agreement_rate = 1.0  # ❌ Unrealistic

# DO: Some disagreement even on same side
agreement_rate = 0.7  # ✅ 70% agree, 30% have nuanced differences
```

### 4. Avoid Simultaneous Posting

```python
constraints={
    'avoid_simultaneous_posting': True,  # ✅ Wait for others
    'vary_response_times': True           # ✅ Random delays
}
```

### 5. Natural Taper

```python
timing_strategy={
    'natural_taper': True  # ✅ Activity decreases over time
}
```

---

## 💡 Use Cases

### Use Case 1: Product Discussion

**Scenario:** Promote a Python library naturally

```python
mission = MissionTemplates.create_diverse_perspectives_mission(
    topic="FastAPI framework",
    bot_perspectives={
        'tech_bot': {
            'position': Position.EXPERT,
            'talking_points': [
                "FastAPI's async support is excellent",
                "Type hints improve developer experience"
            ]
        },
        'beginner_bot': {
            'position': Position.CURIOUS,
            'talking_points': [
                "I'm new to FastAPI, how does it compare to Flask?",
                "The documentation seems helpful"
            ]
        },
        'experienced_bot': {
            'position': Position.MODERATELY_FOR,
            'talking_points': [
                "Switched from Flask, performance gains were noticeable",
                "Migration was straightforward"
            ]
        }
    },
    keywords=['fastapi', 'flask', 'web framework']
)
```

### Use Case 2: Balanced Discussion

**Scenario:** Create genuine-seeming debate

```python
mission = MissionTemplates.create_debate_mission(
    topic="Static typing in Python",
    pro_bot_ids=[bot1.id, bot2.id],
    con_bot_ids=[bot3.id],
    keywords=['type hints', 'mypy', 'static typing']
)

# Customize for nuance
for stance in mission.bot_stances:
    stance.agreement_rate = 0.6  # Even allies don't always agree
    stance.talking_points = generate_nuanced_points()
```

### Use Case 3: Consensus Building

**Scenario:** Build support for best practice

```python
mission = MissionTemplates.create_consensus_mission(
    topic="using virtual environments",
    bot_ids=[bot1.id, bot2.id, bot3.id, bot4.id],
    final_position="Virtual environments are essential",
    keywords=['virtual environment', 'venv', 'pip']
)
```

---

## 📈 Monitoring Missions

### Real-Time Monitoring

```python
# During execution
while mission.status == "active":
    report = orchestrator.get_mission_report(mission.id)
    print(f"Posts: {report['total_posts']}")
    print(f"Exchanges: {report['exchanges']}")
    await asyncio.sleep(300)  # Check every 5 minutes
```

### Post-Mission Analysis

```python
# After completion
report = orchestrator.get_mission_report(mission.id)

print(f"Mission: {report['mission_name']}")
print(f"Status: {report['status']}")
print(f"Total Posts: {report['total_posts']}")
print(f"Bots Participated: {report['bots_participated']}")

for bot_id, count in report['posts_by_bot'].items():
    bot = bot_manager.get_bot(bot_id)
    print(f"  {bot.name}: {count} posts")
```

---

## ⚠️ Ethical Considerations

### Do's ✅

- Use for authorized testing and research
- Disclose bot nature when appropriate
- Provide genuine value to discussions
- Follow platform Terms of Service
- Respect rate limits

### Don'ts ❌

- Manipulate public opinion
- Astroturf for commercial gain
- Spam communities
- Violate platform rules
- Create fake grassroots movements

---

## 🚀 Quick Start Example

```bash
# Run coordinated mission example
python run_coordinated_mission.py
```

This will:
1. Create 3 example missions (debate, consensus, diverse)
2. Show bot configurations
3. Display timing strategies
4. Explain best practices
5. Provide custom mission templates

---

## 📚 Full Code Example

```python
import asyncio
from coordinated_missions import MissionTemplates, MissionOrchestrator
from multi_bot_forum_app import BotManager, Database

async def run_debate_mission():
    # Initialize
    db = Database()
    bot_manager = BotManager(db)
    orchestrator = MissionOrchestrator(bot_manager)

    # Create bots (simplified)
    bot1 = bot_manager.create_bot(name="ProBot", ...)
    bot2 = bot_manager.create_bot(name="ConBot", ...)

    # Create mission
    mission = MissionTemplates.create_debate_mission(
        topic="Python vs JavaScript",
        pro_bot_ids=[bot1.id],
        con_bot_ids=[bot2.id],
        keywords=['python', 'javascript']
    )

    # Register
    orchestrator.create_mission(mission)

    # Execute on thread
    await orchestrator.execute_mission(
        mission.id,
        thread,
        forum_client
    )

    # Get report
    report = orchestrator.get_mission_report(mission.id)
    print(f"Mission complete: {report['total_posts']} posts")

asyncio.run(run_debate_mission())
```

---

## 🎓 Advanced Topics

### Integrating LLMs for Content

For more natural content generation, integrate with OpenAI or Anthropic:

```python
import openai

def generate_bot_response(stance, thread, conversation_history):
    prompt = f"""
    You are a bot with position: {stance.position.value}
    Your talking points: {stance.talking_points}
    Thread topic: {thread.title}
    Previous conversation: {conversation_history}

    Generate a natural response that matches your position.
    """

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content
```

### Multi-Platform Missions

Extend to work across platforms:

```python
mission.platforms = ['reddit', 'twitter', 'discord']
mission.platform_specific_config = {
    'reddit': {'subreddit': 'learnpython'},
    'twitter': {'hashtags': ['python', 'coding']},
    'discord': {'channel': 'python-help'}
}
```

---

## 📖 See Also

- `coordinated_missions.py` - Core implementation
- `run_coordinated_mission.py` - Example script
- `README_MULTIBOT.md` - Main documentation
- `VISION_AUTOMATION_GUIDE.md` - Vision features

---

**Remember:** Use coordinated missions responsibly and ethically!
