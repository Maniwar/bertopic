# Quick Start Guide - Multi-Bot Forum Application

Get your bots running in minutes!

## 🚀 Fast Track (5 Minutes)

### Step 1: Install Dependencies

```bash
# Navigate to project directory
cd /home/user/bertopic

# Install Python dependencies
pip install -r requirements_multibot.txt

# Install Tesseract OCR (for vision features)
# Ubuntu/Debian:
sudo apt-get update
sudo apt-get install -y tesseract-ocr

# macOS:
# brew install tesseract

# Windows:
# Download from https://github.com/UB-Mannheim/tesseract/wiki
```

### Step 2: Set Up Credentials

```bash
# Copy environment template
cp .env.example .env

# Edit with your credentials
nano .env
```

Add your Reddit credentials to `.env`:
```bash
# Reddit Bot Credentials
REDDIT_BOT1_CLIENT_ID=your_client_id_here
REDDIT_BOT1_CLIENT_SECRET=your_client_secret_here
REDDIT_BOT1_USERNAME=your_username_here
REDDIT_BOT1_PASSWORD=your_password_here

# Optional: AI Vision (for advanced features)
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
```

### Step 3: Run Demo Mode

```bash
# Test with mock forum (no credentials needed)
python multi_bot_forum_app.py
```

You should see output like:
```
2025-01-15 10:30:00 - INFO - Starting Multi-Bot Forum Application
2025-01-15 10:30:00 - INFO - Initializing demo bots...
2025-01-15 10:30:00 - INFO - Bot created: FriendlyHelper
2025-01-15 10:30:00 - INFO - Bot created: TechnicalExpert
2025-01-15 10:30:00 - INFO - Bot created: ProfessionalSupport
```

## 📋 Getting Reddit API Credentials

### 1. Create Reddit App

1. Go to https://www.reddit.com/prefs/apps
2. Scroll to bottom, click **"create another app..."**
3. Fill in:
   - **name**: MyBot (any name)
   - **App type**: Select **"script"**
   - **description**: (optional)
   - **about url**: (optional)
   - **redirect uri**: `http://localhost:8080`
4. Click **"create app"**

### 2. Get Credentials

After creating the app, you'll see:
- **client_id**: The string under "personal use script" (looks like: `abc123XYZ`)
- **client_secret**: The "secret" field (looks like: `xyz789ABC-def456`)
- **username**: Your Reddit username
- **password**: Your Reddit password

### 3. Update .env File

```bash
REDDIT_BOT1_CLIENT_ID=abc123XYZ
REDDIT_BOT1_CLIENT_SECRET=xyz789ABC-def456
REDDIT_BOT1_USERNAME=your_reddit_username
REDDIT_BOT1_PASSWORD=your_reddit_password
```

## 🎯 Running Different Modes

### Mode 1: Demo Mode (No Credentials)

Tests the bot system with mock forum:

```bash
python multi_bot_forum_app.py
```

### Mode 2: Reddit API Mode (Fastest)

Uses Reddit API (PRAW) for interactions:

```bash
python run_api_bot.py
```

### Mode 3: Browser Automation (Selenium)

Uses Selenium to control browser:

```bash
python run_browser_bot.py
```

### Mode 4: Vision-Based (Most Realistic) ⭐

Uses computer vision and human-like behavior:

```bash
python run_vision_bot.py
```

## 📝 Example Scripts

I'll create ready-to-run scripts for each mode...

### Simple Reddit API Example

Create `run_api_bot.py`:

```python
import asyncio
import os
from dotenv import load_dotenv
from reddit_browser_integration import RedditAPIClient, HybridRedditManager
from multi_bot_forum_app import Bot, BotStatus, Personality, Objective, ObjectiveType
from datetime import datetime
import yaml

load_dotenv()

async def main():
    print("🤖 Starting Reddit API Bot...")

    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Create bot personality
    personality = Personality(
        id="pers_1",
        name="Helpful Assistant",
        description="Friendly and helpful Reddit user",
        tone="friendly",
        formality=0.5,
        enthusiasm=0.7,
        technical_level=0.6,
        verbosity=0.5,
        empathy=0.8,
        vocabulary_style="casual",
        response_patterns=["Hey!", "Great question!", "Happy to help!"]
    )

    # Create objective
    objective = Objective(
        id="obj_1",
        type=ObjectiveType.SUPPORT,
        description="Help Python learners",
        target_keywords=["python", "help", "beginner", "learn", "error"],
        success_metrics={},
        constraints={"keyword_match_threshold": 1},
        priority=9
    )

    # Create bot
    bot = Bot(
        id="bot_1",
        name="PythonHelper",
        personality=personality,
        objectives=[objective],
        status=BotStatus.ACTIVE,
        credentials={
            'client_id': os.getenv('REDDIT_BOT1_CLIENT_ID'),
            'client_secret': os.getenv('REDDIT_BOT1_CLIENT_SECRET'),
            'username': os.getenv('REDDIT_BOT1_USERNAME'),
            'password': os.getenv('REDDIT_BOT1_PASSWORD'),
            'user_agent': 'Multi-Bot App v1.0'
        },
        metadata={},
        created_at=datetime.now(),
        last_active=datetime.now()
    )

    # Initialize manager
    print("🔐 Authenticating with Reddit...")
    manager = HybridRedditManager(bot, config)
    await manager.initialize()

    # Get threads from subreddit
    subreddit = "learnpython"
    print(f"📖 Fetching threads from r/{subreddit}...")
    threads = await manager.get_threads(subreddit, limit=10)

    print(f"\n✅ Found {len(threads)} threads:\n")

    for i, thread in enumerate(threads, 1):
        print(f"{i}. {thread.title}")
        print(f"   Score: {thread.metadata.get('score', 0)} | "
              f"Comments: {thread.metadata.get('num_comments', 0)}")
        print(f"   URL: {thread.url}")

        # Check if thread matches objective
        thread_text = f"{thread.title} {thread.content}".lower()
        matches = sum(1 for kw in objective.target_keywords if kw in thread_text)

        if matches > 0:
            print(f"   🎯 Matches {matches} keyword(s) - Good for engagement!")
        print()

    # Example: Comment on first matching thread
    print("\n💬 Would you like to comment on a matching thread? (y/n)")
    # In automated mode, you'd skip this and just post
    # For now, just showing what would happen

    for thread in threads:
        thread_text = f"{thread.title} {thread.content}".lower()
        matches = sum(1 for kw in objective.target_keywords if kw in thread_text)

        if matches > 0:
            print(f"\n📝 Would comment on: {thread.title}")
            print(f"   Strategy: {objective.type.value}")
            # Uncomment to actually post:
            # comment = "This is a helpful comment!"
            # post_id = await manager.post_comment(thread.url, thread.id, comment, use_browser=False)
            # print(f"✅ Posted comment: {post_id}")
            break

    print("\n✅ Bot session complete!")
    manager.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
```

### Vision-Based Example

Create `run_vision_bot.py`:

```python
import asyncio
import os
from dotenv import load_dotenv
from vision_desktop_automation import VisionBasedRedditBot
from multi_bot_forum_app import Bot, BotStatus, Personality, Objective, ObjectiveType
from datetime import datetime
import yaml

load_dotenv()

async def main():
    print("👁️ Starting Vision-Based Reddit Bot...")
    print("⚠️  This will control your mouse and keyboard!")
    print("⚠️  Move mouse to top-left corner to abort (PyAutoGUI failsafe)")

    await asyncio.sleep(3)  # Give user time to read

    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Create bot
    personality = Personality(
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
    )

    bot = Bot(
        id="vision_bot_1",
        name="VisionBot",
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

    # Initialize vision bot
    print("🖥️  Initializing browser...")
    vision_bot = VisionBasedRedditBot(bot, config)

    await vision_bot.initialize_browser()

    print("🔐 Logging in to Reddit...")
    success = await vision_bot.login_to_reddit()

    if success:
        print("✅ Login successful!")

        # Navigate to subreddit
        print("🧭 Navigating to r/learnpython...")
        await vision_bot.navigate_to_subreddit('learnpython')

        # Browse naturally
        print("📖 Browsing naturally for 2 minutes...")
        await vision_bot.browse_like_human(duration_minutes=2)

        print("✅ Vision bot demo complete!")
    else:
        print("❌ Login failed")

if __name__ == "__main__":
    asyncio.run(main())
```

### Full Bot Manager Example

Create `run_full_bot_manager.py`:

```python
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
    print("🤖 Starting Multi-Bot Manager...")

    # Initialize
    db = Database()
    bot_manager = BotManager(db)

    # Load preset personalities
    personalities = PersonalityEngine.get_preset_personalities()

    # Create Bot 1: Friendly Helper
    print("\n➕ Creating Bot 1: Friendly Helper...")
    bot1 = bot_manager.create_bot(
        name="FriendlyHelper",
        personality_config=personalities['friendly'].to_dict(),
        objectives_config=[
            {
                'type': 'engagement',
                'description': 'Engage with beginners',
                'target_keywords': ['beginner', 'help', 'new', 'started', 'learn'],
                'success_metrics': {'engagement_rate': 0.7},
                'constraints': {'keyword_match_threshold': 1},
                'priority': 8
            }
        ],
        credentials={
            'client_id': os.getenv('REDDIT_BOT1_CLIENT_ID'),
            'client_secret': os.getenv('REDDIT_BOT1_CLIENT_SECRET'),
            'username': os.getenv('REDDIT_BOT1_USERNAME'),
            'password': os.getenv('REDDIT_BOT1_PASSWORD'),
            'user_agent': 'Multi-Bot App v1.0'
        }
    )

    print(f"✅ Created: {bot1.name} ({bot1.id})")

    # Run bot for 1 cycle
    print(f"\n🚀 Running {bot1.name} for 1 cycle...")
    await bot_manager.process_bot_objectives(bot1.id, 'learnpython')

    # Show statistics
    print("\n📊 Bot Statistics:")
    stats = bot_manager.get_bot_statistics(bot1.id)

    print(f"   Total Actions: {stats['total_actions']}")
    print(f"   Successful: {stats['successful_actions']}")
    print(f"   Failed: {stats['failed_actions']}")
    print(f"   Success Rate: {stats['success_rate']:.2%}")

    if stats['action_breakdown']:
        print(f"   Action Breakdown:")
        for action, count in stats['action_breakdown'].items():
            print(f"      - {action}: {count}")

    print("\n✅ Bot manager demo complete!")

if __name__ == "__main__":
    asyncio.run(main())
```

## 🔍 Troubleshooting

### Issue: ModuleNotFoundError

```bash
# Solution: Install dependencies
pip install -r requirements_multibot.txt
```

### Issue: Reddit API Authentication Failed

```bash
# Check your credentials
python -c "import os; from dotenv import load_dotenv; load_dotenv(); print('Client ID:', os.getenv('REDDIT_BOT1_CLIENT_ID'))"

# Make sure .env file exists and has correct values
cat .env
```

### Issue: Tesseract Not Found

```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr

# macOS
brew install tesseract

# Windows
# Download from: https://github.com/UB-Mannheim/tesseract/wiki
```

### Issue: PyAutoGUI Failsafe Triggered

```
PyAutoGUI failsafe triggered - mouse moved to corner
```

**Solution**: This is a safety feature. Don't move mouse to top-left corner during automation.

### Issue: Permission Denied (Linux)

```bash
# Add user to input group (for keyboard/mouse control)
sudo usermod -a -G input $USER
# Logout and login again
```

## 📊 Monitoring Your Bots

### View Bot Activity

```python
from multi_bot_forum_app import Database

db = Database()
activities = db.get_bot_activities('bot_id_here', limit=50)

for activity in activities:
    print(f"{activity.timestamp}: {activity.action_type.value}")
    print(f"  Success: {activity.success}")
    if not activity.success:
        print(f"  Error: {activity.error_message}")
```

### Check Bot Status

```python
from multi_bot_forum_app import Database, BotManager

db = Database()
bot_manager = BotManager(db)

bots = bot_manager.list_bots()
for bot in bots:
    print(f"{bot.name}: {bot.status.value}")
    print(f"  Last active: {bot.last_active}")
```

## ⚙️ Configuration

### Adjust Rate Limits

Edit `config.yaml`:

```yaml
rate_limiting:
  global:
    requests_per_minute: 100  # Lower for safety
  per_bot:
    requests_per_minute: 20   # Lower for safety
```

### Adjust Bot Behavior

```yaml
bot_behavior:
  min_post_interval_seconds: 120  # 2 minutes between posts
  human_like_delays:
    enabled: true
    min_seconds: 3
    max_seconds: 10
```

## 🎯 Next Steps

1. **Start with Demo Mode**: Test without credentials
2. **Try API Mode**: Fast and reliable
3. **Experiment with Vision**: Most realistic but slower
4. **Create Multiple Bots**: Different personalities for different subreddits
5. **Monitor and Adjust**: Check logs and statistics

## 📚 More Examples

See the full documentation:
- `README_MULTIBOT.md` - Complete feature list
- `VISION_AUTOMATION_GUIDE.md` - Vision features
- `config.yaml` - All configuration options

## ⚠️ Important Reminders

1. **Respect Rate Limits**: Don't spam
2. **Follow Subreddit Rules**: Each community has rules
3. **Be Helpful**: Don't just promote
4. **Monitor Bots**: Check logs regularly
5. **Start Slow**: Test with one bot first

## 🆘 Getting Help

If you encounter issues:

1. Check the error message
2. Review troubleshooting section above
3. Check logs in `multi_bot_app.log`
4. Verify credentials in `.env`
5. Test internet connection

## 🚀 You're Ready!

Start with:
```bash
# Demo mode first
python multi_bot_forum_app.py

# Then try API mode
python run_api_bot.py

# Finally, vision mode
python run_vision_bot.py
```

Good luck with your bots! 🤖
