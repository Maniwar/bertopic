# Multi-Bot Forum Application - Getting Started

**A production-ready, vision-powered bot system for automated forum interactions**

---

## 🚀 Quick Start (3 Steps)

### 1. Install

```bash
# Run setup script
chmod +x setup.sh
./setup.sh

# Or manually:
pip install -r requirements_multibot.txt
```

### 2. Configure

```bash
# Set up credentials
cp .env.example .env
nano .env  # Add your Reddit API credentials
```

Get Reddit credentials: https://www.reddit.com/prefs/apps

### 3. Run

```bash
# Demo mode (no credentials needed)
python multi_bot_forum_app.py

# Reddit API mode
python run_api_bot.py

# Vision mode (human-like)
python run_vision_bot.py

# Full bot manager
python run_full_bot_manager.py
```

---

## 📁 Project Structure

```
bertopic/
├── multi_bot_forum_app.py          # Core bot system
├── reddit_browser_integration.py   # Reddit API + Selenium
├── vision_desktop_automation.py    # Vision-based automation ⭐
│
├── run_api_bot.py                  # Example: API mode
├── run_vision_bot.py               # Example: Vision mode
├── run_full_bot_manager.py         # Example: Full manager
│
├── requirements_multibot.txt       # Python dependencies
├── config.yaml                     # Configuration
├── .env.example                    # Credentials template
├── setup.sh                        # Setup script
│
├── QUICKSTART.md                   # Quick start guide ← START HERE
├── README_MULTIBOT.md              # Full documentation
├── VISION_AUTOMATION_GUIDE.md      # Vision features
└── README_MULTIBOT_START.md        # This file
```

---

## 🎯 What Can This Do?

### Core Features
✅ **Multiple Bot Personalities** - Friendly, Professional, Technical, Casual
✅ **Objective-Driven** - Engagement, Promotion, Support, Information
✅ **Reddit Integration** - Full API support + browser automation
✅ **Vision-Based** - Uses computer vision to interact like a human
✅ **Production-Ready** - Rate limiting, error handling, logging

### Advanced Features
✅ **AI Vision** - GPT-4 Vision / Claude Vision for UI understanding
✅ **Human-Like Behavior** - Bezier curves, typos, realistic timing
✅ **OCR Detection** - Finds UI elements visually
✅ **Template Matching** - Image-based element detection
✅ **Anti-Detection** - No Selenium markers, natural behavior

---

## 📖 Documentation

| Document | Purpose | Read When |
|----------|---------|-----------|
| **QUICKSTART.md** | Step-by-step setup | First time setup |
| **README_MULTIBOT.md** | Complete documentation | Learning all features |
| **VISION_AUTOMATION_GUIDE.md** | Vision automation details | Using vision mode |
| **README_MULTIBOT_START.md** | This file | Quick orientation |

---

## 🎬 Usage Examples

### Example 1: Simple API Bot

```python
from run_api_bot import main
import asyncio

# Fetches threads from r/learnpython
# Shows which ones match bot objectives
# Demonstrates what bot would comment
asyncio.run(main())
```

**Output:**
```
🤖 Reddit API Bot - Starting...
✅ Authentication successful!
📖 Fetching threads from r/learnpython...
✅ Found 10 threads

1. How do I learn Python?
   Score: 45 ⬆ | Comments: 12 💬
   🎯 MATCH! Keywords: python, learn, help
```

### Example 2: Vision-Based Bot

```python
from run_vision_bot import main
import asyncio

# Opens browser like a human
# Logs in with realistic typing
# Browses naturally (scroll, read, click)
# Uses computer vision to find elements
asyncio.run(main())
```

**What happens:**
- Mouse moves along Bezier curves (natural paths)
- Types with variable speed and occasional typos
- Scrolls with realistic rhythm
- Pauses to "read" content
- No Selenium markers - undetectable!

### Example 3: Full Bot Manager

```python
from run_full_bot_manager import main
import asyncio

# Creates 3 bots with different personalities
# Assigns objectives to each
# Runs them all concurrently
# Shows statistics and activity logs
asyncio.run(main())
```

---

## 🔧 Configuration

### Bot Personality

```python
personality_config = {
    'description': 'Friendly helper',
    'tone': 'friendly',
    'formality': 0.5,      # 0=casual, 1=formal
    'enthusiasm': 0.7,     # 0=reserved, 1=enthusiastic
    'technical_level': 0.6,
    'verbosity': 0.5,
    'empathy': 0.8,
}
```

### Bot Objectives

```python
objectives_config = [{
    'type': 'support',
    'description': 'Help Python learners',
    'target_keywords': ['python', 'help', 'beginner'],
    'priority': 9
}]
```

### Rate Limiting

```yaml
# config.yaml
rate_limiting:
  global:
    requests_per_minute: 100
  per_bot:
    requests_per_minute: 20
```

---

## 🛡️ Anti-Detection Features

### Why This Is Hard to Detect

| Feature | Traditional Bot | Vision Bot |
|---------|----------------|------------|
| Mouse Movement | Instant clicks | Bezier curves |
| Typing | Instant input | Variable WPM + typos |
| Selenium Markers | ✅ Present | ❌ None |
| Timing | Consistent | Random delays |
| Behavior | Scripted | Human-like |
| UI Detection | CSS selectors | Computer vision |

### Vision Mode Advantages

```python
# Traditional (detectable)
driver.find_element(By.ID, "login").click()

# Vision (undetectable)
screenshot = vision.capture_screenshot()
buttons = vision.find_text_on_screen("Login")
await mouse.click(*buttons[0].center)  # Natural movement
```

---

## ⚠️ Important Notes

### Ethics & Legal
- ✅ Educational/research use
- ✅ Automating your own content
- ✅ Authorized testing
- ❌ Spam or manipulation
- ❌ Ban evasion
- ❌ Terms of Service violations

### Best Practices
1. **Start Slow** - Test with one bot first
2. **Respect Limits** - Use built-in rate limiting
3. **Monitor Logs** - Check `multi_bot_app.log`
4. **Follow Rules** - Respect subreddit guidelines
5. **Be Helpful** - Provide value, not spam

---

## 🐛 Troubleshooting

### "ModuleNotFoundError"
```bash
pip install -r requirements_multibot.txt
```

### "Reddit authentication failed"
```bash
# Check credentials
cat .env
# Make sure client_id, client_secret, username, password are set
```

### "Tesseract not found"
```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr

# macOS
brew install tesseract
```

### "PyAutoGUI failsafe triggered"
Don't move mouse to top-left corner during vision automation

---

## 📊 Features Comparison

| Feature | API Mode | Browser Mode | Vision Mode |
|---------|----------|--------------|-------------|
| Speed | ⚡⚡⚡ Fast | ⚡⚡ Medium | ⚡ Slow |
| Detection Risk | 🟡 Medium | 🟠 High | 🟢 Very Low |
| Setup | ✅ Easy | ✅ Easy | 🔧 Moderate |
| Reliability | ⭐⭐⭐ | ⭐⭐ | ⭐⭐ |
| Human-Like | ❌ | 🟡 | ✅ |
| Cost | Free | Free | AI API costs |

**Recommendation:**
- **Development/Testing**: API mode
- **Reading/Monitoring**: API mode
- **Posting/Commenting**: Vision mode
- **Production**: Hybrid (API + Vision)

---

## 🎓 Learning Path

1. **Day 1**: Run demo mode, understand bot personalities
2. **Day 2**: Try API mode with your Reddit credentials
3. **Day 3**: Experiment with vision mode (watch it work!)
4. **Day 4**: Create custom bots with different objectives
5. **Day 5**: Deploy to production with monitoring

---

## 🚀 Next Steps

After basic setup:

1. **Customize Personalities**
   - Edit personality traits in run scripts
   - Test different tones and styles

2. **Set Up Objectives**
   - Define target keywords
   - Set priorities
   - Configure constraints

3. **Deploy Multiple Bots**
   - Different bots for different subreddits
   - Varied personalities to avoid patterns
   - Staggered scheduling

4. **Monitor & Optimize**
   - Check success rates
   - Review activity logs
   - Adjust based on performance

5. **Advanced Features**
   - Integrate OpenAI for content generation
   - Set up scheduled runs (cron)
   - Build dashboard for monitoring
   - Add more platforms (Twitter, Discord)

---

## 🆘 Getting Help

1. Check documentation (QUICKSTART.md, README_MULTIBOT.md)
2. Review example scripts (run_*.py)
3. Check logs (multi_bot_app.log)
4. Test in demo mode first

---

## 📚 Additional Resources

### Reddit API
- API Credentials: https://www.reddit.com/prefs/apps
- PRAW Docs: https://praw.readthedocs.io/
- Reddit API Rules: https://www.reddit.com/wiki/api

### Computer Vision
- Tesseract: https://github.com/tesseract-ocr/tesseract
- EasyOCR: https://github.com/JaidedAI/EasyOCR
- OpenCV: https://opencv.org/

### AI Vision
- GPT-4 Vision: https://platform.openai.com/docs/guides/vision
- Claude Vision: https://www.anthropic.com/claude

---

## ✅ You're Ready!

```bash
# Start with demo mode
python multi_bot_forum_app.py

# Then try API mode
python run_api_bot.py

# Finally, vision mode
python run_vision_bot.py
```

**Good luck with your bots!** 🤖

---

*For detailed documentation, see:*
- *QUICKSTART.md - Setup guide*
- *README_MULTIBOT.md - Full features*
- *VISION_AUTOMATION_GUIDE.md - Vision details*
