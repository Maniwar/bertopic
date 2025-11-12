# Vision-Based Desktop Automation Guide

## Overview

This enhanced version of the Multi-Bot Forum Application uses **computer vision and AI** to interact with Reddit (and other forums) through the browser UI **exactly like a real human user**.

Instead of relying on Selenium selectors (which can be detected), this system:
- ✅ Uses **computer vision (OCR)** to "see" the screen
- ✅ Uses **AI vision models** (GPT-4 Vision / Claude Vision) to understand UI layouts
- ✅ Moves the mouse with **Bezier curves** for natural movement
- ✅ Types with **human-like variations** (typos, corrections, variable speed)
- ✅ Takes **screenshots** and analyzes them in real-time
- ✅ Simulates **realistic browsing behavior** (scrolling, reading, clicking)

## Key Features

### 🖱️ Human-Like Mouse Movement
- **Bezier Curve Paths**: Mouse moves along realistic curves, not straight lines
- **Variable Speed**: Speed varies based on distance and context
- **Micro-movements**: Small jitters and corrections like real humans
- **Reaction Time**: Delays simulate human thinking/reaction time

### ⌨️ Human-Like Keyboard Typing
- **Variable WPM**: Typing speed varies (40-80 WPM)
- **Typos & Corrections**: Simulates mistakes and backspacing
- **Punctuation Pauses**: Longer delays after periods, commas
- **Thinking Pauses**: Random pauses mid-sentence
- **Nearby Key Errors**: Realistic typos (e.g., "teh" instead of "the")

### 👁️ Computer Vision
- **OCR (Optical Character Recognition)**: Reads text from screenshots using EasyOCR
- **Template Matching**: Finds UI elements by image similarity
- **Text Search**: Locates buttons, links, text on screen
- **Non-Max Suppression**: Removes duplicate detections

### 🤖 AI Vision Integration
- **GPT-4 Vision**: Understands complex UI layouts
- **Claude Vision**: Alternative AI vision model
- **Context Understanding**: AI describes what it sees and where elements are
- **Adaptive Navigation**: Handles dynamic/changing UIs

### 📸 Screenshot Analysis
- **Real-time Capture**: Takes screenshots of screen or regions
- **Element Detection**: Finds buttons, text fields, links
- **Bounding Boxes**: Annotates screenshots with detected elements
- **Visual Debugging**: Saves annotated images for analysis

## Installation

### Additional Requirements

Beyond the base requirements, install vision dependencies:

```bash
pip install -r requirements_multibot.txt
```

### System Dependencies

**For OCR (Tesseract):**

**Windows:**
```bash
# Download installer from: https://github.com/UB-Mannheim/tesseract/wiki
# Install and add to PATH
```

**Linux:**
```bash
sudo apt-get install tesseract-ocr
```

**macOS:**
```bash
brew install tesseract
```

**For Chrome (if not installed):**
- Windows: Download from google.com/chrome
- Linux: `sudo apt-get install chromium-browser`
- macOS: `brew install --cask google-chrome`

### API Keys

For AI vision capabilities, set environment variables:

```bash
# .env file
OPENAI_API_KEY=your_openai_key_here          # For GPT-4 Vision
ANTHROPIC_API_KEY=your_anthropic_key_here    # For Claude Vision
```

## How It Works

### 1. Vision-Based Navigation

Instead of using Selenium CSS selectors:

```python
# OLD WAY (Selenium - detectable)
button = driver.find_element(By.CSS_SELECTOR, ".login-button")
button.click()

# NEW WAY (Vision - undetectable)
screenshot = vision.capture_screenshot()
login_buttons = vision.find_text_on_screen("Log In", screenshot)
await mouse.click(*login_buttons[0].center)
```

### 2. AI-Powered UI Understanding

When the UI changes or is complex:

```python
# Ask AI what it sees
screenshot = vision.capture_screenshot()
response = await vision.analyze_with_ai_vision(
    screenshot,
    "Where is the comment input box on this Reddit post?"
)
# AI: "The comment box is in the center-left area, below the post content..."
```

### 3. Human-Like Mouse Movement

```python
# Move to a button with Bezier curve
await mouse.move_to(
    x=500, y=300,
    duration=None,      # Random duration
    human_like=True     # Use Bezier curve
)

# The mouse will:
# 1. Take a curved path (not straight line)
# 2. Vary speed (faster in middle, slower at ends)
# 3. Add micro-movements at the end
# 4. Have realistic acceleration
```

### 4. Human-Like Typing

```python
# Type with human characteristics
await keyboard.type_text(
    "This is my comment on the thread.",
    wpm=None,           # Random 40-80 WPM
    error_rate=0.02     # 2% typo rate
)

# The typing will:
# 1. Vary speed per character
# 2. Make occasional typos and fix them
# 3. Pause longer after punctuation
# 4. Have random "thinking" pauses
```

## Usage Examples

### Basic Vision Bot

```python
from vision_desktop_automation import VisionBasedRedditBot
from multi_bot_forum_app import Bot, BotStatus, Personality

# Create bot
bot = Bot(
    id="vision_bot_1",
    name="VisionBot",
    personality=create_personality(...),
    objectives=[...],
    status=BotStatus.ACTIVE,
    credentials={
        'username': 'my_reddit_username',
        'password': 'my_password'
    },
    metadata={},
    created_at=datetime.now(),
    last_active=datetime.now()
)

# Initialize vision bot
config = {
    'openai': {'enabled': True},
    'anthropic': {'enabled': False}
}

vision_bot = VisionBasedRedditBot(bot, config)

# Open browser and login
await vision_bot.initialize_browser()
await vision_bot.login_to_reddit()

# Navigate to subreddit
await vision_bot.navigate_to_subreddit('learnpython')

# Browse naturally for 5 minutes
await vision_bot.browse_like_human(duration_minutes=5)

# Find and comment on a thread
await vision_bot.read_thread_and_post_comment(
    thread_title="How do I get started with Python?",
    comment_text="Great question! I'd recommend..."
)
```

### Advanced: Custom Vision Tasks

```python
from vision_desktop_automation import VisionAnalyzer, HumanMouseController

vision = VisionAnalyzer(config)
mouse = HumanMouseController()

# Take screenshot
screenshot = vision.capture_screenshot()

# Find specific text
elements = vision.find_text_on_screen("Comment", screenshot)

# Use AI to understand layout
ai_response = await vision.analyze_with_ai_vision(
    screenshot,
    "List all the post titles visible on this Reddit page"
)

# Click on first match with human-like movement
if elements:
    await mouse.click(*elements[0].center)
```

### Find UI Elements by Image

```python
# Save a screenshot of a button/icon you want to find
# Then use template matching

vision = VisionAnalyzer(config)
matches = vision.find_image_on_screen(
    template_path="./templates/upvote_button.png",
    threshold=0.8
)

if matches:
    await mouse.click(*matches[0].center)
```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│           Vision-Based Automation Layer                  │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  ┌────────────────┐  ┌──────────────┐  ┌─────────────┐ │
│  │ Human Mouse    │  │ Human        │  │  Vision     │ │
│  │ Controller     │  │ Keyboard     │  │  Analyzer   │ │
│  │                │  │ Controller   │  │             │ │
│  │ • Bezier curves│  │ • Variable   │  │ • OCR       │ │
│  │ • Jitter       │  │   WPM        │  │ • AI Vision │ │
│  │ • Delays       │  │ • Typos      │  │ • Templates │ │
│  └────────────────┘  └──────────────┘  └─────────────┘ │
│                                                           │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
            ┌─────────────────────────┐
            │   Browser UI (Reddit)    │
            │   • No Selenium markers  │
            │   • No automation flags  │
            │   • Real user behavior   │
            └─────────────────────────┘
```

## Anti-Detection Features

### What Makes This Undetectable

1. **No Selenium Markers**
   - Doesn't use WebDriver
   - No `navigator.webdriver` flag
   - No automation extensions

2. **Human-Like Behavior**
   - Natural mouse movements (Bezier curves)
   - Variable typing speed with errors
   - Realistic delays and pauses
   - Random browsing patterns

3. **Computer Vision**
   - Doesn't query DOM directly
   - Sees the page like a human
   - Works even if HTML structure changes

4. **AI Understanding**
   - Adapts to UI changes
   - Understands context
   - Natural language reasoning

### Detection Risks (Minimal)

While this approach is much harder to detect, platforms can still:
- Track unusual usage patterns (posting frequency, timing)
- Detect if you use the same IP for multiple accounts
- Flag suspicious content patterns

**Mitigation:**
- Use rate limiting (built into the app)
- Rotate IPs/use proxies
- Vary bot personalities and posting patterns
- Follow human activity schedules

## Configuration

### Enable AI Vision

```yaml
# config.yaml
openai:
  enabled: true
  model: "gpt-4-vision-preview"

anthropic:
  enabled: false
  model: "claude-3-sonnet-20240229"
```

### Adjust Human Behavior

```python
# Mouse movement
await mouse.move_to(x, y, duration=0.5)  # Faster
await mouse.move_to(x, y, duration=2.0)  # Slower

# Typing speed
await keyboard.type_text(text, wpm=80)   # Fast typer
await keyboard.type_text(text, wpm=40)   # Slow typer

# Error rate
await keyboard.type_text(text, error_rate=0.0)   # Perfect typing
await keyboard.type_text(text, error_rate=0.05)  # 5% errors
```

## Vision Capabilities

### OCR Text Detection

```python
# Find any text on screen
elements = vision.find_text_on_screen("Login")
elements = vision.find_text_on_screen("Submit Comment")
elements = vision.find_text_on_screen("r/learnpython")

# Case-sensitive search
elements = vision.find_text_on_screen("Python", case_sensitive=True)
```

### Template Matching

```python
# Find UI element by image
# 1. Take screenshot of element
# 2. Save as template
# 3. Match on screen

matches = vision.find_image_on_screen(
    "templates/upvote_arrow.png",
    threshold=0.8  # 80% similarity
)
```

### AI Vision Queries

```python
# Ask AI to find elements
response = await vision.analyze_with_ai_vision(
    screenshot,
    "Where is the username field?"
)

response = await vision.analyze_with_ai_vision(
    screenshot,
    "List all post titles on this page"
)

response = await vision.analyze_with_ai_vision(
    screenshot,
    "Is there a 'Load More' button? Where is it?"
)
```

## Complete Workflow Example

```python
async def automated_reddit_workflow():
    # 1. Create bot
    bot = create_bot_with_personality(...)

    # 2. Initialize vision bot
    vision_bot = VisionBasedRedditBot(bot, config)

    # 3. Open browser (looks like human opening browser)
    await vision_bot.initialize_browser()

    # 4. Login (types like human, moves mouse naturally)
    success = await vision_bot.login_to_reddit()

    if not success:
        return

    # 5. Navigate to target subreddit
    await vision_bot.navigate_to_subreddit('learnpython')

    # 6. Browse naturally (scroll, read, click posts)
    await vision_bot.browse_like_human(duration_minutes=5)

    # 7. Find threads matching bot's objectives
    screenshot = vision_bot.vision.capture_screenshot()

    # Use AI to find relevant threads
    ai_response = await vision_bot.vision.analyze_with_ai_vision(
        screenshot,
        "List all threads about Python beginners asking for help"
    )

    # 8. Engage with a thread
    await vision_bot.read_thread_and_post_comment(
        thread_title="How do I learn Python?",
        comment_text="Welcome! Python is great for beginners..."
    )

    # 9. Continue browsing to look natural
    await vision_bot.browse_like_human(duration_minutes=2)

    # 10. Close gracefully
    pyautogui.hotkey('alt', 'f4')
```

## Performance Considerations

### Speed vs. Human-Like Behavior

The more human-like the behavior, the slower the execution:

**Fast Mode** (less human-like):
```python
await mouse.move_to(x, y, duration=0.1, human_like=False)
await keyboard.type_text(text, wpm=100, error_rate=0.0)
```

**Slow Mode** (more human-like):
```python
await mouse.move_to(x, y, duration=None, human_like=True)  # Random 0.5-2s
await keyboard.type_text(text, wpm=50, error_rate=0.02)   # With typos
```

**Recommendation**: Use realistic settings (40-80 WPM, error_rate=0.01-0.03)

### Resource Usage

- **Screenshots**: ~2-5 MB per screenshot (RGB)
- **AI Vision**: Requires API calls ($0.01-0.03 per image)
- **OCR**: CPU-intensive (use regions when possible)

**Optimization:**
```python
# Capture specific regions instead of full screen
region = (0, 0, 1920, 500)  # Top portion only
screenshot = vision.capture_screenshot(region=region)
```

## Troubleshooting

### OCR Not Finding Text

**Problem**: `find_text_on_screen()` returns empty list

**Solutions:**
1. Check text is actually visible on screen
2. Try case-insensitive search
3. Use partial text matching
4. Take screenshot and verify: `vision.save_screenshot_with_annotations(...)`
5. Adjust OCR language if not English

### AI Vision Errors

**Problem**: API errors or timeout

**Solutions:**
1. Check API key is set: `echo $OPENAI_API_KEY`
2. Verify API quota/billing
3. Reduce image size before sending
4. Add retry logic with backoff

### Mouse Click Misses Target

**Problem**: Clicks in wrong location

**Solutions:**
1. Verify element coordinates with annotated screenshot
2. Add delay before click: `await asyncio.sleep(0.5)`
3. Use smaller regions for faster/accurate detection
4. Check screen scaling settings (should be 100%)

### Bot Detected / Account Banned

**Problem**: Reddit flags the account

**Solutions:**
1. Reduce posting frequency
2. Increase random delays
3. Use residential proxies
4. Vary activities (browse, upvote, comment ratio)
5. Follow subreddit rules
6. Don't spam identical content

## Best Practices

### 1. Always Browse First

```python
# DON'T immediately post after login
await login()
await post_comment()  # ❌ Suspicious

# DO browse naturally first
await login()
await browse_like_human(5)  # ✅ Looks natural
await post_comment()
```

### 2. Vary Your Behavior

```python
# Use random values
wpm = random.randint(40, 80)
browse_time = random.randint(3, 10)
scroll_amount = random.randint(200, 600)
```

### 3. Respect Rate Limits

```python
# Built-in rate limiting from main app
await rate_limiter.acquire(bot.id)
await post_comment(...)

# Also add human delays
await asyncio.sleep(random.uniform(60, 180))  # 1-3 mins between posts
```

### 4. Handle Errors Gracefully

```python
try:
    await vision_bot.post_comment(...)
except Exception as e:
    logger.error(f"Failed to post: {e}")
    # Don't retry immediately
    await asyncio.sleep(300)  # Wait 5 minutes
```

### 5. Monitor Your Bots

```python
# Log all actions
logger.info(f"Bot {bot.name} posted comment at {datetime.now()}")

# Track metrics
stats = bot_manager.get_bot_statistics(bot.id)
if stats['success_rate'] < 0.8:
    logger.warning(f"Bot {bot.name} has low success rate!")
```

## Advanced Features

### Custom Template Library

Build a library of UI element templates:

```
templates/
├── reddit/
│   ├── upvote_button.png
│   ├── downvote_button.png
│   ├── comment_button.png
│   ├── reply_button.png
│   └── save_button.png
└── discord/
    ├── send_button.png
    └── emoji_picker.png
```

Usage:
```python
upvote_matches = vision.find_image_on_screen("templates/reddit/upvote_button.png")
```

### Multi-Monitor Support

```python
from mss import mss

with mss() as sct:
    # List all monitors
    for i, monitor in enumerate(sct.monitors):
        print(f"Monitor {i}: {monitor}")

    # Capture specific monitor
    screenshot = sct.grab(sct.monitors[2])  # Second monitor
```

### Record and Replay

```python
# Record a sequence of actions
actions = []

# Move mouse
actions.append(('move', 500, 300))
# Click
actions.append(('click', 500, 300))
# Type
actions.append(('type', 'Hello world'))

# Replay actions
for action in actions:
    if action[0] == 'move':
        await mouse.move_to(action[1], action[2])
    elif action[0] == 'click':
        await mouse.click(action[1], action[2])
    elif action[0] == 'type':
        await keyboard.type_text(action[1])
```

## Security & Ethics

### ⚠️ Important Disclaimers

1. **Respect Terms of Service**: Most platforms prohibit automation
2. **Don't Spam**: Respect communities and their rules
3. **Be Transparent**: Consider disclosing bot nature when appropriate
4. **Rate Limit**: Don't overwhelm servers
5. **Privacy**: Don't scrape personal information
6. **Legal**: Ensure compliance with local laws

### Ethical Use Cases

✅ **Acceptable:**
- Automated moderation for your own subreddit
- Research with permission
- Personal automation for your own account
- Educational purposes

❌ **Not Acceptable:**
- Spam/manipulation
- Vote brigading
- Astroturfing
- Harassment
- Ban evasion

## Conclusion

Vision-based desktop automation provides the most realistic bot behavior possible. By "seeing" the screen like a human and moving the mouse naturally, these bots are extremely difficult to detect.

**Key Advantages:**
- ✅ No Selenium markers or automation flags
- ✅ Works even when UI changes
- ✅ Natural, human-like behavior
- ✅ AI-powered adaptability

**Key Limitations:**
- ⏱️ Slower than API-based automation
- 💰 Requires AI API costs for vision features
- 🖥️ Requires active desktop (can't run headless easily)
- 📊 More resource-intensive

For production use, consider hybrid approach:
- Use API for reading/monitoring (fast, efficient)
- Use vision for posting/commenting (realistic, undetectable)

---

**Remember**: With great power comes great responsibility. Use this tool ethically and legally.
