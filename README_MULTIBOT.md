# Multi-Bot Forum Interaction Application

A scalable, production-ready application for managing multiple bot personalities that can post on forums (Reddit, etc.) and interact with threads based on configurable objectives.

## Features

### Core Capabilities
- **Multiple Bot Personalities** - Manage unlimited bots with distinct personalities, tones, and styles
- **Objective-Driven Behavior** - Bots work toward specific goals (engagement, promotion, support, etc.)
- **Reddit Integration** - Full Reddit API support + browser automation for human-like behavior
- **Comprehensive Logging** - Track all bot activities, metrics, and performance
- **Production-Ready** - Rate limiting, circuit breakers, error handling, retry logic

### Advanced Features
- **Hybrid Interaction** - Uses both Reddit API and browser automation (Selenium)
- **Human-Like Behavior** - Random delays, browsing patterns, realistic interactions
- **Rate Limiting** - Multi-level rate limiting (global, per-bot, per-platform)
- **Circuit Breaker Pattern** - Fault tolerance for API failures
- **Database Persistence** - SQLite (easily upgradable to PostgreSQL)
- **Personality Engine** - Define formal/casual, technical/simple, enthusiastic/reserved bots
- **Content Generation** - Template-based (easily integrate with OpenAI/Anthropic)

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Bot Manager                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │   Bot 1     │  │   Bot 2     │  │   Bot 3     │         │
│  │ Personality │  │ Personality │  │ Personality │         │
│  │ Objectives  │  │ Objectives  │  │ Objectives  │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
└─────────────────────────────────────────────────────────────┘
                           │
          ┌────────────────┼────────────────┐
          │                │                │
   ┌──────▼──────┐  ┌─────▼──────┐  ┌─────▼──────┐
   │ Reddit API  │  │  Browser    │  │  Database  │
   │   Client    │  │ Automation  │  │  Layer     │
   └─────────────┘  └─────────────┘  └────────────┘
          │                │                │
          └────────────────┼────────────────┘
                           │
                    ┌──────▼──────┐
                    │   Reddit    │
                    │   Forums    │
                    └─────────────┘
```

## Installation

### Prerequisites
- Python 3.8+
- Chrome/Chromium browser (for browser automation)
- Reddit account(s) and API credentials

### Setup Steps

1. **Clone the repository**
```bash
cd /path/to/your/project
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Configure environment variables**
```bash
cp .env.example .env
# Edit .env with your Reddit credentials
```

4. **Get Reddit API Credentials**

For each bot, you need to create a Reddit app:
- Go to https://www.reddit.com/prefs/apps
- Click "create another app..."
- Select "script"
- Fill in name and redirect URI (http://localhost:8080)
- Copy `client_id` and `client_secret`

5. **Update configuration**
```bash
# Edit config.yaml with your preferences
nano config.yaml
```

## Usage

### Quick Start - Demo Mode

Run the application in demo mode with mock forum:

```bash
python multi_bot_forum_app.py
```

This will:
- Create 3 demo bots with different personalities
- Simulate forum interactions
- Generate logs and statistics

### Production Mode - Reddit Integration

```python
import asyncio
from reddit_browser_integration import HybridRedditManager
from multi_bot_forum_app import BotManager, Database

async def main():
    # Initialize
    db = Database()
    bot_manager = BotManager(db)

    # Create a bot
    bot = bot_manager.create_bot(
        name="PythonHelper",
        personality_config={
            'description': 'Helpful Python expert',
            'tone': 'friendly',
            'formality': 0.5,
            'enthusiasm': 0.7,
            'technical_level': 0.8,
            'verbosity': 0.6,
            'empathy': 0.8,
            'vocabulary_style': 'professional',
            'response_patterns': ['Happy to help!', 'Great question!']
        },
        objectives_config=[{
            'type': 'support',
            'description': 'Help Python beginners',
            'target_keywords': ['python', 'help', 'error', 'beginner'],
            'success_metrics': {},
            'constraints': {'keyword_match_threshold': 1},
            'priority': 9
        }],
        credentials={
            'client_id': 'your_client_id',
            'client_secret': 'your_client_secret',
            'username': 'your_username',
            'password': 'your_password',
            'user_agent': 'MyBot v1.0'
        }
    )

    # Process objectives for a subreddit
    await bot_manager.process_bot_objectives(bot.id, 'learnpython')

if __name__ == "__main__":
    asyncio.run(main())
```

### Creating Bots

#### Personality Configuration

Personalities control how your bot communicates:

```python
personality_config = {
    'description': 'Professional expert',
    'tone': 'professional',      # formal, casual, friendly, technical
    'formality': 0.8,            # 0.0 (casual) to 1.0 (formal)
    'enthusiasm': 0.5,           # 0.0 (reserved) to 1.0 (enthusiastic)
    'technical_level': 0.7,      # 0.0 (simple) to 1.0 (technical)
    'verbosity': 0.6,            # 0.0 (concise) to 1.0 (verbose)
    'empathy': 0.7,              # 0.0 (factual) to 1.0 (empathetic)
    'vocabulary_style': 'professional',
    'response_patterns': [
        'In my experience,',
        'I would recommend',
    ]
}
```

#### Preset Personalities

```python
from multi_bot_forum_app import PersonalityEngine

personalities = PersonalityEngine.get_preset_personalities()
# Available: 'professional', 'friendly', 'technical', 'casual'
```

#### Objective Types

```python
ObjectiveType.ENGAGEMENT    # Maximize engagement and discussion
ObjectiveType.PROMOTION     # Promote specific topics/products
ObjectiveType.INFORMATION   # Share information and education
ObjectiveType.MONITORING    # Monitor and respond to keywords
ObjectiveType.SUPPORT       # Provide support and help
```

### Browser Automation vs API

The application intelligently uses both:

**Reddit API (PRAW)**
- Faster
- More reliable
- Better for reading/monitoring
- Lower detection risk

**Browser Automation (Selenium)**
- More human-like
- Can perform visual actions
- Handles complex interactions
- Better for stealth operations

Configure in `config.yaml`:
```yaml
browser:
  headless: true           # Run without visible browser
  enabled: true            # Enable browser automation

bot_behavior:
  human_like_delays:
    enabled: true          # Add random delays
    min_seconds: 2
    max_seconds: 8
```

## Configuration

### Rate Limiting

Protect your bots from being rate-limited or banned:

```yaml
rate_limiting:
  global:
    requests_per_minute: 100
  per_bot:
    requests_per_minute: 20
  reddit:
    requests_per_minute: 60    # Reddit's limit
```

### Bot Behavior

```yaml
bot_behavior:
  min_post_interval_seconds: 60    # Minimum time between posts
  max_post_length: 5000            # Maximum post length
  context_window: 10               # Previous messages to consider
  human_like_delays:
    enabled: true
    min_seconds: 2
    max_seconds: 8
  random_browsing:
    enabled: true
    probability: 0.3               # 30% chance to browse before posting
```

### Circuit Breaker

Prevent cascading failures:

```yaml
circuit_breaker:
  failure_threshold: 5             # Open after N failures
  timeout_seconds: 60              # Wait before retry
  half_open_max_calls: 3           # Test calls in half-open state
```

## Database Schema

### Bots Table
- Bot information, personality, objectives
- Status tracking
- Credentials (encrypted)

### Threads Table
- Forum thread information
- Metadata (scores, comments, etc.)
- Last checked timestamp

### Posts Table
- Bot posts history
- Status (pending, posted, failed)
- Associated thread and objective

### Activity Logs Table
- All bot actions
- Success/failure tracking
- Error messages
- Timestamps for analytics

## Monitoring & Analytics

### View Bot Statistics

```python
stats = bot_manager.get_bot_statistics(bot_id)
print(stats)
# {
#     'total_actions': 150,
#     'successful_actions': 142,
#     'failed_actions': 8,
#     'success_rate': 0.947,
#     'action_breakdown': {
#         'reply_to_thread': 45,
#         'create_post': 12,
#         'like_post': 85
#     },
#     'last_activity': '2025-01-15T10:30:00'
# }
```

### Activity Logs

```python
activities = database.get_bot_activities(bot_id, limit=100)
for activity in activities:
    print(f"{activity.timestamp}: {activity.action_type} - {activity.success}")
```

## Production Deployment

### Scaling Considerations

1. **Multiple Bot Instances**
   - Run different bots on different machines/containers
   - Share database (upgrade to PostgreSQL)
   - Centralized logging (ELK stack)

2. **Database**
   ```yaml
   database:
     type: "postgresql"
     host: "db.example.com"
     port: 5432
     database: "multi_bot_db"
     user: "bot_user"
     password: "secure_password"
   ```

3. **Credential Management**
   - Use AWS Secrets Manager or HashiCorp Vault
   - Rotate credentials regularly
   - Enable encryption at rest

4. **Monitoring**
   - Set up Prometheus + Grafana
   - Alert on failure rates
   - Monitor rate limit usage

5. **Rate Limiting**
   - Use Redis for distributed rate limiting
   - Implement backoff strategies
   - Monitor API quotas

### Docker Deployment

```dockerfile
FROM python:3.9-slim

# Install Chrome for Selenium
RUN apt-get update && apt-get install -y \
    chromium \
    chromium-driver \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "multi_bot_forum_app.py"]
```

## Security Best Practices

1. **Never Commit Credentials**
   - Use `.env` files (in `.gitignore`)
   - Use secrets management systems

2. **Rotate Credentials**
   - Set expiry dates
   - Automate rotation

3. **Rate Limiting**
   - Respect platform limits
   - Implement exponential backoff

4. **Error Handling**
   - Never expose credentials in logs
   - Sanitize error messages

5. **Browser Fingerprinting**
   - Use residential proxies
   - Rotate user agents
   - Clear cookies periodically

## Troubleshooting

### Common Issues

**Rate Limiting Errors**
```
Solution: Reduce requests_per_minute in config.yaml
Check: Reddit allows 60 requests/minute
```

**Authentication Failures**
```
Solution: Verify credentials in .env file
Check: Reddit app credentials at reddit.com/prefs/apps
```

**Browser Automation Fails**
```
Solution: Ensure Chrome/Chromium is installed
Run: python -m webdriver_manager.chrome
```

**Database Locked**
```
Solution: Only one process can write to SQLite at a time
Upgrade to PostgreSQL for concurrent access
```

## API Integration

### OpenAI Integration (Optional)

For AI-powered content generation:

```python
import openai

openai.api_key = os.getenv('OPENAI_API_KEY')

def generate_ai_content(prompt, personality):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": f"You are a {personality.description}"},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )
    return response.choices[0].message.content
```

### Anthropic Integration (Optional)

```python
from anthropic import Anthropic

client = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))

def generate_ai_content(prompt, personality):
    response = client.messages.create(
        model="claude-3-sonnet-20240229",
        max_tokens=500,
        messages=[{
            "role": "user",
            "content": prompt
        }]
    )
    return response.content[0].text
```

## Contributing

This is a production-ready foundation. Recommended enhancements:

1. **AI Integration** - Connect OpenAI/Anthropic for dynamic content
2. **More Platforms** - Add Twitter, Discord, Telegram support
3. **ML Models** - Train models on successful interactions
4. **A/B Testing** - Test different personalities and strategies
5. **Dashboard** - Web UI for bot management
6. **Scheduling** - Cron-like scheduling for bot activities

## License

MIT License - Use at your own risk. Ensure compliance with platform ToS.

## Disclaimer

This tool is for educational and authorized use only. Users are responsible for:
- Complying with Reddit's Terms of Service
- Following platform rules and guidelines
- Not engaging in spam or manipulation
- Respecting rate limits and community standards

Misuse of this tool may result in account bans or legal consequences.

## Support

For issues, questions, or contributions:
1. Check existing documentation
2. Review common troubleshooting steps
3. Check platform API documentation
4. Ensure credentials are correctly configured

---

**Built with Production-Ready Best Practices:**
- ✅ Modular architecture
- ✅ Comprehensive error handling
- ✅ Rate limiting and circuit breakers
- ✅ Database persistence
- ✅ Detailed logging
- ✅ Security considerations
- ✅ Scalable design
- ✅ Human-like behavior simulation
