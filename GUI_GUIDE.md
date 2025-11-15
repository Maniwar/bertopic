# Multi-Bot Web GUI Guide

## 🎨 Overview

The Multi-Bot Control Center is a modern web-based interface for managing your bot fleet, creating missions, and monitoring activity.

Built with **Streamlit** for a clean, intuitive experience.

---

## 🚀 Quick Start

### Launch the GUI

```bash
streamlit run bot_control_gui.py
```

The GUI will open in your browser at `http://localhost:8501`

---

## 📱 Interface Overview

### Main Pages

1. **📊 Dashboard** - Overview of all bots and recent activity
2. **🤖 Bot Manager** - Create, edit, and manage bots
3. **🎯 Missions** - Create coordinated multi-bot missions
4. **📈 Analytics** - View performance charts and statistics
5. **⚙️ Settings** - Configure credentials and settings

---

## 🤖 Bot Manager

### Creating a Bot

1. Navigate to **Bot Manager** → **Create Bot** tab
2. Fill in bot details:
   - **Name**: Unique identifier for your bot
   - **Personality Preset**: Choose from friendly, professional, technical, casual, or custom
   - **Objective Type**: Select bot's goal (engagement, support, information, etc.)
   - **Keywords**: Comma-separated trigger keywords
   - **Credentials**: Reddit API credentials (optional for demo mode)

3. Click **Create Bot**

### Personality Presets

| Preset | Description | Best For |
|--------|-------------|----------|
| **Friendly** | Warm, enthusiastic, casual | Welcoming beginners |
| **Professional** | Formal, measured, authoritative | Business/technical discussions |
| **Technical** | Highly technical, detailed | Expert-level topics |
| **Casual** | Laid-back, simple language | Informal conversations |
| **Custom** | Full control over all parameters | Specific use cases |

### Custom Personality

If you select "Custom", you can fine-tune:

- **Formality** (0.0-1.0): Casual ↔ Formal
- **Enthusiasm** (0.0-1.0): Reserved ↔ Enthusiastic
- **Technical Level** (0.0-1.0): Simple ↔ Technical
- **Verbosity** (0.0-1.0): Concise ↔ Verbose
- **Empathy** (0.0-1.0): Factual ↔ Empathetic

### Managing Bots

**From the Bot List:**

- **View Stats** - See detailed statistics
- **Activate/Pause** - Control bot status
- **Delete** - Remove bot (coming soon)

**Quick Actions:**
- Pause/activate directly from dashboard
- Real-time status updates
- Activity monitoring

---

## 🎯 Mission Control

### Creating a Mission

1. Go to **Missions** → **Create Mission**
2. Select **Mission Type**:

   - **Debate** - Bots argue different sides
   - **Consensus** - Bots gradually agree
   - **Diverse Perspectives** - Each bot offers unique angle

3. Configure mission:
   - **Mission Name**: Descriptive name
   - **Topic**: What bots discuss
   - **Target Keywords**: Thread triggers
   - **Bot Selection**: Choose participants
   - **Timing Strategy**: Delays and intervals

4. Click **Create Mission**

### Mission Types Explained

#### Debate Mission

Bots take opposing positions and argue:

```
Setup:
- PRO Side: Bot1, Bot2
- CON Side: Bot3, Bot4

Result:
Bot1: I think Python is better because...
Bot3: I disagree, JavaScript has advantages...
Bot2: Building on Bot1's point...
Bot4: That's fair, but consider...
```

**Configuration:**
- Select PRO bots (argue FOR topic)
- Select CON bots (argue AGAINST topic)
- Set target keywords
- Configure timing

#### Consensus Building

Bots start with different views, gradually agree:

```
Setup:
- Bots: Bot1, Bot2, Bot3
- Final Position: "Type hints are helpful"

Result:
Bot1: I'm not sure about type hints...
Bot2: I was skeptical too, but...
Bot3: After trying them, I see benefits
Bot1: You've convinced me!
```

**Configuration:**
- Select all participating bots
- Define final consensus position
- Set discussion flow timing

#### Diverse Perspectives

Each bot offers a unique angle:

```
Setup:
- TechBot: Technical perspective
- BusinessBot: Business perspective
- BeginnerBot: Beginner questions

Result:
TechBot: From a performance standpoint...
BusinessBot: Consider the ROI implications...
BeginnerBot: Can someone explain this simply?
```

**Configuration:**
- Select bots with different personalities
- Each automatically gets unique angle
- Natural, multi-faceted discussion

### Timing Strategy

Control when bots post:

- **Initial Delay**: Wait X-Y minutes before first bot posts
- **Between Posts**: Wait X-Y minutes between bot responses
- **Max Exchanges**: Limit total rounds of back-and-forth

**Recommendations:**
- **Natural**: 10-30 min delays, 3-5 exchanges
- **Active**: 5-15 min delays, 5-10 exchanges
- **Slow Burn**: 30-60 min delays, 2-3 exchanges

### Mission Status

Track missions in real-time:

- **Pending**: Created but not executing
- **Active**: Currently running
- **Completed**: Finished successfully
- **Failed**: Encountered errors

---

## 📈 Analytics Dashboard

### Performance Metrics

**Bot Performance Chart:**
- Success rate by bot
- Visual comparison of all bots
- Color-coded by status

**Activity Distribution:**
- Pie chart of action types
- See what bots are doing most
- Identify patterns

### Key Metrics

- **Total Actions**: All bot activities
- **Success Rate**: Percentage of successful actions
- **Active Bots**: Currently running
- **Posts by Bot**: Breakdown per bot

---

## ⚙️ Settings

### Credentials Management

**Reddit API:**
- Client ID
- Client Secret
- Username
- Password

**AI APIs (Optional):**
- OpenAI API Key (for GPT-4)
- Anthropic API Key (for Claude)

### Configuration

View and edit `config.yaml`:
- Rate limiting settings
- Bot behavior parameters
- Timing configurations

### Database

- View database location
- Check bot count
- Database management (coming soon)

---

## 💡 Common Workflows

### Workflow 1: Quick Bot Setup

```
1. Click "Bot Manager"
2. Click "Create Bot" tab
3. Enter name: "PythonHelper"
4. Select preset: "Friendly"
5. Set objective: "Support"
6. Keywords: "python, help, beginner"
7. Add credentials (or use demo)
8. Click "Create Bot"
```

### Workflow 2: Create Debate

```
1. Create 2 bots with different personalities
2. Go to "Missions" → "Create Mission"
3. Select "Debate"
4. Topic: "Python vs JavaScript"
5. PRO: Bot1
6. CON: Bot2
7. Keywords: "python, javascript, beginner"
8. Timing: 15-30 min delays
9. Click "Create Mission"
```

### Workflow 3: Monitor Performance

```
1. Go to "Dashboard"
2. View active bots
3. Check recent activity
4. Click "View Stats" on any bot
5. See detailed metrics
6. Go to "Analytics" for charts
```

---

## 🎨 UI Features

### Dashboard

- **Real-time metrics** at top
- **Bot fleet overview** on left
- **Recent activity stream** on right
- **Quick action buttons** for each bot

### Color Coding

- 🟢 **Green** - Active/Success
- 🟡 **Orange** - Paused/Warning
- 🔴 **Red** - Error/Failed
- 🔵 **Blue** - Info/Neutral

### Interactive Elements

- **Expandable cards** for details
- **Hover tooltips** for help
- **Real-time updates** with refresh
- **Responsive design** for all screens

---

## 🔧 Advanced Features

### Bot Filtering

Filter bots by:
- Status (active, paused, error)
- Personality type
- Objective type

### Mission Templates

Quick-create missions with presets:
- Product discussion
- Technical debate
- Community building
- Information sharing

### Export/Import

(Coming soon)
- Export bot configurations
- Import bot templates
- Share mission setups

---

## ⚡ Keyboard Shortcuts

- `R` - Refresh data
- `Ctrl+B` - Go to Bot Manager
- `Ctrl+M` - Go to Missions
- `Ctrl+D` - Go to Dashboard

---

## 🐛 Troubleshooting

### GUI Won't Start

```bash
# Install dependencies
pip install streamlit plotly pandas

# Try running again
streamlit run bot_control_gui.py
```

### "No Bots Found"

1. Create a bot in **Bot Manager**
2. Check database file exists: `multi_bot_app.db`
3. Click refresh button

### Mission Not Showing

1. Ensure mission was created successfully
2. Check **Active Missions** tab
3. Click refresh

### Charts Not Displaying

1. Ensure plotly is installed: `pip install plotly`
2. Check browser console for errors
3. Try refreshing page

---

## 📱 Mobile Access

The GUI is responsive and works on mobile:

1. Start Streamlit with network access:
   ```bash
   streamlit run bot_control_gui.py --server.address=0.0.0.0
   ```

2. Access from mobile:
   ```
   http://YOUR_IP:8501
   ```

---

## 🎓 Tips & Tricks

### Tip 1: Test with Demo Bots

Create bots without real credentials:
- Leave credential fields empty
- GUI uses demo mode
- Perfect for testing interface

### Tip 2: Use Personality Presets

Start with presets, then customize:
- Presets are proven combinations
- Modify after seeing results
- Create variations for A/B testing

### Tip 3: Monitor Missions Live

Keep missions tab open:
- Auto-refresh to see updates
- Watch bot interactions
- Track success rates

### Tip 4: Dashboard as Home

Pin dashboard for quick access:
- Bookmark the dashboard URL
- See everything at a glance
- Quick bot control

---

## 🚀 Production Deployment

### Deploy on Server

```bash
# Install dependencies
pip install -r requirements_multibot.txt

# Run with custom port
streamlit run bot_control_gui.py --server.port=8080

# Run in background
nohup streamlit run bot_control_gui.py &
```

### Secure with Password

Add to `.streamlit/config.toml`:
```toml
[server]
headless = true
port = 8501

[browser]
gatherUsageStats = false

[theme]
primaryColor = "#667eea"
```

### Reverse Proxy (Nginx)

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

---

## 📚 Related Documentation

- **QUICKSTART.md** - General setup guide
- **README_MULTIBOT.md** - Full feature documentation
- **COORDINATED_MISSIONS_GUIDE.md** - Mission details
- **VISION_AUTOMATION_GUIDE.md** - Vision features

---

## 🆘 Support

If you encounter issues:

1. Check this guide
2. Review error messages in terminal
3. Check Streamlit logs
4. Ensure dependencies are installed
5. Verify database file exists

---

## ✨ Future Features

Coming soon:
- [ ] Bot template library
- [ ] Mission scheduling
- [ ] Export/import configs
- [ ] Multi-user support
- [ ] Advanced filtering
- [ ] Custom dashboards
- [ ] Mobile app

---

**Enjoy the GUI!** 🎉

For questions or feedback, check the main documentation.
