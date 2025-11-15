"""
Multi-Bot Forum Application - Web GUI
======================================

Streamlit-based web interface for managing bots, missions, and monitoring activity.

Features:
- Bot creation and management
- Personality configuration
- Mission planning and execution
- Real-time monitoring
- Statistics dashboard
- Activity logs viewer

Usage:
    streamlit run bot_control_gui.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import asyncio
import json
from typing import List, Dict, Any
import os
from dotenv import load_dotenv

# Import bot system
from multi_bot_forum_app import (
    Database, BotManager, PersonalityEngine,
    ObjectiveType, BotStatus, Personality, Objective
)
from coordinated_missions import (
    MissionOrchestrator, MissionTemplates, Mission,
    Position, InteractionStyle, MissionType, BotStance
)

load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Multi-Bot Control Center",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
    }
    .status-active {
        color: #00cc00;
        font-weight: bold;
    }
    .status-paused {
        color: #ff9900;
        font-weight: bold;
    }
    .status-error {
        color: #cc0000;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
if 'db' not in st.session_state:
    st.session_state.db = Database()
if 'bot_manager' not in st.session_state:
    st.session_state.bot_manager = BotManager(st.session_state.db)
if 'orchestrator' not in st.session_state:
    st.session_state.orchestrator = MissionOrchestrator(st.session_state.bot_manager)
if 'refresh_trigger' not in st.session_state:
    st.session_state.refresh_trigger = 0


def main():
    """Main application"""

    # Header
    st.markdown('<div class="main-header">🤖 Multi-Bot Control Center</div>', unsafe_allow_html=True)

    # Sidebar navigation
    with st.sidebar:
        st.image("https://via.placeholder.com/150x50/667eea/ffffff?text=MultiBot", use_container_width=True)
        st.markdown("---")

        page = st.radio(
            "Navigation",
            ["📊 Dashboard", "🤖 Bot Manager", "🎯 Missions", "📈 Analytics", "⚙️ Settings"],
            label_visibility="collapsed"
        )

        st.markdown("---")

        # Quick stats
        st.subheader("Quick Stats")
        bots = st.session_state.bot_manager.list_bots()
        active_bots = sum(1 for b in bots if b.status == BotStatus.ACTIVE)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Bots", len(bots))
        with col2:
            st.metric("Active", active_bots)

        # Refresh button
        if st.button("🔄 Refresh", use_container_width=True):
            st.session_state.refresh_trigger += 1
            st.rerun()

    # Main content based on page selection
    if page == "📊 Dashboard":
        show_dashboard()
    elif page == "🤖 Bot Manager":
        show_bot_manager()
    elif page == "🎯 Missions":
        show_missions()
    elif page == "📈 Analytics":
        show_analytics()
    elif page == "⚙️ Settings":
        show_settings()


def show_dashboard():
    """Dashboard overview"""
    st.header("📊 Dashboard Overview")

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)

    bots = st.session_state.bot_manager.list_bots()
    active_missions = [m for m in st.session_state.orchestrator.active_missions.values()
                      if m.status == "active"]

    # Calculate total actions
    total_actions = 0
    success_count = 0
    for bot in bots:
        stats = st.session_state.bot_manager.get_bot_statistics(bot.id)
        total_actions += stats['total_actions']
        success_count += stats['successful_actions']

    with col1:
        st.metric("Total Bots", len(bots), delta=None)
    with col2:
        active_count = sum(1 for b in bots if b.status == BotStatus.ACTIVE)
        st.metric("Active Bots", active_count, delta=None)
    with col3:
        st.metric("Total Actions", total_actions, delta=None)
    with col4:
        success_rate = (success_count / total_actions * 100) if total_actions > 0 else 0
        st.metric("Success Rate", f"{success_rate:.1f}%", delta=None)

    st.markdown("---")

    # Two columns for bot list and recent activity
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader("🤖 Bot Fleet")

        if not bots:
            st.info("No bots created yet. Go to Bot Manager to create your first bot!")
        else:
            for bot in bots:
                with st.expander(f"{bot.name} - {bot.status.value.upper()}", expanded=False):
                    col_a, col_b, col_c = st.columns(3)

                    with col_a:
                        st.write("**Personality:**", bot.personality.tone.title())
                        st.write("**Objectives:**", len(bot.objectives))

                    with col_b:
                        stats = st.session_state.bot_manager.get_bot_statistics(bot.id)
                        st.write("**Total Actions:**", stats['total_actions'])
                        st.write("**Success Rate:**", f"{stats['success_rate']:.1%}")

                    with col_c:
                        if bot.status == BotStatus.ACTIVE:
                            if st.button("⏸️ Pause", key=f"pause_{bot.id}"):
                                st.session_state.bot_manager.update_bot_status(bot.id, BotStatus.PAUSED)
                                st.rerun()
                        else:
                            if st.button("▶️ Activate", key=f"activate_{bot.id}"):
                                st.session_state.bot_manager.update_bot_status(bot.id, BotStatus.ACTIVE)
                                st.rerun()

    with col_right:
        st.subheader("📜 Recent Activity")

        # Get recent activities from all bots
        all_activities = []
        for bot in bots[:5]:  # Last 5 bots
            activities = st.session_state.db.get_bot_activities(bot.id, limit=5)
            for activity in activities:
                all_activities.append({
                    'bot': bot.name,
                    'action': activity.action_type.value,
                    'time': activity.timestamp,
                    'success': activity.success
                })

        # Sort by time
        all_activities.sort(key=lambda x: x['time'], reverse=True)

        for activity in all_activities[:10]:
            icon = "✅" if activity['success'] else "❌"
            time_str = activity['time'].strftime("%H:%M:%S")
            st.text(f"{icon} {time_str} - {activity['bot']}: {activity['action']}")


def show_bot_manager():
    """Bot management interface"""
    st.header("🤖 Bot Manager")

    tab1, tab2, tab3 = st.tabs(["📋 Bot List", "➕ Create Bot", "✏️ Edit Bot"])

    with tab1:
        show_bot_list()

    with tab2:
        show_create_bot()

    with tab3:
        show_edit_bot()


def show_bot_list():
    """Display list of bots"""
    bots = st.session_state.bot_manager.list_bots()

    if not bots:
        st.info("No bots created yet. Use the 'Create Bot' tab to get started!")
        return

    # Create DataFrame for display
    bot_data = []
    for bot in bots:
        stats = st.session_state.bot_manager.get_bot_statistics(bot.id)
        bot_data.append({
            'Name': bot.name,
            'Status': bot.status.value,
            'Personality': bot.personality.tone,
            'Objectives': len(bot.objectives),
            'Actions': stats['total_actions'],
            'Success Rate': f"{stats['success_rate']:.1%}",
            'ID': bot.id
        })

    df = pd.DataFrame(bot_data)

    # Display table
    st.dataframe(
        df[['Name', 'Status', 'Personality', 'Objectives', 'Actions', 'Success Rate']],
        use_container_width=True,
        hide_index=True
    )

    # Bot actions
    st.subheader("Bot Actions")

    selected_bot = st.selectbox(
        "Select bot for actions:",
        options=[b.name for b in bots],
        key="action_bot_select"
    )

    if selected_bot:
        bot = next(b for b in bots if b.name == selected_bot)

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            if st.button("▶️ Activate", use_container_width=True):
                st.session_state.bot_manager.update_bot_status(bot.id, BotStatus.ACTIVE)
                st.success(f"Activated {bot.name}")
                st.rerun()

        with col2:
            if st.button("⏸️ Pause", use_container_width=True):
                st.session_state.bot_manager.update_bot_status(bot.id, BotStatus.PAUSED)
                st.warning(f"Paused {bot.name}")
                st.rerun()

        with col3:
            if st.button("📊 View Stats", use_container_width=True):
                show_bot_statistics(bot)

        with col4:
            if st.button("🗑️ Delete", use_container_width=True, type="secondary"):
                # TODO: Implement delete
                st.error("Delete functionality not yet implemented")


def show_create_bot():
    """Create new bot interface"""
    st.subheader("Create New Bot")

    with st.form("create_bot_form"):
        # Basic info
        col1, col2 = st.columns(2)

        with col1:
            bot_name = st.text_input("Bot Name*", placeholder="e.g., PythonHelper")

        with col2:
            preset = st.selectbox(
                "Personality Preset",
                ["friendly", "professional", "technical", "casual", "custom"]
            )

        # Personality configuration
        st.subheader("Personality Settings")

        if preset == "custom":
            col1, col2, col3 = st.columns(3)

            with col1:
                tone = st.text_input("Tone", value="friendly")
                formality = st.slider("Formality", 0.0, 1.0, 0.5)
                enthusiasm = st.slider("Enthusiasm", 0.0, 1.0, 0.7)

            with col2:
                technical = st.slider("Technical Level", 0.0, 1.0, 0.5)
                verbosity = st.slider("Verbosity", 0.0, 1.0, 0.5)
                empathy = st.slider("Empathy", 0.0, 1.0, 0.7)

            with col3:
                vocab_style = st.selectbox("Vocabulary", ["casual", "professional", "technical", "simple"])
        else:
            # Use preset
            personalities = PersonalityEngine.get_preset_personalities()
            selected_personality = personalities[preset]

            st.info(f"**{preset.title()}**: {selected_personality.description}")

            # Show preset values
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Formality", f"{selected_personality.formality:.1f}")
                st.metric("Enthusiasm", f"{selected_personality.enthusiasm:.1f}")
            with col2:
                st.metric("Technical", f"{selected_personality.technical_level:.1f}")
                st.metric("Verbosity", f"{selected_personality.verbosity:.1f}")
            with col3:
                st.metric("Empathy", f"{selected_personality.empathy:.1f}")

        # Objectives
        st.subheader("Objectives")

        objective_type = st.selectbox(
            "Objective Type",
            [o.value for o in ObjectiveType]
        )

        objective_desc = st.text_area(
            "Objective Description",
            placeholder="What should this bot try to accomplish?"
        )

        keywords = st.text_input(
            "Target Keywords (comma-separated)",
            placeholder="python, beginner, help, learn"
        )

        priority = st.slider("Priority", 1, 10, 5)

        # Credentials
        st.subheader("Reddit Credentials")

        col1, col2 = st.columns(2)

        with col1:
            client_id = st.text_input("Client ID", type="password")
            username = st.text_input("Username")

        with col2:
            client_secret = st.text_input("Client Secret", type="password")
            password = st.text_input("Password", type="password")

        # Submit
        submitted = st.form_submit_button("Create Bot", use_container_width=True)

        if submitted:
            if not bot_name:
                st.error("Bot name is required!")
                return

            # Create personality config
            if preset == "custom":
                personality_config = {
                    'description': f'{bot_name} personality',
                    'tone': tone,
                    'formality': formality,
                    'enthusiasm': enthusiasm,
                    'technical_level': technical,
                    'verbosity': verbosity,
                    'empathy': empathy,
                    'vocabulary_style': vocab_style,
                    'response_patterns': []
                }
            else:
                personalities = PersonalityEngine.get_preset_personalities()
                personality_config = personalities[preset].to_dict()

            # Create objectives config
            objectives_config = [{
                'type': objective_type,
                'description': objective_desc or f"Default objective for {bot_name}",
                'target_keywords': [k.strip() for k in keywords.split(',') if k.strip()],
                'success_metrics': {},
                'constraints': {},
                'priority': priority
            }]

            # Create credentials
            credentials = {
                'client_id': client_id or os.getenv('REDDIT_BOT1_CLIENT_ID', 'demo'),
                'client_secret': client_secret or os.getenv('REDDIT_BOT1_CLIENT_SECRET', 'demo'),
                'username': username or 'demo_user',
                'password': password or 'demo_pass',
                'user_agent': 'Multi-Bot App v1.0'
            }

            # Create bot
            try:
                bot = st.session_state.bot_manager.create_bot(
                    name=bot_name,
                    personality_config=personality_config,
                    objectives_config=objectives_config,
                    credentials=credentials
                )

                st.success(f"✅ Bot '{bot.name}' created successfully! ID: {bot.id}")
                st.balloons()

            except Exception as e:
                st.error(f"Error creating bot: {e}")


def show_edit_bot():
    """Edit existing bot"""
    bots = st.session_state.bot_manager.list_bots()

    if not bots:
        st.info("No bots to edit. Create a bot first!")
        return

    selected_bot_name = st.selectbox(
        "Select bot to edit:",
        [b.name for b in bots]
    )

    if selected_bot_name:
        bot = next(b for b in bots if b.name == selected_bot_name)

        st.info("Bot editing interface coming soon! For now, you can view bot details below.")

        # Display bot details
        st.json(bot.to_dict())


def show_bot_statistics(bot):
    """Show detailed statistics for a bot"""
    stats = st.session_state.bot_manager.get_bot_statistics(bot.id)

    st.subheader(f"Statistics for {bot.name}")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Actions", stats['total_actions'])
    with col2:
        st.metric("Successful", stats['successful_actions'])
    with col3:
        st.metric("Failed", stats['failed_actions'])

    # Action breakdown chart
    if stats['action_breakdown']:
        st.subheader("Action Breakdown")

        df_actions = pd.DataFrame([
            {'Action': action, 'Count': count}
            for action, count in stats['action_breakdown'].items()
        ])

        fig = px.bar(df_actions, x='Action', y='Count', title="Actions by Type")
        st.plotly_chart(fig, use_container_width=True)


def show_missions():
    """Mission control interface"""
    st.header("🎯 Mission Control")

    tab1, tab2, tab3 = st.tabs(["📋 Active Missions", "➕ Create Mission", "📊 Mission Reports"])

    with tab1:
        show_active_missions()

    with tab2:
        show_create_mission()

    with tab3:
        show_mission_reports()


def show_active_missions():
    """Display active missions"""
    missions = list(st.session_state.orchestrator.active_missions.values())

    if not missions:
        st.info("No active missions. Create a mission in the 'Create Mission' tab!")
        return

    for mission in missions:
        with st.expander(f"{mission.name} - {mission.status.upper()}", expanded=True):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.write("**Type:**", mission.mission_type.value)
                st.write("**Bots:**", len(mission.bot_stances))

            with col2:
                st.write("**Keywords:**", ", ".join(mission.target_thread_keywords[:3]))
                st.write("**Status:**", mission.status)

            with col3:
                report = st.session_state.orchestrator.get_mission_report(mission.id)
                st.metric("Total Posts", report.get('total_posts', 0))

            # Bot stances
            st.write("**Bot Positions:**")
            for stance in mission.bot_stances:
                bot = st.session_state.bot_manager.get_bot(stance.bot_id)
                st.text(f"• {bot.name}: {stance.position.value} ({stance.interaction_style.value})")


def show_create_mission():
    """Create new mission interface"""
    st.subheader("Create Coordinated Mission")

    bots = st.session_state.bot_manager.list_bots()

    if len(bots) < 2:
        st.warning("You need at least 2 bots to create a coordinated mission!")
        return

    with st.form("create_mission_form"):
        # Mission type
        mission_type = st.selectbox(
            "Mission Type",
            ["debate", "consensus", "diverse_perspectives"]
        )

        # Basic info
        mission_name = st.text_input("Mission Name", placeholder="e.g., Python vs JavaScript Debate")

        topic = st.text_input("Topic", placeholder="e.g., Python for beginners")

        keywords = st.text_input(
            "Target Keywords (comma-separated)",
            placeholder="python, javascript, beginner"
        )

        # Bot selection based on mission type
        if mission_type == "debate":
            st.subheader("Select Bots for Each Side")

            col1, col2 = st.columns(2)

            with col1:
                pro_bots = st.multiselect(
                    "PRO Side (For)",
                    [b.name for b in bots]
                )

            with col2:
                con_bots = st.multiselect(
                    "CON Side (Against)",
                    [b.name for b in bots if b.name not in pro_bots]
                )

        elif mission_type == "consensus":
            selected_bots = st.multiselect(
                "Select Bots for Consensus Building",
                [b.name for b in bots]
            )

            final_position = st.text_input(
                "Final Consensus Position",
                placeholder="What should bots eventually agree on?"
            )

        else:  # diverse_perspectives
            st.info("Each bot will offer a unique perspective on the topic")
            selected_bots = st.multiselect(
                "Select Bots",
                [b.name for b in bots]
            )

        # Timing settings
        st.subheader("Timing Strategy")

        col1, col2 = st.columns(2)

        with col1:
            initial_delay_min = st.number_input("Initial Delay Min (minutes)", 5, 60, 10)
            between_posts_min = st.number_input("Between Posts Min (minutes)", 5, 60, 15)

        with col2:
            initial_delay_max = st.number_input("Initial Delay Max (minutes)", 10, 120, 20)
            between_posts_max = st.number_input("Between Posts Max (minutes)", 10, 120, 30)

        max_exchanges = st.slider("Max Exchanges", 1, 10, 5)

        # Submit
        submitted = st.form_submit_button("Create Mission", use_container_width=True)

        if submitted:
            keyword_list = [k.strip() for k in keywords.split(',') if k.strip()]

            try:
                if mission_type == "debate":
                    # Get bot IDs
                    pro_bot_ids = [b.id for b in bots if b.name in pro_bots]
                    con_bot_ids = [b.id for b in bots if b.name in con_bots]

                    mission = MissionTemplates.create_debate_mission(
                        topic=topic,
                        pro_bot_ids=pro_bot_ids,
                        con_bot_ids=con_bot_ids,
                        keywords=keyword_list
                    )

                elif mission_type == "consensus":
                    bot_ids = [b.id for b in bots if b.name in selected_bots]

                    mission = MissionTemplates.create_consensus_mission(
                        topic=topic,
                        bot_ids=bot_ids,
                        final_position=final_position,
                        keywords=keyword_list
                    )

                else:  # diverse_perspectives
                    bot_perspectives = {}
                    for bot in bots:
                        if bot.name in selected_bots:
                            bot_perspectives[bot.id] = {
                                'position': Position.NEUTRAL,
                                'talking_points': [f"My perspective on {topic}..."],
                                'priority': 5
                            }

                    mission = MissionTemplates.create_diverse_perspectives_mission(
                        topic=topic,
                        bot_perspectives=bot_perspectives,
                        keywords=keyword_list
                    )

                # Update timing
                mission.timing_strategy.update({
                    'initial_delay_min': initial_delay_min,
                    'initial_delay_max': initial_delay_max,
                    'between_posts_min': between_posts_min,
                    'between_posts_max': between_posts_max,
                    'max_exchanges': max_exchanges
                })

                # Register mission
                st.session_state.orchestrator.create_mission(mission)

                st.success(f"✅ Mission '{mission.name}' created successfully!")
                st.info("Mission is ready but not executing. Use the API to execute on specific threads.")

            except Exception as e:
                st.error(f"Error creating mission: {e}")


def show_mission_reports():
    """Show mission execution reports"""
    missions = list(st.session_state.orchestrator.active_missions.values())

    if not missions:
        st.info("No missions to report on yet.")
        return

    for mission in missions:
        report = st.session_state.orchestrator.get_mission_report(mission.id)

        with st.expander(f"📊 {report.get('mission_name', 'Unknown')}", expanded=False):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Total Posts", report.get('total_posts', 0))
            with col2:
                st.metric("Bots Participated", report.get('bots_participated', 0))
            with col3:
                st.metric("Status", report.get('status', 'unknown'))

            # Posts by bot
            if report.get('posts_by_bot'):
                st.write("**Posts by Bot:**")
                for bot_id, count in report['posts_by_bot'].items():
                    bot = st.session_state.bot_manager.get_bot(bot_id)
                    st.text(f"• {bot.name}: {count} posts")


def show_analytics():
    """Analytics dashboard"""
    st.header("📈 Analytics")

    bots = st.session_state.bot_manager.list_bots()

    if not bots:
        st.info("No data to analyze yet. Create and run some bots first!")
        return

    # Success rate over time
    st.subheader("Bot Performance")

    bot_stats = []
    for bot in bots:
        stats = st.session_state.bot_manager.get_bot_statistics(bot.id)
        bot_stats.append({
            'Bot': bot.name,
            'Actions': stats['total_actions'],
            'Success Rate': stats['success_rate'] * 100,
            'Status': bot.status.value
        })

    df = pd.DataFrame(bot_stats)

    # Bar chart
    fig = px.bar(df, x='Bot', y='Success Rate', color='Status',
                 title="Bot Success Rates")
    st.plotly_chart(fig, use_container_width=True)

    # Activity distribution
    st.subheader("Activity Distribution")

    all_actions = {}
    for bot in bots:
        stats = st.session_state.bot_manager.get_bot_statistics(bot.id)
        for action, count in stats.get('action_breakdown', {}).items():
            all_actions[action] = all_actions.get(action, 0) + count

    if all_actions:
        df_actions = pd.DataFrame([
            {'Action': action, 'Count': count}
            for action, count in all_actions.items()
        ])

        fig = px.pie(df_actions, names='Action', values='Count',
                     title="Action Type Distribution")
        st.plotly_chart(fig, use_container_width=True)


def show_settings():
    """Settings page"""
    st.header("⚙️ Settings")

    tab1, tab2, tab3 = st.tabs(["🔑 Credentials", "⚙️ Configuration", "💾 Database"])

    with tab1:
        st.subheader("Reddit API Credentials")

        st.info("Credentials are stored in .env file. Never commit this file to version control!")

        with st.form("credentials_form"):
            st.text_input("Client ID", type="password", value="***")
            st.text_input("Client Secret", type="password", value="***")
            st.text_input("Username", value="***")
            st.text_input("Password", type="password", value="***")

            st.form_submit_button("Update Credentials (Not Implemented)")

        st.markdown("---")

        st.subheader("OpenAI/Anthropic API Keys")

        has_openai = bool(os.getenv('OPENAI_API_KEY'))
        has_anthropic = bool(os.getenv('ANTHROPIC_API_KEY'))

        col1, col2 = st.columns(2)

        with col1:
            st.metric("OpenAI", "✅ Configured" if has_openai else "❌ Not Set")
        with col2:
            st.metric("Anthropic", "✅ Configured" if has_anthropic else "❌ Not Set")

    with tab2:
        st.subheader("Application Configuration")

        st.info("Configuration is stored in config.yaml")

        if st.button("View config.yaml"):
            try:
                with open('config.yaml', 'r') as f:
                    st.code(f.read(), language='yaml')
            except:
                st.error("config.yaml not found")

    with tab3:
        st.subheader("Database Management")

        st.info(f"Database: {st.session_state.db.db_path}")

        bots = st.session_state.bot_manager.list_bots()

        st.metric("Total Bots", len(bots))

        st.markdown("---")

        st.warning("⚠️ Danger Zone")

        if st.button("🗑️ Delete All Bots", type="secondary"):
            st.error("This feature is not yet implemented for safety")


if __name__ == "__main__":
    main()
