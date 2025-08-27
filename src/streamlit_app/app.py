import streamlit as st
from dotenv import load_dotenv
import sys
import os

load_dotenv()

# Add parent directory to path to import agent
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from src.langchain_modules.agent import AIAgent
from push_notifications import render_push_notification_interface

st.set_page_config(
    page_title="RegTech AI v2 - Push Notification System", 
    page_icon="📢",
    layout="wide",
    initial_sidebar_state="collapsed"
)

@st.cache_resource
def initialize_agent():
    """Initialize the AI agent once at startup"""
    try:
        print("🚀 Starting AI agent initialization...")
        agent = AIAgent()
        print("✅ AI agent initialized successfully!")
        return agent
    except Exception as e:
        st.error(f"Configuration error: {e}")
        st.stop()
        return None

def main_app():
    """Main application - Push Notification System Demo"""
    
    st.title("🚀 RegTech AI v2: Push Notification System")
    
    # Initialize and store agent for the push notification system
    agent = initialize_agent()
    if agent is None:
        st.error("Failed to initialize AI agent")
        return
    
    st.session_state.agent = agent
    
    # Render the push notification interface
    render_push_notification_interface()

# Run the main app
main_app()