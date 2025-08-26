import streamlit as st
from dotenv import load_dotenv
import sys
import os
import hashlib

load_dotenv()

# Add parent directory to path to import agent
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from src.langchain_modules.agent import AIAgent

st.set_page_config(page_title="Problem-First AI Capstone - RegTech AI")

# Get authentication credentials from environment variables
def get_valid_credentials():
    """Get authentication credentials from environment variables"""
    username = os.getenv("STREAMLIT_USERNAME")
    password = os.getenv("STREAMLIT_PASSWORD")
    
    if not username or not password:
        st.error("Authentication credentials not configured. Please set STREAMLIT_USERNAME and STREAMLIT_PASSWORD in .env file.")
        st.stop()
    
    return {
        username: hash_password(password)
    }

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

def hash_password(password):
    """Hash password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()

def check_password():
    """Check if user is authenticated"""
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    
    if st.session_state.authenticated:
        return True
    
    # Get valid credentials from environment
    valid_credentials = get_valid_credentials()
    
    st.title("🔐 Authentication Required")
    st.markdown("Please login to access the RegTech AI application.")
    
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")
        
        if submit:
            if username in valid_credentials and hash_password(password) == valid_credentials[username]:
                st.session_state.authenticated = True
                st.session_state.username = username
                st.success("Login successful!")
                st.rerun()
            else:
                st.error("Invalid username or password")
    
    st.markdown("---")
    return False

def main_app():
    """Main chatbot application"""
    # Show logout button and user info in the header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("🤖 RegTech AI Agent - v1")
    with col2:
        st.write(f"Logged in as: **{st.session_state.username}**")
        if st.button("Logout"):
            st.session_state.authenticated = False
            st.session_state.pop('username', None)
            st.rerun()
    
    # Get the cached agent
    agent = initialize_agent()
    if agent is None:
        st.error("Failed to initialize AI agent")
        return
    
    # Initialize chat history in session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Sidebar controls
    with st.sidebar:
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            agent.clear_history()
            st.rerun()
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("What would you like to know?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = agent.run(prompt)
                st.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

# Check authentication before running the main app
if check_password():
    main_app()