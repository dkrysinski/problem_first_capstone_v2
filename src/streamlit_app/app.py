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
        st.subheader("👥 User Selection")
        
        # User selector
        all_users = agent.list_all_users()
        current_user = agent.current_user_email
        
        # Find current user index for default selection
        try:
            current_index = all_users.index(current_user)
        except ValueError:
            current_index = 0
        
        selected_user = st.selectbox(
            "Select User Profile:",
            all_users,
            index=current_index,
            format_func=lambda x: x.split('@')[0].title() + f" ({x.split('@')[1]})"
        )
        
        # Switch user if selection changed
        if selected_user != current_user:
            agent.switch_user(selected_user)
            st.rerun()
        
        st.divider()
        st.subheader("🧠 User Profile")
        
        # Display user profile information
        user_profile = agent.get_user_profile()
        if user_profile:
            st.write(f"**Email:** {user_profile.email}")
            if user_profile.industry:
                st.write(f"**Industry:** {user_profile.industry}")
            if user_profile.company:
                st.write(f"**Company:** {user_profile.company}")
            if user_profile.country:
                st.write(f"**Country:** {user_profile.country}")
            if user_profile.regulatory_focus:
                st.write(f"**Regulatory Focus:** {', '.join(user_profile.regulatory_focus)}")
            
            # Show interaction count
            user_context = agent.get_user_context()
            st.write(f"**Total Interactions:** {user_context['interaction_count']}")
            
            if user_context['recent_topics']:
                st.write(f"**Recent Topics:** {', '.join(user_context['recent_topics'])}")
        
        st.divider()
        
        # Recent interactions summary
        if st.checkbox("Show Recent Interactions"):
            st.subheader("📝 Recent Questions")
            interactions = agent.get_user_interactions(limit=5)
            for i, interaction in enumerate(interactions):
                with st.expander(f"Query {i+1}: {interaction.question[:50]}..."):
                    st.write(f"**Question:** {interaction.question}")
                    st.write(f"**Frameworks:** {', '.join(interaction.frameworks_analyzed)}")
                    st.write(f"**Time:** {interaction.timestamp[:19].replace('T', ' ')}")
                    if interaction.topic_category:
                        st.write(f"**Category:** {interaction.topic_category}")
        
        st.divider()
        
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