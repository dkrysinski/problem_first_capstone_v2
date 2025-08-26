# RegTech AI - Multi-Framework Regulatory Compliance Assistant

A sophisticated AI-powered regulatory compliance assistant that analyzes business questions against multiple EU regulatory frameworks (GDPR, NIS2, DORA, CER) with user memory capabilities.

## 🌟 Features

### Core Capabilities
- **Multi-Framework Analysis**: Simultaneously analyzes GDPR, NIS2, DORA, and CER regulations
- **Intelligent Classification**: Automatically determines which regulatory frameworks apply to your business case
- **Document Retrieval**: Vector-based similarity search through regulatory documents
- **Synthesis Engine**: Combines insights from multiple frameworks for comprehensive compliance guidance

### User Memory System
- **Profile Management**: Remembers user details (email, industry, company, role)
- **Interaction History**: Tracks questions and regulatory focus areas
- **Smart Inference**: Automatically detects industry and regulatory complexity from questions
- **Personalized Responses**: Uses historical context for tailored compliance advice

### Advanced Features
- **Real-time Debugging**: Comprehensive logging of user context and decision processes
- **Opik Tracing**: Optional AI observability and performance monitoring
- **Streamlit Web Interface**: User-friendly chat interface with profile display
- **Authentication System**: Secure login with environment-based credentials

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Azure OpenAI API access
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/[username]/regtech-ai.git
   cd regtech-ai
   ```

2. **Set up virtual environment**
   ```bash
   # Use the convenient activation script
   ./activate_env.sh
   
   # Or manually
   python -m venv regtech_env
   source regtech_env/bin/activate  # On Windows: regtech_env\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Configure environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your credentials:
   # AZURE_OPENAI_API_KEY=your_key_here
   # AZURE_OPENAI_ENDPOINT=your_endpoint_here
   # STREAMLIT_USERNAME=your_username
   # STREAMLIT_PASSWORD=your_password
   ```

4. **Run the application**
   ```bash
   streamlit run src/streamlit_app/app.py
   ```

## 📁 Project Structure

```
regtech-ai/
├── src/
│   ├── langchain_modules/
│   │   ├── agent.py          # Main AI agent with LangGraph workflow
│   │   ├── user_memory.py    # User profile and interaction management
│   │   ├── prompts.py        # LLM prompts for different frameworks
│   │   └── tools.py          # Custom tools and utilities
│   └── streamlit_app/
│       └── app.py            # Web interface
├── data/
│   ├── *.pdf                 # Regulatory documents
│   └── user_memory/          # User profiles and interactions (gitignored)
├── requirements.txt          # Python dependencies
├── activate_env.sh          # Environment activation script
└── README.md
```

## 🛠️ Usage

### Web Interface
1. Start the Streamlit app: `streamlit run src/streamlit_app/app.py`
2. Login with your configured credentials
3. Ask regulatory compliance questions in the chat interface
4. View your profile and interaction history in the sidebar

### Programmatic Usage
```python
from src.langchain_modules.agent import AIAgent

# Initialize agent
agent = AIAgent()

# Ask a question
response = agent.run("How does GDPR apply to our cloud data processing?")
print(response)

# Access user profile
profile = agent.get_user_profile()
print(f"User industry: {profile.industry}")

# View interaction history
interactions = agent.get_user_interactions(limit=5)
```

## 🔧 Configuration

### Environment Variables
- `AZURE_OPENAI_API_KEY`: Your Azure OpenAI API key
- `AZURE_OPENAI_ENDPOINT`: Your Azure OpenAI endpoint
- `STREAMLIT_USERNAME`: Web interface username
- `STREAMLIT_PASSWORD`: Web interface password
- `OPIK_API_KEY`: (Optional) Opik tracing API key
- `OPIK_WORKSPACE`: (Optional) Opik workspace name
- `OPIK_USE_LOCAL`: (Optional) Set to "true" for self-hosted Opik

### Supported Regulatory Frameworks
- **GDPR**: General Data Protection Regulation
- **NIS2**: Network and Information Security Directive 2
- **DORA**: Digital Operational Resilience Act
- **CER**: Critical Entities Resilience Directive

## 🧠 Memory System

The agent automatically remembers:
- **User Profile**: Email, industry, company, regulatory focus areas
- **Interaction History**: Questions, frameworks analyzed, timestamps
- **Industry Intelligence**: Automatically infers industry from question content
- **Usage Patterns**: Tracks most consulted frameworks and topics

## 📊 Debugging & Monitoring

Enable comprehensive debugging output that shows:
- User context and profile information
- Regulatory framework classification decisions
- Document retrieval results
- Memory storage operations
- Opik tracing integration (optional)

## 🛡️ Security

- Environment-based configuration
- Gitignored sensitive data
- No hardcoded credentials
- User data stored locally in JSON format

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙋‍♀️ Support

For questions or support, please [open an issue](https://github.com/[username]/regtech-ai/issues) on GitHub.

---

**Built with**: LangChain, LangGraph, Streamlit, Azure OpenAI, FAISS, Pydantic