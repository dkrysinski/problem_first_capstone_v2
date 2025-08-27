# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Environment Setup
```bash
# Activate virtual environment (recommended method)
./activate_env.sh

# Manual activation
source regtech_env/bin/activate
pip install -r requirements.txt
```

### Running the Application
```bash
# Start the Streamlit web application
streamlit run src/streamlit_app/app.py

# Test agent initialization
python -c "from src.langchain_modules.agent import AIAgent; agent = AIAgent()"
```

### Development
```bash
# Deactivate virtual environment
deactivate
```

## Architecture Overview

This is a **RegTech AI system** that provides regulatory compliance assistance across multiple EU frameworks (GDPR, NIS2, DORA, CER) with user memory capabilities.

### Core Architecture

**LangGraph Workflow**: The main AI agent (`src/langchain_modules/agent.py`) uses LangGraph to orchestrate a multi-step workflow:
1. **Classification**: Determines which regulatory frameworks apply to a user question
2. **Document Retrieval**: Vector-based similarity search through regulatory PDFs using FAISS
3. **Framework Analysis**: Parallel processing of relevant frameworks (GDPR, NIS2, DORA, CER)
4. **Synthesis**: Combines insights from multiple frameworks into comprehensive guidance

**State Management**: Uses `AgentState` TypedDict to track workflow state including:
- User question and retrieved documents
- Framework applicability flags (gdpr, nis2, dora, cer, exec_order)
- Analysis results and final synthesized answer

### Key Components

**User Memory System** (`src/langchain_modules/user_memory.py`):
- `UserProfile` dataclass: Stores email, industry, company, role, regulatory focus
- `Interaction` dataclass: Tracks questions and frameworks analyzed
- JSON-based persistence in `data/user_memory/`
- Automatic industry inference from question content

**Document Processing**:
- Regulatory PDFs stored in `data/` directory
- FAISS vector store for document similarity search
- Azure OpenAI embeddings for document vectorization

**Web Interface** (`src/streamlit_app/app.py`):
- Authentication system using environment variables
- Session state management for user profiles
- Real-time chat interface with sidebar profile display

### Framework Integration

**Prompts** (`src/langchain_modules/prompts.py`): Contains specialized prompts for:
- `CLASSIFICATION_PROMPT`: Determines framework applicability
- `RAG_PROMPT`: General document retrieval and analysis
- Framework-specific prompts: `NIS2_PROMPT`, `DORA_PROMPT`, `CER_PROMPT`
- `SYNTHESIS_PROMPT`: Combines multi-framework analysis

**AI Models**: Uses Azure OpenAI:
- GPT-4 for text generation and analysis
- OpenAI embeddings for document vectorization

### Data Flow

1. User submits question via Streamlit interface
2. `AIAgent.run()` initiates LangGraph workflow
3. Classification node determines relevant frameworks
4. Document retrieval fetches relevant regulatory text
5. Framework-specific analysis nodes process in parallel
6. Synthesis node combines results
7. User memory system updates profile and interaction history
8. Response displayed in chat interface

### Configuration

**Environment Variables Required**:
- `AZURE_OPENAI_API_KEY`: Azure OpenAI API access
- `AZURE_OPENAI_ENDPOINT`: Azure OpenAI endpoint URL  
- `STREAMLIT_USERNAME`/`STREAMLIT_PASSWORD`: Web interface authentication
- `OPIK_API_KEY` (optional): AI observability tracing

**File Structure**:
- `src/langchain_modules/`: Core AI logic and workflow
- `src/streamlit_app/`: Web interface
- `data/`: Regulatory PDFs and user memory storage
- `regtech_env/`: Python virtual environment