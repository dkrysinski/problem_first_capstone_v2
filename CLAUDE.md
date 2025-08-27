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

This is a **RegTech AI system** that provides regulatory compliance assistance across multiple frameworks (GDPR, NIS2, DORA, CER, US Executive Orders) with advanced user memory capabilities and push notification functionality.

### Core Architecture

**LangGraph Workflow**: The main AI agent (`src/langchain_modules/agent.py`) uses LangGraph to orchestrate a multi-step workflow:
1. **Classification**: Determines which regulatory frameworks apply to a user question using structured LLM outputs
2. **Document Retrieval**: Vector-based similarity search with query expansion and context-aware retrieval
3. **Framework Analysis**: Parallel processing of relevant frameworks with specialized prompts
4. **Synthesis**: Combines insights from multiple frameworks into comprehensive guidance
5. **Memory Storage**: Automatic user profile updates and interaction tracking

**State Management**: Uses `AgentState` TypedDict to track workflow state including:
- User question and retrieved documents per framework
- Framework applicability flags (gdpr, nis2, dora, cer, exec_order)
- Off-topic detection and rejection handling
- Analysis results and final synthesized answer

**Push Notification System**: New document analysis workflow (`src/streamlit_app/push_notifications.py`):
- Document upload and processing via separate interface
- AI-powered impact assessment for existing users
- Personalized email generation based on user profiles
- Bulk notification processing with relevance scoring

### Key Components

**User Memory System** (`src/langchain_modules/user_memory.py`):
- `UserProfile` dataclass: Stores email, industry, company, role, country, regulatory focus
- `Interaction` dataclass: Tracks questions, frameworks analyzed, topics, and timestamps
- JSON-based persistence in `data/user_memory/` with user_profiles.json and user_interactions.json
- Advanced topic tagging with 60+ granular business categories
- Industry and country inference from question content
- Regulatory complexity assessment and usage pattern tracking

**Document Processing**:
- Regulatory PDFs stored in `data/` directory with framework-specific vector stores
- FAISS vector stores: `gdpr_vector_store`, `nis2_vector_store`, `dora_vector_store`, `cer_vector_store`, `exec_order_vector_store`
- Executive Orders directory (`data/executive_orders/`) supporting multiple PDF documents
- Azure OpenAI embeddings (text-embedding-3-small) for document vectorization
- Batch processing with error handling and content validation

**Web Interface** (`src/streamlit_app/app.py`):
- Dual interface system: main chat-based Q&A and push notification document analysis
- Authentication system using environment variables (STREAMLIT_USERNAME/PASSWORD)
- Session state management for user profiles and agent initialization
- Real-time chat interface with sidebar profile display
- `@st.cache_resource` for efficient agent initialization

### Framework Integration

**Prompts** (`src/langchain_modules/prompts.py`): Contains specialized prompts for:
- `CLASSIFICATION_PROMPT`: Determines framework applicability with confidence scoring
- `RAG_PROMPT`: General document retrieval and analysis
- Framework-specific prompts: `NIS2_PROMPT`, `DORA_PROMPT`, `CER_PROMPT`, `EXEC_ORDER_PROMPT`
- `SYNTHESIS_PROMPT`: Combines multi-framework analysis with user context integration

**AI Models**: Uses Azure OpenAI:
- GPT-4.1 (deployment: "gpt-4.1-maven-course") with temperature=0.3 for text generation and analysis
- OpenAI embeddings (text-embedding-3-small) for document vectorization
- Structured outputs using Pydantic models (`RagGenerationResponse`, `ClassificationResponse`, `RegulationAssessment`)

### Data Flow

1. User submits question via Streamlit interface
2. `AIAgent.run()` initiates LangGraph workflow with optional Opik tracing
3. Classification node determines relevant frameworks using user context
4. Query expansion enhances retrieval with industry/user-specific terms
5. Framework-specific analysis nodes process in parallel with specialized vector stores
6. Synthesis node combines results with user memory context
7. User memory system updates profile, infers industry/country, and tracks interaction history
8. Response displayed with comprehensive debugging information

### Configuration

**Environment Variables Required**:
- `AZURE_OPENAI_API_KEY`: Azure OpenAI API access
- `AZURE_OPENAI_ENDPOINT`: Azure OpenAI endpoint URL  
- `STREAMLIT_USERNAME`/`STREAMLIT_PASSWORD`: Web interface authentication
- `OPIK_API_KEY` (optional): AI observability tracing for cloud instance
- `OPIK_WORKSPACE` (optional): Opik workspace name
- `OPIK_USE_LOCAL` (optional): Set to "true" for self-hosted Opik instance

**File Structure**:
- `src/langchain_modules/`: Core AI logic and workflow
  - `agent.py`: Main LangGraph workflow with 1200+ lines of advanced logic
  - `user_memory.py`: User profile and interaction management
  - `prompts.py`: Framework-specific LLM prompts
  - `document_analyzer.py`: Push notification document processing
- `src/streamlit_app/`: Web interface with dual app modes
  - `app.py`: Main chat interface
  - `push_notifications.py`: Document analysis interface  
- `data/`: Regulatory PDFs, vector stores, and user memory storage
  - Individual framework vector stores (gdpr_vector_store/, nis2_vector_store/, etc.)
  - `executive_orders/`: Directory for US Executive Order PDFs
  - `user_memory/`: JSON-based user profiles and interactions
- `regtech_env/`: Python virtual environment

### Advanced Features

**Query Expansion**: Context-aware query enhancement functions:
- `_expand_query_for_gdpr()`: Adds cross-border, industry-specific terms
- `_expand_query_for_exec_orders()`: Adds China-related, technology, business expansion terms
- User profile integration for personalized expansion

**Debugging & Monitoring**:
- Comprehensive console logging for classification decisions, retrieval results, and memory operations
- User context display showing industry, regulatory patterns, and interaction history
- Framework routing debug with detailed state tracking
- Optional Opik tracing integration with performance metrics and error logging

**Error Handling**:
- Graceful fallback for vector store creation failures
- Batch processing with document validation
- Off-topic question detection and polite rejection
- Missing file and configuration error handling

**Performance Optimizations**:
- `@st.cache_resource` for agent initialization
- Framework-specific vector stores to reduce search scope
- Parallel framework analysis nodes
- Content validation and cleaning during document processing