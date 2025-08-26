from langchain_openai import AzureChatOpenAI, OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.vectorstores import FAISS
from langchain.schema import HumanMessage, AIMessage
from langchain_core.documents import Document
from pydantic import BaseModel, Field
from typing import List, TypedDict
from langgraph.graph import StateGraph, START, END
import os
import os.path as osp
import time

from .prompts import RAG_PROMPT

# Opik tracing imports
try:
    from opik import track, configure, opik_context
    from opik.api_objects import opik_client
    OPIK_AVAILABLE = True
except ImportError:
    OPIK_AVAILABLE = False

class AgentState(TypedDict):
    question: str
    retrieved_docs: list[Document]
    answer: str
    
class RagGenerationResponse(BaseModel):
    """Response to the question with answer and sources. Sources are
    names of the documents. Sources should be None if the answer is not
    found in the context."""
    answer: str = Field(description="Answer to the question.")
    sources: list[str] = Field(
        description="Names of the documents that contain the answer.",
        default_factory=list
    )


FILE_PATH = "/home/dan/capstone_project/data/GDPR_Regulation.pdf"
VECTOR_STORE_PATH = "/home/dan/capstone_project/data/vector_store"

class AIAgent:
    def __init__(self):
        self.llm = None
        self.messages = []
        self._setup_opik()
        self.initialize()  # Automatically initialize LLM
    
    def _setup_opik(self):
        """Initialize Opik tracing if available and configured"""
        if OPIK_AVAILABLE:
            use_local = os.getenv("OPIK_USE_LOCAL", "false").lower() == "true"
            api_key = os.getenv("OPIK_API_KEY")
            workspace = os.getenv("OPIK_WORKSPACE")
            
            if use_local:
                try:
                    # Self-hosted local instance - no API key required
                    configure(use_local=True)
                    print("✅ Opik tracing configured for self-hosted local instance")
                    self.opik_enabled = True
                except Exception as e:
                    print(f"⚠️  Failed to configure Opik self-hosted instance: {e}")
                    self.opik_enabled = False
            elif api_key and workspace and api_key != "your_opik_api_key_here":
                try:
                    # Cloud instance - requires API key and workspace
                    configure(
                        api_key=api_key,
                        workspace=workspace
                    )
                    print("✅ Opik tracing configured for cloud instance")
                    self.opik_enabled = True
                except Exception as e:
                    print(f"⚠️  Failed to configure Opik cloud instance: {e}")
                    self.opik_enabled = False
            else:
                print("ℹ️  Opik not configured (set OPIK_USE_LOCAL=true for self-hosted or provide API key/workspace for cloud)")
                self.opik_enabled = False
        else:
            print("ℹ️  Opik not available")
            self.opik_enabled = False
    
    def initialize(self) -> None:
        AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
        AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
        
        self.llm = AzureChatOpenAI(
            azure_deployment = "gpt-4.1-maven-course",
            api_version = "2024-12-01-preview",
            temperature=0.2,
            api_key=AZURE_OPENAI_API_KEY,
            azure_endpoint=AZURE_OPENAI_ENDPOINT
        )
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        
        # Check if vector store already exists
        if os.path.exists(VECTOR_STORE_PATH):
            print("📦 Loading existing vector store from disk...")
            try:
                self.vector_store = FAISS.load_local(VECTOR_STORE_PATH, self.embeddings, allow_dangerous_deserialization=True)
                print("✅ Vector store loaded successfully!")
            except Exception as e:
                print(f"❌ Error loading vector store: {e}")
                print("🔄 Creating new vector store...")
                self._create_new_vector_store()
        else:
            print("🔄 No existing vector store found. Creating new one...")
            self._create_new_vector_store()
                
        graph = StateGraph(AgentState)
        
        graph.add_node("retrieval", self._create_retrieval_node)
        graph.add_node("answering", self._create_answering_node)
        
        graph.add_edge(START, "retrieval")
        graph.add_edge("retrieval", "answering")
        graph.add_edge("answering", END)
        
        self.graph = graph.compile()
    
    def _create_new_vector_store(self) -> None:
        """Create a new vector store and save it to disk"""
        docs = self._load_and_process_documents()
        
        # Print first document content for debugging
        print(f"First document content type: {type(docs[0].page_content)}")
        print(f"First document content (first 100 chars): {docs[0].page_content[:100]}")
        
        # Create vector store from documents
        print("🚀 Creating vector store from documents...")
        try:
            self.vector_store = FAISS.from_documents(docs, self.embeddings)
            print("💾 Saving vector store to disk...")
            self.vector_store.save_local(VECTOR_STORE_PATH)
            print("✅ Vector store created and saved successfully!")
        except Exception as e:
            print(f"❌ Error creating vector store: {e}")
            # Fallback to batch processing if bulk creation fails
            self._create_vector_store_batch_processing(docs)
    
    def _create_vector_store_batch_processing(self, docs: List[Document]) -> None:
        """Fallback method to create vector store with batch processing"""
        print("🔄 Falling back to batch processing...")
        self.vector_store = FAISS.from_documents([docs[0]], self.embeddings)  # Initialize with first doc
        
        batch_size = 100
        for i in range(1, len(docs), batch_size):
            print(f"Adding documents {i} to {i + batch_size} to vector store")
            batch = docs[i:i + batch_size]
            
            # Validate batch before sending to embeddings
            validated_batch = []
            for doc in batch:
                if isinstance(doc.page_content, str) and doc.page_content.strip():
                    # Ensure content is clean and doesn't contain problematic characters
                    cleaned_content = doc.page_content.encode('utf-8', errors='ignore').decode('utf-8')
                    if cleaned_content.strip():
                        validated_doc = Document(page_content=cleaned_content, metadata=doc.metadata)
                        validated_batch.append(validated_doc)
                else:
                    print(f"Skipping invalid document: type={type(doc.page_content)}, content={repr(doc.page_content)}")
            
            if validated_batch:
                try:
                    self.vector_store.add_documents(validated_batch)
                except Exception as e:
                    print(f"Error adding documents to vector store: {e}")
                    # Try adding documents one by one to isolate the problematic one
                    for j, doc in enumerate(validated_batch):
                        try:
                            self.vector_store.add_documents([doc])
                        except Exception as doc_error:
                            print(f"Failed to add document {i+j}: {doc_error}")
                            print(f"Document content type: {type(doc.page_content)}")
                            print(f"Document content preview: {repr(doc.page_content[:100])}")
            
            if i + batch_size < len(docs): 
                time.sleep(10)
        
        # Save the vector store after batch processing
        print("💾 Saving vector store to disk...")
        self.vector_store.save_local(VECTOR_STORE_PATH)
        print("✅ Vector store created and saved successfully!")
    
    def _load_and_process_documents(self) -> List[Document]:
        """Load and process documents from the specified directory"""
        from langchain_community.document_loaders import PyPDFLoader
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        
        docs = []
        
        print(f"Loading document from {FILE_PATH}")
        loader = PyPDFLoader(FILE_PATH)
        page_docs = loader.load()
        
        combined_doc = "\n".join([doc.page_content for doc in page_docs])

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_text(combined_doc)
        
        # Filter out empty chunks and ensure they're strings
        valid_chunks = [chunk for chunk in chunks if chunk and chunk.strip()]
        
        # Create documents with additional validation
        for chunk in valid_chunks:
            content = str(chunk).strip()
            # Ensure content is not empty and is a valid string
            if content and isinstance(content, str):
                docs.append(Document(page_content=content, metadata={"source": FILE_PATH}))
        return docs
    
    @track(name="document_retrieval")
    def _create_retrieval_node(self, state: AgentState):
        retriever = self.vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
        
        # Perform retrieval with detailed logging
        docs = retriever.invoke(state["question"])
        
        # Log retrieval details to current span if Opik is enabled
        if self.opik_enabled and OPIK_AVAILABLE:
            try:
                current_span = opik_context.get_current_span()
                if current_span:
                    # Log retrieval metadata
                    current_span.log_metadata({
                        "search_query": state["question"],
                        "num_docs_retrieved": len(docs),
                        "search_type": "similarity",
                        "search_k": 3
                    })
                    
                    # Log retrieved documents (content snippets for readability)
                    doc_summaries = []
                    for i, doc in enumerate(docs):
                        doc_summaries.append({
                            "doc_index": i,
                            "source": doc.metadata.get("source", "unknown"),
                            "content_preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                            "content_length": len(doc.page_content)
                        })
                    
                    current_span.log_metadata({
                        "retrieved_documents": doc_summaries
                    })
            except Exception as e:
                print(f"Warning: Failed to log retrieval details to Opik: {e}")
        
        return {"retrieved_docs": docs}
    
    @track(name="answer_generation")
    def _create_answering_node(self, state: AgentState):
        prompt = RAG_PROMPT
        
        model_response_with_structure = self.llm.with_structured_output(RagGenerationResponse)
        chain = prompt | model_response_with_structure
        
        # Log generation details to current span if Opik is enabled
        if self.opik_enabled and OPIK_AVAILABLE:
            try:
                current_span = opik_context.get_current_span()
                if current_span:
                    # Log generation input details
                    current_span.log_metadata({
                        "question": state["question"],
                        "num_context_docs": len(state["retrieved_docs"]),
                        "llm_model": "gpt-4.1-maven-course",
                        "temperature": 0.2,
                        "prompt_template": "RAG_PROMPT"  # Could expand this to show actual prompt
                    })
            except Exception as e:
                print(f"Warning: Failed to log generation input to Opik: {e}")
        
        response = chain.invoke({
            "retrieved_docs": state["retrieved_docs"],
            "question": state["question"]
        })
        
        # Log generation output details
        if self.opik_enabled and OPIK_AVAILABLE:
            try:
                current_span = opik_context.get_current_span()
                if current_span:
                    current_span.log_metadata({
                        "structured_response": {
                            "answer": response.answer,
                            "sources": response.sources,
                            "answer_length": len(response.answer),
                            "num_sources_identified": len(response.sources) if response.sources else 0
                        }
                    })
            except Exception as e:
                print(f"Warning: Failed to log generation output to Opik: {e}")
        
        print(response)
        response_str = f"Answer: {response.answer}\n"
        if response.sources:
            # Remove the file paths from the sources
            response.sources = [osp.basename(source) for source in response.sources]
            # Make sources one per line
            response_str += "\nSources:\n" + "\n".join(f"- {source}" for source in response.sources)
        return {"answer": response_str}
    
    def run(self, query: str) -> str:
        """Send query to LLM and return response"""
        if self.opik_enabled and OPIK_AVAILABLE:
            return self._run_with_tracing(query)
        else:
            return self._run_without_tracing(query)
    
    @track(name="agent_query", project_name="RegTechAI")
    def _run_with_tracing(self, query: str) -> str:
        """Run with Opik tracing enabled with comprehensive logging"""
        import time
        start_time = time.time()
        
        try:
            # Log initial query details
            if OPIK_AVAILABLE:
                try:
                    current_span = opik_context.get_current_span()
                    if current_span:
                        current_span.log_metadata({
                            "user_query": query,
                            "query_length": len(query),
                            "pipeline_type": "RAG",
                            "vector_store_type": "FAISS",
                            "embedding_model": "text-embedding-3-small"
                        })
                except Exception as e:
                    print(f"Warning: Failed to log initial query details: {e}")
            
            # Execute the RAG pipeline
            response = self.graph.invoke({"question": query})
            
            # Log final results and performance
            end_time = time.time()
            processing_time = end_time - start_time
            
            if OPIK_AVAILABLE:
                try:
                    current_span = opik_context.get_current_span()
                    if current_span:
                        current_span.log_metadata({
                            "final_response": response["answer"],
                            "total_processing_time_seconds": processing_time,
                            "pipeline_success": True
                        })
                except Exception as e:
                    print(f"Warning: Failed to log final results: {e}")
            
            return response["answer"]
            
        except Exception as e:
            # Log error details
            if OPIK_AVAILABLE:
                try:
                    current_span = opik_context.get_current_span()
                    if current_span:
                        current_span.log_metadata({
                            "error_occurred": True,
                            "error_message": str(e),
                            "error_type": type(e).__name__
                        })
                except Exception as log_error:
                    print(f"Warning: Failed to log error details: {log_error}")
            
            return f"Error: {str(e)}"
    
    def _run_without_tracing(self, query: str) -> str:
        """Run without tracing"""
        try:
            response = self.graph.invoke({"question": query})
            return response["answer"]
        except Exception as e:
            return f"Error: {str(e)}"
    
    def clear_history(self):
        """Clear conversation history"""
        self.messages = []