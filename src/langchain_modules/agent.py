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

from .prompts import RAG_PROMPT, CLASSIFICATION_PROMPT, NIS2_PROMPT, DORA_PROMPT, CER_PROMPT, EXEC_ORDER_PROMPT, SYNTHESIS_PROMPT
from .user_memory import UserMemory

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
    gdpr_docs: list[Document]
    nis2_docs: list[Document]
    dora_docs: list[Document]
    cer_docs: list[Document]
    exec_order_docs: list[Document]
    answer: str
    gdpr: bool
    nis2: bool
    dora: bool
    cer: bool
    exec_order: bool
    is_off_topic: bool
    rejection_reason: str
    
class RagGenerationResponse(BaseModel):
    """Response to the question with answer and sources. Sources are
    names of the documents. Sources should be None if the answer is not
    found in the context."""
    answer: str = Field(description="Answer to the question.")
    sources: list[str] = Field(
        description="Names of the documents that contain the answer.",
        default_factory=list
    )

class RegulationAssessment(BaseModel):
    """Assessment for a single regulation."""
    applies: bool = Field(description="Whether the regulation applies to this business case.")
    confidence: float = Field(description="Confidence level between 0 and 1.")
    explanation: str = Field(description="Brief explanation of why the regulation applies or doesn't apply.")

class ClassificationResponse(BaseModel):
    """Response indicating which regulations apply with confidence and explanation."""
    is_business_regulatory_question: bool = Field(description="Whether this question is about regulatory compliance for a business use case.")
    question_type: str = Field(description="Type of question: 'business_regulatory', 'off_topic', or 'unclear'")
    rejection_reason: str = Field(description="If off-topic, brief explanation of why this question cannot be answered.", default="")
    GDPR: RegulationAssessment = Field(description="GDPR regulation assessment.")
    NIS2: RegulationAssessment = Field(description="NIS2 regulation assessment.")
    DORA: RegulationAssessment = Field(description="DORA regulation assessment.")
    CER: RegulationAssessment = Field(description="CER regulation assessment.")
    EXEC_ORDER: RegulationAssessment = Field(description="US Executive Order regulatory assessment.")


GDPR_FILE_PATH = "/home/dan/capstone_project_v2/data/GDPR_Regulation.pdf"
NIS2_FILE_PATH = "/home/dan/capstone_project_v2/data/NIS2_Regulation.pdf"
DORA_FILE_PATH = "/home/dan/capstone_project_v2/data/DORA_Regulation.pdf"
CER_FILE_PATH = "/home/dan/capstone_project_v2/data/CER_Regulation.pdf"
EXEC_ORDER_DATA_DIR = "/home/dan/capstone_project_v2/data/executive_orders/"

GDPR_VECTOR_STORE_PATH = "/home/dan/capstone_project_v2/data/gdpr_vector_store"
NIS2_VECTOR_STORE_PATH = "/home/dan/capstone_project_v2/data/nis2_vector_store" 
DORA_VECTOR_STORE_PATH = "/home/dan/capstone_project_v2/data/dora_vector_store"
CER_VECTOR_STORE_PATH = "/home/dan/capstone_project_v2/data/cer_vector_store"
EXEC_ORDER_VECTOR_STORE_PATH = "/home/dan/capstone_project_v2/data/exec_order_vector_store"

class AIAgent:
    def __init__(self):
        self.llm = None
        self.messages = []
        self._setup_opik()
        self.user_memory = UserMemory()  # Initialize user memory system
        self.current_user_email = "danno@ragequitlab.com"  # POC default user
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
            temperature=0.3,
            api_key=AZURE_OPENAI_API_KEY,
            azure_endpoint=AZURE_OPENAI_ENDPOINT
        )
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        
        # Initialize vector stores for each framework
        self.gdpr_vector_store = self._load_or_create_vector_store("GDPR", GDPR_VECTOR_STORE_PATH, GDPR_FILE_PATH)
        self.nis2_vector_store = self._load_or_create_vector_store("NIS2", NIS2_VECTOR_STORE_PATH, NIS2_FILE_PATH)
        self.dora_vector_store = self._load_or_create_vector_store("DORA", DORA_VECTOR_STORE_PATH, DORA_FILE_PATH)
        self.cer_vector_store = self._load_or_create_vector_store("CER", CER_VECTOR_STORE_PATH, CER_FILE_PATH)
        self.exec_order_vector_store = self._load_or_create_exec_order_vector_store("EXEC_ORDER", EXEC_ORDER_VECTOR_STORE_PATH, EXEC_ORDER_DATA_DIR)
        
        # For backward compatibility, set the main vector store to GDPR
        self.vector_store = self.gdpr_vector_store
                
        graph = StateGraph(AgentState)
        
        # Add all nodes
        graph.add_node("classification", self._create_classification_node)
        graph.add_node("gdpr_analysis", self._create_gdpr_analysis_node)
        graph.add_node("nis2_analysis", self._create_nis2_analysis_node)
        graph.add_node("dora_analysis", self._create_dora_analysis_node)
        graph.add_node("cer_analysis", self._create_cer_analysis_node)
        graph.add_node("exec_order_analysis", self._create_exec_order_analysis_node)
        graph.add_node("synthesis", self._create_synthesis_node)
        graph.add_node("answering", self._create_answering_node)
        graph.add_node("rejection", self._create_rejection_node)
        
        # Start with classification
        graph.add_edge(START, "classification")
        
        # Use conditional edges to route to applicable framework analysis or rejection
        graph.add_conditional_edges(
            "classification",
            self._route_to_applicable_frameworks,
            {
                "gdpr_analysis": "gdpr_analysis",
                "nis2_analysis": "nis2_analysis", 
                "dora_analysis": "dora_analysis",
                "cer_analysis": "cer_analysis",
                "exec_order_analysis": "exec_order_analysis",
                "rejection": "rejection"
            }
        )
        
        # All analysis nodes feed into synthesis
        graph.add_edge("gdpr_analysis", "synthesis")
        graph.add_edge("nis2_analysis", "synthesis")
        graph.add_edge("dora_analysis", "synthesis")
        graph.add_edge("cer_analysis", "synthesis")
        graph.add_edge("exec_order_analysis", "synthesis")
        
        # Synthesis feeds into answering, then end
        graph.add_edge("synthesis", "answering")
        graph.add_edge("answering", END)
        
        # Rejection goes directly to end
        graph.add_edge("rejection", END)
        
        self.graph = graph.compile()
    
    def _load_or_create_vector_store(self, framework_name: str, store_path: str, file_path: str):
        """Load existing vector store or create new one for a specific framework"""
        if os.path.exists(store_path):
            print(f"📦 Loading existing {framework_name} vector store from disk...")
            try:
                vector_store = FAISS.load_local(store_path, self.embeddings, allow_dangerous_deserialization=True)
                print(f"✅ {framework_name} vector store loaded successfully!")
                return vector_store
            except Exception as e:
                print(f"❌ Error loading {framework_name} vector store: {e}")
                print(f"🔄 Creating new {framework_name} vector store...")
                return self._create_framework_vector_store(framework_name, store_path, file_path)
        else:
            print(f"🔄 No existing {framework_name} vector store found. Creating new one...")
            return self._create_framework_vector_store(framework_name, store_path, file_path)
    
    def _create_framework_vector_store(self, framework_name: str, store_path: str, file_path: str):
        """Create a new vector store for a specific regulatory framework"""
        docs = self._load_and_process_documents(file_path, framework_name)
        
        if not docs:
            print(f"⚠️ No documents found for {framework_name}, creating empty store")
            # Create minimal vector store with placeholder
            placeholder_doc = Document(page_content=f"{framework_name} placeholder document", metadata={"source": file_path, "framework": framework_name})
            docs = [placeholder_doc]
        
        # Print first document content for debugging
        print(f"First {framework_name} document content type: {type(docs[0].page_content)}")
        print(f"First {framework_name} document content (first 100 chars): {docs[0].page_content[:100]}")
        
        # Create vector store from documents
        print(f"🚀 Creating {framework_name} vector store from documents...")
        try:
            vector_store = FAISS.from_documents(docs, self.embeddings)
            print(f"💾 Saving {framework_name} vector store to disk...")
            vector_store.save_local(store_path)
            print(f"✅ {framework_name} vector store created and saved successfully!")
            return vector_store
        except Exception as e:
            print(f"❌ Error creating {framework_name} vector store: {e}")
            # Fallback to batch processing if bulk creation fails
            return self._create_vector_store_batch_processing(docs, store_path, framework_name)
    
    def _create_new_vector_store(self) -> None:
        """Create a new vector store and save it to disk (legacy method for backward compatibility)"""
        docs = self._load_and_process_documents(GDPR_FILE_PATH, "GDPR")
        
        if not docs:
            print("⚠️ No documents found for GDPR")
            return
            
        # Print first document content for debugging
        print(f"First document content type: {type(docs[0].page_content)}")
        print(f"First document content (first 100 chars): {docs[0].page_content[:100]}")
        
        # Create vector store from documents
        print("🚀 Creating vector store from documents...")
        try:
            self.vector_store = FAISS.from_documents(docs, self.embeddings)
            print("💾 Saving vector store to disk...")
            self.vector_store.save_local(GDPR_VECTOR_STORE_PATH)
            print("✅ Vector store created and saved successfully!")
        except Exception as e:
            print(f"❌ Error creating vector store: {e}")
            # Fallback to batch processing if bulk creation fails
            self._create_vector_store_batch_processing(docs, GDPR_VECTOR_STORE_PATH, "GDPR")
    
    def _create_vector_store_batch_processing(self, docs: List[Document], store_path: str = None, framework_name: str = "GDPR"):
        """Fallback method to create vector store with batch processing"""
        print(f"🔄 Falling back to batch processing for {framework_name}...")
        vector_store = FAISS.from_documents([docs[0]], self.embeddings)  # Initialize with first doc
        
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
                    vector_store.add_documents(validated_batch)
                except Exception as e:
                    print(f"Error adding documents to vector store: {e}")
                    # Try adding documents one by one to isolate the problematic one
                    for j, doc in enumerate(validated_batch):
                        try:
                            vector_store.add_documents([doc])
                        except Exception as doc_error:
                            print(f"Failed to add document {i+j}: {doc_error}")
                            print(f"Document content type: {type(doc.page_content)}")
                            print(f"Document content preview: {repr(doc.page_content[:100])}")
            
            if i + batch_size < len(docs): 
                time.sleep(10)
        
        # Save the vector store after batch processing
        save_path = store_path if store_path else GDPR_VECTOR_STORE_PATH
        print(f"💾 Saving {framework_name} vector store to disk...")
        vector_store.save_local(save_path)
        print(f"✅ {framework_name} vector store created and saved successfully!")
        return vector_store
    
    def _load_and_process_documents(self, file_path: str = None, framework_name: str = "GDPR") -> List[Document]:
        """Load and process documents from the specified file path"""
        from langchain_community.document_loaders import PyPDFLoader
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        
        if file_path is None:
            file_path = GDPR_FILE_PATH  # Default to GDPR for backward compatibility
            
        docs = []
        
        if not os.path.exists(file_path):
            print(f"⚠️ File not found: {file_path}")
            return docs
        
        print(f"Loading {framework_name} document from {file_path}")
        try:
            loader = PyPDFLoader(file_path)
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
                    docs.append(Document(page_content=content, metadata={"source": file_path, "framework": framework_name}))
            
            print(f"✅ Loaded {len(docs)} document chunks for {framework_name}")
        except Exception as e:
            print(f"❌ Error loading {framework_name} document: {e}")
            
        return docs
    
    def _load_or_create_exec_order_vector_store(self, framework_name: str, store_path: str, data_dir: str):
        """Load existing Executive Order vector store or create new one from directory of PDFs"""
        if os.path.exists(store_path):
            print(f"📦 Loading existing {framework_name} vector store from disk...")
            try:
                vector_store = FAISS.load_local(store_path, self.embeddings, allow_dangerous_deserialization=True)
                print(f"✅ {framework_name} vector store loaded successfully!")
                return vector_store
            except Exception as e:
                print(f"❌ Error loading {framework_name} vector store: {e}")
                print(f"🔄 Creating new {framework_name} vector store...")
                return self._create_exec_order_vector_store(framework_name, store_path, data_dir)
        else:
            print(f"🔄 No existing {framework_name} vector store found. Creating new one...")
            return self._create_exec_order_vector_store(framework_name, store_path, data_dir)
    
    def _create_exec_order_vector_store(self, framework_name: str, store_path: str, data_dir: str):
        """Create a new vector store for Executive Orders from directory of PDFs"""
        docs = self._load_and_process_exec_order_documents(data_dir, framework_name)
        
        if not docs:
            print(f"⚠️ No Executive Order documents found in {data_dir}, creating placeholder store")
            # Create minimal vector store with placeholder
            placeholder_doc = Document(
                page_content=f"{framework_name} placeholder - Executive Order documents will be loaded when PDFs are added to {data_dir}",
                metadata={"source": data_dir, "framework": framework_name, "document_type": "placeholder"}
            )
            docs = [placeholder_doc]
        
        # Print debug info
        print(f"First {framework_name} document content type: {type(docs[0].page_content)}")
        print(f"First {framework_name} document content (first 100 chars): {docs[0].page_content[:100]}")
        
        # Create vector store from documents
        print(f"🚀 Creating {framework_name} vector store from {len(docs)} documents...")
        try:
            vector_store = FAISS.from_documents(docs, self.embeddings)
            print(f"💾 Saving {framework_name} vector store to disk...")
            vector_store.save_local(store_path)
            print(f"✅ {framework_name} vector store created and saved successfully!")
            return vector_store
        except Exception as e:
            print(f"❌ Error creating {framework_name} vector store: {e}")
            return self._create_vector_store_batch_processing(docs, store_path, framework_name)
    
    def _expand_query_universal(self, original_query: str, framework: str, user_profile: dict = None) -> str:
        """
        Universal AI-powered query expansion that dynamically generates context-aware terms
        for any regulatory framework based on the user's query and profile.
        """
        
        # Framework-specific context mapping
        framework_contexts = {
            "GDPR": "data protection, privacy rights, personal data processing, cross-border transfers, consent management, data controller obligations",
            "NIS2": "cybersecurity, network security, incident reporting, essential services, digital infrastructure, cyber resilience", 
            "DORA": "operational resilience, financial services, ICT risk management, third-party providers, business continuity",
            "CER": "critical infrastructure, essential entities, resilience measures, security requirements, operational continuity",
            "EXEC_ORDER": "presidential directives, federal compliance, trade restrictions, national security, regulatory compliance"
        }
        
        # Build context for expansion
        context_parts = [
            f"Original question: {original_query}",
            f"Regulatory framework: {framework}",
            f"Framework focus: {framework_contexts.get(framework, 'regulatory compliance')}"
        ]
        
        # Add user context if available
        if user_profile:
            if user_profile.get('industry'):
                context_parts.append(f"User industry: {user_profile['industry']}")
            if user_profile.get('company'):
                context_parts.append(f"Company context: {user_profile['company']}")
            if user_profile.get('country'):
                context_parts.append(f"Country: {user_profile['country']}")
            if user_profile.get('regulatory_focus'):
                context_parts.append(f"Previous regulatory interests: {', '.join(user_profile['regulatory_focus'])}")
        
        # Create expansion prompt
        expansion_prompt = f"""
Based on the following context, generate 8-12 relevant technical and regulatory terms that would improve document retrieval for this {framework} query. Focus on:

1. Industry-specific terminology related to the user's business
2. Regulatory concepts and legal terminology
3. Technical terms and operational aspects
4. Cross-references to related compliance areas
5. Business process terminology
6. Geographic and jurisdictional terms if relevant

Context:
{' | '.join(context_parts)}

Generate only the expansion terms as a comma-separated list, no explanations:
"""

        try:
            # Use the LLM to generate dynamic expansion terms
            response = self.llm.invoke(expansion_prompt)
            expansion_terms_text = response.content.strip()
            
            # Parse the response and clean up terms
            expansion_terms = [term.strip() for term in expansion_terms_text.split(',')]
            expansion_terms = [term for term in expansion_terms if term and len(term) > 2]
            
            # Limit to reasonable number of terms
            expansion_terms = expansion_terms[:10]
            
            # Create expanded query
            if expansion_terms:
                expanded_query = f"{original_query} {' '.join(expansion_terms)}"
            else:
                # Fallback to basic framework context
                expanded_query = f"{original_query} {framework_contexts.get(framework, 'regulatory compliance')}"
                
        except Exception as e:
            print(f"⚠️ Error in universal query expansion: {e}")
            # Fallback to basic framework context
            expanded_query = f"{original_query} {framework_contexts.get(framework, 'regulatory compliance')}"
        
        return expanded_query

    def _load_and_process_exec_order_documents(self, data_dir: str, framework_name: str = "EXEC_ORDER") -> List[Document]:
        """Load and process multiple Executive Order PDF documents from a directory"""
        from langchain_community.document_loaders import PyPDFLoader
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        
        docs = []
        
        if not os.path.exists(data_dir):
            print(f"⚠️ Directory not found: {data_dir}")
            return docs
        
        # Find all PDF files in the directory
        pdf_files = [f for f in os.listdir(data_dir) if f.lower().endswith('.pdf')]
        
        if not pdf_files:
            print(f"⚠️ No PDF files found in {data_dir}")
            return docs
        
        print(f"📄 Found {len(pdf_files)} Executive Order PDFs in {data_dir}")
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        
        for pdf_file in sorted(pdf_files):  # Process in alphabetical order
            file_path = os.path.join(data_dir, pdf_file)
            print(f"Loading Executive Order document: {pdf_file}")
            
            try:
                loader = PyPDFLoader(file_path)
                page_docs = loader.load()
                
                # Combine all pages into one document
                combined_doc = "\n".join([doc.page_content for doc in page_docs])
                
                # Split into chunks
                chunks = text_splitter.split_text(combined_doc)
                
                # Filter out empty chunks
                valid_chunks = [chunk for chunk in chunks if chunk and chunk.strip()]
                
                # Create documents with metadata
                for chunk in valid_chunks:
                    content = str(chunk).strip()
                    if content and isinstance(content, str):
                        docs.append(Document(
                            page_content=content, 
                            metadata={
                                "source": file_path, 
                                "framework": framework_name,
                                "document_name": pdf_file,
                                "document_type": "executive_order"
                            }
                        ))
                
                print(f"✅ Processed {pdf_file}: {len(valid_chunks)} chunks")
                
            except Exception as e:
                print(f"❌ Error loading {pdf_file}: {e}")
                continue
        
        print(f"✅ Total Executive Order chunks loaded: {len(docs)}")
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
    
    @track(name="gdpr_analysis")
    def _create_gdpr_analysis_node(self, state: AgentState):
        print(f"\n🏛️ GDPR ANALYSIS NODE EXECUTING")
        print(f"📝 Question: {state['question']}")
        
        # Get user context for query expansion
        user_context = self.user_memory.get_user_context(self.current_user_email)
        user_profile = user_context.get('profile')
        
        # Generate expanded query for better retrieval using universal expansion
        expanded_query = self._expand_query_universal(state["question"], "GDPR", user_profile)
        print(f"🔍 Expanded GDPR query: {expanded_query}")
        
        # Retrieve GDPR documents with expanded query
        retriever = self.gdpr_vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
        docs = retriever.invoke(expanded_query)
        print(f"📄 GDPR analysis: {len(docs)} documents retrieved")
        
        # Generate GDPR-specific response
        model_response_with_structure = self.llm.with_structured_output(RagGenerationResponse)
        chain = RAG_PROMPT | model_response_with_structure
        
        response = chain.invoke({
            "retrieved_docs": docs,
            "question": state["question"]
        })
        
        # Tag documents with framework
        for doc in docs:
            doc.metadata["framework"] = "GDPR"
            doc.metadata["analysis"] = response.answer
        
        return {"gdpr_docs": docs}
    
    @track(name="nis2_analysis")
    def _create_nis2_analysis_node(self, state: AgentState):
        print(f"\n🔒 NIS2 ANALYSIS NODE EXECUTING")
        print(f"📝 Question: {state['question']}")
        
        # Get user context for query expansion
        user_context = self.user_memory.get_user_context(self.current_user_email)
        user_profile = user_context.get('profile')
        
        # Generate expanded query for better retrieval using universal expansion
        expanded_query = self._expand_query_universal(state["question"], "NIS2", user_profile)
        print(f"🔍 Expanded NIS2 query: {expanded_query}")
        
        # Retrieve NIS2 documents with expanded query
        retriever = self.nis2_vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
        docs = retriever.invoke(expanded_query)
        print(f"📄 NIS2 analysis: {len(docs)} documents retrieved")
        
        # Generate NIS2-specific response
        model_response_with_structure = self.llm.with_structured_output(RagGenerationResponse)
        chain = NIS2_PROMPT | model_response_with_structure
        
        response = chain.invoke({
            "retrieved_docs": docs,
            "question": state["question"]
        })
        
        # Tag documents with framework and analysis
        for doc in docs:
            doc.metadata["framework"] = "NIS2"
            doc.metadata["analysis"] = response.answer
        
        return {"nis2_docs": docs}
    
    @track(name="dora_analysis")
    def _create_dora_analysis_node(self, state: AgentState):
        print(f"\n🏦 DORA ANALYSIS NODE EXECUTING")
        print(f"📝 Question: {state['question']}")
        
        # Get user context for query expansion
        user_context = self.user_memory.get_user_context(self.current_user_email)
        user_profile = user_context.get('profile')
        
        # Generate expanded query for better retrieval using universal expansion
        expanded_query = self._expand_query_universal(state["question"], "DORA", user_profile)
        print(f"🔍 Expanded DORA query: {expanded_query}")
        
        # Retrieve DORA documents with expanded query
        retriever = self.dora_vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
        docs = retriever.invoke(expanded_query)
        print(f"📄 DORA analysis: {len(docs)} documents retrieved")
        
        # Generate DORA-specific response
        model_response_with_structure = self.llm.with_structured_output(RagGenerationResponse)
        chain = DORA_PROMPT | model_response_with_structure
        
        response = chain.invoke({
            "retrieved_docs": docs,
            "question": state["question"]
        })
        
        # Tag documents with framework and analysis
        for doc in docs:
            doc.metadata["framework"] = "DORA"
            doc.metadata["analysis"] = response.answer
        
        return {"dora_docs": docs}
    
    @track(name="cer_analysis")
    def _create_cer_analysis_node(self, state: AgentState):
        print(f"\n🏗️ CER ANALYSIS NODE EXECUTING")
        print(f"📝 Question: {state['question']}")
        
        # Get user context for query expansion
        user_context = self.user_memory.get_user_context(self.current_user_email)
        user_profile = user_context.get('profile')
        
        # Generate expanded query for better retrieval using universal expansion
        expanded_query = self._expand_query_universal(state["question"], "CER", user_profile)
        print(f"🔍 Expanded CER query: {expanded_query}")
        
        # Retrieve CER documents with expanded query
        retriever = self.cer_vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
        docs = retriever.invoke(expanded_query)
        print(f"📄 CER analysis: {len(docs)} documents retrieved")
        
        # Generate CER-specific response
        model_response_with_structure = self.llm.with_structured_output(RagGenerationResponse)
        chain = CER_PROMPT | model_response_with_structure
        
        response = chain.invoke({
            "retrieved_docs": docs,
            "question": state["question"]
        })
        
        # Tag documents with framework and analysis
        for doc in docs:
            doc.metadata["framework"] = "CER"
            doc.metadata["analysis"] = response.answer
        
        return {"cer_docs": docs}
    
    @track(name="exec_order_analysis")
    def _create_exec_order_analysis_node(self, state: AgentState):
        print(f"\n🇺🇸 EXECUTIVE ORDER ANALYSIS NODE EXECUTING")
        print(f"📝 Question: {state['question']}")
        
        # Get user context for query expansion
        user_context = self.user_memory.get_user_context(self.current_user_email)
        user_profile = user_context.get('profile')
        
        # Generate expanded query for better retrieval using universal expansion
        expanded_query = self._expand_query_universal(state["question"], "EXEC_ORDER", user_profile)
        print(f"🔍 Expanded Executive Order query: {expanded_query}")
        
        # Retrieve Executive Order documents with expanded query
        retriever = self.exec_order_vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
        docs = retriever.invoke(expanded_query)
        print(f"📄 Executive Order analysis: {len(docs)} documents retrieved")
        
        # Generate Executive Order-specific response
        model_response_with_structure = self.llm.with_structured_output(RagGenerationResponse)
        chain = EXEC_ORDER_PROMPT | model_response_with_structure
        
        response = chain.invoke({
            "retrieved_docs": docs,
            "question": state["question"]
        })
        
        # Tag documents with framework and analysis
        for doc in docs:
            doc.metadata["framework"] = "EXEC_ORDER"
            doc.metadata["analysis"] = response.answer
        
        return {"exec_order_docs": docs}
    
    def _route_to_applicable_frameworks(self, state: AgentState):
        """Determine which framework analysis nodes should be executed based on classification"""
        print(f"\n🚏 ROUTING DEBUG:")
        
        # Check if question was rejected as off-topic
        if state.get("is_off_topic", False):
            print(f"🚫 QUESTION REJECTED AS OFF-TOPIC")
            print(f"❌ Rejection reason: {state.get('rejection_reason', 'Unknown')}")
            return ["rejection"]
        
        print(f"📥 Input state booleans:")
        print(f"   🏛️  GDPR: {state.get('gdpr', 'NOT_SET')}")
        print(f"   🔒 NIS2: {state.get('nis2', 'NOT_SET')}")
        print(f"   🏦 DORA: {state.get('dora', 'NOT_SET')}")
        print(f"   🏗️  CER: {state.get('cer', 'NOT_SET')}")
        print(f"   🇺🇸 EXEC_ORDER: {state.get('exec_order', 'NOT_SET')}")
        
        applicable_frameworks = []
        
        if state.get("gdpr", False):
            applicable_frameworks.append("gdpr_analysis")
            print(f"   ✅ Adding GDPR analysis")
        else:
            print(f"   ❌ Skipping GDPR analysis (value: {state.get('gdpr', 'NOT_SET')})")
            
        if state.get("nis2", False):
            applicable_frameworks.append("nis2_analysis")
            print(f"   ✅ Adding NIS2 analysis")
        else:
            print(f"   ❌ Skipping NIS2 analysis (value: {state.get('nis2', 'NOT_SET')})")
            
        if state.get("dora", False):
            applicable_frameworks.append("dora_analysis")
            print(f"   ✅ Adding DORA analysis")
        else:
            print(f"   ❌ Skipping DORA analysis (value: {state.get('dora', 'NOT_SET')})")
            
        if state.get("cer", False):
            applicable_frameworks.append("cer_analysis")
            print(f"   ✅ Adding CER analysis")
        else:
            print(f"   ❌ Skipping CER analysis (value: {state.get('cer', 'NOT_SET')})")
            
        if state.get("exec_order", False):
            applicable_frameworks.append("exec_order_analysis")
            print(f"   ✅ Adding Executive Order analysis")
        else:
            print(f"   ❌ Skipping Executive Order analysis (value: {state.get('exec_order', 'NOT_SET')})")
        
        # If no frameworks are applicable, default to GDPR
        if not applicable_frameworks:
            applicable_frameworks = ["gdpr_analysis"]
            print(f"   🔄 No frameworks selected, defaulting to GDPR")
            
        print(f"\n🎯 FINAL ROUTING DECISION: {applicable_frameworks}")
        return applicable_frameworks
    
    @track(name="classify_regulation_types")
    def _create_classification_node(self, state: AgentState):
        print(f"\n🔍 CLASSIFICATION DEBUG:")
        print(f"📝 Input question: {state['question']}")
        
        # Get and display user context for debugging
        user_context = self.user_memory.get_user_context(self.current_user_email)
        user_profile = user_context.get('profile')
        
        print(f"\n👤 USER CONTEXT DEBUG:")
        if user_profile:
            print(f"   📧 Email: {user_profile.get('email', 'Not set')}")
            print(f"   🏢 Industry: {user_profile.get('industry', 'Not set')}")
            print(f"   🏛️  Company: {user_profile.get('company', 'Not set')}")
            print(f"   🌎 Country: {user_profile.get('country', 'Not set')}")
            print(f"   📊 Regulatory Focus: {user_profile.get('regulatory_focus', [])}")
            print(f"   📈 Total Interactions: {user_context.get('interaction_count', 0)}")
            if user_context.get('recent_topics'):
                print(f"   🔖 Recent Topics: {user_context.get('recent_topics', [])}")
            if user_context.get('topic_tags'):
                top_tags = list(user_context.get('topic_tags', {}).keys())[:5]
                print(f"   🏷️  Top Topic Tags: {top_tags}")
            if user_context.get('regulatory_patterns'):
                print(f"   📋 Framework Usage Patterns: {user_context.get('regulatory_patterns', {})}")
        else:
            print(f"   ⚠️  No user profile found for {self.current_user_email}")
        
        prompt = CLASSIFICATION_PROMPT
        
        model_response_with_structure = self.llm.with_structured_output(ClassificationResponse)
        chain = prompt | model_response_with_structure
        
        # Prepare user context for classification
        user_context_str = "No user context available"
        if user_profile:
            context_parts = []
            if user_profile.get('industry'):
                context_parts.append(f"Industry: {user_profile['industry']}")
            if user_profile.get('company'):
                context_parts.append(f"Company: {user_profile['company']}")
            if user_profile.get('country'):
                context_parts.append(f"Country: {user_profile['country']}")
            if user_profile.get('regulatory_focus'):
                context_parts.append(f"Known regulatory interests: {', '.join(user_profile['regulatory_focus'])}")
            if user_context.get('regulatory_patterns'):
                most_used = max(user_context['regulatory_patterns'].items(), key=lambda x: x[1], default=None)
                if most_used:
                    context_parts.append(f"Most consulted framework: {most_used[0]}")
            
            if context_parts:
                user_context_str = "; ".join(context_parts)
        
        print(f"🤖 Sending to LLM for classification with user context...")
        print(f"👤 User context: {user_context_str}")
        response = chain.invoke({
            "question": state["question"],
            "user_context": user_context_str
        })
        
        # Check if question is off-topic first
        print(f"\n🎯 TOPIC CLASSIFICATION:")
        print(f"📋 Is business regulatory question: {response.is_business_regulatory_question}")
        print(f"📋 Question type: {response.question_type}")
        if response.rejection_reason:
            print(f"❌ Rejection reason: {response.rejection_reason}")
        
        # If off-topic, return rejection result
        if not response.is_business_regulatory_question:
            print(f"\n🚫 REJECTING OFF-TOPIC QUESTION")
            return {
                "gdpr": False,
                "nis2": False,
                "dora": False,
                "cer": False,
                "is_off_topic": True,
                "rejection_reason": response.rejection_reason
            }
        
        # Debug each framework classification
        print(f"\n📊 REGULATORY CLASSIFICATION RESULTS:")
        print(f"🏛️  GDPR: applies={response.GDPR.applies}, confidence={response.GDPR.confidence:.2f}")
        print(f"   └─ Explanation: {response.GDPR.explanation}")
        print(f"🔒 NIS2: applies={response.NIS2.applies}, confidence={response.NIS2.confidence:.2f}")
        print(f"   └─ Explanation: {response.NIS2.explanation}")
        print(f"🏦 DORA: applies={response.DORA.applies}, confidence={response.DORA.confidence:.2f}")
        print(f"   └─ Explanation: {response.DORA.explanation}")
        print(f"🏗️  CER: applies={response.CER.applies}, confidence={response.CER.confidence:.2f}")
        print(f"   └─ Explanation: {response.CER.explanation}")
        print(f"🇺🇸 EXEC_ORDER: applies={response.EXEC_ORDER.applies}, confidence={response.EXEC_ORDER.confidence:.2f}")
        print(f"   └─ Explanation: {response.EXEC_ORDER.explanation}")
        
        result = {
            "gdpr": response.GDPR.applies,
            "nis2": response.NIS2.applies,
            "dora": response.DORA.applies,
            "cer": response.CER.applies,
            "exec_order": response.EXEC_ORDER.applies,
            "is_off_topic": False,
            "rejection_reason": ""
        }
        
        print(f"\n✅ CLASSIFICATION FINAL RESULT: {result}")
        return result
    
    @track(name="synthesis")
    def _create_synthesis_node(self, state: AgentState):
        """Combine all framework analyses and documents for final synthesis"""
        print(f"\n🔄 SYNTHESIS NODE EXECUTING")
        print(f"📝 Question: {state['question']}")
        print(f"🗃️  Available state keys: {list(state.keys())}")
        print(f"📊 Framework flags in state:")
        print(f"   🏛️  GDPR: {state.get('gdpr', 'NOT_SET')}")
        print(f"   🔒 NIS2: {state.get('nis2', 'NOT_SET')}")  
        print(f"   🏦 DORA: {state.get('dora', 'NOT_SET')}")
        print(f"   🏗️  CER: {state.get('cer', 'NOT_SET')}")
        
        # Show user context for synthesis awareness
        user_context = self.user_memory.get_user_context(self.current_user_email)
        user_profile = user_context.get('profile')
        
        print(f"\n👤 USER CONTEXT FOR SYNTHESIS:")
        if user_profile:
            print(f"   🏢 User Industry: {user_profile.get('industry', 'Unknown')}")
            print(f"   📊 Historical Regulatory Focus: {user_profile.get('regulatory_focus', [])}")
            if user_context.get('regulatory_patterns'):
                most_used = max(user_context['regulatory_patterns'].items(), key=lambda x: x[1], default=("None", 0))
                print(f"   🎯 Most Consulted Framework: {most_used[0]} ({most_used[1]}x)")
            if user_context.get('recent_topics'):
                print(f"   📋 Recent Topic Categories: {user_context.get('recent_topics', [])}")
        else:
            print(f"   ⚠️  No user profile available for synthesis personalization")
        
        all_docs = []
        framework_analyses = []
        
        # Collect documents and analyses from each framework that was classified as applicable
        if state.get("gdpr", False) and "gdpr_docs" in state:
            gdpr_docs = state.get("gdpr_docs", [])
            all_docs.extend(gdpr_docs)
            if gdpr_docs and "analysis" in gdpr_docs[0].metadata:
                framework_analyses.append(f"**GDPR Analysis:**\n{gdpr_docs[0].metadata['analysis']}")
            print(f"✅ Added {len(gdpr_docs)} GDPR documents to synthesis")
        else:
            print(f"❌ No GDPR documents to add (gdpr={state.get('gdpr', 'NOT_SET')}, has_docs={'gdpr_docs' in state})")
        
        if state.get("nis2", False) and "nis2_docs" in state:
            nis2_docs = state.get("nis2_docs", [])
            all_docs.extend(nis2_docs)
            if nis2_docs and "analysis" in nis2_docs[0].metadata:
                framework_analyses.append(f"**NIS2 Analysis:**\n{nis2_docs[0].metadata['analysis']}")
            print(f"✅ Added {len(nis2_docs)} NIS2 documents to synthesis")
        else:
            print(f"❌ No NIS2 documents to add (nis2={state.get('nis2', 'NOT_SET')}, has_docs={'nis2_docs' in state})")
        
        if state.get("dora", False) and "dora_docs" in state:
            dora_docs = state.get("dora_docs", [])
            all_docs.extend(dora_docs)
            if dora_docs and "analysis" in dora_docs[0].metadata:
                framework_analyses.append(f"**DORA Analysis:**\n{dora_docs[0].metadata['analysis']}")
            print(f"✅ Added {len(dora_docs)} DORA documents to synthesis")
        else:
            print(f"❌ No DORA documents to add (dora={state.get('dora', 'NOT_SET')}, has_docs={'dora_docs' in state})")
        
        if state.get("cer", False) and "cer_docs" in state:
            cer_docs = state.get("cer_docs", [])
            all_docs.extend(cer_docs)
            if cer_docs and "analysis" in cer_docs[0].metadata:
                framework_analyses.append(f"**CER Analysis:**\n{cer_docs[0].metadata['analysis']}")
            print(f"✅ Added {len(cer_docs)} CER documents to synthesis")
        else:
            print(f"❌ No CER documents to add (cer={state.get('cer', 'NOT_SET')}, has_docs={'cer_docs' in state})")
        
        if state.get("exec_order", False) and "exec_order_docs" in state:
            exec_order_docs = state.get("exec_order_docs", [])
            all_docs.extend(exec_order_docs)
            if exec_order_docs and "analysis" in exec_order_docs[0].metadata:
                framework_analyses.append(f"**Executive Order Analysis:**\n{exec_order_docs[0].metadata['analysis']}")
            print(f"✅ Added {len(exec_order_docs)} Executive Order documents to synthesis")
        else:
            print(f"❌ No Executive Order documents to add (exec_order={state.get('exec_order', 'NOT_SET')}, has_docs={'exec_order_docs' in state})")
        
        # Create a combined context that includes both documents and individual analyses
        combined_context = "\n\n".join([
            "## Individual Framework Analyses:",
            *framework_analyses,
            "\n## Supporting Regulatory Documents:",
            *[f"Framework: {doc.metadata.get('framework', 'Unknown')}\nContent: {doc.page_content}" for doc in all_docs[:10]]  # Limit to avoid context overflow
        ])
        
        # Create a synthetic document containing the combined analysis context
        synthesis_doc = Document(
            page_content=combined_context,
            metadata={"source": "multi_framework_synthesis", "frameworks": [doc.metadata.get('framework') for doc in all_docs]}
        )
        
        print(f"\n🔄 SYNTHESIS COMPLETE:")
        print(f"📄 Total documents combined: {len(all_docs)}")
        print(f"🧠 Framework analyses: {len(framework_analyses)}")
        print(f"📋 Frameworks with analyses: {[analysis.split(' Analysis:')[0].replace('**', '') for analysis in framework_analyses]}")
        print(f"✅ Returning synthesis document to answering node")
        return {"retrieved_docs": [synthesis_doc]}
    
    @track(name="rejection")
    def _create_rejection_node(self, state: AgentState):
        """Handle off-topic questions with polite rejection"""
        print(f"\n🚫 REJECTION NODE EXECUTING")
        print(f"📝 Question: {state['question']}")
        print(f"❌ Reason: {state.get('rejection_reason', 'Question is off-topic')}")
        
        rejection_message = f"""I apologize, but I'm a specialized regulatory compliance assistant focused on EU regulatory frameworks (GDPR, NIS2, DORA, and CER). 

{state.get('rejection_reason', 'Your question appears to be outside the scope of business regulatory compliance.')}

I can help you with:
- Understanding GDPR compliance for your business
- NIS2 requirements for essential/important entities
- DORA obligations for financial services
- CER requirements for critical infrastructure operators
- Regulatory compliance guidance for business operations

Please feel free to ask about regulatory compliance requirements for your business activities."""

        print(f"🔄 Generated rejection message")
        return {"answer": rejection_message}
    
    @track(name="answer_generation")
    def _create_answering_node(self, state: AgentState):
        print(f"\n🤖 ANSWER GENERATION DEBUG:")
        print(f"📝 Question: {state['question']}")
        
        # Get user context for personalized responses
        user_context = self.user_memory.get_user_context(self.current_user_email)
        user_profile = user_context.get('profile')
        
        print(f"\n👤 USER MEMORY CONTEXT:")
        if user_profile:
            print(f"   📧 User: {user_profile.get('email', 'Not set')}")
            print(f"   🏢 Industry: {user_profile.get('industry', 'Not set')}")
            print(f"   🌎 Country: {user_profile.get('country', 'Not set')}")
            print(f"   📊 Known Regulatory Interests: {user_profile.get('regulatory_focus', [])}")
            print(f"   📈 Previous Interactions: {user_context.get('interaction_count', 0)}")
            if user_context.get('regulatory_patterns'):
                print(f"   🎯 Most Used Frameworks: {user_context.get('regulatory_patterns', {})}")
        
        # Use synthesis prompt for multi-framework responses
        prompt = SYNTHESIS_PROMPT
        
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
                        "prompt_template": "SYNTHESIS_PROMPT",
                        "user_context": user_context
                    })
            except Exception as e:
                print(f"Warning: Failed to log generation input to Opik: {e}")
        
        response = chain.invoke({
            "retrieved_docs": state["retrieved_docs"],
            "question": state["question"]
        })
        
        # Store interaction in user memory
        frameworks_analyzed = []
        if state.get("gdpr", False):
            frameworks_analyzed.append("GDPR")
        if state.get("nis2", False):
            frameworks_analyzed.append("NIS2") 
        if state.get("dora", False):
            frameworks_analyzed.append("DORA")
        if state.get("cer", False):
            frameworks_analyzed.append("CER")
        if state.get("exec_order", False):
            frameworks_analyzed.append("EXEC_ORDER")
        
        # Infer user details from question and store interaction
        inferred_details = self.user_memory.infer_user_details(
            self.current_user_email, 
            state["question"], 
            frameworks_analyzed
        )
        
        # Update user profile with inferred details if applicable
        profile_updates = {}
        if inferred_details.get("potential_industry"):
            profile_updates["industry"] = inferred_details["potential_industry"]
        if inferred_details.get("potential_country"):
            profile_updates["country"] = inferred_details["potential_country"]
        
        if profile_updates:
            self.user_memory.update_user_profile(
                self.current_user_email,
                **profile_updates
            )
        
        self.user_memory.add_interaction(
            email=self.current_user_email,
            question=state["question"],
            frameworks_analyzed=frameworks_analyzed,
            answer_summary=response.answer,
            topic_category=inferred_details.get("potential_industry")
        )
        
        print(f"\n💾 MEMORY STORAGE DEBUG:")
        print(f"   📧 Stored for user: {self.current_user_email}")
        print(f"   📊 Frameworks recorded: {frameworks_analyzed}")
        print(f"   🏷️  Topic category: {inferred_details.get('potential_industry', 'None inferred')}")
        print(f"   🔍 Inferred industry: {inferred_details.get('potential_industry', 'None')}")
        print(f"   🌍 Inferred country: {inferred_details.get('potential_country', 'None')}")
        print(f"   📈 Regulatory complexity: {inferred_details.get('regulatory_complexity', 'medium')}")
        
        # Show updated profile
        updated_profile = self.user_memory.get_user_profile(self.current_user_email)
        if updated_profile:
            print(f"   📋 Updated regulatory focus: {updated_profile.regulatory_focus}")
            print(f"   🏢 Current industry: {updated_profile.industry or 'Not set'}")
            print(f"   🌎 Current country: {updated_profile.country or 'Not set'}")
        print(f"   ✅ Memory storage complete")
        
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
                        },
                        "memory_stored": True,
                        "frameworks_analyzed": frameworks_analyzed
                    })
            except Exception as e:
                print(f"Warning: Failed to log generation output to Opik: {e}")
        
        print(response)
        response_str = response.answer
        return {"answer": response_str}
    
    def run(self, query: str) -> str:
        """Send query to LLM and return response"""
        if self.opik_enabled and OPIK_AVAILABLE:
            return self._run_with_tracing(query)
        else:
            return self._run_without_tracing(query)
    
    @track(name="agent_query", project_name="RegTechAI-v2")
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
    
    def get_user_profile(self):
        """Get current user's profile"""
        return self.user_memory.get_user_profile(self.current_user_email)
    
    def get_user_interactions(self, limit=10):
        """Get current user's recent interactions"""
        return self.user_memory.get_user_interactions(self.current_user_email, limit)
    
    def get_user_context(self):
        """Get current user's context for personalized responses"""
        return self.user_memory.get_user_context(self.current_user_email)
    
    def update_user_info(self, **kwargs):
        """Update current user's profile information"""
        return self.user_memory.update_user_profile(self.current_user_email, **kwargs)
    
    def switch_user(self, email: str):
        """Switch to a different user profile"""
        profile = self.user_memory.get_user_profile(email)
        if profile:
            self.current_user_email = email
            print(f"🔄 Switched to user: {email}")
            return profile
        else:
            print(f"❌ User profile not found: {email}")
            return None
    
    def list_all_users(self):
        """List all user profiles in the system"""
        profiles = self.user_memory._load_profiles()
        return list(profiles.keys())
    
    def get_user_summary(self, email: str = None):
        """Get a summary of user profile and interactions"""
        target_email = email or self.current_user_email
        profile = self.user_memory.get_user_profile(target_email)
        context = self.user_memory.get_user_context(target_email)
        interactions = self.user_memory.get_user_interactions(target_email, limit=3)
        
        if not profile:
            return f"No profile found for {target_email}"
        
        summary = f"""
👤 **User Profile: {profile.email}**
🏢 Industry: {profile.industry or 'Not set'}
🏛️  Company: {profile.company or 'Not set'}  
👔 Role: {profile.role or 'Not set'}
🌎 Country: {profile.country or 'Not set'}
📊 Regulatory Focus: {', '.join(profile.regulatory_focus) if profile.regulatory_focus else 'None'}
📈 Total Interactions: {context['interaction_count']}
🔖 Recent Topics: {', '.join(context['recent_topics'][:3]) if context['recent_topics'] else 'None'}
📋 Framework Usage: {context['regulatory_patterns']}
        """
        
        if interactions:
            summary += "\n🕒 **Recent Questions:**\n"
            for i, interaction in enumerate(interactions[:2], 1):
                summary += f"{i}. {interaction.question[:80]}...\n"
        
        return summary.strip()