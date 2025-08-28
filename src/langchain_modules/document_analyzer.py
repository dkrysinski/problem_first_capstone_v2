"""
Document Analyzer for Push Notifications
Analyzes new regulatory documents to identify impacted users
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import pypdf
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI
from pydantic import BaseModel
from .user_memory import UserMemory
from .agent import AIAgent
import os

# Opik tracing imports
try:
    from opik import track, opik_context
    from opik.integrations.langchain import OpikTracer
    OPIK_AVAILABLE = True
except ImportError:
    OPIK_AVAILABLE = False
    def track(func):
        return func
    opik_context = None

@dataclass
class DocumentAnalysis:
    """Analysis results for a document"""
    frameworks: List[str]
    topic_tags: List[str]
    business_relevance: str
    geographic_scope: List[str]
    industry_focus: List[str]
    summary: str
    key_requirements: List[str]

@dataclass
class UserImpactScore:
    """Impact scoring for a user"""
    email: str
    impact_score: float  # 0.0 to 1.0
    relevance_reasons: List[str]
    applicable_frameworks: List[str]
    matched_topics: List[str]

class DocumentImpactResponse(BaseModel):
    """Structured response for document impact analysis"""
    frameworks: List[str]
    topic_tags: List[str] 
    business_relevance: str
    geographic_scope: List[str]
    industry_focus: List[str]
    summary: str
    key_requirements: List[str]

class DocumentAnalyzer:
    """Analyzes regulatory documents and identifies impacted users"""
    
    def __init__(self, agent: AIAgent = None):
        self.agent = agent or AIAgent()
        self.user_memory = self.agent.user_memory
        
        # Initialize LLM
        self.llm = AzureChatOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version="2024-02-01",
            model="gpt-4",
            temperature=0.1,
        )
        
        # Initialize Opik tracer if available
        self.opik_tracer = None
        if OPIK_AVAILABLE:
            try:
                self.opik_tracer = OpikTracer()
                print("✅ Opik tracing initialized for DocumentAnalyzer")
            except Exception as e:
                print(f"⚠️ Could not initialize Opik tracer: {e}")
                self.opik_tracer = None
        
        # Document analysis prompt
        self.analysis_prompt = ChatPromptTemplate.from_template("""
You are a regulatory document analyzer specialized in identifying the business impact and applicability of regulatory documents.

Analyze the following document content and provide a structured assessment:

**DOCUMENT CONTENT:**
{document_content}

**ANALYSIS REQUIREMENTS:**

1. **Frameworks**: Which regulatory frameworks does this relate to? (GDPR, NIS2, DORA, CER, EXEC_ORDER)

2. **Topic Tags**: Extract relevant business topic tags from this content. Consider:
   - Business operations (data-processing, cross-border, cloud-services, AI-ML, cybersecurity, compliance, contracts, business-expansion)
   - Technology aspects (blockchain, iot, mobile-apps, websites, databases, apis, analytics)
   - Regulatory context (data-transfers, consent, rights, lawful-basis, risk-assessment, training, documentation)
   - Geographic scope (us, eu, china, uk, germany, france)
   - Industry specifics (banking, healthcare-provider, energy-sector, financial, government, technology)

3. **Business Relevance**: Describe what types of businesses this would impact (1-2 sentences)

4. **Geographic Scope**: Which countries/regions are primarily affected?

5. **Industry Focus**: Which industries would be most impacted by this document?

6. **Summary**: Provide a concise 2-3 sentence summary of the document's key regulatory changes or requirements

7. **Key Requirements**: List 3-5 specific actionable requirements or obligations this document creates

Provide your analysis in a structured format that matches the expected response model.
""")

    @track(name="extract_text_from_pdf", project_name="RegTechAI-v2")
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text content from a PDF file"""
        if OPIK_AVAILABLE and opik_context:
            opik_context.update_current_trace(
                name="extract_pdf_text",
                tags=["document_processing", "pdf"],
                metadata={"file_path": pdf_path}
            )
        
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = pypdf.PdfReader(file)
                text = ""
                
                # Extract text from first few pages (limit for analysis)
                max_pages = min(5, len(pdf_reader.pages))
                for page_num in range(max_pages):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text() + "\n"
                
                extracted_text = text.strip()
                
                if OPIK_AVAILABLE and opik_context:
                    opik_context.update_current_trace(
                        metadata={
                            "file_path": pdf_path,
                            "pages_processed": max_pages,
                            "total_pages": len(pdf_reader.pages),
                            "text_length": len(extracted_text),
                            "extraction_success": True
                        }
                    )
                
                return extracted_text
        except Exception as e:
            error_msg = f"Error extracting text from PDF: {e}"
            print(error_msg)
            
            if OPIK_AVAILABLE and opik_context:
                opik_context.update_current_trace(
                    metadata={
                        "file_path": pdf_path,
                        "extraction_success": False,
                        "error": str(e)
                    }
                )
            
            return ""

    @track(name="analyze_document", project_name="RegTechAI-v2")
    def analyze_document(self, document_path: str, document_content: str = None) -> DocumentAnalysis:
        """Analyze a regulatory document to understand its impact"""
        
        if OPIK_AVAILABLE and opik_context:
            opik_context.update_current_trace(
                name="analyze_document", 
                tags=["document_analysis", "regulatory"],
                metadata={
                    "document_path": document_path if document_path else "content_provided",
                    "has_content": document_content is not None
                }
            )
        
        if document_content is None:
            # Extract content from PDF
            if document_path.lower().endswith('.pdf'):
                document_content = self.extract_text_from_pdf(document_path)
            else:
                # For text files
                with open(document_path, 'r', encoding='utf-8') as f:
                    document_content = f.read()
        
        if not document_content or len(document_content) < 100:
            error_msg = "Document content is too short or empty for analysis"
            if OPIK_AVAILABLE and opik_context:
                opik_context.update_current_trace(
                    metadata={
                        "error": error_msg,
                        "content_length": len(document_content) if document_content else 0
                    }
                )
            raise ValueError(error_msg)
        
        # Limit content length for analysis
        max_content_length = 8000
        original_length = len(document_content)
        was_truncated = False
        if len(document_content) > max_content_length:
            document_content = document_content[:max_content_length] + "... [truncated]"
            was_truncated = True
        
        # Analyze document with LLM
        model_with_structure = self.llm.with_structured_output(DocumentImpactResponse)
        
        # Add tracing to the chain if available
        if self.opik_tracer:
            chain = self.analysis_prompt | model_with_structure.with_config({"callbacks": [self.opik_tracer]})
        else:
            chain = self.analysis_prompt | model_with_structure
        
        response = chain.invoke({"document_content": document_content})
        
        analysis_result = DocumentAnalysis(
            frameworks=response.frameworks,
            topic_tags=response.topic_tags,
            business_relevance=response.business_relevance,
            geographic_scope=response.geographic_scope,
            industry_focus=response.industry_focus,
            summary=response.summary,
            key_requirements=response.key_requirements
        )
        
        if OPIK_AVAILABLE and opik_context:
            opik_context.update_current_trace(
                metadata={
                    "document_path": document_path if document_path else "content_provided",
                    "original_content_length": original_length,
                    "processed_content_length": len(document_content),
                    "was_truncated": was_truncated,
                    "frameworks_found": len(analysis_result.frameworks),
                    "topics_found": len(analysis_result.topic_tags),
                    "geographic_scope": len(analysis_result.geographic_scope),
                    "industry_focus": len(analysis_result.industry_focus),
                    "key_requirements": len(analysis_result.key_requirements),
                    "analysis_success": True
                }
            )
        
        return analysis_result

    @track(name="calculate_user_impact", project_name="RegTechAI-v2")
    def calculate_user_impact(self, document_analysis: DocumentAnalysis, user_email: str) -> UserImpactScore:
        """Calculate impact score for a specific user based on document analysis"""
        
        if OPIK_AVAILABLE and opik_context:
            opik_context.update_current_trace(
                name="calculate_user_impact",
                tags=["scoring", "user_matching"],
                metadata={
                    "user_email": user_email,
                    "document_frameworks": len(document_analysis.frameworks),
                    "document_topics": len(document_analysis.topic_tags)
                }
            )
        
        # Get user profile and context
        user_profile = self.user_memory.get_user_profile(user_email)
        user_context = self.user_memory.get_user_context(user_email)
        
        if not user_profile:
            if OPIK_AVAILABLE and opik_context:
                opik_context.update_current_trace(
                    metadata={
                        "user_email": user_email,
                        "has_profile": False,
                        "impact_score": 0.0,
                        "scoring_reason": "no_user_profile"
                    }
                )
            
            return UserImpactScore(
                email=user_email,
                impact_score=0.0,
                relevance_reasons=[],
                applicable_frameworks=[],
                matched_topics=[]
            )
        
        impact_score = 0.0
        relevance_reasons = []
        applicable_frameworks = []
        matched_topics = []
        
        # Framework matching (high weight)
        user_frameworks = set(user_profile.regulatory_focus or [])
        doc_frameworks = set(document_analysis.frameworks)
        framework_overlap = user_frameworks.intersection(doc_frameworks)
        
        if framework_overlap:
            framework_weight = min(len(framework_overlap) * 0.3, 0.6)
            impact_score += framework_weight
            applicable_frameworks = list(framework_overlap)
            relevance_reasons.append(f"Works with {', '.join(framework_overlap)} framework(s)")
        
        # Industry matching (high weight)
        user_industry = (user_profile.industry or "").lower()
        doc_industries = [ind.lower() for ind in document_analysis.industry_focus]
        
        if user_industry and any(user_industry in doc_ind or doc_ind in user_industry for doc_ind in doc_industries):
            impact_score += 0.25
            relevance_reasons.append(f"Industry match: {user_profile.industry}")
        
        # Geographic matching (medium weight)
        user_country = (user_profile.country or "").lower()
        doc_geography = [geo.lower() for geo in document_analysis.geographic_scope]
        
        geographic_match = False
        if user_country:
            # Check for direct country match or regional match (EU, US)
            if any(user_country in geo or geo in user_country for geo in doc_geography):
                geographic_match = True
            elif user_country in ["germany", "france", "netherlands", "italy", "spain"] and "eu" in doc_geography:
                geographic_match = True
            elif user_country == "united states" and "us" in doc_geography:
                geographic_match = True
                
        if geographic_match:
            impact_score += 0.15
            relevance_reasons.append(f"Geographic relevance: {user_profile.country}")
        
        # Topic tag matching (medium weight)
        user_topics = set(user_context.get('topic_tags', {}).keys())
        doc_topics = set(document_analysis.topic_tags)
        topic_overlap = user_topics.intersection(doc_topics)
        
        if topic_overlap:
            topic_weight = min(len(topic_overlap) * 0.05, 0.2)
            impact_score += topic_weight
            matched_topics = list(topic_overlap)
            relevance_reasons.append(f"Topic relevance: {', '.join(list(topic_overlap)[:3])}")
        
        # Business relevance keywords (low weight)
        business_keywords = document_analysis.business_relevance.lower()
        if user_industry and user_industry in business_keywords:
            impact_score += 0.1
        
        # Cap the impact score at 1.0
        impact_score = min(impact_score, 1.0)
        
        result = UserImpactScore(
            email=user_email,
            impact_score=impact_score,
            relevance_reasons=relevance_reasons,
            applicable_frameworks=applicable_frameworks,
            matched_topics=matched_topics
        )
        
        if OPIK_AVAILABLE and opik_context:
            opik_context.update_current_trace(
                metadata={
                    "user_email": user_email,
                    "has_profile": True,
                    "user_industry": user_profile.industry,
                    "user_country": user_profile.country,
                    "user_frameworks": user_profile.regulatory_focus,
                    "impact_score": impact_score,
                    "framework_matches": len(applicable_frameworks),
                    "topic_matches": len(matched_topics),
                    "relevance_reasons_count": len(relevance_reasons),
                    "scoring_success": True
                }
            )
        
        return result

    @track(name="identify_impacted_users", project_name="RegTechAI-v2")
    def identify_impacted_users(self, document_analysis: DocumentAnalysis, 
                              min_impact_score: float = 0.2) -> List[UserImpactScore]:
        """Identify all users who would be impacted by this document"""
        
        if OPIK_AVAILABLE and opik_context:
            opik_context.update_current_trace(
                name="identify_impacted_users",
                tags=["user_identification", "batch_scoring"],
                metadata={
                    "min_impact_score": min_impact_score,
                    "document_frameworks": document_analysis.frameworks,
                    "document_topics": len(document_analysis.topic_tags)
                }
            )
        
        # Get all users from the agent
        try:
            all_users = self.agent.list_all_users()
        except Exception as e:
            error_msg = f"Error loading user list: {e}"
            print(error_msg)
            all_users = []
            
            if OPIK_AVAILABLE and opik_context:
                opik_context.update_current_trace(
                    metadata={
                        "error": error_msg,
                        "users_loaded": 0
                    }
                )
        
        impacted_users = []
        users_processed = 0
        users_above_threshold = 0
        
        for user_email in all_users:
            impact_score = self.calculate_user_impact(document_analysis, user_email)
            users_processed += 1
            
            if impact_score.impact_score >= min_impact_score:
                impacted_users.append(impact_score)
                users_above_threshold += 1
        
        # Sort by impact score (highest first)
        impacted_users.sort(key=lambda x: x.impact_score, reverse=True)
        
        if OPIK_AVAILABLE and opik_context:
            avg_score = sum(u.impact_score for u in impacted_users) / len(impacted_users) if impacted_users else 0
            opik_context.update_current_trace(
                metadata={
                    "min_impact_score": min_impact_score,
                    "total_users": len(all_users),
                    "users_processed": users_processed,
                    "users_above_threshold": users_above_threshold,
                    "average_impact_score": round(avg_score, 3),
                    "identification_success": True
                }
            )
        
        return impacted_users

    @track(name="generate_personalized_email", project_name="RegTechAI-v2")
    def generate_personalized_email(self, document_analysis: DocumentAnalysis, 
                                   user_impact: UserImpactScore, document_name: str) -> str:
        """Generate a personalized email notification for a user"""
        
        if OPIK_AVAILABLE and opik_context:
            opik_context.update_current_trace(
                name="generate_personalized_email",
                tags=["email_generation", "personalization"],
                metadata={
                    "user_email": user_impact.email,
                    "document_name": document_name,
                    "impact_score": user_impact.impact_score,
                    "applicable_frameworks": user_impact.applicable_frameworks,
                    "matched_topics_count": len(user_impact.matched_topics)
                }
            )
        
        user_profile = self.user_memory.get_user_profile(user_impact.email)
        
        email_prompt = ChatPromptTemplate.from_template("""
You are a regulatory compliance specialist writing personalized email notifications to business professionals about new regulatory developments.

**USER PROFILE:**
- Name: {user_name}
- Company: {company}
- Industry: {industry}
- Country: {country}
- Regulatory Focus: {regulatory_focus}

**DOCUMENT INFORMATION:**
- Document: {document_name}
- Summary: {document_summary}
- Key Requirements: {key_requirements}
- Applicable Frameworks: {applicable_frameworks}
- Relevance Reasons: {relevance_reasons}

**EMAIL REQUIREMENTS:**
- Professional, concise tone
- Personalize based on user's industry and regulatory focus
- Highlight specific relevance to their business
- Include key action items
- End with offer for consultation

Write a personalized email notification (subject + body) that would be valuable for this specific user.
""")
        
        user_name = user_impact.email.split('@')[0].replace('.', ' ').title()
        company = user_profile.company if user_profile else "your organization"
        industry = user_profile.industry if user_profile else "your industry"
        country = user_profile.country if user_profile else "your region"
        regulatory_focus = ", ".join(user_profile.regulatory_focus) if user_profile and user_profile.regulatory_focus else "compliance"
        
        # Use tracing for LLM call if available
        if self.opik_tracer:
            response = self.llm.with_config({"callbacks": [self.opik_tracer]}).invoke(
                email_prompt.format(
                    user_name=user_name,
                    company=company,
                    industry=industry,
                    country=country,
                    regulatory_focus=regulatory_focus,
                    document_name=document_name,
                    document_summary=document_analysis.summary,
                    key_requirements="; ".join(document_analysis.key_requirements),
                    applicable_frameworks=", ".join(user_impact.applicable_frameworks),
                    relevance_reasons="; ".join(user_impact.relevance_reasons)
                )
            )
        else:
            response = self.llm.invoke(
                email_prompt.format(
                    user_name=user_name,
                    company=company,
                    industry=industry,
                    country=country,
                    regulatory_focus=regulatory_focus,
                    document_name=document_name,
                    document_summary=document_analysis.summary,
                    key_requirements="; ".join(document_analysis.key_requirements),
                    applicable_frameworks=", ".join(user_impact.applicable_frameworks),
                    relevance_reasons="; ".join(user_impact.relevance_reasons)
                )
            )
        
        email_content = response.content
        
        if OPIK_AVAILABLE and opik_context:
            opik_context.update_current_trace(
                metadata={
                    "user_email": user_impact.email,
                    "user_name": user_name,
                    "user_company": company,
                    "user_industry": industry,
                    "user_country": country,
                    "document_name": document_name,
                    "email_length": len(email_content),
                    "personalization_success": True,
                    "has_user_profile": user_profile is not None
                }
            )
        
        return email_content