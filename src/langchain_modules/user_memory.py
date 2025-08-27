import json
import os
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from pathlib import Path

@dataclass
class UserProfile:
    email: str
    industry: Optional[str] = None
    company: Optional[str] = None
    role: Optional[str] = None
    country: Optional[str] = None
    regulatory_focus: List[str] = None
    created_at: str = None
    last_active: str = None
    
    def __post_init__(self):
        if self.regulatory_focus is None:
            self.regulatory_focus = []
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
        self.last_active = datetime.now().isoformat()

@dataclass 
class Interaction:
    timestamp: str
    question: str
    frameworks_analyzed: List[str]
    answer_summary: str
    topic_category: Optional[str] = None
    topic_tags: List[str] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
        if self.topic_tags is None:
            self.topic_tags = []

class UserMemory:
    def __init__(self, data_dir: str = "data/user_memory"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.profiles_file = self.data_dir / "user_profiles.json"
        self.interactions_file = self.data_dir / "user_interactions.json"
        
        # Initialize default user for POC
        self._initialize_default_user()
    
    def _initialize_default_user(self):
        """Initialize diverse set of users for demonstration purposes"""
        self._initialize_sample_profiles()
    
    def _initialize_sample_profiles(self):
        """Initialize diverse sample profiles for demonstration"""
        sample_profiles = [
            # Original default user
            {
                "email": "danno@ragequitlab.com",
                "industry": "Technology/Software",
                "company": "RageQuit Labs", 
                "role": "Developer/Researcher",
                "country": "United States",
                "regulatory_focus": ["GDPR", "NIS2"]
            },
            
            # Financial Services - EU
            {
                "email": "marie.dubois@banquecentrale.fr",
                "industry": "Financial",
                "company": "Banque Centrale Paris",
                "role": "Chief Risk Officer",
                "country": "France",
                "regulatory_focus": ["GDPR", "DORA", "NIS2"]
            },
            
            # Healthcare - Germany
            {
                "email": "dr.mueller@gesundheitsklinik.de",
                "industry": "Healthcare",
                "company": "Gesundheitsklinik München",
                "role": "Data Protection Officer",
                "country": "Germany",
                "regulatory_focus": ["GDPR", "NIS2"]
            },
            
            # US Federal Contractor
            {
                "email": "james.wilson@defensetech.com",
                "industry": "Government",
                "company": "DefenseTech Solutions",
                "role": "Compliance Director",
                "country": "United States",
                "regulatory_focus": ["EXEC_ORDER", "GDPR"]
            },
            
            # Energy - Netherlands
            {
                "email": "anna.vanderberg@energienl.com",
                "industry": "Energy",
                "company": "Nederlandse Energie Maatschappij",
                "role": "Infrastructure Security Manager",
                "country": "Netherlands",
                "regulatory_focus": ["NIS2", "CER", "GDPR"]
            },
            
            # Manufacturing - Poland
            {
                "email": "piotr.kowalski@przemysl.pl",
                "industry": "Manufacturing",
                "company": "Przemysł Automotive SA",
                "role": "Operations Manager",
                "country": "Poland",
                "regulatory_focus": ["GDPR", "NIS2"]
            },
            
            # FinTech Startup - UK
            {
                "email": "sarah.thompson@fintechuk.co.uk",
                "industry": "Financial",
                "company": "FinTech Innovations Ltd",
                "role": "CEO & Founder",
                "country": "United Kingdom",
                "regulatory_focus": ["GDPR", "DORA"]
            },
            
            # Transportation - Sweden
            {
                "email": "erik.larsson@transportse.com",
                "industry": "Transportation",
                "company": "Transport Sverige AB",
                "role": "Digital Transformation Lead",
                "country": "Sweden",
                "regulatory_focus": ["GDPR", "NIS2", "CER"]
            },
            
            # Multinational Tech - US/EU
            {
                "email": "alex.rodriguez@globaltechcorp.com",
                "industry": "Technology/Software",
                "company": "GlobalTech Corporation",
                "role": "VP of Privacy & Security",
                "country": "United States",
                "regulatory_focus": ["GDPR", "EXEC_ORDER", "NIS2"]
            },
            
            # Insurance - Italy
            {
                "email": "giulia.rossi@assicurazioneit.it",
                "industry": "Financial",
                "company": "Assicurazioni Italiane SpA",
                "role": "Head of Regulatory Affairs",
                "country": "Italy",
                "regulatory_focus": ["GDPR", "DORA", "NIS2"]
            },
            
            # Healthcare Tech - Canada
            {
                "email": "michael.chen@healthtechca.com",
                "industry": "Healthcare",
                "company": "HealthTech Canada Inc",
                "role": "Chief Privacy Officer",
                "country": "Canada",
                "regulatory_focus": ["GDPR"]
            },
            
            # Critical Infrastructure - Spain
            {
                "email": "carlos.martinez@infraestructura.es",
                "industry": "Energy",
                "company": "Infraestructura Crítica España",
                "role": "Security Operations Director",
                "country": "Spain",
                "regulatory_focus": ["CER", "NIS2", "GDPR"]
            },
            
            # Digital Services - Belgium
            {
                "email": "sophie.lambert@digitalservices.be",
                "industry": "Technology/Software",
                "company": "Digital Services Belgium",
                "role": "Head of Compliance",
                "country": "Belgium",
                "regulatory_focus": ["GDPR", "NIS2", "DORA"]
            },
            
            # Aviation - Australia (doing business in EU)
            {
                "email": "david.smith@aviationau.com",
                "industry": "Transportation",
                "company": "Aviation Australia Pty Ltd",
                "role": "International Compliance Manager",
                "country": "Australia",
                "regulatory_focus": ["GDPR", "NIS2"]
            },
            
            # AI/ML Research - US
            {
                "email": "dr.patel@airesearch.edu",
                "industry": "Technology/Software",
                "company": "AI Research Institute",
                "role": "Principal Research Scientist",
                "country": "United States",
                "regulatory_focus": ["EXEC_ORDER", "GDPR"]
            }
        ]
        
        # Only create profiles that don't already exist
        for profile_data in sample_profiles:
            if not self.get_user_profile(profile_data["email"]):
                profile = UserProfile(**profile_data)
                self._save_user_profile(profile)
                print(f"✅ Created sample profile for {profile_data['email']}")
        
        # Create sample interactions to demonstrate usage patterns
        self._create_sample_interactions()
    
    def _create_sample_interactions(self):
        """Create realistic sample interactions for demonstration profiles"""
        sample_interactions = [
            # French Banking - DORA compliance
            {
                "email": "marie.dubois@banquecentrale.fr",
                "question": "What operational resilience requirements does DORA impose on our payment processing systems?",
                "frameworks": ["DORA", "GDPR"],
                "answer_summary": "DORA requires comprehensive ICT risk management frameworks, operational resilience testing, and third-party risk oversight for payment systems...",
                "topic_category": "financial"
            },
            
            # German Healthcare - NIS2 implementation
            {
                "email": "dr.mueller@gesundheitsklinik.de", 
                "question": "How do we implement NIS2 cybersecurity measures for our hospital's patient data systems?",
                "frameworks": ["NIS2", "GDPR"],
                "answer_summary": "Healthcare entities under NIS2 must implement cybersecurity risk management, incident reporting within 24 hours, and ensure business continuity...",
                "topic_category": "healthcare"
            },
            
            # US Defense Contractor - Executive Orders
            {
                "email": "james.wilson@defensetech.com",
                "question": "What Executive Order 14028 requirements apply to our federal software contracts?",
                "frameworks": ["EXEC_ORDER"],
                "answer_summary": "EO 14028 mandates critical software identification, SBOM provision, security attestations, and vulnerability disclosure programs...",
                "topic_category": "government"
            },
            
            # Dutch Energy - Critical Infrastructure
            {
                "email": "anna.vanderberg@energienl.com",
                "question": "What resilience measures must we implement under CER for our power grid operations?",
                "frameworks": ["CER", "NIS2", "GDPR"],
                "answer_summary": "CER requires risk assessments, business continuity plans, incident reporting, and resilience measures for critical energy infrastructure...",
                "topic_category": "energy"
            },
            
            # Polish Manufacturing - Supply Chain
            {
                "email": "piotr.kowalski@przemysl.pl",
                "question": "How does GDPR apply to our automotive supply chain data sharing with German partners?",
                "frameworks": ["GDPR"],
                "answer_summary": "Cross-border automotive data transfers require appropriate safeguards, data processing agreements, and compliance with GDPR Article 44-49...",
                "topic_category": "manufacturing"
            },
            
            # UK FinTech - Post-Brexit compliance
            {
                "email": "sarah.thompson@fintechuk.co.uk",
                "question": "As a UK fintech serving EU customers, what DORA requirements apply to our operations?",
                "frameworks": ["DORA", "GDPR"],
                "answer_summary": "UK fintech serving EU markets must comply with DORA as a third-party ICT service provider, requiring operational resilience frameworks...",
                "topic_category": "financial"
            },
            
            # Swedish Transport - Digital transformation
            {
                "email": "erik.larsson@transportse.com",
                "question": "What NIS2 obligations do we have for our smart transportation management systems?",
                "frameworks": ["NIS2", "GDPR", "CER"],
                "answer_summary": "Transportation entities must implement NIS2 cybersecurity measures, CER resilience requirements, and GDPR compliance for passenger data...",
                "topic_category": "transportation"
            },
            
            # US Multinational - Cross-border compliance
            {
                "email": "alex.rodriguez@globaltechcorp.com",
                "question": "How do US Executive Orders on AI governance interact with EU GDPR requirements?",
                "frameworks": ["EXEC_ORDER", "GDPR", "NIS2"],
                "answer_summary": "US AI governance orders and EU GDPR create overlapping obligations for algorithmic accountability, risk assessments, and privacy rights...",
                "topic_category": "technology"
            },
            
            # Italian Insurance - DORA preparation
            {
                "email": "giulia.rossi@assicurazioneit.it",
                "question": "What digital operational resilience testing must we conduct under DORA?",
                "frameworks": ["DORA", "GDPR"],
                "answer_summary": "Insurance firms must conduct threat-led penetration testing, scenario-based testing, and ensure third-party provider resilience under DORA...",
                "topic_category": "financial"
            },
            
            # Canadian HealthTech - EU expansion
            {
                "email": "michael.chen@healthtechca.com",
                "question": "What GDPR requirements apply when expanding our health app to European markets?",
                "frameworks": ["GDPR"],
                "answer_summary": "Health apps processing EU personal data must comply with GDPR including lawful basis, special category data protection, and privacy by design...",
                "topic_category": "healthcare"
            },
            
            # Spanish Critical Infrastructure
            {
                "email": "carlos.martinez@infraestructura.es",
                "question": "How do CER and NIS2 requirements overlap for our energy distribution network?",
                "frameworks": ["CER", "NIS2", "GDPR"],
                "answer_summary": "Energy operators face dual obligations under CER for resilience measures and NIS2 for cybersecurity, requiring integrated compliance approaches...",
                "topic_category": "energy"
            },
            
            # Belgian Digital Services
            {
                "email": "sophie.lambert@digitalservices.be",
                "question": "What DORA requirements apply to our cloud services for financial institutions?",
                "frameworks": ["DORA", "NIS2", "GDPR"],
                "answer_summary": "Cloud service providers to financial entities must meet DORA's critical ICT third-party requirements, including oversight and exit strategies...",
                "topic_category": "technology"
            },
            
            # Australian Aviation - EU operations
            {
                "email": "david.smith@aviationau.com",
                "question": "What NIS2 requirements apply to our flight management systems operating in EU airports?",
                "frameworks": ["NIS2", "GDPR"],
                "answer_summary": "Aviation operators in EU must comply with NIS2 cybersecurity measures for flight management systems and GDPR for passenger data processing...",
                "topic_category": "transportation"
            },
            
            # US AI Research - Executive Order compliance
            {
                "email": "dr.patel@airesearch.edu",
                "question": "What AI safety requirements does Executive Order 14110 impose on our research models?",
                "frameworks": ["EXEC_ORDER"],
                "answer_summary": "EO 14110 requires safety evaluations, red team testing, and reporting for AI models exceeding computational thresholds in research contexts...",
                "topic_category": "technology"
            }
        ]
        
        # Add interactions only if they don't already exist for the user
        existing_interactions = self._load_interactions()
        
        for interaction_data in sample_interactions:
            email = interaction_data["email"]
            # Only add if user has no existing interactions
            if email not in existing_interactions or len(existing_interactions[email]) == 0:
                self.add_interaction(
                    email=email,
                    question=interaction_data["question"],
                    frameworks_analyzed=interaction_data["frameworks"],
                    answer_summary=interaction_data["answer_summary"],
                    topic_category=interaction_data["topic_category"]
                )
                print(f"✅ Added sample interaction for {email}")
    
    def _load_profiles(self) -> Dict[str, Dict]:
        """Load user profiles from JSON file"""
        if self.profiles_file.exists():
            try:
                with open(self.profiles_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                return {}
        return {}
    
    def _save_profiles(self, profiles: Dict[str, Dict]):
        """Save user profiles to JSON file"""
        with open(self.profiles_file, 'w', encoding='utf-8') as f:
            json.dump(profiles, f, indent=2, ensure_ascii=False)
    
    def _load_interactions(self) -> Dict[str, List[Dict]]:
        """Load user interactions from JSON file"""
        if self.interactions_file.exists():
            try:
                with open(self.interactions_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                return {}
        return {}
    
    def _save_interactions(self, interactions: Dict[str, List[Dict]]):
        """Save user interactions to JSON file"""
        with open(self.interactions_file, 'w', encoding='utf-8') as f:
            json.dump(interactions, f, indent=2, ensure_ascii=False)
    
    def get_user_profile(self, email: str) -> Optional[UserProfile]:
        """Get user profile by email"""
        profiles = self._load_profiles()
        if email in profiles:
            profile_data = profiles[email]
            return UserProfile(**profile_data)
        return None
    
    def _save_user_profile(self, profile: UserProfile):
        """Save user profile"""
        profiles = self._load_profiles()
        profiles[profile.email] = asdict(profile)
        self._save_profiles(profiles)
    
    def update_user_profile(self, email: str, **kwargs):
        """Update user profile with new information"""
        profile = self.get_user_profile(email)
        if not profile:
            profile = UserProfile(email=email)
        
        # Update profile fields
        for key, value in kwargs.items():
            if hasattr(profile, key) and value is not None:
                setattr(profile, key, value)
        
        profile.last_active = datetime.now().isoformat()
        self._save_user_profile(profile)
        return profile
    
    def add_interaction(self, email: str, question: str, frameworks_analyzed: List[str], 
                       answer_summary: str, topic_category: Optional[str] = None):
        """Add a new user interaction"""
        interactions = self._load_interactions()
        
        if email not in interactions:
            interactions[email] = []
        
        # Extract topic tags
        topic_tags = self.extract_topic_tags(question, frameworks_analyzed, answer_summary)
        
        interaction = Interaction(
            timestamp=datetime.now().isoformat(),
            question=question,
            frameworks_analyzed=frameworks_analyzed,
            answer_summary=answer_summary[:500] + "..." if len(answer_summary) > 500 else answer_summary,
            topic_category=topic_category,
            topic_tags=topic_tags
        )
        
        interactions[email].append(asdict(interaction))
        
        # Keep only last 50 interactions per user to avoid memory bloat
        interactions[email] = interactions[email][-50:]
        
        self._save_interactions(interactions)
        
        # Update user's last activity and regulatory focus
        self._update_user_activity(email, frameworks_analyzed)
    
    def _update_user_activity(self, email: str, frameworks_analyzed: List[str]):
        """Update user's last activity and regulatory interests"""
        profile = self.get_user_profile(email)
        if profile:
            # Update regulatory focus based on recent queries
            for framework in frameworks_analyzed:
                if framework not in profile.regulatory_focus:
                    profile.regulatory_focus.append(framework)
            
            profile.last_active = datetime.now().isoformat()
            self._save_user_profile(profile)
    
    def get_user_interactions(self, email: str, limit: int = 10) -> List[Interaction]:
        """Get recent user interactions"""
        interactions = self._load_interactions()
        user_interactions = interactions.get(email, [])
        
        # Return most recent interactions first
        recent_interactions = user_interactions[-limit:][::-1]
        return [Interaction(**interaction) for interaction in recent_interactions]
    
    def get_user_context(self, email: str) -> Dict:
        """Get comprehensive user context for personalized responses"""
        profile = self.get_user_profile(email)
        recent_interactions = self.get_user_interactions(email, limit=5)
        
        context = {
            "profile": asdict(profile) if profile else None,
            "recent_topics": [],
            "topic_tags": {},
            "regulatory_patterns": {},
            "interaction_count": 0
        }
        
        if recent_interactions:
            context["interaction_count"] = len(self.get_user_interactions(email, limit=100))
            
            # Extract recent topics
            context["recent_topics"] = [
                interaction.topic_category 
                for interaction in recent_interactions 
                if interaction.topic_category
            ][:3]
            
            # Count regulatory framework patterns
            for interaction in recent_interactions:
                for framework in interaction.frameworks_analyzed:
                    context["regulatory_patterns"][framework] = \
                        context["regulatory_patterns"].get(framework, 0) + 1
            
            # Get topic tag trends
            context["topic_tags"] = self.get_topic_tag_trends(email, limit=20)
        
        return context
    
    def infer_user_details(self, email: str, question: str, frameworks_analyzed: List[str]) -> Dict:
        """Infer user industry/interests from their questions"""
        industry_keywords = {
            "financial": ["bank", "finance", "payment", "trading", "investment"],
            "healthcare": ["medical", "patient", "health", "hospital", "clinical"],
            "technology": ["software", "data", "AI", "platform", "system", "cloud"],
            "manufacturing": ["factory", "production", "supply chain", "industrial"],
            "energy": ["power", "electricity", "grid", "utility", "energy"],
            "transportation": ["logistics", "shipping", "transport", "vehicle"],
            "government": ["federal", "government", "contractor", "agency", "defense", "national security"]
        }
        
        inferred = {
            "potential_industry": None,
            "potential_country": None,
            "regulatory_complexity": "medium"
        }
        
        question_lower = question.lower()
        
        # Country detection keywords and patterns
        country_keywords = {
            "United States": ["us", "usa", "united states", "american", "federal", "usc", "cfr"],
            "Germany": ["germany", "german", "deutschland", "de"],
            "France": ["france", "french", "français", "fr"],
            "United Kingdom": ["uk", "britain", "british", "england", "scotland", "wales"],
            "Canada": ["canada", "canadian", "ontario", "quebec"],
            "Australia": ["australia", "australian", "sydney", "melbourne"],
            "Netherlands": ["netherlands", "dutch", "holland", "nl"],
            "Sweden": ["sweden", "swedish", "stockholm", "se"],
            "Italy": ["italy", "italian", "italia", "it"],
            "Spain": ["spain", "spanish", "españa", "es"],
            "Poland": ["poland", "polish", "warszawa", "pl"],
            "Belgium": ["belgium", "belgian", "brussels", "be"]
        }
        
        # EU member countries (for context)
        eu_countries = {
            "Austria", "Belgium", "Bulgaria", "Croatia", "Cyprus", "Czech Republic",
            "Denmark", "Estonia", "Finland", "France", "Germany", "Greece",
            "Hungary", "Ireland", "Italy", "Latvia", "Lithuania", "Luxembourg",
            "Malta", "Netherlands", "Poland", "Portugal", "Romania", "Slovakia",
            "Slovenia", "Spain", "Sweden"
        }
        
        # Infer country from question content
        for country, keywords in country_keywords.items():
            if any(keyword in question_lower for keyword in keywords):
                inferred["potential_country"] = country
                break
        
        # Framework-based country inference
        if not inferred["potential_country"]:
            if "EXEC_ORDER" in frameworks_analyzed:
                inferred["potential_country"] = "United States"
            elif any(fw in frameworks_analyzed for fw in ["GDPR", "NIS2", "DORA", "CER"]):
                # If only EU frameworks, likely EU-based but don't assume specific country
                inferred["potential_country"] = "European Union"
        
        # Infer industry from question content
        for industry, keywords in industry_keywords.items():
            if any(keyword in question_lower for keyword in keywords):
                inferred["potential_industry"] = industry
                break
        
        # Special case: If Executive Orders are involved, likely government/contractor
        if "EXEC_ORDER" in frameworks_analyzed:
            if not inferred["potential_industry"]:
                inferred["potential_industry"] = "government"
        
        # Determine regulatory complexity based on frameworks
        if len(frameworks_analyzed) > 2:
            inferred["regulatory_complexity"] = "high"
        elif len(frameworks_analyzed) == 1:
            inferred["regulatory_complexity"] = "low"
        
        return inferred
    
    def extract_topic_tags(self, question: str, frameworks_analyzed: List[str], answer_summary: str = "") -> List[str]:
        """Extract detailed topic tags from user question and analysis"""
        tags = set()
        text_to_analyze = f"{question} {answer_summary}".lower()
        
        # Business operation tags
        operation_keywords = {
            "data-processing": ["data processing", "personal data", "data collection", "data storage", "data analytics"],
            "cross-border": ["international", "cross-border", "third country", "global", "overseas", "export", "import"],
            "cloud-services": ["cloud", "saas", "paas", "iaas", "aws", "azure", "google cloud", "hosting"],
            "AI-ML": ["artificial intelligence", "machine learning", "ai", "ml", "algorithm", "automated", "neural"],
            "cybersecurity": ["cybersecurity", "security", "cyber", "breach", "incident", "vulnerability", "threat"],
            "compliance": ["compliance", "audit", "assessment", "requirement", "obligation", "mandate"],
            "contracts": ["contract", "agreement", "vendor", "supplier", "third party", "outsourcing"],
            "business-expansion": ["expand", "expansion", "new market", "launch", "scaling", "growing"],
            "mergers-acquisitions": ["merger", "acquisition", "m&a", "consolidation", "integration"],
            "employee-data": ["employee", "hr", "workforce", "personnel", "payroll", "recruitment"],
            "customer-data": ["customer", "client", "user", "consumer", "subscriber", "member"],
            "marketing": ["marketing", "advertising", "campaign", "email marketing", "newsletter", "analytics"],
            "payment-processing": ["payment", "transaction", "billing", "financial transaction", "payment card"],
            "authentication": ["authentication", "login", "password", "biometric", "two-factor", "access control"],
            "incident-response": ["incident", "breach", "response", "notification", "reporting", "disclosure"]
        }
        
        # Technology tags
        tech_keywords = {
            "blockchain": ["blockchain", "cryptocurrency", "bitcoin", "ethereum", "distributed ledger"],
            "iot": ["iot", "internet of things", "sensors", "connected devices", "smart devices"],
            "mobile-apps": ["mobile app", "smartphone", "ios", "android", "mobile application"],
            "websites": ["website", "web application", "online platform", "web portal", "e-commerce"],
            "databases": ["database", "sql", "mongodb", "data warehouse", "data lake"],
            "apis": ["api", "rest", "graphql", "integration", "interface", "webhook"],
            "analytics": ["analytics", "reporting", "dashboard", "metrics", "kpi", "business intelligence"]
        }
        
        # Regulatory context tags
        regulatory_keywords = {
            "data-transfers": ["data transfer", "adequacy", "standard contractual clauses", "binding corporate rules"],
            "consent": ["consent", "opt-in", "opt-out", "withdrawal", "agreement"],
            "rights": ["rights", "access", "rectification", "erasure", "portability", "objection"],
            "lawful-basis": ["lawful basis", "legitimate interest", "contract", "legal obligation", "vital interest"],
            "special-categories": ["special category", "sensitive data", "health data", "biometric", "genetic"],
            "risk-assessment": ["risk assessment", "impact assessment", "dpia", "pia", "risk management"],
            "training": ["training", "awareness", "education", "staff training", "employee training"],
            "documentation": ["documentation", "records", "policy", "procedure", "register"],
            "penalties": ["penalty", "fine", "sanction", "enforcement", "violation", "breach"]
        }
        
        # Geographic tags
        geographic_keywords = {
            "us": ["united states", "usa", "american", "federal"],
            "eu": ["european union", "eu", "europe", "member state"],
            "china": ["china", "chinese", "prc", "people's republic"],
            "uk": ["united kingdom", "uk", "britain", "british"],
            "germany": ["germany", "german", "deutschland"],
            "france": ["france", "french"]
        }
        
        # Apply all keyword categories
        all_keyword_categories = [operation_keywords, tech_keywords, regulatory_keywords, geographic_keywords]
        
        for keyword_dict in all_keyword_categories:
            for tag, keywords in keyword_dict.items():
                if any(keyword in text_to_analyze for keyword in keywords):
                    tags.add(tag)
        
        # Framework-specific tags
        framework_tags = {
            "GDPR": ["gdpr-compliance", "data-protection"],
            "NIS2": ["nis2-compliance", "essential-entities"],
            "DORA": ["dora-compliance", "financial-resilience"],
            "CER": ["cer-compliance", "critical-infrastructure"],
            "EXEC_ORDER": ["executive-orders", "federal-compliance"]
        }
        
        for framework in frameworks_analyzed:
            if framework in framework_tags:
                tags.update(framework_tags[framework])
        
        # Industry-specific context tags based on question content
        if "bank" in text_to_analyze or "financial institution" in text_to_analyze:
            tags.add("banking")
        if "hospital" in text_to_analyze or "healthcare provider" in text_to_analyze:
            tags.add("healthcare-provider")
        if "energy" in text_to_analyze or "utility" in text_to_analyze:
            tags.add("energy-sector")
        
        return sorted(list(tags))
    
    def get_topic_tag_trends(self, email: str, limit: int = 20) -> Dict[str, int]:
        """Get trending topic tags for a user"""
        interactions = self.get_user_interactions(email, limit=limit)
        tag_counts = {}
        
        for interaction in interactions:
            if hasattr(interaction, 'topic_tags') and interaction.topic_tags:
                for tag in interaction.topic_tags:
                    tag_counts[tag] = tag_counts.get(tag, 0) + 1
        
        # Sort by frequency, descending
        return dict(sorted(tag_counts.items(), key=lambda x: x[1], reverse=True))