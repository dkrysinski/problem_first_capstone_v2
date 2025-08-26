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
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

class UserMemory:
    def __init__(self, data_dir: str = "data/user_memory"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.profiles_file = self.data_dir / "user_profiles.json"
        self.interactions_file = self.data_dir / "user_interactions.json"
        
        # Initialize default user for POC
        self._initialize_default_user()
    
    def _initialize_default_user(self):
        """Initialize default user for POC purposes"""
        default_email = "danno@ragequitlab.com"
        if not self.get_user_profile(default_email):
            profile = UserProfile(
                email=default_email,
                industry="Technology/Software",
                company="RageQuit Labs", 
                role="Developer/Researcher",
                regulatory_focus=["GDPR", "NIS2"]
            )
            self._save_user_profile(profile)
    
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
        
        interaction = Interaction(
            timestamp=datetime.now().isoformat(),
            question=question,
            frameworks_analyzed=frameworks_analyzed,
            answer_summary=answer_summary[:500] + "..." if len(answer_summary) > 500 else answer_summary,
            topic_category=topic_category
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
        
        return context
    
    def infer_user_details(self, email: str, question: str, frameworks_analyzed: List[str]) -> Dict:
        """Infer user industry/interests from their questions"""
        industry_keywords = {
            "financial": ["bank", "finance", "payment", "trading", "investment"],
            "healthcare": ["medical", "patient", "health", "hospital", "clinical"],
            "technology": ["software", "data", "AI", "platform", "system", "cloud"],
            "manufacturing": ["factory", "production", "supply chain", "industrial"],
            "energy": ["power", "electricity", "grid", "utility", "energy"],
            "transportation": ["logistics", "shipping", "transport", "vehicle"]
        }
        
        inferred = {
            "potential_industry": None,
            "regulatory_complexity": "medium"
        }
        
        question_lower = question.lower()
        
        # Infer industry from question content
        for industry, keywords in industry_keywords.items():
            if any(keyword in question_lower for keyword in keywords):
                inferred["potential_industry"] = industry
                break
        
        # Determine regulatory complexity based on frameworks
        if len(frameworks_analyzed) > 2:
            inferred["regulatory_complexity"] = "high"
        elif len(frameworks_analyzed) == 1:
            inferred["regulatory_complexity"] = "low"
        
        return inferred