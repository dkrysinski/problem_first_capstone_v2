"""
Push Notification System Interface
Streamlit interface for document upload and user impact analysis
"""

import streamlit as st
import sys
import os
from pathlib import Path
import tempfile
from typing import List
import time

# Add parent directory to path to import modules
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from src.langchain_modules.document_analyzer import DocumentAnalyzer, DocumentAnalysis, UserImpactScore
from src.langchain_modules.agent import AIAgent

# Opik tracing imports
try:
    from opik import track, opik_context, Opik
    OPIK_AVAILABLE = True
    opik_client = Opik()
except ImportError:
    OPIK_AVAILABLE = False
    def track(func):
        return func
    opik_context = None
    opik_client = None

@track(name="analyze_document_workflow", project_name="RegTechAI-v2")
def analyze_document_workflow(analyzer, document_to_analyze, document_name, min_impact_score):
    """Analyze document and identify impacted users with tracing"""
    if OPIK_AVAILABLE and opik_context:
        opik_context.update_current_trace(
            name="push_notification_workflow",
            tags=["streamlit", "push_notifications", "workflow"],
            metadata={
                "document_name": document_name,
                "document_type": "sample" if isinstance(document_to_analyze, dict) else "uploaded",
                "min_impact_score": min_impact_score
            }
        )
    
    # Document analysis phase
    if isinstance(document_to_analyze, dict):
        # Sample document - use content directly
        document_analysis = analyzer.analyze_document(
            document_path="", 
            document_content=document_to_analyze['content']
        )
    else:
        # Uploaded file - save temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(document_to_analyze.name).suffix) as tmp_file:
            tmp_file.write(document_to_analyze.getvalue())
            tmp_file_path = tmp_file.name
        
        # Analyze document
        document_analysis = analyzer.analyze_document(tmp_file_path)
        
        # Clean up temporary file
        os.unlink(tmp_file_path)
    
    # User identification phase
    impacted_users = analyzer.identify_impacted_users(
        document_analysis, 
        min_impact_score=min_impact_score
    )
    
    # Update trace with final results
    if OPIK_AVAILABLE and opik_context:
        high_impact_users = len([u for u in impacted_users if u.impact_score >= 0.7])
        avg_score = sum(u.impact_score for u in impacted_users) / len(impacted_users) if impacted_users else 0
        
        opik_context.update_current_trace(
            metadata={
                "document_name": document_name,
                "document_type": "sample" if isinstance(document_to_analyze, dict) else "uploaded",
                "min_impact_score": min_impact_score,
                "frameworks_identified": len(document_analysis.frameworks),
                "topics_identified": len(document_analysis.topic_tags),
                "total_impacted_users": len(impacted_users),
                "high_impact_users": high_impact_users,
                "average_impact_score": round(avg_score, 3),
                "workflow_success": True
            }
        )
    
    return document_analysis, impacted_users

@track(name="generate_bulk_emails_workflow", project_name="RegTechAI-v2")
def generate_bulk_emails_workflow(analyzer, document_analysis, impacted_users, document_name, max_users=10):
    """Generate bulk emails with tracing"""
    bulk_users = impacted_users[:max_users]
    
    if OPIK_AVAILABLE and opik_context:
        opik_context.update_current_trace(
            name="bulk_email_generation",
            tags=["email", "bulk_generation", "streamlit"],
            metadata={
                "document_name": document_name,
                "total_users_to_process": len(bulk_users)
            }
        )
    
    bulk_emails = {}
    successful_emails = 0
    failed_emails = 0
    
    for user_impact in bulk_users:
        try:
            email_content = analyzer.generate_personalized_email(
                document_analysis, 
                user_impact, 
                document_name
            )
            bulk_emails[user_impact.email] = email_content
            successful_emails += 1
            
        except Exception as e:
            bulk_emails[user_impact.email] = f"Error: {str(e)}"
            failed_emails += 1
    
    # Update trace with bulk generation results
    if OPIK_AVAILABLE and opik_context:
        opik_context.update_current_trace(
            metadata={
                "document_name": document_name,
                "total_users_processed": len(bulk_users),
                "successful_emails": successful_emails,
                "failed_emails": failed_emails,
                "bulk_generation_success": failed_emails == 0
            }
        )
    
    return bulk_emails, successful_emails, failed_emails

def render_push_notification_interface():
    """Render the push notification interface"""
    
    
    # Initialize document analyzer with agent
    if 'document_analyzer' not in st.session_state:
        with st.spinner("Initializing document analyzer..."):
            # Initialize with the same agent instance from the main app
            if 'agent' in st.session_state:
                agent = st.session_state.agent
            else:
                agent = AIAgent()
                st.session_state.agent = agent
            st.session_state.document_analyzer = DocumentAnalyzer(agent=agent)
    
    analyzer = st.session_state.document_analyzer
    
    # File upload section
    col1, col2 = st.columns([1, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Upload Regulatory Document",
            type=['pdf', 'txt']
        )
    
    with col2:
        if st.button("📋 Use Sample Executive Order", type="secondary"):
            try:
                sample_path = os.path.join(os.path.dirname(__file__), '../../sample_regulation.txt')
                with open(sample_path, 'r', encoding='utf-8') as f:
                    sample_content = f.read()
                
                st.session_state.sample_document = {
                    'name': 'Sample_Executive_Order_Financial_Cybersecurity.txt',
                    'content': sample_content
                }
                st.success("✅ Sample loaded")
            except Exception as e:
                st.error(f"Error: {e}")
    
    # Check if we have a document (uploaded or sample)
    document_to_analyze = None
    document_name = None
    
    if uploaded_file is not None:
        document_to_analyze = uploaded_file
        document_name = uploaded_file.name
        st.success(f"✅ {uploaded_file.name}")
    elif 'sample_document' in st.session_state:
        document_to_analyze = st.session_state.sample_document
        document_name = st.session_state.sample_document['name']
        st.success(f"✅ {document_name}")
    
    # Set static parameters
    min_impact_score = 0.40
    
    if document_to_analyze is not None:        
        # Analyze button
        if st.button("🔍 Analyze & Identify Users", type="primary"):
            with st.spinner("Analyzing..."):
                try:
                    # Use the traced workflow function
                    document_analysis, impacted_users = analyze_document_workflow(
                        analyzer, document_to_analyze, document_name, min_impact_score
                    )
                    
                    # Store results in session state
                    st.session_state.document_analysis = document_analysis
                    st.session_state.impacted_users = impacted_users
                    st.session_state.document_name = document_name
                    
                    st.success(f"✅ Found {len(impacted_users)} impacted users")
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    
    # Display results if available
    if 'document_analysis' in st.session_state and 'impacted_users' in st.session_state:
        
        document_analysis = st.session_state.document_analysis
        impacted_users = st.session_state.impacted_users
        document_name = st.session_state.document_name
        
        # Document Analysis Results
        st.subheader("📋 Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if document_analysis.frameworks:
                st.markdown("**Frameworks:** " + " | ".join([f"`{fw}`" for fw in document_analysis.frameworks]))
            if document_analysis.geographic_scope:
                st.markdown("**Geography:** " + " | ".join([f"`{geo}`" for geo in document_analysis.geographic_scope]))
        
        with col2:
            if document_analysis.industry_focus:
                st.markdown("**Industries:** " + " | ".join([f"`{ind}`" for ind in document_analysis.industry_focus]))
            if document_analysis.topic_tags:
                tags_to_show = document_analysis.topic_tags[:6]
                st.markdown("**Topics:** " + " | ".join([f"`{tag}`" for tag in tags_to_show]))
        
        with st.expander("Document Summary"):
            st.write(document_analysis.summary)
            for i, req in enumerate(document_analysis.key_requirements, 1):
                st.write(f"{i}. {req}")
        
        # Impacted Users Results
        st.subheader("👥 Users")
        
        if not impacted_users:
            st.warning("No users above threshold")
        else:
            # Business metrics
            col1, col2, col3 = st.columns(3)
            high_impact = len([u for u in impacted_users if u.impact_score >= 0.7])
            
            with col1:
                st.metric("Impacted", len(impacted_users))
            with col2:
                avg_score = sum(user.impact_score for user in impacted_users) / len(impacted_users)
                st.metric("Avg Score", f"{avg_score:.2f}")
            with col3:
                st.metric("High Priority", high_impact)
            
            st.divider()
            
            # Display user table
            users_to_show = impacted_users
            
            for i, user_impact in enumerate(users_to_show):
                user_profile = analyzer.user_memory.get_user_profile(user_impact.email)
                
                with st.expander(f"👤 {user_impact.email} (Impact: {user_impact.impact_score:.2f})", expanded=i < 3):
                    
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        st.write("**Profile:**")
                        if user_profile:
                            st.write(f"• {user_profile.company or 'Unknown'}")
                            st.write(f"• {user_profile.industry or 'Unknown'}")
                            st.write(f"• {user_profile.country or 'Unknown'}")
                            if user_profile.regulatory_focus:
                                st.write(f"• {', '.join(user_profile.regulatory_focus)}")
                        else:
                            st.write("No profile")
                        
                        st.write(f"**Score:** {user_impact.impact_score:.2f}")
                    
                    with col2:
                        st.write("**Reasons:**")
                        for reason in user_impact.relevance_reasons:
                            st.write(f"• {reason}")
                        
                        if user_impact.applicable_frameworks:
                            fw_text = " | ".join([f"`{fw}`" for fw in user_impact.applicable_frameworks])
                            st.markdown(fw_text)
                        
                        if user_impact.matched_topics:
                            topics_text = " | ".join([f"`{topic}`" for topic in user_impact.matched_topics[:5]])
                            st.markdown(topics_text)
                    
                    if st.button(f"📧 Email", key=f"email_{i}"):
                        with st.spinner("Generating..."):
                            try:
                                email_content = analyzer.generate_personalized_email(
                                    document_analysis, 
                                    user_impact, 
                                    document_name
                                )
                                
                                st.subheader(f"📧 {user_impact.email}")
                                st.code(email_content, language="text")
                                
                            except Exception as e:
                                st.error(f"Error: {str(e)}")
        
        # Bulk email generation
        if impacted_users:
            st.subheader("📬 Bulk Emails")
            
            if st.button("📧 Generate All Emails", type="secondary"):
                with st.spinner("Generating emails..."):
                    progress_bar = st.progress(0)
                    
                    # Use the traced workflow function
                    bulk_emails, successful_emails, failed_emails = generate_bulk_emails_workflow(
                        analyzer, document_analysis, impacted_users, document_name
                    )
                    
                    # Update progress bar
                    progress_bar.progress(1.0)
                    
                    st.session_state.bulk_emails = bulk_emails
                    
                    if failed_emails > 0:
                        st.warning(f"⚠️ {successful_emails} emails generated, {failed_emails} failed")
                    else:
                        st.success(f"✅ {successful_emails} emails generated successfully")
                    
            # Display bulk emails if generated
            if 'bulk_emails' in st.session_state:
                st.subheader("📧 Generated")
                
                for email, content in st.session_state.bulk_emails.items():
                    with st.expander(f"📧 {email}"):
                        st.code(content, language="text")
        
        # Clear results button
        if st.button("🗑️ Clear"):
            for key in ['document_analysis', 'impacted_users', 'document_name', 'bulk_emails']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

if __name__ == "__main__":
    st.set_page_config(page_title="RegTech AI - Push Notifications", layout="wide")
    render_push_notification_interface()