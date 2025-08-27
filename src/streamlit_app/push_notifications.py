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

# Add parent directory to path to import modules
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from src.langchain_modules.document_analyzer import DocumentAnalyzer, DocumentAnalysis, UserImpactScore
from src.langchain_modules.agent import AIAgent

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
    
    # Analysis parameters
    if document_to_analyze:
        col1, col2 = st.columns(2)
        with col1:
            min_impact_score = st.slider("Impact Threshold", 0.0, 1.0, 0.2, 0.05)
        with col2:
            max_users_display = st.number_input("Max Users", 1, 50, 10)
    
    if document_to_analyze is not None:        
        # Analyze button
        if st.button("🔍 Analyze & Identify Users", type="primary"):
            
            with st.spinner("Analyzing..."):
                try:
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
                    
                    # Identify impacted users
                    impacted_users = analyzer.identify_impacted_users(
                        document_analysis, 
                        min_impact_score=min_impact_score
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
            col1, col2, col3, col4 = st.columns(4)
            high_impact = len([u for u in impacted_users if u.impact_score >= 0.7])
            estimated_hours = high_impact * 2.5
            
            with col1:
                st.metric("Impacted", len(impacted_users))
            with col2:
                avg_score = sum(user.impact_score for user in impacted_users) / len(impacted_users)
                st.metric("Avg Score", f"{avg_score:.2f}")
            with col3:
                st.metric("High Priority", high_impact)
            with col4:
                st.metric("Est. Hours", f"{estimated_hours:.1f}h")
            
            if high_impact > 0:
                potential_revenue = estimated_hours * 450
                st.success(f"💰 ${potential_revenue:,.0f} opportunity")
            
            st.divider()
            
            # Display user table
            users_to_show = impacted_users[:max_users_display]
            
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
                bulk_users = impacted_users[:10]
                
                with st.spinner(f"Generating {len(bulk_users)} emails..."):
                    progress_bar = st.progress(0)
                    
                    bulk_emails = {}
                    for i, user_impact in enumerate(bulk_users):
                        try:
                            email_content = analyzer.generate_personalized_email(
                                document_analysis, 
                                user_impact, 
                                document_name
                            )
                            bulk_emails[user_impact.email] = email_content
                            
                        except Exception as e:
                            st.error(f"Error: {user_impact.email}")
                            bulk_emails[user_impact.email] = f"Error: {str(e)}"
                        
                        progress_bar.progress((i + 1) / len(bulk_users))
                    
                    st.session_state.bulk_emails = bulk_emails
                    st.success(f"✅ {len(bulk_emails)} emails generated")
                    
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