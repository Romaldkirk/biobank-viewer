"""
Enhanced Biobank Viewer with AI Analysis and Feedback Collection
Includes inline Claude Haiku integration and anonymous feedback system
"""

import streamlit as st
import pandas as pd
import os
import uuid
from datetime import datetime
import json
from anthropic import Anthropic

# Page configuration
st.set_page_config(
    page_title="Biobank Partnership Opportunities",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Hide Streamlit UI for clean embedding
hide_streamlit_style = """
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}
    section.main > div:has(~ footer) {
        padding-bottom: 1rem;
    }
    /* Improve button styling for AI interface */
    .stButton > button {
        width: 100%;
        margin-top: 0.5rem;
    }
    /* Feedback buttons styling */
    .feedback-container {
        margin-top: 1rem;
        padding-top: 1rem;
        border-top: 1px solid #e0e0e0;
    }
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    """Initialize all session state variables"""
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    
    if 'ai_analyses' not in st.session_state:
        st.session_state.ai_analyses = {}
    
    if 'feedback_data' not in st.session_state:
        st.session_state.feedback_data = []
    
    if 'anthropic_client' not in st.session_state:
        # Initialize Anthropic client with API key from secrets
        try:
            api_key = st.secrets.get("ANTHROPIC_API_KEY", None)
            if api_key:
                st.session_state.anthropic_client = Anthropic(api_key=api_key)
            else:
                st.session_state.anthropic_client = None
                st.warning("AI Analysis unavailable - API key not configured")
        except Exception as e:
            st.session_state.anthropic_client = None
            st.error(f"Failed to initialize AI client: {e}")

# Data loading functions
@st.cache_data
def load_match_data():
    """Load enriched match scores with context fields"""
    try:
        df = pd.read_csv('data/pair_scores_enriched.csv')
        # Filter out irrelevant matches (s_disease < 2.0)
        df = df[df['s_disease'] >= 2.0].copy()
        return df
    except FileNotFoundError:
        st.error("Data file not found. Please ensure pair_scores_enriched.csv is in the data folder.")
        return pd.DataFrame()

@st.cache_data
def load_knowledge_base():
    """Load biobanking knowledge base for AI context"""
    knowledge_content = ""
    knowledge_path = 'config/knowledge/Playbook_ALL.md'
    
    if os.path.exists(knowledge_path):
        try:
            with open(knowledge_path, 'r', encoding='utf-8') as f:
                knowledge_content = f.read()
        except Exception as e:
            st.warning(f"Could not load knowledge base: {e}")
    
    return knowledge_content

# AI Analysis functions
def generate_ai_prompt(match, biobank_name, request_title, knowledge_base):
    """Generate comprehensive prompt for AI analysis"""
    
    # Extract match details
    disease_score = match.get('s_disease', 0)
    type_score = match.get('s_sample_type', 0) 
    format_score = match.get('s_sample_format', 0)
    lead_score = match.get('LeadScore', 0)
    
    # Request context
    r_disease = match.get('r_disease', 'Not specified')
    r_sample_type = match.get('r_sample_type', 'Not specified')
    r_sample_format = match.get('r_sample_format', 'Not specified')
    r_country = match.get('r_country', 'Not specified')
    r_collaboration = match.get('r_collaboration', 'Not specified')
    
    # Additional request context if available
    r_post_content = match.get('r_post_content', '')
    r_no_cases = match.get('r_no_cases', '')
    r_data_required = match.get('r_data_required', '')
    r_inclusion_criteria = match.get('r_inclusion_criteria', '')
    r_exclusion_criteria = match.get('r_exclusion_criteria', '')
    
    # Biobank context
    b_disease = match.get('b_disease', 'Not specified')
    b_sample_type = match.get('b_sample_type', 'Not specified')
    b_sample_format = match.get('b_sample_format', 'Not specified')
    b_country = match.get('b_country', 'Not specified')
    b_collaboration = match.get('b_collaboration', 'Not specified')
    biobank_specialty = match.get('biobank_specialty', 'Not specified')
    
    # Additional biobank context if available
    b_post_content = match.get('b_post_content', '')
    b_clinical_information = match.get('b_clinical_information', '')
    b_research_services = match.get('b_research_services', '')
    b_certifications = match.get('b_certifications', '')
    
    prompt = f"""You are a biobank partnership specialist analyzing compatibility between a research request and a biobank.
Use the provided knowledge base to give specific, actionable guidance.

## KNOWLEDGE BASE CONTEXT
{knowledge_base[:3000]}  # Truncate to manage token usage

## MATCH OVERVIEW
Biobank: {biobank_name}
Research Request: {request_title}
LeadScore: {lead_score:.1f}/10
- Disease Match: {disease_score:.1f}/6
- Sample Type Match: {type_score:.1f}/2
- Sample Format Match: {format_score:.1f}/2

## REQUEST DETAILS
Disease Focus: {r_disease}
Sample Types Needed: {r_sample_type}
Sample Format: {r_sample_format}
Location: {r_country}
Collaboration Preference: {r_collaboration}

Additional Context:
{f'Project Overview: {r_post_content[:500]}' if r_post_content else ''}
{f'Study Scale: {r_no_cases}' if r_no_cases else ''}
{f'Data Requirements: {r_data_required}' if r_data_required else ''}

## BIOBANK DETAILS
Specialty: {biobank_specialty}
Disease Focus: {b_disease}
Sample Types Available: {b_sample_type}
Sample Formats: {b_sample_format}
Location: {b_country}
Collaboration Requirement: {b_collaboration}

Additional Context:
{f'Description: {b_post_content[:500]}' if b_post_content else ''}
{f'Clinical Data: {b_clinical_information}' if b_clinical_information else ''}
{f'Services: {b_research_services}' if b_research_services else ''}
{f'Certifications: {b_certifications}' if b_certifications else ''}

## ANALYSIS REQUEST
Provide a concise partnership assessment covering:
1. **Match Strengths**: What makes this a good/poor match?
2. **Regulatory Considerations**: Key requirements based on jurisdictions
3. **Next Steps**: Specific actions for moving forward
4. **Potential Challenges**: Any issues to address

Keep response under 300 words. Be specific and actionable."""

    return prompt

def get_ai_analysis(match, biobank_name, request_title):
    """Get AI analysis for a specific match"""
    if not st.session_state.anthropic_client:
        return "AI analysis unavailable - API key not configured"
    
    try:
        knowledge_base = load_knowledge_base()
        prompt = generate_ai_prompt(match, biobank_name, request_title, knowledge_base)
        
        # Use Claude Haiku for cost efficiency (~$0.001 per analysis)
        response = st.session_state.anthropic_client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=500,
            temperature=0.7,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        return response.content[0].text
        
    except Exception as e:
        return f"Analysis failed: {str(e)}"

def handle_followup_question(original_analysis, question, match_context):
    """Handle follow-up questions about the analysis"""
    if not st.session_state.anthropic_client:
        return "AI unavailable for follow-up questions"
    
    try:
        prompt = f"""Previous analysis:
{original_analysis}

Match context:
- Biobank: {match_context['biobank_name']}
- Request: {match_context['request_title']}
- Score: {match_context['lead_score']}/10

User's follow-up question: {question}

Provide a specific, helpful answer based on the context. Keep under 200 words."""

        response = st.session_state.anthropic_client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=300,
            temperature=0.7,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        return response.content[0].text
        
    except Exception as e:
        return f"Follow-up failed: {str(e)}"

# Feedback functions
def save_feedback(feedback_type, context, details=""):
    """Save feedback to CSV file"""
    feedback_entry = {
        'timestamp': datetime.now().isoformat(),
        'session_id': st.session_state.session_id,
        'feedback_type': feedback_type,
        'biobank': context.get('biobank_name', ''),
        'request': context.get('request_title', ''),
        'lead_score': context.get('lead_score', 0),
        'had_ai_analysis': context.get('had_ai_analysis', False),
        'details': details
    }
    
    # Append to session state
    st.session_state.feedback_data.append(feedback_entry)
    
    # Save to CSV
    feedback_file = 'feedback_data.csv'
    df_feedback = pd.DataFrame([feedback_entry])
    
    if os.path.exists(feedback_file):
        df_feedback.to_csv(feedback_file, mode='a', header=False, index=False)
    else:
        df_feedback.to_csv(feedback_file, mode='w', header=True, index=False)
    
    return True

# Display functions
def display_scoring_breakdown(match):
    """Display the simplified scoring breakdown"""
    disease_score = match.get('s_disease', 0)
    type_score = match.get('s_sample_type', 0)
    format_score = match.get('s_sample_format', 0)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Request Requirements:**")
        r_disease = match.get('r_disease', 'Not specified')
        if r_disease != 'Not specified':
            r_disease = r_disease.replace(',', ', ')
        st.write(f"• **Specific Focus:** {r_disease}")
        
        r_sample_type = match.get('r_sample_type', 'Not specified')
        if r_sample_type != 'Not specified':
            r_sample_type = r_sample_type.replace(',', ', ')
        st.write(f"• **Sample Type:** {r_sample_type}")
        
        r_sample_format = match.get('r_sample_format', 'Not specified')
        if r_sample_format != 'Not specified':
            r_sample_format = r_sample_format.replace(',', ', ')
        st.write(f"• **Sample Format:** {r_sample_format}")
    
    with col2:
        st.markdown("**Biobank Capabilities:**")
        b_disease = match.get('b_disease', 'Not specified')
        if b_disease != 'Not specified':
            b_disease = str(b_disease).replace(',', ', ')
        st.write(f"• **Specific Focus:** {b_disease}")
        
        b_category = match.get('b_category', 'Not specified')
        if b_category != 'Not specified':
            b_category = str(b_category).replace(',', ', ')
            st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;• ***Disease Categories:*** {b_category}", 
                       unsafe_allow_html=True)
        
        biobank_specialty = match.get('biobank_specialty', 'Not specified')
        st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;• ***Biobank Specialty:*** {biobank_specialty}", 
                   unsafe_allow_html=True)
        
        b_sample_type = match.get('b_sample_type', 'Not specified')
        if b_sample_type != 'Not specified':
            b_sample_type = b_sample_type.replace(',', ', ')
        st.write(f"• **Sample Types:** {b_sample_type}")
        
        b_sample_format = match.get('b_sample_format', 'Not specified')
        if b_sample_format != 'Not specified':
            b_sample_format = b_sample_format.replace(',', ', ')
        st.write(f"• **Sample Formats:** {b_sample_format}")
    
    st.markdown("---")
    st.markdown("**Disease Matching Logic:**")
    
    # Display disease score explanation
    if abs(disease_score - 6) < 0.1:
        st.write(f"**Specific Focus:** 6/6 - Exact match")
    elif abs(disease_score - 4) < 0.1:
        st.write(f"**Specific Focus:** 4/6 - Category match")
    elif abs(disease_score - 2) < 0.1:
        st.write(f"**Specific Focus:** 2/6 - General hospital match")
    else:
        st.write(f"**Specific Focus:** {disease_score:.1f}/6")
    
    st.write(f"**Sample Type:** {int(type_score)}/2")
    st.write(f"**Sample Format:** {int(format_score)}/2")
    
    total = disease_score + type_score + format_score
    st.markdown(f"**Total LeadScore: {total:.1f}/10**")

def display_ai_analysis_section(match, biobank_name, request_title, match_key):
    """Display AI analysis section with follow-up capability"""
    
    # Create unique keys for this match
    analysis_key = f"analysis_{match_key}"
    qa_key = f"qa_{match_key}"
    
    # Initialize analysis state for this match if needed
    if analysis_key not in st.session_state.ai_analyses:
        st.session_state.ai_analyses[analysis_key] = {
            'analysis': None,
            'qa_history': [],
            'feedback_given': False
        }
    
    # AI Analysis button
    if st.button("🤖 Get AI Analysis", key=f"ai_btn_{match_key}", type="secondary"):
        with st.spinner("Analyzing partnership compatibility..."):
            analysis = get_ai_analysis(match, biobank_name, request_title)
            st.session_state.ai_analyses[analysis_key]['analysis'] = analysis
            st.rerun()
    
    # Display analysis if available
    if st.session_state.ai_analyses[analysis_key]['analysis']:
        st.markdown("### 🤖 AI Partnership Assessment")
        st.write(st.session_state.ai_analyses[analysis_key]['analysis'])
        
        # Display Q&A history
        for qa in st.session_state.ai_analyses[analysis_key]['qa_history']:
            st.info(f"**Q:** {qa['question']}")
            st.write(f"**A:** {qa['answer']}")
        
        # Follow-up question form
        with st.form(key=f"qa_form_{match_key}_{len(st.session_state.ai_analyses[analysis_key]['qa_history'])}"):
            st.markdown("#### 💬 Ask a follow-up question")
            question = st.text_input(
                "Your question:",
                placeholder="e.g., What are the specific regulatory requirements for this transfer?"
            )
            submitted = st.form_submit_button("Ask", type="primary")
            
            if submitted and question:
                match_context = {
                    'biobank_name': biobank_name,
                    'request_title': request_title,
                    'lead_score': match.get('LeadScore', 0)
                }
                
                with st.spinner("Getting answer..."):
                    answer = handle_followup_question(
                        st.session_state.ai_analyses[analysis_key]['analysis'],
                        question,
                        match_context
                    )
                    
                    # Store Q&A
                    st.session_state.ai_analyses[analysis_key]['qa_history'].append({
                        'question': question,
                        'answer': answer
                    })
                    st.rerun()
        
        # Feedback section
        if not st.session_state.ai_analyses[analysis_key]['feedback_given']:
            st.markdown("---")
            st.markdown("#### 📝 Feedback")
            
            col1, col2, col3 = st.columns(3)
            
            context = {
                'biobank_name': biobank_name,
                'request_title': request_title,
                'lead_score': match.get('LeadScore', 0),
                'had_ai_analysis': True
            }
            
            with col1:
                if st.button("👍 Helpful", key=f"helpful_{match_key}"):
                    if save_feedback('helpful', context):
                        st.session_state.ai_analyses[analysis_key]['feedback_given'] = True
                        st.success("Thank you for your feedback!")
                        st.rerun()
            
            with col2:
                if st.button("📊 Scoring Issue", key=f"scoring_{match_key}"):
                    if save_feedback('scoring_issue', context):
                        st.session_state.ai_analyses[analysis_key]['feedback_given'] = True
                        st.warning("Thank you. We'll review the scoring.")
                        st.rerun()
            
            with col3:
                if st.button("🤔 AI Quality Issue", key=f"ai_issue_{match_key}"):
                    if save_feedback('ai_quality_issue', context):
                        st.session_state.ai_analyses[analysis_key]['feedback_given'] = True
                        st.info("Thank you. We'll improve the AI responses.")
                        st.rerun()

# Main application
def main():
    # Initialize session state
    init_session_state()
    
    st.title("🏥 Biobank Partnership Opportunities")
    st.info("Select a biobank to explore matching research requests and get AI-powered partnership assessments")
    
    # Load data
    match_scores = load_match_data()
    
    if match_scores.empty:
        st.error("No data available. Please check data files.")
        return
    
    # Get unique biobanks
    unique_biobanks = sorted(match_scores['biobank_name'].unique())
    
    # Biobank selector
    col1, col2 = st.columns([3, 1])
    
    with col1:
        selected_biobank = st.selectbox(
            "Select Biobank",
            options=unique_biobanks,
            key="biobank_selector"
        )
    
    with col2:
        st.metric("Total Biobanks", len(unique_biobanks))
    
    if selected_biobank:
        # Get matches for selected biobank
        biobank_matches = match_scores[
            match_scores['biobank_name'] == selected_biobank
        ].copy()
        
        # Sort by LeadScore
        if 'LeadScore' in biobank_matches.columns:
            biobank_matches = biobank_matches.sort_values('LeadScore', ascending=False)
        
        st.markdown(f"## {selected_biobank}")
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Matching Requests", len(biobank_matches))
        with col2:
            avg_score = biobank_matches['LeadScore'].mean()
            st.metric("Average Score", f"{avg_score:.1f}/10")
        with col3:
            high_matches = len(biobank_matches[biobank_matches['LeadScore'] >= 7])
            st.metric("High Matches (7+)", high_matches)
        
        st.markdown("---")
        st.markdown("### Research Requests Matching This Biobank")
        
        # Display each match
        for idx, (_, match) in enumerate(biobank_matches.iterrows()):
            request_title = match.get('post_title', 'Unknown Request')
            lead_score = match.get('LeadScore', 0)
            
            # Create unique key for this match
            match_key = f"{selected_biobank}_{request_title}_{idx}"
            
            with st.expander(
                f"**Match {idx + 1}:** {request_title} (Score: {lead_score:.1f}/10)",
                expanded=(idx == 0)  # Expand first match only
            ):
                st.markdown(f"## {request_title}")
                
                # Display scoring breakdown
                display_scoring_breakdown(match)
                
                # Add separator before AI section
                st.markdown("---")
                
                # Display AI analysis section
                display_ai_analysis_section(
                    match,
                    selected_biobank,
                    request_title,
                    match_key
                )
    
    # Footer with session info (for debugging)
    with st.sidebar:
        st.caption(f"Session: {st.session_state.session_id[:8]}...")
        if st.button("Download Feedback Data"):
            if os.path.exists('feedback_data.csv'):
                df = pd.read_csv('feedback_data.csv')
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"feedback_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

if __name__ == "__main__":
    main()