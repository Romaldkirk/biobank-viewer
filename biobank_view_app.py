"""
Enhanced Biobank Viewer with AI Analysis and Feedback Collection
Includes inline Claude Haiku integration and anonymous feedback system
Updated with card-style view selector and refined UI
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
    /* Custom styling for AI Analysis button - always orange */
    div.stButton > button {
        background-color: #FF6B35 !important;
        color: white !important;
        font-weight: bold !important;
        border: none !important;
        padding: 0.75rem 1.5rem !important;
        font-size: 16px !important;
    }
    div.stButton > button:hover {
        background-color: #FF5722 !important;
        transform: scale(1.02) !important;
    }
    /* Text area styling */
    .stTextArea textarea {
        min-height: 100px;
    }
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Initialize session state
import os  # Add this at the top of your file if not already there

import os  # Add this at the top of your file if not already there

def init_session_state():
    """Initialize all session state variables"""
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    
    if 'ai_analyses' not in st.session_state:
        st.session_state.ai_analyses = {}
    
    if 'feedback_data' not in st.session_state:
        st.session_state.feedback_data = []
    
    if 'view_mode' not in st.session_state:
        st.session_state.view_mode = 'biobank'
    
    if 'anthropic_client' not in st.session_state:
        env_key = os.environ.get("ANTHROPIC_API_KEY")
        
        try:
            if env_key:
                # Workaround for the proxies issue
                import anthropic
                # Use base_url instead of problematic parameters
                st.session_state.anthropic_client = anthropic.Client(
                    api_key=env_key
                )
                st.write("Debug: Anthropic client created successfully")
            else:
                st.session_state.anthropic_client = None
                st.write("Debug: No API key found")
                
        except Exception as e:
            # Try alternative initialization
            try:
                st.session_state.anthropic_client = Anthropic(
                    api_key=env_key
                )
                st.write("Debug: Client created with alternative method")
            except:
                st.session_state.anthropic_client = None
                st.write(f"Debug: Both methods failed: {str(e)}")

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
    r_prospective = match.get('r_prospective', 'Not specified')
    
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
    b_prospective = match.get('b_prospective', 'Not specified')
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
**Collaboration Terms Preference: {r_collaboration}**
**Prospective Collection Required: {r_prospective}**

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
**Collaboration Terms Requirement: {b_collaboration}**
**Prospective Collection Capability: {b_prospective}**

Additional Context:
{f'Description: {b_post_content[:500]}' if b_post_content else ''}
{f'Clinical Data: {b_clinical_information}' if b_clinical_information else ''}
{f'Services: {b_research_services}' if b_research_services else ''}
{f'Certifications: {b_certifications}' if b_certifications else ''}

## CRITICAL COMPATIBILITY CHECKS

### COLLABORATION TERMS:
Request prefers: {r_collaboration}
Biobank requires: {b_collaboration}
**IMPORTANT:** If biobank requires "Yes" (mandatory collaboration) but request prefers "fee-for-service", these are INCOMPATIBLE and will require negotiation.

### PROSPECTIVE COLLECTION:
Request needs: {r_prospective}
Biobank offers: {b_prospective}
**IMPORTANT:** If request requires prospective collection ("Yes") but biobank cannot provide it ("No"), this is a MAJOR INCOMPATIBILITY.

## ANALYSIS REQUEST
Provide a concise partnership assessment covering:

1. **Collaboration Compatibility**: Are the collaboration terms aligned? Flag any conflicts.
2. **Prospective Collection Match**: Can the biobank meet prospective collection needs if required?
3. **Match Strengths**: What makes this a good/poor match overall?
4. **Regulatory Considerations**: Key requirements based on jurisdictions
5. **Next Steps**: Specific actions, especially addressing any term conflicts
6. **Potential Deal-Breakers**: Highlight any critical incompatibilities

Keep response under 300 words. Be specific about collaboration and prospective collection issues."""

    return prompt

def get_ai_analysis(match, biobank_name, request_title):
    """Get AI analysis for a specific match"""
    if not st.session_state.anthropic_client:
        return "AI analysis unavailable - API key not configured"
    
    try:
        # Add debug info
        st.write("Debug: API client exists")  # Temporary debug line
        
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
        
        st.write("Debug: Response received")  # Temporary debug line
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
def save_feedback(feedback_type, context, comment=""):
    """Save feedback to CSV file with optional comment"""
    feedback_entry = {
        'timestamp': datetime.now().isoformat(),
        'session_id': st.session_state.session_id,
        'feedback_type': feedback_type,
        'biobank': context.get('biobank_name', ''),
        'request': context.get('request_title', ''),
        'lead_score': context.get('lead_score', 0),
        'had_ai_analysis': context.get('had_ai_analysis', False),
        'comment': comment  # User's text comment
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
    """Display the simplified scoring breakdown in table format"""
    disease_score = match.get('s_disease', 0)
    type_score = match.get('s_sample_type', 0)
    format_score = match.get('s_sample_format', 0)
    
    # Prepare data for display
    st.markdown("### Match Details Comparison")
    
    # Create columns for the table-like display - no header for first column
    col1, col2, col3, col4 = st.columns([1.5, 2, 2, 1.5])
    
    with col1:
        st.markdown("")  # Empty header for Requirement column
    with col2:
        st.markdown("**Request**")
    with col3:
        st.markdown("**Biobank**")
    with col4:
        st.markdown("**Matching Logic**")
    
    st.markdown("---")
    
    # Specific Focus row
    col1, col2, col3, col4 = st.columns([1.5, 2, 2, 1.5])
    
    r_disease = match.get('r_disease', 'Not specified')
    if r_disease != 'Not specified':
        r_disease = r_disease.replace(',', ', ')
    
    b_disease = match.get('b_disease', 'Not specified')
    if b_disease != 'Not specified':
        b_disease = str(b_disease).replace(',', ', ')
    
    # Disease matching logic
    if abs(disease_score - 6) < 0.1:
        disease_logic = "Exact match (6/6)"
    elif abs(disease_score - 4) < 0.1:
        matched_category = match.get('disease_matched_category', '')
        if matched_category:
            disease_logic = f"Category: {matched_category} (4/6)"
        else:
            disease_logic = "Category match (4/6)"
    elif abs(disease_score - 2) < 0.1:
        disease_logic = "General hospital (2/6)"
    else:
        disease_logic = f"No alignment ({disease_score:.1f}/6)"
    
    with col1:
        st.write("**Specific Focus:**")
    with col2:
        st.write(r_disease)
    with col3:
        st.write(b_disease)
        # Add category/specialty info if available
        b_category = match.get('b_category', '')
        biobank_specialty = match.get('biobank_specialty', '')
        if b_category and b_category != 'Not specified':
            st.caption(f"Categories: {str(b_category).replace(',', ', ')}")
        if biobank_specialty and biobank_specialty != 'Not specified':
            st.caption(f"Specialty: {biobank_specialty}")
    with col4:
        st.write(disease_logic)
    
    # Sample Type row
    col1, col2, col3, col4 = st.columns([1.5, 2, 2, 1.5])
    
    r_sample_type = match.get('r_sample_type', 'Not specified')
    if r_sample_type != 'Not specified':
        r_sample_type = r_sample_type.replace(',', ', ')
    
    b_sample_type = match.get('b_sample_type', 'Not specified')
    if b_sample_type != 'Not specified':
        b_sample_type = b_sample_type.replace(',', ', ')
    
    with col1:
        st.write("**Sample Type:**")
    with col2:
        st.write(r_sample_type)
    with col3:
        st.write(b_sample_type)
    with col4:
        st.write(f"Match score: {int(type_score)}/2")
    
    # Sample Format row
    col1, col2, col3, col4 = st.columns([1.5, 2, 2, 1.5])
    
    r_sample_format = match.get('r_sample_format', 'Not specified')
    if r_sample_format != 'Not specified':
        r_sample_format = r_sample_format.replace(',', ', ')
    
    b_sample_format = match.get('b_sample_format', 'Not specified')
    if b_sample_format != 'Not specified':
        b_sample_format = b_sample_format.replace(',', ', ')
    
    with col1:
        st.write("**Sample Format:**")
    with col2:
        st.write(r_sample_format)
    with col3:
        st.write(b_sample_format)
    with col4:
        st.write(f"Match score: {int(format_score)}/2")
    
    # Geographic Location row
    col1, col2, col3, col4 = st.columns([1.5, 2, 2, 1.5])
    
    r_country = match.get('r_country', 'Not specified')
    b_country = match.get('b_country', 'Not specified')
    
    # Determine geographic compatibility
    if r_country == b_country and r_country != 'Not specified':
        geo_logic = "Same country"
    elif r_country == 'Not specified' or b_country == 'Not specified':
        geo_logic = "Location unclear"
    else:
        # Check for common regional blocks
        eu_countries = ['Germany', 'France', 'Spain', 'Italy', 'Netherlands', 'Belgium', 'Austria', 'Poland']
        if r_country in eu_countries and b_country in eu_countries:
            geo_logic = "Both in EU"
        elif (r_country == 'United States' and b_country == 'Canada') or (r_country == 'Canada' and b_country == 'United States'):
            geo_logic = "US-Canada"
        else:
            geo_logic = f"Cross-border ({calculate_distance(r_country, b_country)})"
    
    with col1:
        st.write("**Location:**")
    with col2:
        st.write(r_country)
    with col3:
        st.write(b_country)
    with col4:
        st.write(geo_logic)
    
    # Preferred Terms of Engagement row
    col1, col2, col3, col4 = st.columns([1.5, 2, 2, 1.5])
    
    r_collaboration = match.get('r_collaboration', 'Not specified')
    b_collaboration = match.get('b_collaboration', 'Not specified')
    
    # Format request collaboration for display with proper capitalization
    if r_collaboration.lower() == 'fee-for-service':
        r_collab_display = 'Fee-for-service'
    elif r_collaboration.lower() == 'collaboration & co-publication':
        r_collab_display = 'Collaboration & co-publication'
    elif r_collaboration.lower() == 'open to discussion':
        r_collab_display = 'Open to discussion'
    elif r_collaboration.lower() == 'it depends':
        r_collab_display = 'It depends'
    elif r_collaboration.lower() == 'other':
        r_collab_display = 'Other'
    else:
        r_collab_display = r_collaboration
    
    # Convert biobank collaboration values to new terminology
    if b_collaboration == 'Yes':
        b_collab_display = 'Collaboration required'
    elif b_collaboration == 'Yes if possible':
        b_collab_display = 'Collaboration if possible'
    elif b_collaboration == 'No':
        b_collab_display = 'Collaboration not required'
    elif b_collaboration == 'Sometimes':
        b_collab_display = 'Collaboration sometimes required'
    else:
        b_collab_display = b_collaboration if b_collaboration else 'Not specified'
    
    # Determine compatibility logic for collaboration
    if r_collaboration.lower() == 'open to discussion':
        collab_logic = "Flexible"
    elif r_collaboration.lower() in ['it depends', 'other']:
        collab_logic = "Check specifics"
    elif r_collaboration.lower() == 'fee-for-service' and b_collaboration == 'Yes':
        collab_logic = "Conflict"
    elif r_collaboration.lower() == 'collaboration & co-publication' and b_collaboration == 'No':
        collab_logic = "Conflict"
    elif b_collaboration in ['Not specified', '', None]:
        collab_logic = "Terms unclear"
    else:
        collab_logic = "Check compatibility"
    
    with col1:
        st.write("**Preferred Terms of Engagement:**")
    with col2:
        st.write(r_collab_display)
    with col3:
        st.write(b_collab_display)
    with col4:
        st.write(collab_logic)
    
    # Prospective Collection row
    col1, col2, col3, col4 = st.columns([1.5, 2, 2, 1.5])
    
    r_prospective = match.get('r_prospective', 'Not specified')
    b_prospective = match.get('b_prospective', 'Not specified')
    
    # Format request prospective for display
    if r_prospective == 'Yes':
        r_prosp_display = 'Requires prospective collection'
    elif r_prospective == 'No':
        r_prosp_display = 'Does not require prospective collection'
    else:
        r_prosp_display = 'Not specified'
    
    # Format biobank prospective for display
    if b_prospective == 'Yes':
        b_prosp_display = 'Can do prospective collection'
    elif b_prospective == 'No':
        b_prosp_display = 'Cannot do prospective collection'
    elif b_prospective == 'Sometimes':
        b_prosp_display = 'Can sometimes do prospective collection'
    else:
        b_prosp_display = 'Not specified'
    
    # Determine prospective compatibility
    if r_prospective == 'Yes' and b_prospective == 'No':
        prospective_logic = "Cannot meet"
    elif r_prospective == 'Yes' and b_prospective == 'Sometimes':
        prospective_logic = "Conditional"
    elif r_prospective == b_prospective and r_prospective != 'Not specified':
        prospective_logic = "Aligned"
    elif b_prospective == 'Not specified' or r_prospective == 'Not specified':
        prospective_logic = "Unclear"
    else:
        prospective_logic = "Compatible"
    
    with col1:
        st.write("**Prospective Collection:**")
    with col2:
        st.write(r_prosp_display)
    with col3:
        st.write(b_prosp_display)
    with col4:
        st.write(prospective_logic)
    
    # Total LeadScore summary
    st.markdown("---")
    total = disease_score + type_score + format_score
    total_display = f"{total:.1f}" if total != int(total) else str(int(total))
    
    # Score summary using columns for better control
    score_col1, score_col2 = st.columns([1, 3])
    
    with score_col1:
        # Color-coded score
        if total >= 7:
            st.success(f"**LeadScore: {total_display}/10**")
        elif total >= 5:
            st.warning(f"**LeadScore: {total_display}/10**")
        else:
            st.error(f"**LeadScore: {total_display}/10**")
    
    with score_col2:
        st.write(f"Disease: {disease_score:.1f} + Sample Type: {type_score} + Sample Format: {format_score}")
        
        # Add interpretation
        if total >= 7:
            st.caption("Strong match - High compatibility")
        elif total >= 5:
            st.caption("Moderate match - Review specific requirements")
        else:
            st.caption("Weak match - Significant gaps in alignment")

def calculate_distance(country1, country2):
    """Simple helper to describe geographic relationship"""
    if country1 == 'Not specified' or country2 == 'Not specified':
        return "unknown"
    
    # Define some common relationships
    same_continent = {
        'Europe': ['Germany', 'France', 'Spain', 'Italy', 'United Kingdom', 'Netherlands', 'Belgium', 'Austria'],
        'North America': ['United States', 'Canada', 'Mexico'],
        'Asia': ['China', 'Japan', 'Singapore', 'India'],
        'Oceania': ['Australia', 'New Zealand']
    }
    
    for continent, countries in same_continent.items():
        if country1 in countries and country2 in countries:
            return "same region"
    
    return "different regions"

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
    
    # AI Analysis button - styled without columns
    st.markdown("---")  # Add a separator line above
    if st.button("Get AI Analysis", key=f"ai_btn_{match_key}"):
        with st.spinner("Analyzing partnership compatibility..."):
            analysis = get_ai_analysis(match, biobank_name, request_title)
            st.session_state.ai_analyses[analysis_key]['analysis'] = analysis
            st.rerun()
        
    # Display analysis if available
    if st.session_state.ai_analyses[analysis_key]['analysis']:
        st.markdown("### AI Partnership Assessment")
        # Display analysis directly without word count
        st.write(st.session_state.ai_analyses[analysis_key]['analysis'])
        
        # Display Q&A history
        for qa in st.session_state.ai_analyses[analysis_key]['qa_history']:
            st.info(f"**Q:** {qa['question']}")
            st.write(f"**A:** {qa['answer']}")
        
        # Follow-up question form
        with st.form(key=f"qa_form_{match_key}_{len(st.session_state.ai_analyses[analysis_key]['qa_history'])}"):
            st.markdown("#### Ask a follow-up question")
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
        
        # Enhanced feedback section with text input
        if not st.session_state.ai_analyses[analysis_key]['feedback_given']:
            st.markdown("---")
            st.markdown("### Feedback")
            st.write("Help us improve by sharing your feedback about this match and AI analysis:")
            
            # Feedback form with text input
            with st.form(key=f"feedback_form_{match_key}"):
                # Radio button for feedback type
                feedback_type = st.radio(
                    "How was this analysis?",
                    ["Helpful", "Scoring Issue", "AI Quality Issue", "Other"],
                    horizontal=True,
                    key=f"feedback_type_{match_key}"
                )
                
                # Text area for detailed feedback
                feedback_comment = st.text_area(
                    "Your comments (optional):",
                    placeholder="Tell us more about your experience, any issues you noticed, or suggestions for improvement...",
                    height=100,
                    key=f"feedback_comment_{match_key}"
                )
                
                # Submit button
                submit_feedback = st.form_submit_button("Submit Feedback", type="primary")
                
                if submit_feedback:
                    context = {
                        'biobank_name': biobank_name,
                        'request_title': request_title,
                        'lead_score': match.get('LeadScore', 0),
                        'had_ai_analysis': True
                    }
                    
                    # Map radio selection to feedback type
                    feedback_type_map = {
                        "Helpful": "helpful",
                        "Scoring Issue": "scoring_issue",
                        "AI Quality Issue": "ai_quality_issue",
                        "Other": "other"
                    }
                    
                    if save_feedback(feedback_type_map[feedback_type], context, feedback_comment):
                        st.session_state.ai_analyses[analysis_key]['feedback_given'] = True
                        st.success("Thank you for your feedback! Your input helps us improve the system.")
                        st.rerun()
        
        # If feedback was already given, show thank you message
        elif st.session_state.ai_analyses[analysis_key]['feedback_given']:
            st.markdown("---")
            st.info("Thank you for providing feedback on this match!")

def render_view_selector():
    """Render card-style view selector with clickable cards"""
    
    # Center the title
    st.markdown("<h3 style='text-align: center; margin-bottom: 30px;'>Choose Your View Mode</h3>", unsafe_allow_html=True)
    
    # Create centered layout
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        # Create two sub-columns for the cards
        card1, spacer, card2 = st.columns([1, 0.2, 1])
        
        with card1:
            # Biobank View Card
            if st.session_state.view_mode == 'biobank':
                # Selected state - show as highlighted card with a simple icon
                st.markdown("""
                <div style='text-align: center; padding: 20px; border: 3px solid #FF6B35; 
                            background-color: #FFF5F0; border-radius: 10px; cursor: pointer;'>
                    <div style='font-size: 40px; margin: 10px;'>🏛</div>
                    <div style='font-weight: bold; font-size: 16px; margin: 5px;'>BIOBANK VIEW</div>
                    <div style='color: #666; font-size: 12px;'>For biobanks to find requests</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                # Unselected state - clickable button
                if st.button("🏛\n**BIOBANK VIEW**\nFor biobanks to find requests", 
                           key="biobank_btn", 
                           use_container_width=True):
                    st.session_state.view_mode = 'biobank'
                    st.rerun()
        
        with card2:
            # Request View Card  
            if st.session_state.view_mode == 'request':
                # Selected state - show as highlighted card with a simple icon
                st.markdown("""
                <div style='text-align: center; padding: 20px; border: 3px solid #FF6B35; 
                            background-color: #FFF5F0; border-radius: 10px; cursor: pointer;'>
                    <div style='font-size: 40px; margin: 10px;'>📋</div>
                    <div style='font-weight: bold; font-size: 16px; margin: 5px;'>REQUEST VIEW</div>
                    <div style='color: #666; font-size: 12px;'>For researchers to find biobanks</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                # Unselected state - clickable button
                if st.button("📋\n**REQUEST VIEW**\nFor researchers to find biobanks", 
                           key="request_btn", 
                           use_container_width=True):
                    st.session_state.view_mode = 'request'
                    st.rerun()
    
    # Add CSS to style the unselected buttons
    st.markdown("""
    <style>
        /* Style the buttons to look more card-like */
        [data-testid="stButton"] button {
            height: 140px !important;
            border: 2px solid #E0E0E0 !important;
            border-radius: 10px !important;
            background-color: white !important;
            white-space: pre-line !important;
            transition: all 0.2s ease !important;
        }
        [data-testid="stButton"] button:hover {
            border-color: #FF6B35 !important;
            transform: translateY(-2px) !important;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1) !important;
        }
    </style>
    """, unsafe_allow_html=True)
            
def render_biobank_view(match_scores):
    """Render the biobank-centric view"""
    st.info("Select a biobank to explore matching research requests")
    
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
        st.metric("Biobanks with Relevant Matches", len(unique_biobanks))
    
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
            match_key = f"biobank_{selected_biobank}_{request_title}_{idx}"
            
            with st.expander(
                f"**Match {idx + 1}:** {request_title} (Score: {lead_score:.1f}/10)",
                expanded=(idx == 0)  # Only expand the first match
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

def render_request_view(match_scores):
    """Render the request-centric view"""
    st.info("Select a research request to see the best matching biobanks")
    
    # Get unique requests
    unique_requests = sorted(match_scores['post_title'].unique())
    
    # Request selector
    col1, col2 = st.columns([3, 1])
    
    with col1:
        selected_request = st.selectbox(
            "Select Research Request",
            options=unique_requests,
            key="request_selector"
        )
    
    with col2:
        st.metric("Total Requests", len(unique_requests))
    
    if selected_request:
        # Get matches for selected request
        request_matches = match_scores[
            match_scores['post_title'] == selected_request
        ].copy()
        
        # Sort by LeadScore descending (best matches first)
        if 'LeadScore' in request_matches.columns:
            request_matches = request_matches.sort_values('LeadScore', ascending=False)
        
        st.markdown(f"## {selected_request}")
        
        # Display request details if available
        if not request_matches.empty:
            first_match = request_matches.iloc[0]
            
            # Show request requirements in a clean box
            with st.container():
                st.markdown("### Request Requirements")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    r_disease = first_match.get('r_disease', 'Not specified')
                    if r_disease != 'Not specified':
                        r_disease = r_disease.replace(',', ', ')
                    st.write(f"**Disease Focus:** {r_disease}")
                    
                    r_sample_type = first_match.get('r_sample_type', 'Not specified')
                    if r_sample_type != 'Not specified':
                        r_sample_type = r_sample_type.replace(',', ', ')
                    st.write(f"**Sample Type:** {r_sample_type}")
                
                with col2:
                    r_sample_format = first_match.get('r_sample_format', 'Not specified')
                    if r_sample_format != 'Not specified':
                        r_sample_format = r_sample_format.replace(',', ', ')
                    st.write(f"**Sample Format:** {r_sample_format}")
                    
                    r_country = first_match.get('r_country', 'Not specified')
                    st.write(f"**Location:** {r_country}")
                
                with col3:
                    r_collaboration = first_match.get('r_collaboration', 'Not specified')
                    st.write(f"**Collaboration:** {r_collaboration}")
                    
                    r_prospective = first_match.get('r_prospective', 'Not specified')
                    st.write(f"**Prospective:** {r_prospective}")
        
        # Display metrics
        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Matching Biobanks", len(request_matches))
        with col2:
            avg_score = request_matches['LeadScore'].mean()
            st.metric("Average Score", f"{avg_score:.1f}/10")
        with col3:
            high_matches = len(request_matches[request_matches['LeadScore'] >= 7])
            st.metric("High Matches (7+)", high_matches)
        with col4:
            perfect_matches = len(request_matches[request_matches['LeadScore'] >= 9.5])
            st.metric("Near-Perfect (9.5+)", perfect_matches)
        
        st.markdown("---")
        
        # Display top 10 matches (removed filter controls)
        st.markdown(f"### Top 10 Biobanks for This Request")
        
        # Get top 10 matches with score >= 5
        filtered_matches = request_matches[request_matches['LeadScore'] >= 5.0].head(10)
        
        if len(filtered_matches) == 0:
            st.warning("No biobanks found with LeadScore >= 5.0")
        else:
            # Display each biobank match
            for idx, (_, match) in enumerate(filtered_matches.iterrows()):
                biobank_name = match.get('biobank_name', 'Unknown Biobank')
                lead_score = match.get('LeadScore', 0)
                
                # Create unique key for this match
                match_key = f"request_{selected_request}_{biobank_name}_{idx}"
                
                # Color code the expander based on score
                if lead_score >= 8:
                    score_emoji = "HIGH"
                elif lead_score >= 6:
                    score_emoji = "MEDIUM"
                else:
                    score_emoji = "LOW"
                
                with st.expander(
                    f"{score_emoji} **Rank {idx + 1}:** {biobank_name} (Score: {lead_score:.1f}/10)",
                    expanded=(idx == 0)  # Only expand the first (top-ranked) match
                ):
                    st.markdown(f"## {biobank_name}")
                    
                    # Skip location/specialty box - go straight to table
                    # Display scoring breakdown
                    display_scoring_breakdown(match)
                    
                    # Add separator before AI section
                    st.markdown("---")
                    
                    # Display AI analysis section
                    display_ai_analysis_section(
                        match,
                        biobank_name,
                        selected_request,
                        match_key
                    )
                    
# Main application
def main():
    # Initialize session state
    init_session_state()
    
    # Temporary debug - remove after testing
    if st.session_state.anthropic_client:
        st.success("API client initialized successfully")
    else:
        st.error("API client not initialized")

    # No title - will be added on webpage
    
    # Render view selector cards
    render_view_selector()
    
    st.markdown("---")
    
    # Load data
    match_scores = load_match_data()
    
    if match_scores.empty:
        st.error("No data available. Please check data files.")
        return
    
    if st.session_state.view_mode == 'biobank':
        render_biobank_view(match_scores)
    else:
        render_request_view(match_scores)
    
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