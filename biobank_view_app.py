import streamlit as st
import pandas as pd

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
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

@st.cache_data
def load_data():
    # Load your enriched match scores
    df = pd.read_csv('data/pair_scores_enriched.csv')
    # Filter out irrelevant matches
    df = df[df['s_disease'] >= 2.0].copy()
    return df

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

# Main App
def main():
    st.title("Biobank Partnership Opportunities")
    st.info("Select a biobank to see all research requests that match with it")
    
    # Load data
    match_scores = load_data()
    
    # Get unique biobanks
    unique_biobanks = sorted(match_scores['biobank_name'].unique())
    
    # Biobank selector
    selected_biobank = st.selectbox(
        "Select Biobank",
        options=unique_biobanks,
        key="biobank_selector"
    )
    
    if selected_biobank:
        # Get matches for selected biobank
        biobank_matches = match_scores[
            match_scores['biobank_name'] == selected_biobank
        ].copy()
        
        # Sort by LeadScore
        if 'LeadScore' in biobank_matches.columns:
            biobank_matches = biobank_matches.sort_values('LeadScore', ascending=False)
        
        st.markdown(f"## {selected_biobank}")
        st.markdown(f"#### {len(biobank_matches)} Research Requests Match This Biobank")
        
        # Display each match
        for idx, (_, match) in enumerate(biobank_matches.iterrows()):
            request_title = match.get('post_title', 'Unknown Request')
            lead_score = match.get('LeadScore', 0)
            
            with st.expander(
                f"Match {idx + 1}: {request_title} (Score: {lead_score:.1f}/10)",
                expanded=(idx == 0)  # Expand first match only
            ):
                st.markdown(f"## {request_title}")
                display_scoring_breakdown(match)

if __name__ == "__main__":
    main()