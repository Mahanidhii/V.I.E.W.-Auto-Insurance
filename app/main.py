import streamlit as st

# Must be the first Streamlit command
st.set_page_config(
    page_title="Automated Vehicle Insurance Assessor",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define the pages using Streamlit's native routing
home_page = st.Page(
    "pages/home.py", 
    title="Home",  
    default=True
)
vision_page = st.Page(
    "pages/1_vision_assessment.py", 
    title="Damage Detection", 
    icon="📸"
)
claim_page = st.Page(
    "pages/2_claim_estimation.py", 
    title="Claim Estimation", 
    icon="💰"
)

# Set up the navigation sidebar
pg = st.navigation({
    "Overview": [home_page],
    "Assessment Pipeline": [vision_page, claim_page]
})

# Initialize session state variables if they don't exist yet
if 'damage_detected' not in st.session_state:
    st.session_state.damage_detected = False
if 'damage_class' not in st.session_state:
    st.session_state.damage_class = None

# Run the selected page
pg.run()