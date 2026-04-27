import streamlit as st

st.title("AI Automated Vehicle Insurance System")
st.markdown("---")

st.markdown("""
### Welcome to the V.I.E.W.
*(Visual Inspection and Estimation Workflow)*

This system is an end-to-end multi-modal machine learning pipeline designed to automate vehicle damage assessment and financial claim estimation.

#### How it works:
1. **📸 Phase 1: Computer Vision Assessment**
   * Upload an image of the damaged vehicle.
   * Our fine-tuned **EfficientNet-B0** deep learning model analyzes the image to classify the specific type of damage (e.g., Dent, Scratch, Broken Glass).
2. **💰 Phase 2: Claim Estimation**
   * Provide the vehicle's context (Make, Age, Fuel Type).
   * A **CatBoost Regression Ensemble** calculates the repair cost.
   * A **Hybrid Rule-Based Engine** verifies the ML output against strict actuarial constraints (brand multipliers, depreciation curves) to guarantee a logically sound financial estimate.

---
""")

st.info("**To begin, select 'Damage Detection' from the sidebar menu.**")