import streamlit as st
import pandas as pd
import json
import sys
from pathlib import Path
from catboost import CatBoostRegressor

# Linking the Utils ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT / "app"))

from utils.hybrid_engine import generate_hybrid_claim
from utils.visualizer import create_claim_waterfall

# Configuration Paths
MODEL_PATH    = PROJECT_ROOT / "models" / "catboost_claim_model.cbm"
METADATA_PATH = PROJECT_ROOT / "models" / "model_features.json"

st.title("Claim Estimation")

# Security Check
if not st.session_state.get('damage_detected'):
    st.warning("⚠️ Please complete the Visual Damage Assessment (Step 1) before estimating claims.")
    st.page_link("pages/1_vision_assessment.py", label="Go to Vision Assessment", icon="📸")
    st.stop()

detected_class = st.session_state.damage_class
st.success(f"✅ Linking AI Vision Output: **{detected_class}** detected.")

# Load Regression Model
@st.cache_resource
def load_regression_model():
    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        return None, None
    model = CatBoostRegressor()
    model.load_model(str(MODEL_PATH))
    with open(METADATA_PATH, 'r') as f:
        metadata = json.load(f)
    return model, metadata

model, metadata = load_regression_model()

if model is None:
    st.error("Regression model missing. Please run `src/train_regression.py` first.")
    st.stop()

st.markdown("---")
st.subheader("🚗 Vehicle & Incident Context")

# Helper text
st.caption(
    "Provide accurate details below. The **Collision Severity** and **Vehicle Brand** "
    "have the largest influence on the final estimate."
)

# User Input Form
with st.form("estimation_form"):
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Vehicle Details")
        auto_make = st.selectbox(
            "Vehicle Brand",
            ["Maruti", "Hyundai", "Honda", "Toyota", "Kia", "MG",
             "Skoda", "Volkswagen", "Tata", "Mahindra", "Renault",
             "Ford", "BMW", "Mercedes-Benz", "Audi", "Volvo"],
            help="Brand tier directly affects part and labour costs."
        )
        auto_year = st.number_input(
            "Manufacturing Year", min_value=2000, max_value=2025, value=2018,
            help="Older vehicles attract higher depreciation."
        )
        fuel_type = st.selectbox(
            "Fuel Type",
            ["Petrol", "Diesel", "Hybrid", "Electric"],
            help="EV/Hybrid parts are more expensive to source."
        )
        policy_deductable = st.number_input(
            "Policy Deductible (₹)", min_value=0, value=1000,
            help="Amount the insured bears before the insurer pays."
        )

    with col2:
        st.markdown("#### Incident Details")
        collision_type = st.selectbox(
            "Collision Type",
            ["Front Collision", "Rear Collision", "Side Collision", "Unknown"],
            help="Indicates which part of the vehicle was impacted."
        )
        collision_severity = st.selectbox(
            "Collision Severity",
            ["Minor Collision", "Moderate Collision", "Major Collision"],
            index=1,
            help=(
                "**Minor** – low-speed nudge / parking scrape  \n"
                "**Moderate** – standard road accident  \n"
                "**Major** – high-speed / multi-vehicle impact"
            )
        )
        number_of_vehicles = st.selectbox(
            "Vehicles Involved",
            [1, 2, 3],
            help="Multi-vehicle accidents often mean higher claims."
        )
        bodily_injuries = st.number_input(
            "Bodily Injuries Reported", min_value=0, max_value=10, value=0,
            help="Number of people injured in the incident."
        )

    submit = st.form_submit_button("🔍 Generate Financial Estimate", type="primary")

# Prediction Logic
if submit:
    with st.spinner("Crunching actuarial data & ML predictions…"):

        # Derived variables
        vehicle_age              = 2025 - auto_year
        policy_annual_premium    = 1200.0
        injury_claim             = 0
        property_claim           = 0
        total_other_claims       = injury_claim + property_claim
        premium_deductable_ratio = policy_annual_premium / max(policy_deductable, 1)
        vehicles_x_injuries      = number_of_vehicles * (bodily_injuries + 1)

        # Map UI collision severity → ML feature value
        severity_ordinal_map = {
            "Minor Collision":    1,
            "Moderate Collision": 2,
            "Major Collision":    3,
        }
        severity_ordinal = severity_ordinal_map.get(collision_severity, 2)

        # Incident severity label fed to the ML model
        ml_incident_severity_map = {
            "Minor Collision":    "Minor Damage",
            "Moderate Collision": "Major Damage",
            "Major Collision":    "Total Loss",
        }
        ml_incident_severity = ml_incident_severity_map.get(collision_severity, "Major Damage")

        # Construct 24-feature dictionary for CatBoost
        input_data = {
            'auto_make':                    auto_make,
            'auto_model':                   "Unknown",
            'vehicle_age':                  vehicle_age,
            'incident_severity':            ml_incident_severity,
            'severity_ordinal':             severity_ordinal,
            'incident_type':                "Single Vehicle Collision" if number_of_vehicles == 1 else "Multi-vehicle Collision",
            'collision_type':               collision_type,
            'number_of_vehicles_involved':  number_of_vehicles,
            'bodily_injuries':              bodily_injuries,
            'witnesses':                    0,
            'incident_hour_of_the_day':     12,
            'hour_bucket':                  'afternoon',
            'authorities_contacted':        'None',
            'property_damage':              'NO',
            'police_report_available':      'YES',
            'policy_deductable':            policy_deductable,
            'umbrella_limit':               0,
            'policy_annual_premium':        policy_annual_premium,
            'premium_deductable_ratio':     premium_deductable_ratio,
            'insured_occupation':           'Unknown',
            'insured_relationship':         'Unknown',
            'fraud_flag':                   0,
            'vehicles_x_injuries':          vehicles_x_injuries,
            'total_other_claims':           total_other_claims,
        }

        input_df = pd.DataFrame([input_data])
        for col in metadata['categorical_features']:
            if col in input_df.columns:
                input_df[col] = input_df[col].astype(str)

        # ML Prediction (USD → INR)
        raw_ml_prediction_usd = model.predict(input_df)[0]
        ml_prediction_inr     = raw_ml_prediction_usd * 83.0

        # Hybrid Engine
        final_results = generate_hybrid_claim(
            ml_prediction      = ml_prediction_inr,
            auto_make          = auto_make,
            vehicle_age        = vehicle_age,
            fuel_type          = fuel_type,
            detected_class     = detected_class,
            collision_severity = collision_severity,
        )

    # Displays Results
    st.markdown("---")
    st.subheader("Final Claim Assessment")

    engine_color = "🟢" if "Hybrid" in final_results['method_used'] else "🟡"
    st.info(f"**Engine Used:** {engine_color} {final_results['method_used']}")

    m1, m2, m3 = st.columns(3)
    m1.metric("Recommended Payout",  f"₹{final_results['final_claim']:,.0f}")
    m2.metric("Low Range  (−15%)",   f"₹{final_results['low_range']:,.0f}")
    m3.metric("High Range (+15%)",   f"₹{final_results['high_range']:,.0f}")

    st.markdown("### Estimation Breakdown")
    fig = create_claim_waterfall(final_results['breakdown'], final_results['final_claim'])
    st.plotly_chart(fig, use_container_width=True)

    # Deductible note
    if policy_deductable > 0:
        net_payout = max(final_results['final_claim'] - policy_deductable, 0)
        st.caption(
            f"📋 **After deductible of ₹{policy_deductable:,}:** "
            f"estimated net insurer payout ≈ **₹{net_payout:,.0f}**"
        )