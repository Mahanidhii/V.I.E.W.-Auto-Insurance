import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
from pathlib import Path

# Configuration Paths 
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "efficientnet_damage_type.h5"
MAP_PATH   = PROJECT_ROOT / "models" / "class_mappings.json"

# Translation + severity mapping for the Vietnamese dataset classes 
# Each raw class maps to: (English damage description, severity tier, ordinal)
# Severity tiers: Minor Damage | Moderate Damage | Major Damage
CLASS_TRANSLATION = {
    "tray_son":    ("Paint Scratch / Surface Scuff",   "Minor Damage",    1),
    "rach":        ("Deep Scratch / Laceration",        "Minor Damage",    1),
    "mop_lom":     ("Dented Body Panel",                "Moderate Damage", 2),
    "be_den":      ("Cracked / Broken Headlight",       "Moderate Damage", 2),
    "vo_kinh":     ("Shattered Glass",                  "Major Damage",    3),
    "thung":       ("Punctured / Holed Panel",          "Major Damage",    3),
    "mat_bo_phan": ("Missing / Detached Part",          "Major Damage",    3),
}

SEVERITY_STYLE = {
    "Minor Damage":    {"emoji": "🟡", "color": "#f5c518", "label": "Minor Damage"},
    "Moderate Damage": {"emoji": "🟠", "color": "#ff7b00", "label": "Moderate Damage"},
    "Major Damage":    {"emoji": "🔴", "color": "#e03131", "label": "Major Damage"},
}

st.title("Visual Damage Assessment")
st.markdown("Upload a clear photo of the vehicle damage for AI classification.")

# Load Model & Mappings (Cached for Speed)
@st.cache_resource
def load_vision_model():
    if not MODEL_PATH.exists() or not MAP_PATH.exists():
        return None, None
    model = tf.keras.models.load_model(str(MODEL_PATH))
    with open(MAP_PATH, 'r') as f:
        class_names = json.load(f)
    return model, class_names

model, class_names = load_vision_model()

if model is None:
    st.error("⚠️ Model not found! Please ensure you have trained the model and it is saved at `models/efficientnet_damage_type.h5`")
    st.stop()

# File Uploader
uploaded_file = st.file_uploader("Choose an image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Uploaded Image")
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("AI Analysis")
        if st.button("🔍 Run Damage Detection", type="primary"):
            with st.spinner("Analyzing textures and contours..."):
                # Preprocess the image for EfficientNet
                img_resized = image.resize((224, 224))
                img_array  = tf.keras.utils.img_to_array(img_resized)
                img_array  = tf.expand_dims(img_array, 0)

                # Inference
                predictions = model.predict(img_array)
                predicted_class_index = np.argmax(predictions[0])
                raw_class   = class_names[predicted_class_index]
                confidence  = 100 * np.max(predictions[0])

                # Translate raw Vietnamese class → English description + severity tier
                english_label, severity, severity_ordinal = CLASS_TRANSLATION.get(
                    raw_class,
                    (raw_class.replace("_", " ").title(), "Minor Damage", 1)
                )
                style = SEVERITY_STYLE.get(severity, SEVERITY_STYLE["Minor Damage"])

                # Save to session state so Page 2 can use it
                st.session_state.damage_detected   = True
                st.session_state.damage_class      = severity       # e.g. "Major Damage"
                st.session_state.damage_label      = english_label  # e.g. "Missing / Detached Part"
                st.session_state.damage_raw        = raw_class
                st.session_state.severity_ordinal  = severity_ordinal
                st.session_state.damage_confidence = confidence

                # Displaying Results 
                st.success("Analysis Complete!")

                # Primary result: severity classification (big + coloured)
                st.markdown(
                    f"""
                    <div style="
                        background: {style['color']}22;
                        border: 2px solid {style['color']};
                        border-radius: 10px;
                        padding: 16px 20px;
                        margin: 12px 0;
                    ">
                        <p style="margin:0; font-size:13px; color:#aaa;">Damage Severity</p>
                        <p style="margin:4px 0 0; font-size:28px; font-weight:700; color:{style['color']}">
                            {style['emoji']}&nbsp;{severity}
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # Secondary info: damage type detail + confidence
                c1, c2 = st.columns(2)
                c1.metric("Damage Type", english_label)
                c2.metric("Confidence", f"{confidence:.1f}%")

                st.info("The AI has logged this damage profile. You may now proceed to Step 2 for financial estimation.")

# Navigation
if st.session_state.get('damage_detected', False):
    st.markdown("---")
    st.page_link("pages/2_claim_estimation.py", label="Proceed to Claim Estimation ➔", icon="💰")