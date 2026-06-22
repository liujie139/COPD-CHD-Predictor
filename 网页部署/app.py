import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import os

# ---------------------------------------------------------
# 1. Page Configuration & Custom CSS
# ---------------------------------------------------------
st.set_page_config(
    page_title="COPD-CHD Risk Calculator", 
    page_icon="🫀", 
    layout="wide"
)

st.markdown("""
    <style>
    .main {background-color: #ffffff;}
    .stAlert {border-radius: 8px;}
    .metric-container {text-align: center;}
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. Model & Artifact Loading
# ---------------------------------------------------------
@st.cache_resource
def load_artifacts():
    # Ensure the script looks for the model in the same directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'copd_chd_model.pkl')
    return joblib.load(model_path)

try:
    artifacts = load_artifacts()
    calibrated_model = artifacts['calibrated_model']
    shap_model = artifacts['shap_model']
    scaler = artifacts['scaler']
    model_features = artifacts['features']
except FileNotFoundError:
    st.error("Error: 'copd_chd_model.pkl' not found. Please ensure the model artifacts are saved in the same directory as this app.")
    st.stop()

# ---------------------------------------------------------
# 3. Clinical Feature Mapping
# ---------------------------------------------------------
FEATURE_DICT = {
    'Age': {'type': 'num', 'name': 'Age (years)', 'default': 65.0, 'step': 1.0},
    'Gender': {'type': 'cat', 'name': 'Gender (0=Female, 1=Male)', 'options': {0: 'Female', 1: 'Male'}},
    'Hypertensi': {'type': 'cat', 'name': 'Hypertension', 'options': {0: 'No', 1: 'Yes'}},
    'Diabetes': {'type': 'cat', 'name': 'Diabetes Mellitus', 'options': {0: 'No', 1: 'Yes'}},
    'Chronic ga': {'type': 'cat', 'name': 'Chronic Gastritis', 'options': {0: 'No', 1: 'Yes'}},
    'Hypokalemi': {'type': 'cat', 'name': 'Hypokalemia', 'options': {0: 'No', 1: 'Yes'}},
    'Pulse rate': {'type': 'num', 'name': 'Pulse Rate (bpm)', 'default': 75.0, 'step': 1.0},
    'DBP': {'type': 'num', 'name': 'Diastolic Blood Pressure [DBP] (mmHg)', 'default': 80.0, 'step': 1.0},
    'TC': {'type': 'num', 'name': 'Total Cholesterol [TC] (mmol/L)', 'default': 4.5, 'step': 0.1},
    'PT': {'type': 'num', 'name': 'Prothrombin Time [PT] (s)', 'default': 12.0, 'step': 0.1},
    'FIB': {'type': 'num', 'name': 'Fibrinogen [FIB] (g/L)', 'default': 3.0, 'step': 0.1},
    'PLT count': {'type': 'num', 'name': 'Platelet Count [PLT] (10^9/L)', 'default': 200.0, 'step': 1.0},
    'LYM count': {'type': 'num', 'name': 'Lymphocyte Count [LYM] (10^9/L)', 'default': 2.0, 'step': 0.1},
    'UA': {'type': 'num', 'name': 'Uric Acid [UA] (μmol/L)', 'default': 300.0, 'step': 1.0},
    'PAB': {'type': 'num', 'name': 'Prealbumin [PAB] (mg/L)', 'default': 250.0, 'step': 1.0},
    'P': {'type': 'num', 'name': 'Phosphorus [P] (mmol/L)', 'default': 1.2, 'step': 0.01}
}

# ---------------------------------------------------------
# 4. Main UI Setup
# ---------------------------------------------------------
st.title("🫁🫀 Multi-feature Prediction Model for CHD Comorbidity in COPD")
st.markdown("""
**Objective:** A machine learning-based clinical decision support system utilizing SHAP for predicting the risk of Coronary Heart Disease (CHD) comorbidity in middle-aged and older adults with Chronic Obstructive Pulmonary Disease (COPD).
***
""")

# ---------------------------------------------------------
# 5. Sidebar: Clinical Parameters Input
# ---------------------------------------------------------
st.sidebar.header("📋 Clinical Parameters")
st.sidebar.markdown("Enter patient demographics and laboratory indices:")

input_data = {}

for raw_feat in model_features:
    if raw_feat in FEATURE_DICT:
        config = FEATURE_DICT[raw_feat]
        if config['type'] == 'num':
            input_data[raw_feat] = st.sidebar.number_input(
                config['name'], 
                value=float(config['default']), 
                step=float(config['step']),
                format="%.2f"
            )
        elif config['type'] == 'cat':
            options_list = list(config['options'].keys())
            display_func = lambda x: config['options'][x]
            
            selected_val = st.sidebar.selectbox(
                config['name'],
                options=options_list,
                format_func=display_func
            )
            input_data[raw_feat] = selected_val
    else:
        input_data[raw_feat] = st.sidebar.number_input(f"Enter {raw_feat}", value=0.0)

# ---------------------------------------------------------
# 6. Core Prediction Logic & SHAP Generation
# ---------------------------------------------------------
if st.sidebar.button("Predict Risk & Generate SHAP Analysis", type="primary", use_container_width=True):
    
    input_df = pd.DataFrame([input_data])[model_features]
    X_scaled = scaler.transform(input_df)
    
    prob = calibrated_model.predict_proba(X_scaled)[0][1]
    
    col1, col2 = st.columns([1, 1.8])
    
    with col1:
        st.subheader("📊 Risk Assessment")
        
        THRESHOLD_MODERATE = 0.30
        THRESHOLD_HIGH = 0.70
        
        if prob < THRESHOLD_MODERATE:
            st.success(f"### Probability: {prob*100:.1f}%\n**Stratification: LOW RISK**")
        elif prob < THRESHOLD_HIGH:
            st.warning(f"### Probability: {prob*100:.1f}%\n**Stratification: MODERATE RISK**")
        else:
            st.error(f"### Probability: {prob*100:.1f}%\n**Stratification: HIGH RISK**")
            
        st.info(
            "💡 **Clinical Recommendation:**\n\n"
            "High-risk individuals warrant prioritized cardiovascular screening "
            "(e.g., electrocardiography, echocardiography) and close multidisciplinary observation."
        )

    with col2:
        st.subheader("🔍 SHAP Force/Waterfall Visualization")
        st.write("Visualizing individual feature contributions driving the current prediction deviation from the base expected value.")
        
        with st.spinner('Generating SHAP explanation...'):
            explainer = shap.TreeExplainer(shap_model)
            shap_values = explainer(X_scaled)
            shap_values.data = input_df.values 
            
            display_names = [FEATURE_DICT.get(f, {}).get('name', f) for f in model_features]
            shap_values.feature_names = display_names
            
            fig, ax = plt.subplots(figsize=(10, 6))
            shap.plots.waterfall(shap_values[0], max_display=12, show=False)
            plt.tight_layout()
            st.pyplot(fig)

# ---------------------------------------------------------
# 7. Disclaimer Footer
# ---------------------------------------------------------
st.markdown("---")
st.caption("⚠️ **Disclaimer:** This computational tool is intended strictly for academic research and educational purposes. It does not replace professional medical diagnosis, advice, or treatment. Clinical judgments should exclusively be rendered by certified healthcare professionals.")