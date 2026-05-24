import streamlit as st
import joblib
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from PIL import Image
from load_ecg_model import download_ecg_model

st.set_page_config(
    page_title="CardioSense AI",
    page_icon="❤️",
    layout="wide"
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f0f4f8; }
    .block-container { padding-top: 2rem; }
    .risk-low    { background:#d4edda; border-left:6px solid #28a745;
                   padding:1rem; border-radius:8px; font-size:1.1rem; }
    .risk-mod    { background:#fff3cd; border-left:6px solid #ffc107;
                   padding:1rem; border-radius:8px; font-size:1.1rem; }
    .risk-high   { background:#f8d7da; border-left:6px solid #dc3545;
                   padding:1rem; border-radius:8px; font-size:1.1rem; }
    .card        { background:white; padding:1.5rem; border-radius:12px;
                   box-shadow:0 2px 8px rgba(0,0,0,0.08); margin-bottom:1rem; }
    .stat-box    { background:#1a3c5e; color:white; padding:1rem;
                   border-radius:10px; text-align:center; }
    h1           { color:#1a3c5e; }
    h2, h3       { color:#1a3c5e; }
</style>
""", unsafe_allow_html=True)


# ── Model Loading ─────────────────────────────────────────────
@st.cache_resource
def load_clinical_models():
    try:
        preprocessor = joblib.load("preprocessor.pkl")
        rf_model     = joblib.load("best_rf_tuned.pkl")
        return preprocessor, rf_model
    except Exception as e:
        st.error(f"Error loading clinical models: {e}")
        return None, None

@st.cache_resource
def load_ecg_model():
    try:
        from tensorflow.keras.models import load_model
        model_path = download_ecg_model()
        model = load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading ECG model: {e}")
        return None

@st.cache_resource
def load_class_labels():
    try:
        with open("ecg_class_indices_new.json", "r") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading class labels: {e}")
        return None

RISK_MAP = {
    "ECG Images of Myocardial Infarction Patients (240x12=2880)": 0.95,
    "ECG Images of Patient that have History of MI (172x12=2064)": 0.75,
    "ECG Images of Patient that have abnormal heartbeat (233x12=2796)": 0.60,
    "Normal Person ECG Images (284x12=3408)": 0.05
}

DISPLAY_NAMES = {
    "ECG Images of Myocardial Infarction Patients (240x12=2880)": "Myocardial Infarction",
    "ECG Images of Patient that have History of MI (172x12=2064)": "History of MI",
    "ECG Images of Patient that have abnormal heartbeat (233x12=2796)": "Abnormal Heartbeat",
    "Normal Person ECG Images (284x12=3408)": "Normal"
}


# ── Sidebar Navigation ────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/color/96/heart-with-pulse.png", width=80)
st.sidebar.title("CardioSense AI")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigate",
    ["🏠 Home", "🔬 Predict", "ℹ️ About"]
)
st.sidebar.markdown("---")
st.sidebar.markdown(
    "<small>⚠️ For educational purposes only.<br>Not a substitute for medical advice.</small>",
    unsafe_allow_html=True
)


# ══════════════════════════════════════════════════════════════
# PAGE 1 — HOME
# ══════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.markdown("<h1 style='text-align:center'>❤️ CardioSense AI</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align:center;color:#555'>Cardiovascular Risk Prediction — Powered by ML & Deep Learning</h3>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="stat-box"><h2>4</h2><p>ML Models Trained</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="stat-box"><h2>2</h2><p>Prediction Modes</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="stat-box"><h2>⚡</h2><p>Dual-Modality Fusion</p></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        <div class="card">
            <h3>🧪 Clinical Features Mode</h3>
            <p>Enter patient vitals and clinical measurements such as age, cholesterol,
            blood pressure, ECG readings and more. Our tuned Random Forest model
            analyses these to estimate cardiovascular risk.</p>
        </div>
        """, unsafe_allow_html=True)
    with col_b:
        st.markdown("""
        <div class="card">
            <h3>🫀 ECG Image Mode</h3>
            <p>Upload an ECG image and our VGG16 deep learning model will classify it
            into one of four categories — Normal, Abnormal Heartbeat,
            History of MI, or Myocardial Infarction — and compute a risk score.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <h3>🔗 Combined Mode</h3>
        <p>Use both methods together for a more comprehensive risk assessment.
        The system fuses the two scores using a weighted average
        (60% clinical, 40% ECG) to produce a final cardiovascular risk percentage.</p>
    </div>
    """, unsafe_allow_html=True)

    st.warning("⚠️ This tool is for educational and research purposes only. It does not replace professional medical diagnosis or clinical judgement.")


# ══════════════════════════════════════════════════════════════
# PAGE 2 — PREDICT
# ══════════════════════════════════════════════════════════════
elif page == "🔬 Predict":
    st.title("🔬 Cardiovascular Risk Prediction")
    st.markdown("Fill in one or both sections below, then click **Run Prediction**.")
    st.markdown("---")

    preprocessor, rf_model = load_clinical_models()
    ecg_model               = load_ecg_model()
    class_indices           = load_class_labels()

    clinical_risk = None
    ecg_risk      = None
    ecg_class     = None

    col_left, col_right = st.columns(2)

    # ── LEFT: Clinical Input ──────────────────────────────────
    with col_left:
        st.markdown("### 🧪 Clinical Features")
        st.markdown('<div class="card">', unsafe_allow_html=True)

        use_clinical = st.checkbox("Use Clinical Features", value=True)

        if use_clinical:
            age      = st.number_input("Age (years)", min_value=20, max_value=100, value=50)
            sex_inp  = st.selectbox("Sex", ["Male", "Female"])
            sex      = 1 if sex_inp == "Male" else 0

            cp_inp   = st.selectbox("Chest Pain Type",
                                    ["Typical Angina", "Atypical Angina",
                                     "Non-Anginal Pain", "Asymptomatic"])
            cp_map   = {"Typical Angina": 1, "Atypical Angina": 2,
                        "Non-Anginal Pain": 3, "Asymptomatic": 4}
            cp       = cp_map[cp_inp]

            rbp      = st.slider("Resting Blood Pressure (mm Hg)", 80, 200, 120)
            chol     = st.slider("Cholesterol (mg/dl)", 100, 600, 200)

            fbs_inp  = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
            fbs      = 1 if fbs_inp == "Yes" else 0

            recg_inp = st.selectbox("Resting ECG",
                                    ["Normal", "ST-T Wave Abnormality",
                                     "Left Ventricular Hypertrophy"])
            recg_map = {"Normal": 0, "ST-T Wave Abnormality": 1,
                        "Left Ventricular Hypertrophy": 2}
            recg     = recg_map[recg_inp]

            mhr      = st.slider("Max Heart Rate Achieved", 60, 220, 150)

            ea_inp   = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
            ea       = 1 if ea_inp == "Yes" else 0

            oldpeak  = st.number_input("Oldpeak (ST Depression)", 0.0, 6.0, 1.0, step=0.1)

            slope_inp = st.selectbox("ST Slope",
                                     ["Upsloping", "Flat", "Downsloping"])
            slope_map = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}
            slope     = slope_map[slope_inp]

        st.markdown('</div>', unsafe_allow_html=True)

    # ── RIGHT: ECG Image Input ────────────────────────────────
    with col_right:
        st.markdown("### 🫀 ECG Image")
        st.markdown('<div class="card">', unsafe_allow_html=True)

        use_ecg       = st.checkbox("Use ECG Image", value=False)
        uploaded_file = None

        if use_ecg:
            uploaded_file = st.file_uploader(
                "Upload ECG Image (JPG/PNG)", type=["jpg", "jpeg", "png"]
            )
            if uploaded_file:
                st.image(uploaded_file, caption="Uploaded ECG", use_column_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    run = st.button("🚀 Run Prediction", use_container_width=True)

    if run:
        if not use_clinical and not use_ecg:
            st.warning("Please enable at least one prediction mode.")
        else:
            # ── Clinical Prediction ───────────────────────────
            if use_clinical and preprocessor and rf_model:
                input_df = pd.DataFrame([{
                    'age': age,
                    'sex': sex,
                    'chest pain type': cp,
                    'fasting blood sugar': fbs,
                    'resting ecg': recg,
                    'exercise angina': ea,
                    'ST slope': slope,
                    'resting bp s': rbp,
                    'cholesterol': chol,
                    'max heart rate': mhr,
                    'oldpeak': oldpeak
                }])
                try:
                    processed    = preprocessor.transform(input_df)
                    clinical_risk = rf_model.predict_proba(processed)[0][1]
                except Exception as e:
                    st.error(f"Clinical prediction error: {e}")

            # ── ECG Prediction ────────────────────────────────
            if use_ecg and uploaded_file and ecg_model and class_indices:
                try:
                    img       = Image.open(uploaded_file).convert('RGB')
                    img       = img.resize((100, 100))
                    img_array = np.array(img) / 255.0
                    img_array = np.expand_dims(img_array, axis=0)

                    ecg_probs = ecg_model.predict(img_array)[0]
                    class_names_ordered = sorted(
                        class_indices.keys(), key=lambda x: class_indices[x]
                    )
                    ecg_risk  = sum(
                        ecg_probs[i] * RISK_MAP[class_names_ordered[i]]
                        for i in range(4)
                    )
                    ecg_class = DISPLAY_NAMES[class_names_ordered[np.argmax(ecg_probs)]]
                    ecg_conf  = float(np.max(ecg_probs)) * 100
                except Exception as e:
                    st.error(f"ECG prediction error: {e}")

            # ── Final Risk Score ──────────────────────────────
            if clinical_risk is not None and ecg_risk is not None:
                final_risk = 0.60 * clinical_risk + 0.40 * ecg_risk
                mode_label = "Combined (Clinical + ECG)"
            elif clinical_risk is not None:
                final_risk = clinical_risk
                mode_label = "Clinical Features Only"
            else:
                final_risk = ecg_risk
                mode_label = "ECG Image Only"

            final_pct = final_risk * 100

            # ── Risk Category ─────────────────────────────────
            if final_pct <= 30:
                risk_class = "risk-low"
                risk_msg   = "🟢 LOW RISK — Your indicators suggest a low probability of cardiovascular disease."
            elif final_pct <= 60:
                risk_class = "risk-mod"
                risk_msg   = "🟡 MODERATE RISK — Some indicators warrant attention. Please consult a physician."
            else:
                risk_class = "risk-high"
                risk_msg   = "🔴 HIGH RISK — Multiple indicators suggest elevated cardiovascular risk. Seek medical advice immediately."

            st.markdown("## 📊 Results")

            # Gauge
            fig = go.Figure(go.Indicator(
                mode  = "gauge+number",
                value = final_pct,
                number = {"suffix": "%", "font": {"size": 40}},
                title  = {"text": f"Cardiovascular Risk Score<br><sub>{mode_label}</sub>"},
                gauge  = {
                    "axis": {"range": [0, 100]},
                    "bar":  {"color": "#1a3c5e"},
                    "steps": [
                        {"range": [0,  30], "color": "#d4edda"},
                        {"range": [30, 60], "color": "#fff3cd"},
                        {"range": [60,100], "color": "#f8d7da"},
                    ],
                    "threshold": {
                        "line":  {"color": "red", "width": 4},
                        "thickness": 0.75,
                        "value": final_pct
                    }
                }
            ))
            fig.update_layout(height=350, margin=dict(t=80, b=20))
            st.plotly_chart(fig, use_container_width=True)

            # Risk banner
            st.markdown(f'<div class="{risk_class}">{risk_msg}</div>',
                        unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

            # ECG class result
            if ecg_class:
                st.info(f"🫀 ECG Classification: **{ecg_class}** ({ecg_conf:.1f}% confidence)")

            # Score breakdown table
            if clinical_risk is not None and ecg_risk is not None:
                st.markdown("### 📋 Score Breakdown")
                breakdown = pd.DataFrame({
                    "Source":     ["Clinical Features", "ECG Image", "Combined (Weighted)"],
                    "Risk Score": [f"{clinical_risk*100:.1f}%",
                                   f"{ecg_risk*100:.1f}%",
                                   f"{final_pct:.1f}%"]
                })
                st.table(breakdown)

            # Feature importance chart
            if rf_model and preprocessor:
                try:
                    st.markdown("### 🔍 Top Clinical Feature Contributions")
                    numerical_cols = ['age', 'resting bp s', 'cholesterol',
                                      'max heart rate', 'oldpeak']
                    cat_feature_names = (preprocessor
                                         .named_transformers_['cat_pipeline']
                                         .get_feature_names_out(
                                             ['sex','chest pain type',
                                              'fasting blood sugar','resting ecg',
                                              'exercise angina','ST slope']))
                    all_features = numerical_cols + list(cat_feature_names)
                    importances  = rf_model.feature_importances_

                    feat_df = pd.DataFrame({
                        "Feature":    all_features,
                        "Importance": importances
                    }).sort_values("Importance", ascending=False).head(8)

                    fig2 = go.Figure(go.Bar(
                        x=feat_df["Importance"],
                        y=feat_df["Feature"],
                        orientation='h',
                        marker_color='#1a3c5e'
                    ))
                    fig2.update_layout(
                        title="Top 8 Most Influential Features",
                        xaxis_title="Importance Score",
                        yaxis={"autorange": "reversed"},
                        height=350,
                        margin=dict(l=20, r=20, t=50, b=20)
                    )
                    st.plotly_chart(fig2, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not render feature chart: {e}")

            st.warning("⚠️ This result is for educational purposes only and does not constitute medical advice.")


# ══════════════════════════════════════════════════════════════
# PAGE 3 — ABOUT
# ══════════════════════════════════════════════════════════════
elif page == "ℹ️ About":
    st.title("ℹ️ About CardioSense AI")

    st.markdown("""
    <div class="card">
        <h3>🎯 Project Overview</h3>
        <p>CardioSense AI is a dual-modality cardiovascular risk prediction system built as
        a data science research project. It combines classical machine learning on clinical
        tabular data with deep learning on ECG images to provide a comprehensive risk assessment.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="card">
            <h3>🧪 Clinical Model</h3>
            <ul>
                <li>4 models compared: Logistic Regression, Random Forest, SVM, XGBoost</li>
                <li>SMOTE applied to handle class imbalance</li>
                <li>RandomizedSearchCV hyperparameter tuning</li>
                <li>SHAP explainability analysis</li>
                <li>Best model: Tuned Random Forest with SMOTE</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="card">
            <h3>🫀 ECG Image Model</h3>
            <ul>
                <li>VGG16 pretrained on ImageNet (transfer learning)</li>
                <li>Fine-tuned last 10 layers on ECG dataset</li>
                <li>4 classes: Normal, Abnormal Heartbeat, History of MI, Myocardial Infarction</li>
                <li>Input size: 100×100 RGB</li>
                <li>Augmentation: rotation, zoom, flips, shifts</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <h3>📋 Clinical Features Used</h3>
        <table width="100%">
            <tr><th>Feature</th><th>Medical Significance</th></tr>
            <tr><td>Age</td><td>Risk increases significantly with age</td></tr>
            <tr><td>ST Slope</td><td>Flat/downsloping ST segment indicates ischemia</td></tr>
            <tr><td>Oldpeak</td><td>ST depression — key stress test indicator</td></tr>
            <tr><td>Max Heart Rate</td><td>Lower values may indicate poor cardiac reserve</td></tr>
            <tr><td>Exercise Angina</td><td>Classic symptom of coronary artery disease</td></tr>
            <tr><td>Chest Pain Type</td><td>Asymptomatic type is particularly dangerous</td></tr>
            <tr><td>Cholesterol</td><td>High LDL contributes to atherosclerosis</td></tr>
            <tr><td>Resting BP</td><td>Hypertension is a major modifiable risk factor</td></tr>
        </table>
    </div>
    """, unsafe_allow_html=True)

    st.error("⚠️ Disclaimer: CardioSense AI is built for educational and research purposes only. It is not a certified medical device and should never be used as a substitute for professional clinical diagnosis.")
```

---
