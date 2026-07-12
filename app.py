import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

# Page configuration
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .risk-high {
        background-color: #ffebee;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #d32f2f;
    }
    .risk-low {
        background-color: #e8f5e9;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #388e3c;
    }
    .info-box {
        background-color: #e3f2fd;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Cache model loading for performance
@st.cache_resource
def load_model_and_scaler():
    try:
        model = joblib.load("KNN_heart.pkl")
        scaler = joblib.load("scaler.pkl")
        expected_columns = joblib.load("columns.pkl")
        return model, scaler, expected_columns
    except FileNotFoundError as e:
        st.error(f"Error loading model files: {e}")
        return None, None, None

# Function to calculate risk level and interpretation
def get_risk_interpretation(prediction, confidence=None):
    if prediction == 1:
        risk_level = "High Risk"
        emoji = "⚠️"
        color = "risk-high"
        interpretation = """
        Based on your health metrics, you have indicators associated with higher heart disease risk.
        **Immediate actions:**
        - Consult with a cardiologist for a professional evaluation
        - Schedule a comprehensive cardiac assessment
        - Discuss lifestyle modifications with your healthcare provider
        - Monitor your symptoms closely
        """
    else:
        risk_level = "Low Risk"
        emoji = "✅"
        color = "risk-low"
        interpretation = """
        Your current health metrics suggest a lower risk of heart disease.
        **Recommendations:**
        - Maintain regular exercise (150+ minutes/week)
        - Continue a heart-healthy diet (Mediterranean diet recommended)
        - Monitor your blood pressure and cholesterol regularly
        - Schedule annual checkups with your healthcare provider
        """
    
    return risk_level, emoji, color, interpretation

# Main title
st.title("❤️ Heart Disease Risk Prediction")
st.markdown("---")

# Sidebar for information
with st.sidebar:
    st.header("About This Tool")
    st.info(
        """
        This predictor uses a **K-Nearest Neighbors (KNN)** machine learning model 
        trained on cardiac health data to assess heart disease risk.
        
        **Disclaimer:** This tool is for informational purposes only and should NOT 
        replace professional medical advice. Always consult a cardiologist for diagnosis.
        """
    )
    
    st.header("Understanding the Metrics")
    with st.expander("📚 Learn about each metric"):
        st.markdown("""
        - **Age:** Your current age in years
        - **Sex:** Biological sex (M/F)
        - **Chest Pain Type:** 
          - ASY = Asymptomatic
          - ATA = Atypical Angina
          - NAP = Non-anginal Pain
          - TA = Typical Angina
        - **Resting BP:** Blood pressure at rest (normal: < 120)
        - **Cholesterol:** Total cholesterol level (optimal: < 200)
        - **Fasting BS:** Blood sugar level after fasting
        - **Max Heart Rate:** Maximum HR achieved during exercise
        - **Oldpeak:** ST depression induced by exercise
        - **ST Slope:** Slope of ST segment (Up/Flat/Down)
        """)

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Input Your Health Metrics")
    
    # Organize inputs in columns for better UX
    input_col1, input_col2 = st.columns(2)
    
    with input_col1:
        st.markdown("**Basic Information**")
        age = st.slider("Age", 18, 100, 40, help="Your current age")
        sex = st.selectbox("Sex", ["M", "F"], help="Biological sex")
        
        st.markdown("**Cardiovascular Measurements**")
        resting_bp = st.number_input(
            "Resting BP (mm Hg)", 
            60, 200, 120, 
            help="Blood pressure when at rest (Normal: <120)"
        )
        cholesterol = st.number_input(
            "Cholesterol (mg/dL)", 
            0, 600, 200,
            help="Total cholesterol (Optimal: <200)"
        )
        max_hr = st.slider(
            "Max Heart Rate", 
            40, 220, 150,
            help="Maximum heart rate achieved during exercise"
        )
    
    with input_col2:
        st.markdown("**Cardiac Indicators**")
        chest_pain = st.selectbox(
            "Chest Pain Type", 
            ["ASY", "ATA", "NAP", "TA"],
            help="Type of chest pain experienced"
        )
        resting_ecg = st.selectbox(
            "Resting ECG", 
            ["Normal", "ST", "LVH"],
            help="Resting electrocardiogram results"
        )
        st_slope = st.selectbox(
            "ST Slope", 
            ["Up", "Flat", "Down"],
            help="Slope of the ST segment"
        )
        
        st.markdown("**Other Factors**")
        fasting_bs = st.selectbox(
            "Fasting BS > 120 mg/dL", 
            [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No",
            help="Blood sugar level after fasting"
        )
        exercise_angina = st.selectbox(
            "Exercise-Induced Angina", 
            ["N", "Y"],
            help="Chest pain induced by exercise"
        )
        oldpeak = st.slider(
            "Oldpeak (ST Depression)", 
            0.0, 6.0, 1.0, 0.1,
            help="ST depression induced by exercise relative to rest"
        )

with col2:
    st.subheader("Quick Reference")
    st.markdown("""
    **Normal Ranges:**
    
    🩺 **Resting BP:** < 120
    
    🩸 **Cholesterol:** < 200
    
    ❤️ **Max HR:** 60-100+
    
    **Risk Factors:**
    - Age > 65
    - High BP/Cholesterol
    - Chest pain
    - Diabetes (FastingBS)
    """)

st.markdown("---")

# Prediction button
if st.button("🔍 Predict Risk", use_container_width=True, type="primary"):
    
    # Load model
    model, scaler, expected_columns = load_model_and_scaler()
    
    if model is None:
        st.error("Could not load model. Please check your files.")
    else:
        try:
            # Create input dictionary with proper encoding
            raw_input = {
                'Age': age,
                'RestingBP': resting_bp,
                'Cholesterol': cholesterol,
                'FastingBS': fasting_bs,
                'MaxHR': max_hr,
                'Oldpeak': oldpeak,
                'Sex_' + sex: 1,
                'ChestPainType_' + chest_pain: 1,
                'RestingECG_' + resting_ecg: 1,
                'ExerciseAngina_' + exercise_angina: 1,
                'ST_Slope_' + st_slope: 1
            }

            # Create dataframe
            input_df = pd.DataFrame([raw_input])

            # Fill missing columns with 0
            for col in expected_columns:
                if col not in input_df.columns:
                    input_df[col] = 0

            # Ensure correct column order
            input_df = input_df[expected_columns]

            # Scale input
            scaled_input = scaler.transform(input_df)

            # Make prediction
            prediction = model.predict(scaled_input)[0]
            
            # Get prediction probability (distance to nearest neighbors)
            distances, indices = model.kneighbors(scaled_input)
            confidence = 1 - (distances[0].mean() / distances[0].max()) if len(distances[0]) > 0 else 0.5
            confidence = min(confidence, 1.0)

            # Get interpretation
            risk_level, emoji, css_class, interpretation = get_risk_interpretation(prediction)

            # Display results
            st.markdown("---")
            st.subheader(f"{emoji} Prediction Result")
            
            # Risk level with styling
            if prediction == 1:
                st.markdown(f"""
                <div class="risk-high">
                    <h2 style="color: #d32f2f; margin: 0;">⚠️ HIGH RISK</h2>
                    <p style="font-size: 18px; color: #d32f2f;">Your health metrics indicate higher risk</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="risk-low">
                    <h2 style="color: #388e3c; margin: 0;">✅ LOW RISK</h2>
                    <p style="font-size: 18px; color: #388e3c;">Your health metrics indicate lower risk</p>
                </div>
                """, unsafe_allow_html=True)

            # Display interpretation
            st.markdown("""
            <div class="info-box">
            """, unsafe_allow_html=True)
            st.markdown(interpretation)
            st.markdown("</div>", unsafe_allow_html=True)

            # Show metrics summary
            st.subheader("📊 Your Health Metrics Summary")
            metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
            
            with metrics_col1:
                st.metric("Age", f"{age} years")
            with metrics_col2:
                st.metric("Resting BP", f"{resting_bp} mm Hg", 
                         delta="Normal" if resting_bp < 120 else "Elevated", delta_color="off")
            with metrics_col3:
                st.metric("Cholesterol", f"{cholesterol} mg/dL",
                         delta="Normal" if cholesterol < 200 else "High", delta_color="off")
            with metrics_col4:
                st.metric("Max Heart Rate", f"{max_hr} bpm")

            # Warning disclaimer
            st.warning(
                """
                ⚠️ **Important Disclaimer:** 
                This prediction is generated by a machine learning model and should NOT be used for self-diagnosis. 
                It is intended for informational purposes only. Always consult a qualified healthcare professional 
                for proper medical evaluation and diagnosis.
                """
            )

        except Exception as e:
            st.error(f"An error occurred during prediction: {str(e)}")
            st.info("Please check your inputs and try again.")

# Footer
st.markdown("---")
st.markdown("""
<center>
    <p style="color: gray; font-size: 12px;">
    Heart Disease Prediction by Siddharth | Powered by Machine Learning | 
    <a href="https://github.com/Siddharthcdt25/HeartDisease-" target="_blank">GitHub Repository</a>
    </p>
</center>
""", unsafe_allow_html=True)
