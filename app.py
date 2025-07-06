import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from io import StringIO
import plotly.graph_objects as go


with open("diabetes_model.pkl", "rb") as f:
    model = pickle.load(f)


# Set Streamlit config
st.set_page_config(page_title="Diabetes Prediction App", layout="centered")
st.markdown("""
    <style>
        .title-style {
            text-align: center;
            font-size: 2.5rem;
            color: #FFA500;
        }
        .subtitle-style {
            font-size: 1.3rem;
            color: #D3D3D3;
            margin-top: 20px;
        }
        .result-box {
            background-color: #1a1a1a;
            padding: 20px;
            border-radius: 10px;
            border: 1px solid #444;
            font-size: 1.1rem;
            color: #ffffff;
        }
        .stButton button, .stDownloadButton button {
            background-color: #FFA500;
            color: black;
            font-weight: bold;
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 8px;
        }
        .stDownloadButton button:hover {
            background-color: #e69500;
        }
    </style>
""", unsafe_allow_html=True)


st.markdown('<h1 class="title-style">🏥 Diabetes Prediction App</h1>', unsafe_allow_html=True)

# Initialize session state
if 'submitted' not in st.session_state:
    st.session_state.submitted = False
if 'edited' not in st.session_state:
    st.session_state.edited = False
if 'input_data' not in st.session_state:
    st.session_state.input_data = {
        'Pregnancies': 0, 'Glucose': 0, 'BloodPressure': 0, 'SkinThickness': 0,
        'Insulin': 0, 'BMI': 0.0, 'DiabetesPedigreeFunction': 0.0, 'Age': 0
    }

def clear_edit_mode():
    st.session_state.edited = False

# Page 1: Input form
if not st.session_state.submitted:
    st.markdown("<h3 class='subtitle-style'>🔢 Enter Patient Health Details</h3>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        pregnancies = st.slider("Pregnancies", 0, 17, 0)
        glucose = st.slider("Glucose", 0, 200, 0)
        blood_pressure = st.slider("Blood Pressure", 0, 140, 0)
        skin_thickness = st.slider("Skin Thickness", 0, 100, 0)

    with col2:
        insulin = st.slider("Insulin", 0, 850, 0)
        bmi = st.slider("BMI", 0.0, 67.1, 0.0)
        dpf = st.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.0)
        age = st.slider("Age", 10, 100, 0)

    if st.button("Submit"):
        st.session_state.input_data = {
            'Pregnancies': pregnancies,
            'Glucose': glucose,
            'BloodPressure': blood_pressure,
            'SkinThickness': skin_thickness,
            'Insulin': insulin,
            'BMI': bmi,
            'DiabetesPedigreeFunction': dpf,
            'Age': age
        }
        st.session_state.submitted = True

# Page 2: Show result
else:
    input_df = pd.DataFrame([st.session_state.input_data])
    prediction = model.predict(input_df)[0]
    prediction_proba = model.predict_proba(input_df)[0]

    result = "🟢 Not Diabetic" if prediction == 0 else "🔴 Diabetic"

    st.markdown("<h3 class='subtitle-style'>📊 Prediction Result</h3>", unsafe_allow_html=True)
    st.markdown(f"""
        <div class='result-box'>
            <strong>Result:</strong> {result}<br><br>
            <strong>Probability:</strong><br>
            - Not Diabetic: {prediction_proba[0]*100:.2f}%<br>
            - Diabetic: {prediction_proba[1]*100:.2f}%<br>
        </div>
    """, unsafe_allow_html=True)

    # Show bar chart for probabilities
    st.markdown("<h3 class='subtitle-style'>📈 Probability Chart</h3>", unsafe_allow_html=True)
    prob_df = pd.DataFrame({
        "Outcome": ["Not Diabetic", "Diabetic"],
        "Probability": prediction_proba
    })
    st.bar_chart(prob_df.set_index("Outcome"))

    st.markdown("<h3 class='subtitle-style'>📥 Download Result</h3>", unsafe_allow_html=True)
    csv_output = input_df.copy()
    csv_output['Prediction'] = result
    csv = csv_output.to_csv(index=False)
    st.download_button("Download CSV", csv, "diabetes_prediction.csv", "text/csv")

    st.markdown("<h3 class='subtitle-style'>✏️ Edit Inputs</h3>", unsafe_allow_html=True)
    if st.button("Edit"):
        st.session_state.edited = True

    if st.session_state.edited:
        with st.sidebar:
            st.markdown("### 🔄 Update Health Details")
            pregnancies = st.slider("Pregnancies", 0, 17, st.session_state.input_data['Pregnancies'], key='edit_pregnancies')
            glucose = st.slider("Glucose", 0, 200, st.session_state.input_data['Glucose'], key='edit_glucose')
            blood_pressure = st.slider("Blood Pressure", 0, 140, st.session_state.input_data['BloodPressure'], key='edit_bp')
            skin_thickness = st.slider("Skin Thickness", 0, 100, st.session_state.input_data['SkinThickness'], key='edit_skin')
            insulin = st.slider("Insulin", 0, 850, st.session_state.input_data['Insulin'], key='edit_insulin')
            bmi = st.slider("BMI", 0.0, 67.1, st.session_state.input_data['BMI'], key='edit_bmi')
            dpf = st.slider("Diabetes Pedigree Function", 0.0, 2.5, st.session_state.input_data['DiabetesPedigreeFunction'], key='edit_dpf')
            age = st.slider("Age", 10, 100, st.session_state.input_data['Age'], key='edit_age')

            if st.button("Update Prediction"):
                st.session_state.input_data = {
                    'Pregnancies': pregnancies,
                    'Glucose': glucose,
                    'BloodPressure': blood_pressure,
                    'SkinThickness': skin_thickness,
                    'Insulin': insulin,
                    'BMI': bmi,
                    'DiabetesPedigreeFunction': dpf,
                    'Age': age
                }
                clear_edit_mode()
                st.rerun()
