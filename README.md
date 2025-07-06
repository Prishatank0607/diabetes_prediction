# Diabetes Prediction Web App

An interactive and user-friendly web application built with **Streamlit** that predicts whether a person is diabetic based on health metrics using a **Random Forest Classifier**. The app allows users to input data, receive predictions, visualize the model output, and download results.

---

## About the Project

This project aims to make machine learning models accessible and interpretable by deploying them in an interactive frontend. It is designed as a two-page flow:
- Page 1: User inputs health data using sliders
- Page 2: Displays prediction results with probability breakdown, visualizations, and editing options

---

## Dataset Overview

The model is trained on the **Pima Indians Diabetes Database**, which contains diagnostic measurements of female patients.

- **Total Samples:** 768  
- **Features:**
  - Pregnancies
  - Glucose
  - BloodPressure
  - SkinThickness
  - Insulin
  - BMI
  - DiabetesPedigreeFunction
  - Age  
- **Target:** Binary outcome (1: Diabetic, 0: Not Diabetic)

Dataset Source: [Kaggle - Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)

---

## Features Implemented

- Input patient health data using sliders
- Predict diabetes using Random Forest Classifier
- Show prediction result and class probability
- Beautiful and responsive dark-themed UI with orange accents
- Interactive **bar chart**
- Option to edit inputs via sidebar and rerun prediction
- Download results in `.csv` format with all inputs and output

---

## Tools & Technologies Used

- **Python 3.11+**
- **Streamlit** – Web app framework
- **scikit-learn** – ML modeling
- **Pandas / NumPy** – Data processing
- **Plotly** – Interactive visualizations
- **Pickle** – Model serialization

---

## How to Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/Prishatank0607/diabetes-prediction.git
cd diabetes-prediction

# 2. (Optional) Create and activate virtual environment
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# 3. Install all dependencies
pip install -r requirements.txt

# 4. Run the Streamlit app
streamlit run app.py
```

---

## Live Deployment

The app is deployed using **Streamlit Cloud** and is accessible at:

🔗 [Live App Link](https://your-streamlit-cloud-link)


