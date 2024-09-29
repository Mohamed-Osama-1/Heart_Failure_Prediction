import streamlit as st
import numpy as np
import pickle as pkl


# Load label encoders
le_ST_Slope = pkl.load(open('Deployment_Streamlit\Encoding\ST_Slope_le.pkl','rb'))#C:\My_Files\osama\Deployment_Streamlit\Encoding\ST_Slope_le.pkl
le_ChestPainType = pkl.load(open('Deployment_Streamlit\Encoding/ChestPainType_le.pkl','rb'))
le_RestingECG = pkl.load(open('Deployment_Streamlit\Encoding/RestingECG_le.pkl','rb'))

# Load the scaler and models
scaler = pkl.load(open('Deployment_Streamlit\Scaling\scaler.pkl', 'rb'))#C:\My_Files\osama\Deployment_Streamlit\Scaling\scaler.pkl
models = {
    'Decision Tree': pkl.load(open('Deployment_Streamlit\Models\Decision_Tree.pkl', 'rb')),#C:\My_Files\osama\Deployment_Streamlit\Models\Decision_Tree.pkl
    'Logistic Regression': pkl.load(open('Deployment_Streamlit\Models\Logistic_Regression.pkl', 'rb')),
    'SVC':pkl.load(open('Deployment_Streamlit\Models\svc.pkl','rb')),
    'KNN': pkl.load(open('Deployment_Streamlit\Models\KNN.pkl', 'rb')),
    'Naive Bayes': pkl.load(open('Deployment_Streamlit//Models//Naive_bayes.pkl', 'rb')),#C:\My_Files\osama\Deployment_Streamlit\Models\Naive_bayes.pkl
    'Random Forest': pkl.load(open('Deployment_Streamlit\Models\Random_forest.pkl', 'rb')),
    'Gradient Boosting': pkl.load(open('Deployment_Streamlit\Models\Gradient_boosting.pkl', 'rb')),
    'XGBoost' : pkl.load(open('Deployment_Streamlit//Models//xgboost.pkl','rb')),
    'AdaBoost': pkl.load(open('Deployment_Streamlit\Models\AdaBoost.pkl', 'rb')),
    'Stacking' : pkl.load(open('Deployment_Streamlit\Models\Stacking.pkl','rb'))
}
#photo
page_bg_img="""
<style>
[data-testid="stAppViewContainer"]{

background-image:  url("https://c4.wallpaperflare.com/wallpaper/151/793/264/heartbeat-hd-wallpaper-preview.jpg");
background-image: cover;
</style>

"""
st.markdown(page_bg_img,unsafe_allow_html=True)



# Streamlit App Configuration
st.title("Heart Failure detection  ")
st.markdown('---')


# Input Features
c1, c2, c3 = st.columns(3)

with c1:
   age = st.number_input("Age", min_value=1, max_value=120, value=25)
   sex = st.selectbox("Sex", options=['Male', 'Female'])
   cp_type = st.selectbox("Chest Pain Type", options=le_ChestPainType.classes_)
   resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", min_value=50, max_value=200, value=120)
with c2:
   cholesterol = st.number_input("Serum Cholesterol (mg/dL)", min_value=100, max_value=600, value=200)
   fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", options=['No', 'Yes'])
   resting_ecg = st.selectbox("Resting ECG Results", options=le_RestingECG.classes_)
   max_hr = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=130)
   
with c3:
   exercise_angina = st.selectbox("Exercise-Angina", options=['No', 'Yes'])
   oldpeak = st.number_input("Oldpeak", min_value=0.0, max_value=10.0, value=1.0)
   st_slope = st.selectbox("Slope of the Peak Exercise ST Segment", options=le_ST_Slope.classes_)

# Encode categorical features
sex_encoded = 1 if sex == 'Male' else 0
cp_type_encoded = le_ChestPainType.transform([cp_type])[0]
fasting_bs_encoded = 1 if fasting_bs == 'Yes' else 0
resting_ecg_encoded = le_RestingECG.transform([resting_ecg])[0]
exercise_angina_encoded = 1 if exercise_angina == 'Yes' else 0
st_slope_encoded = le_ST_Slope.transform([st_slope])[0]

# Create feature array
features = np.array([[age, sex_encoded, cp_type_encoded, resting_bp, cholesterol, fasting_bs_encoded,
                      resting_ecg_encoded, max_hr, exercise_angina_encoded, oldpeak, st_slope_encoded]])

# Scale the features
scaled_features = scaler.transform(features)

# Select model for prediction
model_choice = st.selectbox("Choose Model", options=list(models.keys()))

# Predict and display the result
if st.button("Predict"):
    model = models[model_choice]
    prediction = model.predict(scaled_features)
    result = "[Positive] 💔" if prediction[0] == 0 else " [Negative] ❤️"
    st.markdown('---')
    st.subheader(f"Prediction: {result}")
    