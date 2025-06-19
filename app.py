import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

st.set_page_config(page_title="Customer Churn Prediction", layout="centered")
st.title('🔍 Customer Churn Prediction')

# --- Load model and preprocessing files with error handling ---

try:
    model = tf.keras.models.load_model('my_model.h5')
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

try:
    with open('label_gender.pkl', 'rb') as file:
        label_gender = pickle.load(file)
except Exception as e:
    st.error(f"❌ Error loading gender label encoder: {e}")
    st.stop()

try:
    with open('onehot_code_geo.pkl', 'rb') as file:
        onehot_code_geo = pickle.load(file)
except Exception as e:
    st.error(f"❌ Error loading geography encoder: {e}")
    st.stop()

try:
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
except Exception as e:
    st.error(f"❌ Error loading scaler: {e}")
    st.stop()

# --- UI Inputs ---
st.header("📋 Input Customer Information")

geography = st.selectbox('🌍 Geography', onehot_code_geo.categories_[0])
gender = st.selectbox('👤 Gender', label_gender.classes_)
age = st.slider('🎂 Age', 18, 92, 30)
credit_score = st.number_input('💳 Credit Score', value=600)
balance = st.number_input('💰 Balance', value=50000.0)
estimated_salary = st.number_input('💼 Estimated Salary', value=50000.0)
tenure = st.slider('📈 Tenure (Years)', 0, 10, 3)
num_of_products = st.slider('🛍️ Number of Products', 1, 4, 1)
has_cr_card = st.selectbox('💳 Has Credit Card?', [0, 1])
is_active_member = st.selectbox('✅ Is Active Member?', [0, 1])

# --- Preprocessing ---
try:
    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Gender': [label_gender.transform([gender])[0]],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_cr_card],
        'IsActiveMember': [is_active_member],
        'EstimatedSalary': [estimated_salary]
    })

    geo_encoded = onehot_code_geo.transform([[geography]]).toarray()
    geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_code_geo.get_feature_names_out(['Geography']))

    input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)
    input_data_scaled = scaler.transform(input_data)

except Exception as e:
    st.error(f"❌ Error during preprocessing: {e}")
    st.stop()

# --- Prediction ---
try:
    prediction = model.predict(input_data_scaled)
    prediction_proba = prediction[0][0]

    st.subheader(f'📊 Churn Probability: `{prediction_proba:.2f}`')
    if prediction_proba > 0.5:
        st.warning('⚠️ The customer is likely to **churn**.')
    else:
        st.success('✅ The customer is **not likely** to churn.')

except Exception as e:
    st.error(f"❌ Error during prediction: {e}")
