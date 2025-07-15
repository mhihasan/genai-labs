import pickle

import pandas as pd
import streamlit as st
import tensorflow as tf
from tensorflow import keras


# Load the trained moddel
model = keras.models.load_model('model.h5')

# Load the encoders
with open("onehot_encoder_geo.pkl", "rb") as f:
    onehot_encoder_geo = pickle.load(f)

with open("label_encoder_gender.pkl", "rb") as f:
    label_encoder_gender = pickle.load(f)

# Load the scaler
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)


## streamlit app
st.title("Customer Churn Prediction")
st.write("This app predicts whether a customer will churn based on their demographics and usage patterns.")

# Input fields
geography = st.selectbox("Geography", onehot_encoder_geo.categories_[0])
gender = st.selectbox("Gender", label_encoder_gender.classes_)
age = st.slider("Age", 18, 100)
balance = st.number_input("Balance")
credit_score = st.number_input("Credit Score")
estimated_salary = st.number_input("Estimated Salary")
tenure = st.slider("Tenure (months)", 0, 10)
number_of_products = st.slider("Number of Products", 1, 4)
has_cr_card = st.selectbox("Has Credit Card", [0, 1])
is_active_member = st.selectbox("Is Active Member", [0, 1])

# Prepare the input data
input_data = {
    "CreditScore": credit_score,
    "Gender": label_encoder_gender.transform([gender])[0],
    "Age": age,
    "Tenure": tenure,
    "Balance": balance,
    "NumOfProducts": number_of_products,
    "HasCrCard": has_cr_card,
    "IsActiveMember": is_active_member,
    "EstimatedSalary": estimated_salary

}
input_df = pd.DataFrame([input_data])
# Encode one-hot encoded features
geography_encoded = onehot_encoder_geo.transform([[geography]])
geography_encoded_df = pd.DataFrame(geography_encoded, columns=onehot_encoder_geo.get_feature_names_out(["Geography"]))

# Combine the encoded geography with the rest of the input data
input_df = pd.concat([input_df, geography_encoded_df], axis=1)

# Scale the input data
input_df_scaled = scaler.transform(input_df)

# Make prediction
prediction = model.predict(input_df_scaled)

st.write(f"Churn Probability: {prediction[0][0]:.2f}")
# Display the prediction
if prediction[0][0] > 0.5:
    st.write("The customer is likely to churn.")
else:
    st.write("The customer is likely to stay.")

