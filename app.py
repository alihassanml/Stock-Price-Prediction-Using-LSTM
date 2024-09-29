import streamlit as st
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Import TensorFlow and other necessary libraries
import tensorflow as tf
import pickle

# Streamlit App Title and Caption
st.title('Bitcoin Price Prediction')
st.caption('Using LSTM to Enhance BTC Price')

# Load Data and Model
df = pd.read_csv('BTC-Data.csv')  # Load BTC data
model = tf.keras.models.load_model('model.h5')  # Load trained model
scalar_X = pickle.load(open('scalar_X.pkl', 'rb'))  # Load X scaler
scalar_y = pickle.load(open('scalar_y.pkl', 'rb'))  # Load y scaler

# Prediction function
def predict(Open, High, Low, Volume):
    # Prepare the input data
    data = np.array([[Open, High, Low, Volume]])
    
    # Scale the input data
    train = scalar_X.transform(data)
    
    # Predict using the trained model
    y_pred = model.predict(train)
    
    # If needed, reshape y_pred (depending on the model output shape)
    y_pred = y_pred.reshape(-1, 1)
    
    # Inverse transform the prediction to original scale
    y_pred_original = scalar_y.inverse_transform(y_pred)
    
    # Display the prediction
    st.success(f'Model Predicted Close Price: {y_pred_original[0][0]:.2f}')

# Streamlit form for user input
with st.form("my_form"):
    col1, col2 = st.columns(2)  # Create two columns for inputs
    
    # Input fields for Open, High, Low, and Volume values
    Open = col1.number_input("Enter Open Value", format="%.8f")
    High = col2.number_input("Enter High Value", format="%.8f")
    Low = col1.number_input("Enter Low Value", format="%.8f")
    Volume = col2.number_input("Enter Volume Value", format="%.0f")
    
    # When the form is submitted
    if st.form_submit_button("Submit"):
        st.write('Done')
        predict(Open, High, Low, Volume)
