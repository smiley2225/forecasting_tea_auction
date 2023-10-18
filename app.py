import streamlit as st
from model_module import Ml_Model
# Import your machine learning model and any necessary libraries
# Replace 'your_model.pkl' with your model file and import it accordingly
st.set_page_config(page_title='Stratlytics',page_icon=":bar_chart", layout='wide')
# Load your trained machine learning model
import joblib
# model = joblib.load('your_model.pkl')

# Create the Streamlit app
st.title(' :bar_chart: Tea Auction Forecasting Using ML Model')
st.sidebar.header('Input Parameters')

# Add input elements (e.g., sliders, text inputs) to collect user input
Tea_type = st.sidebar.selectbox('Tea_type', options= ['LEAF_AND_ALL_DUST', 'Other(testing)'])
location = st.sidebar.selectbox('location', options=['Kolkata','Other location(testing)'])

# Create a prediction function that uses the model and user inputs
def predict(location,Tea_type):
    st.write('Test Data Predictions...')
    model = Ml_Model(loc=location,tea_type=Tea_type)
    prediction = model.models()
    # Perform any necessary data preprocessing or feature engineering
    # Input your model prediction logic here
    # prediction = model.predict([[parameter1, parameter2]])[0]
    return prediction

# Display the prediction
if st.sidebar.button('Predict'):
    prediction = predict( location,Tea_type)

    st.write('Prediction:', prediction)