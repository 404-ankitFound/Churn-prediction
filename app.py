import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle
input_data = {}

st.title('Customer Churn Prediction')
input_data['Age']=st.number_input('Age',min_value=18,max_value=100,value=25)
input_data['CreditScore']=st.number_input('Credit Score',min_value=300,max_value=850,value=319)
input_data['Geography']=st.selectbox('Geography', ['France', 'Spain', 'Germany'])
input_data['Gender']=st.selectbox('Gender', ['Male', 'Female'])
input_data['Tenure']=st.number_input('Tenure',min_value=0,max_value=10,value=2)
input_data['Balance']=st.number_input('Balance',min_value=0.0,max_value=250000.0,value=0.0)
input_data['NumOfProducts']=st.number_input('Number of Products',min_value=0,max_value=4,value=0)
input_data['HasCrCard']=st.selectbox('Has Credit Card', [0, 1])
input_data['IsActiveMember']=st.selectbox('Is Active Member', [0, 1])
input_data['EstimatedSalary']=st.number_input('Estimated Salary',min_value=0.0,max_value=250000.0,value=101348.88)

# load trained model, scaller pickel and onehot encoded
model = load_model('model3.h5')

with open('encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('gender_encoder.pkl', 'rb') as f:
    gender_encoder = pickle.load(f)

geo_enc=encoder.transform([[input_data['Geography']]])
geo_enc_df=pd.DataFrame(geo_enc, columns=encoder.get_feature_names_out(['Geography']))

input_df=pd.DataFrame([input_data])

# Drop original Geography column
input_df = input_df.drop('Geography', axis=1)

# Add encoded columns
input_df = pd.concat([input_df, geo_enc_df], axis=1)

if input_df['Gender'].iloc[0] == 'Male':
    input_df['Gender'] = 1                      
else:
    input_df['Gender'] = 0

# scaling the input data
num_col=['CreditScore', 'Age', 'Balance', 'Tenure',  'NumOfProducts', 'EstimatedSalary']
input_df[num_col] = scaler.transform(input_df[num_col])


pred=model.predict(input_df)
st.write("probability of Exit bank :",pred[0][0])
if pred[0][0] > 0.5:
    st.error("Customer is likely to Exit the bank.")  
else:
    st.success("Customer is not likely to Exit the bank.")
