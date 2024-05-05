import streamlit as st
import joblib
import numpy as np
import pandas as pd
# from utils import PrepProcesor, columns
model = joblib.load("titanic_model.pkl")

st.title('Did they survive')
# pid = st.text_input('Passenger id')
pcls = st.select_slider('Choose passenger class',[1,2,3])
# name = st.text_input('Input passengers name')
# gender = st.select_slider('Select gender',['male','female'])
age = st.slider('Input Age',0,100)
sib = st.slider('Input siblings',0,10)
parch = st.slider('Input parents/children',0,2)
# ticketid = st.number_input('Ticket number',0,100)
fare = st.number_input('Fare amount',0,100)
# cabin = st.text_input('Enter cabin','C52')
# embarked = st.selectbox('Choose embarkation point',['S','C','Q'])

# column_names = ['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
column_names = ['Pclass','Age', 'SibSp', 'Parch', 'Fare']

def predict():
    row =  np.array([pcls,age,sib,parch,fare])
    X = pd.DataFrame([row],columns = column_names)
    prediction = model.predict(X)[0]
    if prediction ==1:
        st.success('Passenger Survived')
    else:
        st.success('Passenger didnot survive')

st.button('Predict',on_click=predict)