import numpy as np
import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier


st.write(''' # Predicción de la temperatura  ''')
st.image("temperaturafoto.jpg", caption="Predice la temperatura en segundos.")

st.header('Datos')

def user_input_features():
  # Entrada
  City = st.number_input('Ciudad ():', min_value=0, max_value=2, value = 0, step = 1)
  month = st.number_input('Mes:',  min_value=1, max_value=12, value = 1, step = 1)
  year = st.number_input('Año:', min_value=0, max_value=5000, value = 0, step = 1)


  user_input_data = {'City': City,
                     'month': month,
                     'year': year,
                     }

  features = pd.DataFrame(user_input_data, index=[0])

  return features

df = user_input_features()
                    

datos =  pd.read_csv('dftemp.csv', encoding='latin-1')
X = datos.drop(columns='AverageTemperature')
y = datos['AverageTemperature']

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1613080)
LR = LinearRegression()
LR.fit(X_train,y_train)

b1 = LR.coef_
b0 = LR.intercept_
prediccion = b0 + b1[0]*df['City'] + b1[1]*df['month'] + b1[2]*df['year'] 

st.subheader('Cálculo de la temperatura')
st.write('La temperatura es de: ', prediccion)
