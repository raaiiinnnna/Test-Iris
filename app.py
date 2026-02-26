import streamlit as st
import pandas as pd
import pickle
from sklearn.datasets import load_iris

# Load model yang sudah dibuat
model = pickle.load(open('iris_model.pkl', 'rb'))
iris = load_iris()

st.title("Aplikasi Prediksi Bunga Iris")
st.write("Masukkan parameter di bawah ini untuk memprediksi jenis bunga Iris.")

# Membuat sidebar untuk input user
st.sidebar.header("Input Parameter")

def user_input():
    sepal_length = st.sidebar.slider('Sepal Length', 4.3, 7.9, 5.4)
    sepal_width = st.sidebar.slider('Sepal Width', 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider('Petal Length', 1.0, 6.9, 1.3)
    petal_width = st.sidebar.slider('Petal Width', 0.1, 2.5, 0.2)
    data = {
        'sepal length (cm)': sepal_length,
        'sepal width (cm)': sepal_width,
        'petal length (cm)': petal_length,
        'petal width (cm)': petal_width
    }
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input()

st.subheader('Parameter Input:')
st.write(df)

# Melakukan Prediksi
prediction = model.predict(df)
prediction_proba = model.predict_proba(df)

st.subheader('Hasil Prediksi:')
st.write(iris.target_names[prediction][0])

st.subheader('Probabilitas Prediksi:')
st.write(pd.DataFrame(prediction_proba, columns=iris.target_names))
