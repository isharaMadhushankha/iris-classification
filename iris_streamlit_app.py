import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# Load Iris data
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target
model = RandomForestClassifier()
model.fit(X, y)

# Streamlit UI
st.set_page_config(page_title="Iris Flower Classifier", page_icon="🌸")
st.title("🌸 Iris Flower Classification")
st.write("Enter the flower measurements below:")

# Input boxes instead of sliders
sepal_length = st.number_input("Sepal Length (cm)", min_value=4.0, max_value=8.0, value=5.1, step=0.1)
sepal_width = st.number_input("Sepal Width (cm)", min_value=2.0, max_value=4.5, value=3.5, step=0.1)
petal_length = st.number_input("Petal Length (cm)", min_value=1.0, max_value=7.0, value=1.4, step=0.1)
petal_width = st.number_input("Petal Width (cm)", min_value=0.1, max_value=2.5, value=0.2, step=0.1)

# Predict button
if st.button("Predict"):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_data)[0]
    species = iris.target_names[prediction]
    st.success(f"The predicted Iris species is: **{species.capitalize()}**")
