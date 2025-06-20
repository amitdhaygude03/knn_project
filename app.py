import streamlit as st
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

# Load and train model inline to avoid pickle compatibility issues
@st.cache_resource
def train_model():
    data = pd.read_csv("iris.csv")
    X = data.drop(columns=["species"])
    y = data["species"]

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X, y_encoded)

    return model, le

model, label_encoder = train_model()

# Streamlit UI
st.set_page_config(page_title="Iris KNN Classifier 🌼", layout="centered")
st.title("🌼 Iris Flower Species Predictor (KNN)")
st.write("Enter the flower's measurements below to predict its species using K-Nearest Neighbors:")

# Input features
sepal_length = st.number_input("Sepal Length (cm)", value=5.1, format="%.2f")
sepal_width = st.number_input("Sepal Width (cm)", value=3.5, format="%.2f")
petal_length = st.number_input("Petal Length (cm)", value=1.4, format="%.2f")
petal_width = st.number_input("Petal Width (cm)", value=0.2, format="%.2f")

# Prediction
if st.button("Predict"):
    try:
        features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        prediction = model.predict(features)[0]
        flower_name = label_encoder.inverse_transform([prediction])[0]

        st.success(f"🌸 Predicted Flower: **{flower_name}**")
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
