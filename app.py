# app.py
import streamlit as st
from src.pipeline.train_pipeline import Train_Pipeline
from src.pipeline.prediction_pipeline import PredictionPipeline
from src.exception.exception import CustomException
import sys

def main():
    st.title("Machine Learning App")

    # Navigation
    menu = ["Home", "Train", "Predict"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.subheader("Home")
        st.write("Welcome to the Machine Learning App")

    elif choice == "Train":
        st.subheader("Train the Model")

        if st.button('Start Training'):
            try:
                train_pipeline = Train_Pipeline()
                train_pipeline.run_pipeline()
                st.success("Training successful !!")
            except Exception as e:
                st.error(f"Error Occurred! {e}")

    elif choice == "Predict":
        st.subheader("Make a Prediction")

        text = st.text_area("Enter text here:")

        if st.button('Predict'):
            try:
                obj = PredictionPipeline()
                prediction = obj.run_pipeline(text)
                st.write(f'The prediction is: {prediction}')
            except Exception as e:
                st.error(f"Error Occurred! {e}")

if __name__ == '__main__':
    main()
