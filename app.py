import os
import sys

import streamlit as st
from joblib import load

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
sys.path.insert(0, SRC_DIR)

from utils import clean_text


def load_model(path: str):
    return load(path)


def classify_text(model, text: str) -> str:
    cleaned = clean_text(text)
    return model.predict([cleaned])[0]


def main():
    st.title("Fake News Detector")
    st.write("Enter a news headline or article excerpt and the model will predict whether it is fake or real.")

    text_input = st.text_area("News text", height=200)
    model = load_model("models/model.pkl")

    if st.button("Classify"):
        if not text_input.strip():
            st.warning("Please provide some text for classification.")
            return

        label = classify_text(model, text_input)
        st.success(f"Prediction: {label}")


if __name__ == "__main__":
    main()
