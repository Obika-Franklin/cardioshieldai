import os
import gdown
import streamlit as st

MODEL_PATH = "ecg_model_new.keras"
FILE_ID = "10rARyGahx1EbunzvN7HKN1oMB2Q9S4kq"  # replace with your correct File ID

def download_ecg_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading ECG model... this may take a minute ⏳"):
            url = f"https://drive.google.com/uc?id={FILE_ID}"
            gdown.download(url, MODEL_PATH, quiet=False)
    return MODEL_PATH