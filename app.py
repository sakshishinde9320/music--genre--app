import streamlit as st
import librosa
import numpy as np
import joblib
import tempfile

# Load model
model = joblib.load("model.pkl")
encoder = joblib.load("encoder.pkl")

# Page config
st.set_page_config(
    page_title="Music Genre Classifier",
    page_icon="🎵",
    layout="centered"
)

# Title
st.markdown(
    "<h1 style='text-align: center; color: #4CAF50;'>🎵 Music Genre Classification</h1>",
    unsafe_allow_html=True
)

st.write("Upload a WAV song file and predict its genre")

st.divider()

# Upload box
uploaded_file = st.file_uploader(
    "Upload Song",
    type=["wav"]
)

if uploaded_file is not None:

    st.success("File uploaded successfully ✅")

    st.audio(uploaded_file)

    # Save temp file
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        path = tmp.name

    # Feature extraction
    audio, sr = librosa.load(path)

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=13
    )

    mfcc_mean = np.mean(mfcc.T, axis=0)
    mfcc_mean = mfcc_mean.reshape(1, -1)

    # Prediction
    prediction = model.predict(mfcc_mean)
    genre = encoder.inverse_transform(prediction)

    st.markdown(
        f"<h2 style='color: blue;'>Predicted Genre: {genre[0]}</h2>",
        unsafe_allow_html=True
    )