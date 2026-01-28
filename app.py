import streamlit as st
import torch
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import os
from model import CNN1DAttention
from utils import audio_to_mfcc, N_MFCC
from xai import GradCAM1D, build_explanation
from llm import generate_clinical_explanation


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLASS_NAMES = ['Asthma', 'COPD', 'Healthy', 'ILD', 'Infection']


@st.cache_resource
def load_model():
    model = CNN1DAttention(N_MFCC, len(CLASS_NAMES)).to(DEVICE)
    model.load_state_dict(torch.load("best_model_fold0.pth", map_location=DEVICE))
    model.eval()
    return model


model = load_model()

st.set_page_config(page_title="Lung Sound XAI", layout="wide")
st.title("🫁 Lung Sound Diagnosis with Explainable AI")

uploaded = st.file_uploader("Upload lung sound (.wav)", type=["wav"])

if uploaded:
    
    st.subheader("🎧 Lung Sound Playback")
    st.audio(uploaded, format="audio/wav")

    audio_data, sr = sf.read(uploaded)
    duration_sec = len(audio_data) / sr

    st.caption(f"Audio duration: **{duration_sec:.2f} seconds**")
    uploaded.seek(0)
    mfcc = audio_to_mfcc(uploaded)
    x = torch.tensor(mfcc).unsqueeze(0).float().to(DEVICE)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        pred = probs.argmax(dim=1).item()

    gradcam = GradCAM1D(model, model.conv3)
    cam = gradcam.generate(x, pred)
    gradcam.remove()

    st.subheader("🔍 Prediction")
    st.write(f"**Diagnosis:** {CLASS_NAMES[pred]}")
    st.write(f"**Confidence:** {probs[0, pred].item():.3f}")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.imshow(mfcc, aspect="auto", origin="lower", cmap="gray")

    cam_r = np.interp(
        np.linspace(0, 1, mfcc.shape[1]),
        np.linspace(0, 1, len(cam)),
        cam
    )

    ax.imshow(
        cam_r[np.newaxis, :],
        aspect="auto",
        cmap="jet",
        alpha=0.6,
        extent=[0, mfcc.shape[1], 0, mfcc.shape[0]]
    )
    st.pyplot(fig)

    explanation = build_explanation(
        patient_id="uploaded_sample",
        predicted_class=CLASS_NAMES[pred],
        softmax_prob=probs[0, pred].item(),
        cam=cam,
        n_files=1,
        total_duration_sec=duration_sec
        )


    
    


    st.subheader("🧠 Clinician-Friendly Explanation (LLM)")

    if st.button("Generate Explanation"):
        with st.spinner("Generating explanation..."):
            clinical_text = generate_clinical_explanation(explanation)
        st.markdown(clinical_text)



