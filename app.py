import streamlit as st
import torch
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import json

from model import CNN1DAttention
from utils import audio_to_mfcc, N_MFCC
from xai import (
    GradCAM1D, build_explanation, cam_to_time_ranges,
    cam_statistics, attention_cam_agreement,
    prediction_metrics, differential_gradcam
)
from audio_quality import assess_audio_quality
from ensemble import EnsemblePredictor
from rule_engine import generate_clinical_explanation


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLASS_NAMES = ['Asthma', 'COPD', 'Healthy', 'ILD', 'Infection']


def _agreement_label(score):
    """Label for attention-CAM agreement score (for chart title)."""
    if score >= 0.8:
        return "Strong"
    if score >= 0.1:
        return "Moderate"
    return "Weak"


@st.cache_resource
def load_ensemble():
    """Load all fold models as an ensemble."""
    return EnsemblePredictor(
        in_channels=N_MFCC,
        n_classes=len(CLASS_NAMES),
        fold_dir=".",
        device=DEVICE
    )


st.set_page_config(page_title="Lung Sound XAI", layout="wide")

ensemble = load_ensemble()

st.title("Lung Sound Diagnosis with Explainable AI")
st.caption(
    "Edge-ready • Rule-based explanations • "
    "Multi-fold ensemble • Zero cloud dependencies"
)

uploaded = st.file_uploader("Upload lung sound (.wav)", type=["wav"])

if uploaded:

    # ── Audio Playback ────────────────────────────────────────────────
    st.subheader("🎧 Lung Sound Playback")
    st.audio(uploaded, format="audio/wav")

    # ── Read Audio & Assess Quality ───────────────────────────────────
    audio_data, sr = sf.read(uploaded)
    if audio_data.ndim > 1:
        audio_data = audio_data.mean(axis=1)
    duration_sec = len(audio_data) / sr

    st.caption(
        f"Audio duration: **{duration_sec:.2f} seconds** | "
        f"Sample rate: **{sr} Hz**"
    )

    audio_quality = assess_audio_quality(audio_data, sr)

    # Quality indicators
    q1, q2, q3, q4 = st.columns(4)
    with q1:
        st.metric("SNR", f"{audio_quality['snr_db']:.1f} dB")
    with q2:
        st.metric("Silence", f"{audio_quality['silence_ratio']:.0%}")
    with q3:
        st.metric(
            "Clipping",
            "Yes ⚠️" if audio_quality['clipping'] else "No ✅"
        )
    with q4:
        quality_emoji = {
            "good": "✅ Good", "acceptable": "🟡 OK", "poor": "🔴 Poor"
        }
        st.metric(
            "Quality",
            quality_emoji.get(audio_quality['overall_quality'], "Unknown")
        )

    # ── Feature Extraction ────────────────────────────────────────────
    uploaded.seek(0)
    mfcc = audio_to_mfcc(uploaded)
    x = torch.tensor(mfcc).unsqueeze(0).float().to(DEVICE)

    # ── Ensemble Prediction ───────────────────────────────────────────
    ensemble_result = ensemble.predict(x)
    mean_probs = ensemble_result["mean_probs"]
    pred_info = prediction_metrics(mean_probs, CLASS_NAMES)
    pred = pred_info["predicted_idx"]

    # ── Grad-CAM (on primary model) ───────────────────────────────────
    primary_model = ensemble.get_primary_model()
    gradcam = GradCAM1D(primary_model, primary_model.conv3)
    cam = gradcam.generate(x, pred)
    gradcam.remove()

    # ── Attention Weights (from the GradCAM forward pass) ─────────────
    attn_weights = primary_model.attn.last_weights
    attn_agree = attention_cam_agreement(attn_weights, cam)

    # ── CAM Statistics ────────────────────────────────────────────────
    cam_stats = cam_statistics(cam, total_duration_sec=duration_sec)

    # ── Differential Grad-CAM (if top-2 gap is narrow) ────────────────
    differential_info = None
    diff_cam = None
    if pred_info["top2_gap"] < 0.3:
        diff_cam, cam_top1, cam_top2 = differential_gradcam(
            primary_model, primary_model.conv3, x,
            pred_info["predicted_idx"], pred_info["second_idx"]
        )
        pos_diff = np.maximum(diff_cam, 0)
        if pos_diff.max() > 0:
            pos_diff_norm = pos_diff / (pos_diff.max() + 1e-8)
            diff_regions = cam_to_time_ranges(
                pos_diff_norm, top_k=3,
                threshold_pct=0.5,
                total_duration_sec=duration_sec
            )
            differential_info = {
                "compared_classes": [
                    pred_info["predicted_class"],
                    pred_info["second_class"]
                ],
                "distinguishing_regions": [
                    {
                        "start": round(float(s), 2),
                        "end": round(float(e), 2),
                        "diff_score": round(float(sc), 2)
                    }
                    for s, e, sc in diff_regions
                ]
            }

    # ── Ensemble Info ─────────────────────────────────────────────────
    fold_info = ensemble.format_fold_info(ensemble_result, CLASS_NAMES)

    # ── Build Enriched Explanation ────────────────────────────────────
    explanation = build_explanation(
        patient_id="uploaded_sample",
        prediction_info=pred_info,
        cam=cam,
        cam_stats=cam_stats,
        attention_agreement=attn_agree,
        audio_quality=audio_quality,
        fold_info=fold_info,
        differential_info=differential_info,
        n_files=1,
        total_duration_sec=duration_sec
    )

    # ══════════════════════════════════════════════════════════════════
    # DISPLAY RESULTS
    # ══════════════════════════════════════════════════════════════════

    st.divider()
    st.subheader("🔍 Prediction")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Diagnosis", CLASS_NAMES[pred])
    with col2:
        st.metric("Confidence", f"{pred_info['probability']:.1%}")
    with col3:
        consensus_text = (
            f"{fold_info['agreement_count']}/{fold_info['n_folds']} folds"
        )
        st.metric("Ensemble Agreement", consensus_text)

    # ── Probability Distribution Chart ────────────────────────────────
    st.subheader("📊 Class Probabilities")
    prob_data = pred_info["full_distribution"]

    fig_prob, ax_prob = plt.subplots(figsize=(10, 3))
    colors = [
        '#ef4444' if name == CLASS_NAMES[pred] else '#94a3b8'
        for name in prob_data.keys()
    ]
    bars = ax_prob.barh(
        list(prob_data.keys()),
        list(prob_data.values()),
        color=colors
    )
    ax_prob.set_xlim(0, 1)
    ax_prob.set_xlabel("Probability")
    for bar, val in zip(bars, prob_data.values()):
        ax_prob.text(
            bar.get_width() + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f'{val:.1%}', va='center', fontsize=10
        )
    plt.tight_layout()
    st.pyplot(fig_prob)

    # ── Grad-CAM Spectrogram Overlay ──────────────────────────────────
    st.subheader("🔬 Grad-CAM Spectrogram Overlay")
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
    ax.set_xlabel("Time frames")
    ax.set_ylabel("MFCC coefficients")
    ax.set_title(
        f"Grad-CAM: {CLASS_NAMES[pred]} "
        f"({pred_info['probability']:.1%})"
    )
    st.pyplot(fig)

    # ── Attention vs Grad-CAM Comparison ──────────────────────────────
    st.subheader("🧠 Attention vs Grad-CAM Comparison")
    fig_cmp, ax_cmp = plt.subplots(figsize=(10, 3))

    # Normalize attention for plotting
    attn_np = attn_weights.squeeze().cpu().numpy()
    if attn_np.ndim > 1:
        attn_np = attn_np.squeeze()
    attn_interp = np.interp(
        np.linspace(0, 1, len(cam)),
        np.linspace(0, 1, len(attn_np)),
        attn_np
    )
    attn_interp = (
        (attn_interp - attn_interp.min())
        / (attn_interp.max() - attn_interp.min() + 1e-8)
    )

    time_axis = np.linspace(0, duration_sec, len(cam))
    ax_cmp.plot(
        time_axis, cam, label="Grad-CAM",
        color="#ef4444", linewidth=2
    )
    ax_cmp.plot(
        time_axis, attn_interp, label="Attention",
        color="#3b82f6", linewidth=2, alpha=0.8
    )
    ax_cmp.fill_between(time_axis, cam, alpha=0.15, color="#ef4444")
    ax_cmp.fill_between(
        time_axis, attn_interp, alpha=0.15, color="#3b82f6"
    )
    ax_cmp.set_xlabel("Time (seconds)")
    ax_cmp.set_ylabel("Activation")
    ax_cmp.set_title(
        f"Agreement: {attn_agree:.2f} "
        f"({_agreement_label(attn_agree)})"
    )
    ax_cmp.legend()
    plt.tight_layout()
    st.pyplot(fig_cmp)

    # ── Differential Grad-CAM (if applicable) ─────────────────────────
    if differential_info and diff_cam is not None:
        st.subheader("⚖️ Differential Grad-CAM")
        st.caption(
            f"What distinguishes {pred_info['predicted_class']} "
            f"from {pred_info['second_class']}"
        )
        fig_diff, ax_diff = plt.subplots(figsize=(10, 3))
        time_diff = np.linspace(0, duration_sec, len(diff_cam))
        ax_diff.fill_between(
            time_diff, diff_cam, where=diff_cam > 0,
            color="#22c55e", alpha=0.5,
            label=f"Favors {pred_info['predicted_class']}"
        )
        ax_diff.fill_between(
            time_diff, diff_cam, where=diff_cam < 0,
            color="#ef4444", alpha=0.5,
            label=f"Favors {pred_info['second_class']}"
        )
        ax_diff.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
        ax_diff.set_xlabel("Time (seconds)")
        ax_diff.set_ylabel("Differential activation")
        ax_diff.legend()
        plt.tight_layout()
        st.pyplot(fig_diff)

    # ── Clinical Explanation (Rule-Based) ─────────────────────────────
    st.divider()
    st.subheader("📋 Clinical Explanation (Rule-Based)")
    clinical_text = generate_clinical_explanation(explanation)
    st.markdown(clinical_text)

    # ── Raw XAI JSON (collapsible) ────────────────────────────────────
    with st.expander("🔧 Raw XAI JSON (for debugging / audit trail)"):
        # Strip non-serializable tensors for display
        display_json = {}
        for k, v in explanation.items():
            if isinstance(v, torch.Tensor):
                display_json[k] = v.tolist()
            else:
                display_json[k] = v
        st.json(display_json)
