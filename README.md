#  Clinical Audio AI & Explainability (XAI) Dashboard

This repository contains a deep learning pipeline for the classification and interpretation of clinical audio signals. It leverages **1D-CNNs**, **Temporal Attention**, and **Explainable AI (XAI)** to provide not just predictions, but actionable insights for clinicians.

---

## Key Features

*   **Deep Learning Pipeline**: A custom 1D-CNN architecture optimized for sequence data like MFCCs.
*   **Temporal Attention**: Integrated attention mechanism to weight significant time-steps in the audio signal.
*   **XAI with Grad-CAM**: Implementation of Gradient-weighted Class Activation Mapping for 1D signals to identify "where" the model is looking.
*   **LLM Synthesis**: Uses **Cohere's Command model** to translate technical XAI outputs and probability scores into natural language clinical summaries.
*   **Audio Preprocessing**: Standardized MFCC extraction and normalization for robust training and inference.

---


## Supported Conditions

The model is trained to identify the following respiratory conditions:
1.  **Asthma**
2.  **COPD** (Chronic Obstructive Pulmonary Disease)
3.  **Healthy**
4.  **ILD** (Interstitial Lung Disease)
5.  **Infection**


##  Model Workflow

### 1. Feature Extraction & Prediction
- **Audio Input**: Users upload a `.wav` file of lung sounds.
- **MFCC Processing**: Standardized MFCC extraction and normalization.
- **Inference**: The `CNN1DAttention` model predicts the condition with a confidence score.

### 2. Interpretability (XAI)
The `GradCAM1D` class hooks into the final convolutional layer (`conv3`). It generates a heatmap overlaid on the MFCC spectrogram, visually highlighting the specific time-frequency regions that influenced the diagnosis.

### 3. Clinical Report Generation
The detected "interest regions" are formatted into JSON and processed by the Cohere LLM to generate:
-   A high-level **Summary** of the predicted class.
-   Direct **Findings** describing specific time regions in the audio.
-   **Suggested next steps** for the clinician.
-   A mandatory medical **Disclaimer**.

---


