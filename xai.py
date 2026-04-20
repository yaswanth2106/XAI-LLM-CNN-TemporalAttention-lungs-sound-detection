import torch
import numpy as np


class GradCAM1D:
    """Gradient-weighted Class Activation Mapping for 1D signals."""

    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None

        self.fwd = target_layer.register_forward_hook(self._forward)
        self.bwd = target_layer.register_full_backward_hook(self._backward)

    def _forward(self, m, i, o):
        self.activations = o.detach()

    def _backward(self, m, gi, go):
        self.gradients = go[0].detach()

    def generate(self, x, class_idx):
        self.model.zero_grad()
        score = self.model(x)[0, class_idx]
        score.backward()

        weights = self.gradients.mean(dim=2, keepdim=True)
        cam = (weights * self.activations).sum(dim=1)
        cam = torch.relu(cam).squeeze().cpu().numpy()
        # FIX: proper min-max normalization (was missing cam.min() in denominator)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

    def remove(self):
        self.fwd.remove()
        self.bwd.remove()


# ── Differential Grad-CAM ─────────────────────────────────────────────────

def differential_gradcam(model, target_layer, x, class1_idx, class2_idx):
    """Compute differential Grad-CAM between two classes.

    Shows which regions distinguish class1 from class2.
    Positive values → favors class1, Negative → favors class2.

    Args:
        model: the CNN model
        target_layer: convolutional layer to hook
        x: input tensor [1, C, T]
        class1_idx: index of the primary (top-1) class
        class2_idx: index of the secondary (top-2) class

    Returns:
        diff_cam: differential activation map (numpy array)
        cam1: Grad-CAM for class1
        cam2: Grad-CAM for class2
    """
    gradcam = GradCAM1D(model, target_layer)
    cam1 = gradcam.generate(x, class1_idx)
    gradcam.remove()

    gradcam = GradCAM1D(model, target_layer)
    cam2 = gradcam.generate(x, class2_idx)
    gradcam.remove()

    diff = cam1 - cam2
    return diff, cam1, cam2


# ── Time Region Extraction ────────────────────────────────────────────────

def cam_to_time_ranges(cam, top_k=3, threshold_pct=0.6, total_duration_sec=5.0):
    """Extract top-K contiguous time regions above threshold from a CAM map."""
    T = len(cam)
    threshold = threshold_pct * cam.max()
    idx = np.where(cam >= threshold)[0]

    if len(idx) == 0:
        return []

    regions, start, prev = [], idx[0], idx[0]
    for i in idx[1:]:
        if i == prev + 1:
            prev = i
        else:
            regions.append((start, prev))
            start, prev = i, i
    regions.append((start, prev))

    results = []
    for s, e in regions:
        score = cam[s:e+1].mean()
        results.append((
            (s / T) * total_duration_sec,
            (e / T) * total_duration_sec,
            score
        ))

    return sorted(results, key=lambda x: x[2], reverse=True)[:top_k]


# ── CAM Statistics ────────────────────────────────────────────────────────

def cam_statistics(cam, total_duration_sec=5.0):
    """Compute statistics over the Grad-CAM activation map.

    Returns metrics describing the shape and spatial distribution
    of model attention — used by the rule engine to describe
    whether findings are focal vs diffuse, clustered vs spread.
    """
    # Coverage: fraction of signal with activation > 0.5
    coverage_ratio = float((cam > 0.5).mean())

    # Peak sharpness: how localized is the highest activation
    peak_sharpness = float(cam.max() / (cam.mean() + 1e-8))

    # Activation entropy (normalized to [0, 1])
    cam_norm = cam / (cam.sum() + 1e-8)
    entropy = -np.sum(cam_norm * np.log(cam_norm + 1e-8))
    max_entropy = np.log(len(cam)) if len(cam) > 1 else 1.0
    normalized_entropy = float(entropy / (max_entropy + 1e-8))

    # Number of activated regions (use generous top_k to count all)
    regions = cam_to_time_ranges(
        cam, top_k=10, total_duration_sec=total_duration_sec
    )
    n_regions = len(regions)

    # Temporal spread of activated regions
    if n_regions > 1:
        centers = [(s + e) / 2 for s, e, _ in regions]
        temporal_spread = float(np.std(centers))
        if temporal_spread > total_duration_sec * 0.3:
            spread_label = "spread"
        elif temporal_spread > total_duration_sec * 0.1:
            spread_label = "moderate"
        else:
            spread_label = "clustered"
    elif n_regions == 1:
        temporal_spread = 0.0
        spread_label = "focal"
    else:
        temporal_spread = 0.0
        spread_label = "none"

    return {
        "coverage_ratio": round(coverage_ratio, 3),
        "peak_sharpness": round(peak_sharpness, 2),
        "activation_entropy": round(normalized_entropy, 3),
        "n_active_regions": n_regions,
        "temporal_spread_sec": round(temporal_spread, 3),
        "spread_label": spread_label
    }


# ── Attention–CAM Agreement ──────────────────────────────────────────────

def attention_cam_agreement(attention_weights, cam):
    """Compute cosine similarity between attention weights and Grad-CAM.

    High agreement (>0.7) means both mechanisms focus on the same regions.
    Low agreement (<0.3) means the model uses different features at
    different layers — findings should be interpreted more cautiously.

    Args:
        attention_weights: tensor from model.attn.last_weights [B, T, 1]
        cam: Grad-CAM numpy array [T']

    Returns:
        float: cosine similarity in [0, 1]
    """
    attn = attention_weights.squeeze().cpu().numpy()
    if attn.ndim > 1:
        attn = attn.squeeze()

    # Interpolate attention to match CAM resolution
    if len(attn) != len(cam):
        attn = np.interp(
            np.linspace(0, 1, len(cam)),
            np.linspace(0, 1, len(attn)),
            attn
        )

    # Normalize both to [0, 1]
    attn = (attn - attn.min()) / (attn.max() - attn.min() + 1e-8)

    # Cosine similarity
    dot = np.dot(attn, cam)
    norms = np.linalg.norm(attn) * np.linalg.norm(cam) + 1e-8
    agreement = float(dot / norms)

    return round(agreement, 3)


# ── Prediction-Level Metrics ─────────────────────────────────────────────

def prediction_metrics(probs, class_names):
    """Compute prediction-level metrics from softmax probabilities.

    Extracts full distribution, entropy, top-2 gap, and confidence tier.
    These are all deterministic and critical for the rule engine.

    Args:
        probs: softmax probabilities tensor [1, n_classes] or [n_classes]
        class_names: list of class name strings

    Returns:
        dict with all prediction-level metrics
    """
    if probs.dim() > 1:
        probs = probs.squeeze(0)

    n_classes = len(class_names)
    probs_np = probs.cpu().numpy()

    # Full distribution
    distribution = {
        name: round(float(p), 4)
        for name, p in zip(class_names, probs_np)
    }

    # Top-1 and top-2
    sorted_indices = torch.argsort(probs, descending=True)
    top1_idx = sorted_indices[0].item()
    top2_idx = sorted_indices[1].item()
    top1_prob = probs[top1_idx].item()
    top2_prob = probs[top2_idx].item()

    # Prediction entropy (normalized to [0, 1])
    entropy = -torch.sum(probs * torch.log(probs + 1e-8)).item()
    max_entropy = np.log(n_classes)
    normalized_entropy = entropy / (max_entropy + 1e-8)

    # Top-2 gap
    top2_gap = top1_prob - top2_prob

    # Confidence tier
    if top1_prob >= 0.9:
        confidence_tier = "high"
    elif top1_prob >= 0.7:
        confidence_tier = "moderate"
    elif top1_prob >= 0.5:
        confidence_tier = "low"
    else:
        confidence_tier = "very_low"

    return {
        "predicted_class": class_names[top1_idx],
        "predicted_idx": top1_idx,
        "probability": round(top1_prob, 4),
        "second_class": class_names[top2_idx],
        "second_idx": top2_idx,
        "second_probability": round(top2_prob, 4),
        "top2_gap": round(top2_gap, 4),
        "prediction_entropy": round(normalized_entropy, 3),
        "confidence_tier": confidence_tier,
        "full_distribution": distribution
    }


# ── Enriched Explanation Builder ──────────────────────────────────────────

def build_explanation(
    patient_id,
    prediction_info,
    cam,
    cam_stats,
    attention_agreement,
    audio_quality,
    fold_info=None,
    differential_info=None,
    n_files=1,
    total_duration_sec=5.0
):
    """Build enriched explanation JSON combining ALL XAI metrics.

    This is the single source of truth consumed by the rule engine.
    Every field is deterministic and fully traceable.

    Args:
        patient_id: patient identifier string
        prediction_info: dict from prediction_metrics()
        cam: Grad-CAM numpy array
        cam_stats: dict from cam_statistics()
        attention_agreement: float from attention_cam_agreement()
        audio_quality: dict from audio_quality module
        fold_info: dict from ensemble (optional)
        differential_info: dict with differential CAM info (optional)
        n_files: number of audio files used
        total_duration_sec: audio duration in seconds
    """
    regions = cam_to_time_ranges(cam, total_duration_sec=total_duration_sec)

    explanation = {
        "patient_id": str(patient_id),

        # ── Prediction ──
        "predicted_class": prediction_info["predicted_class"],
        "probability": prediction_info["probability"],
        "confidence_tier": prediction_info["confidence_tier"],
        "full_distribution": prediction_info["full_distribution"],
        "prediction_entropy": prediction_info["prediction_entropy"],
        "second_class": prediction_info["second_class"],
        "second_probability": prediction_info["second_probability"],
        "top2_gap": prediction_info["top2_gap"],

        # ── Temporal Regions ──
        "top_time_regions_sec": [
            {
                "start": round(float(s), 2),
                "end": round(float(e), 2),
                "severity_score": round(float(scr), 2)
            }
            for s, e, scr in regions
        ],

        # ── CAM Statistics ──
        "cam_statistics": cam_stats,

        # ── Attention–CAM Agreement ──
        "attention_cam_agreement": attention_agreement,

        # ── Audio Quality ──
        "audio_quality": audio_quality,

        # ── Ensemble ──
        "ensemble": fold_info if fold_info else None,

        # ── Differential Diagnosis ──
        "differential": differential_info if differential_info else None,

        # ── Metadata ──
        "n_samples_used": int(n_files),
        "total_duration_sec": round(float(total_duration_sec), 2),
        "notes": "Enhanced XAI: Grad-CAM + Temporal Attention + Ensemble"
    }

    return explanation
