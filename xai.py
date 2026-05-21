import torch
import numpy as np


class GradCAM1D:


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
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

    def remove(self):
        self.fwd.remove()
        self.bwd.remove()



def differential_gradcam(model, target_layer, x, class1_idx, class2_idx):
    gradcam = GradCAM1D(model, target_layer)
    cam1 = gradcam.generate(x, class1_idx)
    gradcam.remove()

    gradcam = GradCAM1D(model, target_layer)
    cam2 = gradcam.generate(x, class2_idx)
    gradcam.remove()

    diff = cam1 - cam2
    return diff, cam1, cam2




def cam_to_time_ranges(cam, top_k=3, threshold_pct=0.6, total_duration_sec=5.0):
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




def cam_statistics(cam, total_duration_sec=5.0):
    coverage_ratio = float((cam > 0.5).mean())
    peak_sharpness = float(cam.max() / (cam.mean() + 1e-8))

    cam_norm = cam / (cam.sum() + 1e-8)
    entropy = -np.sum(cam_norm * np.log(cam_norm + 1e-8))
    max_entropy = np.log(len(cam)) if len(cam) > 1 else 1.0
    normalized_entropy = float(entropy / (max_entropy + 1e-8))

    regions = cam_to_time_ranges(
        cam, top_k=10, total_duration_sec=total_duration_sec
    )
    n_regions = len(regions)

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



def attention_cam_agreement(attention_weights, cam):
    attn = attention_weights.squeeze().cpu().numpy()
    if attn.ndim > 1:
        attn = attn.squeeze()

    if len(attn) != len(cam):
        attn = np.interp(
            np.linspace(0, 1, len(cam)),
            np.linspace(0, 1, len(attn)),
            attn
        )

    attn = (attn - attn.min()) / (attn.max() - attn.min() + 1e-8)

    dot = np.dot(attn, cam)
    norms = np.linalg.norm(attn) * np.linalg.norm(cam) + 1e-8
    agreement = float(dot / norms)

    return round(agreement, 3)




def prediction_metrics(probs, class_names):
    if probs.dim() > 1:
        probs = probs.squeeze(0)

    n_classes = len(class_names)
    probs_np = probs.cpu().numpy()

    distribution = {
        name: round(float(p), 4)
        for name, p in zip(class_names, probs_np)
    }

    sorted_indices = torch.argsort(probs, descending=True)
    top1_idx = sorted_indices[0].item()
    top2_idx = sorted_indices[1].item()
    top1_prob = probs[top1_idx].item()
    top2_prob = probs[top2_idx].item()

    entropy = -torch.sum(probs * torch.log(probs + 1e-8)).item()
    max_entropy = np.log(n_classes)
    normalized_entropy = entropy / (max_entropy + 1e-8)
    top2_gap = top1_prob - top2_prob
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
    
    regions = cam_to_time_ranges(cam, total_duration_sec=total_duration_sec)

    explanation = {
        "patient_id": str(patient_id),
        "predicted_class": prediction_info["predicted_class"],
        "probability": prediction_info["probability"],
        "confidence_tier": prediction_info["confidence_tier"],
        "full_distribution": prediction_info["full_distribution"],
        "prediction_entropy": prediction_info["prediction_entropy"],
        "second_class": prediction_info["second_class"],
        "second_probability": prediction_info["second_probability"],
        "top2_gap": prediction_info["top2_gap"],

        "top_time_regions_sec": [
            {
                "start": round(float(s), 2),
                "end": round(float(e), 2),
                "severity_score": round(float(scr), 2)
            }
            for s, e, scr in regions
        ],

        "cam_statistics": cam_stats,

        "attention_cam_agreement": attention_agreement,

        "audio_quality": audio_quality,

        "ensemble": fold_info if fold_info else None,

        "differential": differential_info if differential_info else None,

        "n_samples_used": int(n_files),
        "total_duration_sec": round(float(total_duration_sec), 2),
        "notes": "Enhanced XAI: Grad-CAM + Temporal Attention + Ensemble"
    }

    return explanation
