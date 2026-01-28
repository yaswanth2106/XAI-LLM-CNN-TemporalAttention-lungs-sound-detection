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
        cam = (cam - cam.min()) / (cam.max() + 1e-8)
        return cam

    def remove(self):
        self.fwd.remove()
        self.bwd.remove()


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


def build_explanation(
    patient_id,
    predicted_class,
    softmax_prob,
    cam,
    n_files,
    total_duration_sec=5.0
):
    regions = cam_to_time_ranges(cam, total_duration_sec=total_duration_sec)

    return {
        "patient_id": str(patient_id),
        "predicted_class": str(predicted_class),
        "probability": float(round(float(softmax_prob), 3)),
        "top_time_regions_sec": [
            {
                "start": round(float(s), 2),
                "end": round(float(e), 2),
                "severity_score": round(float(scr), 2)
            }
            for s, e, scr in regions
        ],
        "n_samples_used": int(n_files),
        "notes": "Grad-CAM averaged over MFCC temporal dimension"
    }
