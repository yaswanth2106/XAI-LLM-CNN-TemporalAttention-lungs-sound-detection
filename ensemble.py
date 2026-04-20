"""
Multi-Fold Ensemble Predictor

Uses all 5 fold checkpoints for:
  1. More accurate predictions (averaged probabilities)
  2. Free uncertainty quantification (inter-fold disagreement)
  3. Consensus reporting ("4/5 models agree on COPD")

Total overhead: ~5× inference time (~250ms on CPU for a 941KB model),
which is still orders of magnitude faster than any LLM API call.
"""

import torch
import os
import glob
from collections import Counter
from model import CNN1DAttention


class EnsemblePredictor:
    """Multi-fold ensemble for robust prediction and uncertainty estimation."""

    def __init__(self, in_channels, n_classes, fold_dir=".", device="cpu"):
        """Load all fold checkpoints from a directory.

        Args:
            in_channels: number of MFCC coefficients (e.g. 40)
            n_classes: number of output classes (e.g. 5)
            fold_dir: directory containing best_model_fold*.pth files
            device: 'cpu' or 'cuda'
        """
        self.device = device
        self.n_classes = n_classes
        self.models = []

        fold_paths = sorted(
            glob.glob(os.path.join(fold_dir, "best_model_fold*.pth"))
        )

        if not fold_paths:
            raise FileNotFoundError(
                f"No fold checkpoints found in {fold_dir}"
            )

        for path in fold_paths:
            # Skip empty / corrupted checkpoint files
            if os.path.getsize(path) == 0:
                continue
            try:
                model = CNN1DAttention(in_channels, n_classes).to(device)
                model.load_state_dict(
                    torch.load(path, map_location=device)
                )
                model.eval()
                self.models.append(model)
            except Exception as e:
                print(f"Warning: skipping {path} — {e}")

        if not self.models:
            raise RuntimeError(
                "No valid fold checkpoints could be loaded"
            )

        self.n_folds = len(self.models)

    def predict(self, x):
        """Run ensemble prediction across all folds.

        Args:
            x: input tensor [1, channels, time]

        Returns:
            dict with ensemble prediction results including
            mean/std probabilities, per-fold predictions,
            agreement count, and attention weights
        """
        all_probs = []
        all_attention_weights = []

        for model in self.models:
            with torch.no_grad():
                logits = model(x)
                probs = torch.softmax(logits, dim=1)
                all_probs.append(probs)

                # Collect attention weights
                if (hasattr(model.attn, 'last_weights')
                        and model.attn.last_weights is not None):
                    all_attention_weights.append(model.attn.last_weights)

        # Stack: [n_folds, 1, n_classes]
        stacked = torch.stack(all_probs)
        mean_probs = stacked.mean(dim=0)    # [1, n_classes]
        std_probs = stacked.std(dim=0)      # [1, n_classes]

        # Ensemble prediction (argmax of averaged probabilities)
        ensemble_pred = mean_probs.argmax(dim=1).item()

        # Per-fold predictions
        fold_preds = [p.argmax(dim=1).item() for p in all_probs]

        # Agreement: how many folds match the ensemble prediction
        agreement_count = sum(1 for p in fold_preds if p == ensemble_pred)

        # Mean attention weights across folds
        mean_attention = None
        if all_attention_weights:
            mean_attention = torch.stack(all_attention_weights).mean(dim=0)

        return {
            "mean_probs": mean_probs,
            "std_probs": std_probs,
            "ensemble_pred": ensemble_pred,
            "fold_preds": fold_preds,
            "agreement_count": agreement_count,
            "total_folds": self.n_folds,
            "agreement_ratio": agreement_count / self.n_folds,
            "individual_probs": all_probs,
            "mean_attention_weights": mean_attention,
            "pred_std": float(std_probs.squeeze()[ensemble_pred].item())
        }

    def get_primary_model(self):
        """Return the first fold model for Grad-CAM analysis."""
        return self.models[0]

    def format_fold_info(self, result, class_names):
        """Format ensemble results for the explanation JSON.

        Args:
            result: dict from predict()
            class_names: list of class name strings

        Returns:
            dict suitable for inclusion in the explanation JSON
        """
        fold_preds_named = [class_names[p] for p in result["fold_preds"]]
        pred_counts = dict(Counter(fold_preds_named))

        return {
            "n_folds": result["total_folds"],
            "agreement_count": result["agreement_count"],
            "agreement_ratio": round(result["agreement_ratio"], 2),
            "fold_predictions": fold_preds_named,
            "prediction_counts": pred_counts,
            "ensemble_std": round(result["pred_std"], 4),
            "consensus": result["agreement_ratio"] >= 0.8
        }
