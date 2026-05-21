
import torch
import os
import glob
from collections import Counter
from model import CNN1DAttention


class EnsemblePredictor:
    

    def __init__(self, in_channels, n_classes, fold_dir="./new", device="cpu"):
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

        all_probs = []
        all_attention_weights = []

        for model in self.models:
            with torch.no_grad():
                logits = model(x)
                probs = torch.softmax(logits, dim=1)
                all_probs.append(probs)

                if (hasattr(model.attn, 'last_weights')
                        and model.attn.last_weights is not None):
                    all_attention_weights.append(model.attn.last_weights)

        stacked = torch.stack(all_probs)
        mean_probs = stacked.mean(dim=0)   
        std_probs = stacked.std(dim=0)      

        ensemble_pred = mean_probs.argmax(dim=1).item()

        fold_preds = [p.argmax(dim=1).item() for p in all_probs]

        agreement_count = sum(1 for p in fold_preds if p == ensemble_pred)

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
        return self.models[0]

    def format_fold_info(self, result, class_names):
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
