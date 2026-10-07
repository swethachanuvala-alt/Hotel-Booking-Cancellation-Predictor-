"""
Re-train the model and overwrite the files in /model.

    python train_model.py

Run this only if you want to regenerate the .pkl files (for example after
changing the dataset). The app also works without running it.
"""
import time

import model_utils as mu

if __name__ == "__main__":
    t = time.time()
    print("Training Random Forest (this takes about a minute)...")
    model, scaler, encoders, meta = mu.train_pipeline()
    mu.save_artifacts(model, scaler, encoders, meta)
    m = meta["metrics"]
    print(f"Done in {time.time() - t:.0f}s")
    print(
        f"Accuracy {m['accuracy']:.4f} | Precision {m['precision']:.4f} | "
        f"Recall {m['recall']:.4f} | F1 {m['f1']:.4f} | ROC-AUC {m['roc_auc']:.4f}"
    )
    print("Saved files to:", mu.MODEL_DIR)
