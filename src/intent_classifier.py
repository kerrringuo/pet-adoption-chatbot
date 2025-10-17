import os
import json
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer

class IntentClassifier:
    def __init__(self, model_dir="../models/intent"):
        """Load MiniLM encoder and SVM classifier."""
        self.model_dir = model_dir
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")

        model_path = os.path.join(model_dir, "minilm_logreg.joblib")
        labels_path = os.path.join(model_dir, "minilm_labels.json")
        meta_path = os.path.join(model_dir, "minilm_meta.json")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Missing model: {model_path}")

        self.clf = joblib.load(model_path)
        self.labels = json.load(open(labels_path, "r"))
        self.meta = json.load(open(meta_path, "r"))
        self.threshold = float(self.meta.get("threshold", 0.7))

    def predict(self, text):
        """Return (intent, confidence) for input text."""
        if not text or not text.strip():
            return "unknown", 0.0

        emb = self.encoder.encode([text])
        probs = self.clf.predict_proba(emb)[0]
        idx = int(np.argmax(probs))
        conf = float(probs[idx])
        intent = self.labels[idx] if conf >= self.threshold else "unknown"
        return intent, conf


if __name__ == "__main__":
    clf = IntentClassifier()
    for msg in [
        "I want to adopt a puppy near KL",
        "How to train my dog?",
        "bye bye see you later",
        "hi i wan a doog"
    ]:
        intent, conf = clf.predict(msg)
        print(f"{msg} → {intent} ({conf:.2f})")
