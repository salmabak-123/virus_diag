"""Mesure les performances : accuracy, précision, rappel, F1, matrice de confusion."""
from utils import metrics


class Evaluator:
    def __init__(self, seuil=0.5):
        self.seuil = seuil

    def evaluate(self, model, X, y):
        y_pred = (model.predict_proba_batch(X) >= self.seuil).astype(int)
        tn, fp, fn, tp = metrics.confusion_matrix(y, y_pred)
        return {"accuracy": metrics.accuracy(y, y_pred),
                "precision": metrics.precision(y, y_pred),
                "recall": metrics.recall(y, y_pred),
                "f1": metrics.f1_score(y, y_pred),
                "confusion": {"VN": tn, "FP": fp, "FN": fn, "VP": tp}}

    @staticmethod
    def report(nom, r):
        c = r["confusion"]
        return (f"{nom:22s} acc={r['accuracy']:.3f}  prec={r['precision']:.3f}  "
                f"rappel={r['recall']:.3f}  F1={r['f1']:.3f}  "
                f"[VN={c['VN']} FP={c['FP']} FN={c['FN']} VP={c['VP']}]")
