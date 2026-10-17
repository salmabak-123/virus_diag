"""Régression logistique avec régularisation L2."""
import numpy as np
from core.model import BaseModel, sigmoid


class LogisticRegression(BaseModel):
    def __init__(self, n_features, l2=0.001):
        self.l2 = l2
        self.params = {"w": np.zeros(n_features), "b": np.zeros(1)}

    def predict_proba_batch(self, X):
        return sigmoid(X @ self.params["w"] + self.params["b"][0])

    def compute_gradients(self, X, y):
        erreur = self.predict_proba_batch(X) - y
        return {"w": X.T @ erreur / len(y) + self.l2 * self.params["w"],
                "b": np.array([erreur.mean()])}
