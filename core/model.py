"""Classe abstraite commune à tous les modèles IA."""
from abc import ABC, abstractmethod
import numpy as np


def sigmoid(z):
    return np.where(z >= 0, 1 / (1 + np.exp(-np.abs(z))),
                    np.exp(-np.abs(z)) / (1 + np.exp(-np.abs(z))))


class BaseModel(ABC):
    """Tout modèle doit savoir : prédire un batch et calculer ses gradients."""

    params: dict

    @abstractmethod
    def predict_proba_batch(self, X) -> np.ndarray:
        """Probabilité d'infection pour chaque ligne de X."""

    @abstractmethod
    def compute_gradients(self, X, y) -> dict:
        """Gradient de la perte par rapport à chaque paramètre."""

    def loss(self, X, y) -> float:
        p = np.clip(self.predict_proba_batch(X), 1e-9, 1 - 1e-9)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    def predict(self, x) -> float:
        """predict() : probabilité d'infection pour UN patient."""
        return float(self.predict_proba_batch(np.atleast_2d(x))[0])
