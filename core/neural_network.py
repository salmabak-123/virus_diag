"""Réseau de neurones multicouche (ReLU + sortie sigmoïde), rétropropagation."""
import numpy as np
from core.model import BaseModel, sigmoid


class NeuralNetwork(BaseModel):
    def __init__(self, n_features, hidden=(16, 8), seed=0):
        rng = np.random.default_rng(seed)
        tailles = [n_features, *hidden, 1]
        self.n_layers = len(tailles) - 1
        self.params = {}
        for i in range(self.n_layers):
            self.params[f"W{i}"] = rng.normal(0, np.sqrt(2 / tailles[i]),
                                              (tailles[i], tailles[i + 1]))
            self.params[f"b{i}"] = np.zeros(tailles[i + 1])

    def _forward(self, X):
        activations, pre = [X], []
        a = X
        for i in range(self.n_layers):
            z = a @ self.params[f"W{i}"] + self.params[f"b{i}"]
            pre.append(z)
            a = sigmoid(z) if i == self.n_layers - 1 else np.maximum(0, z)
            activations.append(a)
        return activations, pre

    def predict_proba_batch(self, X):
        return self._forward(X)[0][-1].ravel()

    def compute_gradients(self, X, y):
        activations, pre = self._forward(X)
        n = len(y)
        delta = (activations[-1].ravel() - y).reshape(-1, 1)
        grads = {}
        for i in reversed(range(self.n_layers)):
            grads[f"W{i}"] = activations[i].T @ delta / n
            grads[f"b{i}"] = delta.mean(axis=0)
            if i > 0:
                delta = (delta @ self.params[f"W{i}"].T) * (pre[i - 1] > 0)
        return grads
