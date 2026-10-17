"""Entraîne le modèle : boucle d'apprentissage par mini-batchs."""
import numpy as np


class Trainer:
    def __init__(self, optimizer, epochs=150, batch_size=32, seed=0, verbose=0):
        self.optimizer = optimizer
        self.epochs, self.batch_size = epochs, batch_size
        self.rng = np.random.default_rng(seed)
        self.verbose = verbose
        self.history = []

    def train(self, model, X, y):
        n = len(y)
        for epoch in range(1, self.epochs + 1):
            ordre = self.rng.permutation(n)
            for debut in range(0, n, self.batch_size):
                idx = ordre[debut:debut + self.batch_size]
                grads = model.compute_gradients(X[idx], y[idx])
                self.optimizer.step(model.params, grads)
            perte = model.loss(X, y)
            self.history.append(perte)
            if self.verbose and epoch % self.verbose == 0:
                print(f"  epoch {epoch:4d}  perte = {perte:.4f}")
        return model
