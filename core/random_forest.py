"""Forêt aléatoire (Random Forest) pour le diagnostic Infecté / Sain."""
import numpy as np
from core.model import BaseModel


class _Noeud:
    __slots__ = ("feature", "seuil", "gauche", "droite", "proba")

    def __init__(self, proba, feature=None, seuil=None, gauche=None, droite=None):
        self.proba, self.feature, self.seuil = proba, feature, seuil
        self.gauche, self.droite = gauche, droite


class _Arbre:
    """Arbre de décision binaire (critère de Gini)."""

    def __init__(self, max_depth, min_samples, n_features_split, rng):
        self.max_depth, self.min_samples = max_depth, min_samples
        self.n_features_split, self.rng = n_features_split, rng
        self.racine = None

    @staticmethod
    def _gini(y):
        if len(y) == 0:
            return 0.0
        p = y.mean()
        return 2 * p * (1 - p)

    def _meilleur_split(self, X, y):
        n, d = X.shape
        meilleur = (None, None, self._gini(y))
        features = self.rng.choice(d, size=min(self.n_features_split, d), replace=False)
        for f in features:
            for seuil in np.unique(np.percentile(X[:, f], [10, 25, 50, 75, 90])):
                masque = X[:, f] <= seuil
                nb_g = masque.sum()
                if nb_g == 0 or nb_g == n:
                    continue
                g = (nb_g * self._gini(y[masque]) +
                     (n - nb_g) * self._gini(y[~masque])) / n
                if g < meilleur[2]:
                    meilleur = (f, seuil, g)
        return meilleur[0], meilleur[1]

    def _construire(self, X, y, profondeur):
        proba = float(y.mean())
        if (profondeur >= self.max_depth or len(y) < self.min_samples
                or proba in (0.0, 1.0)):
            return _Noeud(proba)
        f, seuil = self._meilleur_split(X, y)
        if f is None:
            return _Noeud(proba)
        masque = X[:, f] <= seuil
        return _Noeud(proba, f, seuil,
                      self._construire(X[masque], y[masque], profondeur + 1),
                      self._construire(X[~masque], y[~masque], profondeur + 1))

    def fit(self, X, y):
        self.racine = self._construire(X, y, 0)
        return self

    def _predire_un(self, x):
        n = self.racine
        while n.feature is not None:
            n = n.gauche if x[n.feature] <= n.seuil else n.droite
        return n.proba

    def predict_proba(self, X):
        return np.array([self._predire_un(x) for x in X])


class RandomForest(BaseModel):
    """Ensemble d'arbres entraînés sur des échantillons bootstrap."""

    def __init__(self, n_features, n_trees=25, max_depth=6, min_samples=5, seed=0):
        self.n_trees, self.max_depth, self.min_samples = n_trees, max_depth, min_samples
        self.n_features_split = max(1, int(np.sqrt(n_features)))
        self.rng = np.random.default_rng(seed)
        self.params = {}          # pas de paramètres à descente de gradient
        self.arbres = []

    def fit(self, X, y):
        n = len(y)
        self.arbres = []
        for _ in range(self.n_trees):
            idx = self.rng.integers(0, n, n)          # bootstrap
            arbre = _Arbre(self.max_depth, self.min_samples,
                           self.n_features_split, self.rng)
            self.arbres.append(arbre.fit(X[idx], y[idx]))
        return self

    def predict_proba_batch(self, X):
        return np.mean([a.predict_proba(X) for a in self.arbres], axis=0)

    def compute_gradients(self, X, y):
        return {}                 # pas de gradient : l'apprentissage se fait via fit()