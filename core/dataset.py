import csv
import numpy as np


class MinMaxScaler:
    """Ramène chaque variable dans [0, 1]."""

    def fit(self, X):
        self.min_ = X.min(axis=0)
        self.range_ = X.max(axis=0) - self.min_
        self.range_[self.range_ == 0] = 1.0
        return self

    def transform(self, X):
        return (np.asarray(X, dtype=float) - self.min_) / self.range_


def train_test_split(X, y, test_size=0.2, seed=0, stratify=True):
    rng = np.random.default_rng(seed)
    if not stratify:
        idx = rng.permutation(len(y))
        k = int(len(y) * test_size)
        test, train = idx[:k], idx[k:]
    else:
        test, train = [], []
        for classe in np.unique(y):
            idx = rng.permutation(np.where(y == classe)[0])
            k = int(len(idx) * test_size)
            test.extend(idx[:k]); train.extend(idx[k:])
        test, train = np.array(test), np.array(train)
    return X[train], X[test], y[train], y[test]


class Dataset:
    def __init__(self, path, target="infecte"):
        self.path = path
        self.target = target
        self.feature_names, self.X, self.y = [], None, None

    def load(self):
        with open(self.path, newline="", encoding="utf-8") as f:
            lecteur = csv.DictReader(f)
            lignes = list(lecteur)
            self.feature_names = [c for c in lecteur.fieldnames if c != self.target]
        self.X = np.array([[float(l[c]) for c in self.feature_names] for l in lignes])
        self.y = np.array([int(l[self.target]) for l in lignes])
        return self

    def split(self, test_size=0.2, seed=0):
        return train_test_split(self.X, self.y, test_size, seed, stratify=True)

    def __len__(self):
        return len(self.y)
