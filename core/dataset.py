"""Chargement du CSV patients (module csv standard + numpy)."""
import csv
import numpy as np
from utils.preprocessing import train_test_split


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
