"""Usage :  python main.py            -> entraîne, évalue, démo
            python main.py --cli      -> + interface clinique interactive
            python main.py --regen    -> régénère patient_data.csv"""
import sys
from pathlib import Path

from data.generate_data import generate
from core.dataset import Dataset
from core.logistic_regression import LogisticRegression
from core.neural_network import NeuralNetwork
from core.optimizer import Adam
from pipeline.trainer import Trainer
from pipeline.evaluator import Evaluator
from pipeline.predictor import ClinicalPredictor
from utils.preprocessing import MinMaxScaler
from app.api import ClinicalAPI
from app.interface_clinique import run_cli

CSV = Path(__file__).parent / "data" / "patient_data.csv"

PATIENT_MALADE = {"age": 58, "temperature": 39.1, "frequence_cardiaque": 102,
                  "saturation_o2": 91.5, "globules_blancs": 13.2, "crp": 62.0,
                  "toux": 1, "perte_odorat": 1, "maux_de_tete": 1}
PATIENT_SAIN = {"age": 29, "temperature": 36.8, "frequence_cardiaque": 72,
                "saturation_o2": 98.0, "globules_blancs": 6.5, "crp": 3.0,
                "toux": 0, "perte_odorat": 0, "maux_de_tete": 0}


def main():
    if "--regen" in sys.argv or not CSV.exists():
        generate(CSV)
        print(f"Données générées : {CSV.name}")

    ds = Dataset(CSV).load()
    X_tr, X_te, y_tr, y_te = ds.split()
    scaler = MinMaxScaler().fit(X_tr)
    X_tr, X_te = scaler.transform(X_tr), scaler.transform(X_te)
    d = X_tr.shape[1]
    print(f"{len(ds)} patients, {d} variables, {y_tr.mean():.0%} d'infectés (train)\n")

    modeles = {"Régression logistique": LogisticRegression(d),
               "Réseau de neurones": NeuralNetwork(d, hidden=(16, 8))}
    predictors, scores = {}, {}
    for nom, modele in modeles.items():
        Trainer(Adam(lr=0.01), epochs=150, batch_size=32).train(modele, X_tr, y_tr)
        res = Evaluator().evaluate(modele, X_te, y_te)
        print(Evaluator.report(nom, res))
        predictors[nom] = ClinicalPredictor(modele, scaler, ds.feature_names)
        scores[nom] = res["f1"]

    print("\n--- Diagnostics ---")
    for etiquette, patient in (("Patient malade", PATIENT_MALADE),
                               ("Patient sain", PATIENT_SAIN)):
        for nom, p in predictors.items():
            print(f"{etiquette:15s} | {nom:22s} -> {p.diagnose(patient)}")

    meilleur = max(scores, key=scores.get)
    api = ClinicalAPI(predictors[meilleur])
    print(f"\nAPI ({meilleur}) :")
    print(" GET  /health   ->", api.handle("GET", "/health"))
    print(" POST /diagnose ->", api.handle("POST", "/diagnose", PATIENT_MALADE))
    print(" POST /diagnose ->", api.handle("POST", "/diagnose", {"age": 40}))

    if "--cli" in sys.argv:
        run_cli(predictors[meilleur])


if __name__ == "__main__":
    main()
