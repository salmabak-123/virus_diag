"""Interface clinique en ligne de commande (saisie validée)."""
from app.api import PLAGES


def _lire(nom):
    lo, hi = PLAGES.get(nom, (float("-inf"), float("inf")))
    while True:
        try:
            v = float(input(f"  {nom} [{lo} - {hi}] : ").replace(",", "."))
            if lo <= v <= hi:
                return v
        except ValueError:
            pass
        print("  Valeur invalide, réessayez.")


def run_cli(predictor):
    print("\n=== Interface clinique : saisie d'un patient ===")
    while True:
        patient = {nom: _lire(nom) for nom in predictor.feature_names}
        p = predictor.predict_proba(patient)
        print(f"\n  >>> Diagnostic : {predictor.diagnose(patient)}  (p = {p:.2f})\n")
        if input("Autre patient ? (o/n) ").strip().lower() != "o":
            break
