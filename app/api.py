"""API simulée (sans serveur) : routes GET /health et POST /diagnose."""

PLAGES = {"age": (0, 120), "temperature": (33, 43), "frequence_cardiaque": (30, 220),
          "saturation_o2": (50, 100), "globules_blancs": (0, 60), "crp": (0, 500),
          "toux": (0, 1), "perte_odorat": (0, 1), "maux_de_tete": (0, 1)}


class ClinicalAPI:
    def __init__(self, predictor):
        self.predictor = predictor

    def handle(self, methode, route, payload=None):
        if (methode, route) == ("GET", "/health"):
            return {"status": 200, "message": "ok"}
        if (methode, route) == ("POST", "/diagnose"):
            return self._diagnose(payload or {})
        return {"status": 404, "erreur": "route inconnue"}

    def _diagnose(self, payload):
        for champ in self.predictor.feature_names:
            if champ not in payload:
                return {"status": 422, "erreur": f"champ manquant : {champ}"}
            lo, hi = PLAGES.get(champ, (float("-inf"), float("inf")))
            if not lo <= payload[champ] <= hi:
                return {"status": 422,
                        "erreur": f"{champ} hors plage [{lo}, {hi}]"}
        proba = self.predictor.predict_proba(payload)
        return {"status": 200, "diagnostic": self.predictor.diagnose(payload),
                "probabilite_infection": round(proba, 3)}
