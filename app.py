"""TalentInsight Analytics - application Flask (port 5002)."""
import json

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

import ml_pipeline as mp

# Les fichiers statiques sont dans « public/ » (dossier servi directement par Vercel)
app = Flask(__name__, static_folder="public", static_url_path="")


def load_artifacts():
    """Charge les modeles et resultats enregistres ; entraine une seule fois s'ils sont absents."""
    if not mp.artifacts_ready():
        mp.train_all()
    results = json.loads(mp.RESULTS_PATH.read_text(encoding="utf-8"))
    return (results, joblib.load(mp.PREP_PATH), joblib.load(mp.REG_PATH), joblib.load(mp.CLF_PATH))


RESULTS, PREP, REG, CLF = load_artifacts()


def page(template, active, **kw):
    return render_template(template, r=RESULTS, active=active, options=mp.CATEGORIES,
                           ranges=RESULTS["sim"]["ranges"], **kw)


@app.route("/")
def index():
    return page("index.html", "index")


@app.route("/eda")
def eda():
    return page("eda.html", "eda")


@app.route("/preprocessing")
def preprocessing():
    return page("preprocessing.html", "preprocessing")


@app.route("/regression")
def regression():
    return page("regression.html", "regression")


@app.route("/classification")
def classification():
    return page("classification.html", "classification")


@app.route("/simulation")
def simulation():
    return page("simulation.html", "simulation")


@app.route("/apropos")
def apropos():
    return page("apropos.html", "apropos")


def parse_profile(data):
    """Valide et convertit le profil saisi. Leve ValueError avec un message clair."""
    row = {}
    for col, allowed in mp.CATEGORIES.items():
        if data.get(col) not in allowed:
            raise ValueError(f"Valeur invalide pour « {col} ».")
        row[col] = data[col]
    for col, (lo, hi) in RESULTS["sim"]["ranges"].items():
        try:
            v = float(str(data.get(col, "")).replace(",", "."))
        except ValueError:
            raise ValueError(f"« {col} » doit etre un nombre.")
        if not lo <= v <= hi:
            raise ValueError(f"« {col} » doit etre compris entre {lo} et {hi}.")
        row[col] = v
    if row["anciennete_entreprise"] > row["experience_ans"]:
        raise ValueError("L'anciennete dans l'entreprise ne peut pas depasser l'experience totale.")
    if row["experience_ans"] > row["age"] - 16:
        raise ValueError("L'experience professionnelle est incoherente avec l'age.")
    return row


@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        row = parse_profile(request.get_json(force=True, silent=True) or {})
    except ValueError as e:
        return jsonify(error=str(e)), 400
    X = PREP.transform(pd.DataFrame([row]))  # meme preprocessing qu'a l'entrainement
    salary = float(REG.predict(X)[0])
    label = int(CLF.predict(X)[0])
    proba = float(CLF.predict_proba(X)[0][1]) if hasattr(CLF, "predict_proba") else None
    ratio = (row["satisfaction_travail"] + row["satisfaction_manager"]) / 2
    sim = RESULTS["sim"]
    notes = []
    if ratio < sim["ref_ratio_mean"] - 0.3:
        notes.append(f"Satisfaction moyenne ({ratio:.1f}/5) inferieure a la moyenne de l'echantillon ({sim['ref_ratio_mean']:.1f}/5).")
    elif ratio > sim["ref_ratio_mean"] + 0.3:
        notes.append(f"Satisfaction moyenne ({ratio:.1f}/5) superieure a la moyenne de l'echantillon ({sim['ref_ratio_mean']:.1f}/5).")
    if row["heures_sup_mois"] > sim["ref_overtime_median"] + 5:
        notes.append(f"Heures supplementaires ({row['heures_sup_mois']:.0f} h/mois) au-dessus de la mediane ({sim['ref_overtime_median']:.0f} h).")
    if proba is None:
        text = "Le modele predit un risque de depart." if label else "Le modele predit que l'employe reste."
    elif proba >= 0.6:
        text = "Le modele estime un risque eleve de depart."
    elif proba >= 0.4:
        text = "Le modele est hesitant : risque modere, sans conclusion nette."
    else:
        text = "Le modele estime un risque faible de depart."
    return jsonify(salary=round(salary), attrition=label, proba=None if proba is None else round(proba, 3),
                   model=sim["classifier"], explanation=text, notes=notes)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5002, debug=False)
