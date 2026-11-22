"""TalentInsight Analytics - pipeline Machine Learning.

Lancer `python ml_pipeline.py` pour (re)entrainer les modeles, produire les
graphiques (static/images) et le fichier de resultats (models/results.json).
L'application Flask ne fait ensuite que LIRE ces artefacts.
"""
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

try:  # graphiques : nécessaires seulement pour (ré)entraîner en local, pas pour afficher le site
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:  # ex. sur Vercel, où matplotlib/seaborn ne sont pas installés
    plt = sns = None
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score,
                             mean_absolute_error, mean_squared_error, precision_score,
                             r2_score, recall_score)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent
DATA_PATH = BASE / "data" / "rh_employes.csv"
MODELS_DIR = BASE / "models"
IMG_DIR = BASE / "static" / "images"
RESULTS_PATH = MODELS_DIR / "results.json"
PREP_PATH = MODELS_DIR / "preprocessing.pkl"
REG_PATH = MODELS_DIR / "regression_model.pkl"
CLF_PATH = MODELS_DIR / "classification_model.pkl"

RANDOM_STATE = 42
SAMPLE_SIZE = 200

GREEN_D, GREEN, GREEN_L, GREY = "#145A3A", "#239B69", "#DDF3E7", "#F4F7F5"
ACCENT = "#E8A33D"

CATEGORIES = {
    "genre": ["Femme", "Homme"],
    "ville": ["Agadir", "Casablanca", "Marrakech", "Rabat", "Tanger"],
    "diplome": ["Bac+2", "Bac+3", "Bac+5", "Doctorat"],
    "departement": ["Finance", "Informatique", "Marketing", "Operations", "Ressources Humaines", "Ventes"],
    "niveau_poste": ["Junior", "Intermediaire", "Senior", "Manager"],
}
CAT_COLS = list(CATEGORIES)
MISSING_COLS = ["anciennete_entreprise", "nb_formations_annee", "heures_sup_mois",
                "satisfaction_travail", "satisfaction_manager"]
RAW_FEATURES = ["age", "genre", "ville", "diplome", "departement", "niveau_poste", "experience_ans",
                "anciennete_entreprise", "nb_formations_annee", "heures_sup_mois",
                "satisfaction_travail", "satisfaction_manager"]
FEATURES = RAW_FEATURES + ["ratio_satisfaction"]
SCALE_COLS = ["age", "experience_ans", "anciennete_entreprise", "heures_sup_mois",
              "satisfaction_travail", "satisfaction_manager", "nb_formations_annee",
              "ratio_satisfaction"]
SIM_MODEL = "Random Forest"
LEVEL_ORDER = ["Junior", "Intermediaire", "Senior", "Manager"]


class HRPreprocessor(BaseEstimator, TransformerMixin):
    """Imputation (mediane) -> LabelEncoder -> ratio_satisfaction -> StandardScaler.

    Tout ce qui est appris (medianes, scaler) l'est dans fit(), donc uniquement sur
    les donnees d'entrainement : aucune fuite de donnees (data leakage).
    """

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.medians_ = X[MISSING_COLS].median().to_dict()
        self.encoders_ = {c: LabelEncoder().fit(CATEGORIES[c]) for c in CAT_COLS}
        self.scaler_ = StandardScaler().fit(self._prepare(X)[SCALE_COLS])
        return self

    def _prepare(self, X):
        d = pd.DataFrame(X)[RAW_FEATURES].copy().fillna(self.medians_)
        for c in CAT_COLS:
            d[c] = self.encoders_[c].transform(d[c])
        d["ratio_satisfaction"] = (d["satisfaction_travail"] + d["satisfaction_manager"]) / 2
        return d[FEATURES].astype(float)

    def transform(self, X):
        d = self._prepare(X)
        d[SCALE_COLS] = self.scaler_.transform(d[SCALE_COLS])
        return d


# --------------------------------------------------------------------------- utils
def fmt(x, nd=0):
    return f"{x:,.{nd}f}".replace(",", " ")


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(IMG_DIR / f"{name}.png", dpi=110, facecolor="white")
    plt.close(fig)


def _f(x, nd=4):
    return round(float(x), nd)


def _bar_labels(ax, fmt_="{:.1f}"):
    for p in ax.patches:
        h = p.get_height()
        if np.isfinite(h) and h > 0:
            ax.annotate(fmt_.format(h), (p.get_x() + p.get_width() / 2, h), ha="center", va="bottom", fontsize=9)


def _qual_r2(r2):
    return ("excellente" if r2 >= 0.9 else "bonne" if r2 >= 0.75 else "moderee" if r2 >= 0.5 else "faible")


FEATURE_ADVICE = {
    "satisfaction_travail": "mesurer regulierement la satisfaction au travail et traiter les causes d'insatisfaction",
    "satisfaction_manager": "accompagner et former les managers (la relation manager-employe)",
    "ratio_satisfaction": "suivre un indicateur global de satisfaction (moyenne travail + manager)",
    "heures_sup_mois": "surveiller la charge de travail et les heures supplementaires",
    "anciennete_entreprise": "suivre les parcours selon l'anciennete (integration, evolution)",
    "nb_formations_annee": "proposer un plan de formation et de developpement",
    "age": "adapter les actions de fidelisation aux differentes tranches d'age",
    "experience_ans": "proposer des perspectives de carriere adaptees au niveau d'experience",
    "niveau_poste": "clarifier les perspectives de promotion selon le niveau de poste",
    "departement": "analyser plus finement les departements les plus touches",
    "diplome": "valoriser les parcours selon le niveau d'etudes",
    "ville": "tenir compte des specificites locales (ville)",
    "genre": "verifier l'equite et l'inclusion",
}


# --------------------------------------------------------------------------- pipeline
def train_all():
    MODELS_DIR.mkdir(exist_ok=True)
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", rc={"axes.edgecolor": "#cfd8d3", "axes.titleweight": "bold"})
    sns.set_palette([GREEN_D, GREEN, "#7CCBA2", ACCENT, "#6C8EAD", "#B5654B"])

    # 1-2. Chargement (CSV original, jamais modifie) + exploration
    raw = pd.read_csv(DATA_PATH)
    n_raw = len(raw)
    missing_raw = raw.isnull().sum()
    dup_full = int(raw.duplicated().sum())
    dup_no_id = int(raw.drop(columns="id").duplicated().sum())

    # 3. Doublons : l'id etant unique, on les detecte sur toutes les colonnes SAUF id
    clean = raw.drop_duplicates(subset=[c for c in raw.columns if c != "id"], keep="first").reset_index(drop=True)
    outliers = {
        "age_lt_18": int((clean.age < 18).sum()), "age_gt_65": int((clean.age > 65).sum()),
        "salaire_negatif": int((clean.salaire_mensuel_mad < 0).sum()),
        "salaire_nul": int((clean.salaire_mensuel_mad == 0).sum()),
    }

    # Echantillon representatif (stratifie sur attrition), 200 employes max
    n_sample = min(SAMPLE_SIZE, len(clean))
    sample = train_test_split(clean, train_size=n_sample, stratify=clean["attrition"],
                              random_state=RANDOM_STATE)[0].reset_index(drop=True)

    # 4. Split AVANT d'apprendre le preprocessing (pas de fuite)
    X = sample[RAW_FEATURES]
    X_tr, X_te, yr_tr, yr_te, yc_tr, yc_te = train_test_split(
        X, sample["salaire_mensuel_mad"], sample["attrition"],
        test_size=0.2, random_state=RANDOM_STATE, stratify=sample["attrition"])
    prep = HRPreprocessor().fit(X_tr)
    Xtr, Xte = prep.transform(X_tr), prep.transform(X_te)
    cv_pipe = lambda est: Pipeline([("prep", HRPreprocessor()), ("model", clone(est))])

    res = {"meta": {"random_state": RANDOM_STATE, "n_raw": n_raw, "n_clean": len(clean),
                    "n_sample": n_sample, "n_train": len(X_tr), "n_test": len(X_te),
                    "n_cols": raw.shape[1], "features": FEATURES}}

    # ---------------------------------------------------------------- EDA + graphiques
    sd = sample.copy()
    stay, leave = int((clean.attrition == 0).sum()), int((clean.attrition == 1).sum())
    res["kpi"] = {"n_raw": n_raw, "n_clean": len(clean), "n_sample": n_sample, "n_cols": raw.shape[1],
                  "mean_salary": fmt(clean.salaire_mensuel_mad.mean()),
                  "attr_rate": round(100 * leave / len(clean), 1), "stay": stay, "leave": leave}

    head = json.loads(raw.head(8).to_json(orient="records"))
    desc = raw.describe().T.round(2).reset_index().rename(columns={"index": "variable"})
    res["eda"] = {
        "head": head, "columns": list(raw.columns), "shape": [n_raw, raw.shape[1]],
        "dtypes": [{"col": c, "dtype": str(t).replace("str", "object"), "missing": int(missing_raw[c])}
                   for c, t in raw.dtypes.items()],
        "describe": json.loads(desc.to_json(orient="records")),
        "missing_total": int(missing_raw.sum()), "dup_full": dup_full, "dup_no_id": dup_no_id,
        "attr_counts_sample": {"0": int((sd.attrition == 0).sum()), "1": int((sd.attrition == 1).sum())},
    }
    interp = {}

    # missing
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ms = missing_raw[missing_raw > 0].sort_values()
    ax.barh(ms.index, ms.values, color=GREEN)
    for i, v in enumerate(ms.values):
        ax.text(v + 0.1, i, str(v), va="center")
    ax.set_title("Valeurs manquantes par colonne (CSV original, 612 lignes)")
    ax.set_xlabel("Nombre de valeurs manquantes")
    _save(fig, "missing_bar")
    interp["missing"] = (f"Le CSV contient {int(missing_raw.sum())} valeurs manquantes au total, reparties sur "
                         f"{len(ms)} colonnes numeriques ({', '.join(f'{k}: {v}' for k, v in ms.sort_values(ascending=False).items())}).")

    # attrition countplot
    fig, ax = plt.subplots(figsize=(5, 3.8))
    sns.countplot(x="attrition", data=sd, ax=ax, palette=[GREEN, ACCENT])
    for p in ax.patches:
        ax.annotate(f"{int(p.get_height())} ({100 * p.get_height() / len(sd):.1f}%)",
                    (p.get_x() + p.get_width() / 2, p.get_height()), ha="center", va="bottom")
    ax.set_xticklabels(["0 - Reste", "1 - Quitte"])
    ax.set_title("Repartition de l'attrition (echantillon)")
    ax.set_xlabel("")
    _save(fig, "attrition_count")
    p1 = 100 * sd.attrition.mean()
    interp["attrition"] = (f"Dans l'echantillon de {len(sd)} employes, {p1:.1f}% quittent l'entreprise et {100 - p1:.1f}% restent : "
                           "les classes sont quasi equilibrees, mais on analyse tout de meme les metriques par classe.")

    # salary distribution
    fig, ax = plt.subplots(figsize=(7, 3.8))
    sns.histplot(sd.salaire_mensuel_mad, kde=True, color=GREEN, ax=ax, bins=25)
    ax.set_title("Distribution des salaires mensuels (MAD)")
    ax.set_xlabel("Salaire mensuel (MAD)")
    ax.set_ylabel("Nombre d'employes")
    _save(fig, "salary_dist")
    sk = sd.salaire_mensuel_mad.skew()
    interp["salary"] = (f"Salaire moyen {fmt(sd.salaire_mensuel_mad.mean())} MAD, median {fmt(sd.salaire_mensuel_mad.median())} MAD. "
                        f"Coefficient d'asymetrie = {sk:.2f} : distribution "
                        + ("asymetrique a droite (quelques salaires eleves tirent la moyenne vers le haut)." if sk > 0.5
                           else "asymetrique a gauche." if sk < -0.5 else "approximativement symetrique.")
                        + " Un histogramme ne permet pas de conclure formellement a la normalite.")

    # salary by dept / level
    fig, ax = plt.subplots(figsize=(7, 4.2))
    order = sd.groupby("departement").salaire_mensuel_mad.median().sort_values().index
    sns.boxplot(data=sd, y="departement", x="salaire_mensuel_mad", order=order, color=GREEN, ax=ax)
    ax.set_title("Salaire selon le departement")
    ax.set_xlabel("Salaire mensuel (MAD)")
    ax.set_ylabel("")
    _save(fig, "salary_dept")
    med_d = sd.groupby("departement").salaire_mensuel_mad.median().sort_values()
    interp["salary_dept"] = (f"Mediane la plus haute : {med_d.index[-1]} ({fmt(med_d.iloc[-1])} MAD) ; la plus basse : "
                             f"{med_d.index[0]} ({fmt(med_d.iloc[0])} MAD). Ecart de {fmt(med_d.iloc[-1] - med_d.iloc[0])} MAD entre departements.")
    fig, ax = plt.subplots(figsize=(7, 4.2))
    sns.boxplot(data=sd, x="niveau_poste", y="salaire_mensuel_mad", order=LEVEL_ORDER, color=GREEN, ax=ax)
    ax.set_title("Salaire selon le niveau de poste")
    ax.set_xlabel("")
    ax.set_ylabel("Salaire mensuel (MAD)")
    _save(fig, "salary_level")
    med_l = sd.groupby("niveau_poste").salaire_mensuel_mad.median().reindex(LEVEL_ORDER)
    interp["salary_level"] = "Salaire median par niveau : " + " | ".join(f"{k} {fmt(v)} MAD" for k, v in med_l.items()) + "."

    # correlation heatmap
    num = sd.drop(columns=["id"]).select_dtypes("number")
    corr = num.corr()
    fig, ax = plt.subplots(figsize=(8, 6.2))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap=sns.light_palette(GREEN_D, as_cmap=True), ax=ax,
                annot_kws={"size": 8}, cbar_kws={"shrink": .8})
    ax.set_title("Correlations entre variables numeriques")
    _save(fig, "corr_heatmap")
    cs = corr["salaire_mensuel_mad"].drop(["salaire_mensuel_mad", "attrition"]).sort_values(key=abs, ascending=False)
    interp["corr"] = "Variables les plus correlees avec le salaire : " + ", ".join(f"{k} ({v:+.2f})" for k, v in cs.head(3).items()) + "."
    top_corr = [(k, float(v)) for k, v in cs.head(3).items()]

    # attrition by dept / satisfaction
    rd = sd.groupby("departement").attrition.mean().mul(100).sort_values()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(rd.index, rd.values, color=GREEN)
    for i, v in enumerate(rd.values):
        ax.text(v + 0.5, i, f"{v:.1f}%", va="center")
    ax.set_title("Taux d'attrition par departement (%)")
    ax.set_xlabel("Taux d'attrition (%)")
    _save(fig, "attr_dept")
    interp["attr_dept"] = (f"Taux le plus eleve : {rd.index[-1]} ({rd.iloc[-1]:.1f}%) ; le plus bas : {rd.index[0]} ({rd.iloc[0]:.1f}%). "
                           "Avec ~30 employes par departement dans l'echantillon, ces ecarts sont a interpreter avec prudence.")
    bins = pd.cut(sd.satisfaction_travail, [0.99, 2, 3, 4, 5.01], right=False, labels=["[1-2[", "[2-3[", "[3-4[", "[4-5]"])
    rs = sd.groupby(bins, observed=True).attrition.mean().mul(100)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(rs.index.astype(str), rs.values, color=[GREEN_D, GREEN, "#7CCBA2", ACCENT][: len(rs)])
    _bar_labels(ax, "{:.1f}%")
    ax.set_title("Taux d'attrition selon la satisfaction au travail")
    ax.set_xlabel("Satisfaction au travail")
    ax.set_ylabel("Taux d'attrition (%)")
    _save(fig, "attr_sat")
    interp["attr_sat"] = "Taux d'attrition par tranche de satisfaction : " + " | ".join(f"{k} : {v:.1f}%" for k, v in rs.items()) + "."
    r_sat = float(sd[["satisfaction_travail", "attrition"]].corr().iloc[0, 1])

    # scatter experience vs salary
    fig, ax = plt.subplots(figsize=(7, 4.2))
    sns.scatterplot(data=sd, x="experience_ans", y="salaire_mensuel_mad", hue="niveau_poste", hue_order=LEVEL_ORDER, ax=ax, s=40)
    ax.set_title("Experience vs salaire (par niveau de poste)")
    ax.set_xlabel("Experience (annees)")
    ax.set_ylabel("Salaire mensuel (MAD)")
    _save(fig, "scatter_exp")
    interp["scatter"] = f"Correlation experience / salaire : {sd.experience_ans.corr(sd.salaire_mensuel_mad):+.2f}."
    res["eda"]["interp"] = interp

    # ---------------------------------------------------------------- Preprocessing
    prepared_all = prep._prepare(sample[RAW_FEATURES])
    miss_before = sample[MISSING_COLS].isnull().sum()
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.9))
    xs = np.arange(len(MISSING_COLS))
    axs[0].bar(xs - 0.2, miss_before.values, 0.4, color=ACCENT, label="Avant")
    axs[0].bar(xs + 0.2, [0] * len(xs), 0.4, color=GREEN, label="Apres (mediane)")
    axs[0].set_xticks(xs)
    axs[0].set_xticklabels([c.replace("_", "\n") for c in MISSING_COLS], fontsize=8)
    axs[0].set_title("Valeurs manquantes (echantillon)")
    axs[0].legend()
    sns.histplot(sample.age, color=ACCENT, kde=True, ax=axs[1], label="age brut", stat="density", alpha=.5)
    sns.histplot(prep.transform(sample[RAW_FEATURES])["age"], color=GREEN, kde=True, ax=axs[1], label="age standardise", stat="density", alpha=.5)
    axs[1].set_title("Variable 'age' avant / apres StandardScaler")
    axs[1].legend()
    _save(fig, "before_after")
    ratio_d = prepared_all.assign(attrition=sample.attrition.values)
    corr_ratio = {c: float(ratio_d[c].corr(ratio_d.attrition)) for c in ["satisfaction_travail", "satisfaction_manager", "ratio_satisfaction"]}
    res["prep"] = {
        "dup_removed": n_raw - len(clean), "dup_full": dup_full, "missing_total": int(missing_raw.sum()),
        "medians": {k: _f(v, 2) for k, v in prep.medians_.items()}, "outliers": outliers,
        "age_range": [int(clean.age.min()), int(clean.age.max())],
        "salary_range": [int(clean.salaire_mensuel_mad.min()), int(clean.salaire_mensuel_mad.max())],
        "missing_after_sample": int(prepared_all.isnull().sum().sum()),
        "encoders": {c: {k: int(i) for i, k in enumerate(prep.encoders_[c].classes_)} for c in CAT_COLS},
        "scale_cols": SCALE_COLS, "corr_ratio": {k: round(v, 3) for k, v in corr_ratio.items()},
        "scaler_mean": {c: _f(m, 2) for c, m in zip(SCALE_COLS, prep.scaler_.mean_)},
    }

    # ---------------------------------------------------------------- Regression
    reg = LinearRegression().fit(Xtr, yr_tr)
    pr = reg.predict(Xte)
    mae, mse = mean_absolute_error(yr_te, pr), mean_squared_error(yr_te, pr)
    rmse, r2 = float(np.sqrt(mse)), r2_score(yr_te, pr)
    r2_train = r2_score(yr_tr, reg.predict(Xtr))
    cvr = cross_val_score(cv_pipe(LinearRegression()), X_tr, yr_tr, cv=5, scoring="r2")
    resid = np.asarray(yr_te) - pr
    sh_w, sh_p = stats.shapiro(resid)

    fig, ax = plt.subplots(figsize=(5.6, 5))
    ax.scatter(yr_te, pr, color=GREEN, edgecolor=GREEN_D, s=45, alpha=.85)
    lim = [min(yr_te.min(), pr.min()), max(yr_te.max(), pr.max())]
    ax.plot(lim, lim, "--", color=ACCENT, label="Droite ideale y = x")
    ax.set_title("Salaires reels vs salaires predits (test)")
    ax.set_xlabel("Salaire reel (MAD)")
    ax.set_ylabel("Salaire predit (MAD)")
    ax.legend()
    _save(fig, "reg_actual_pred")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(resid, kde=True, color=GREEN, ax=ax, bins=15)
    ax.axvline(0, color=ACCENT, ls="--")
    ax.set_title("Distribution des residus (reel - predit)")
    ax.set_xlabel("Residu (MAD)")
    _save(fig, "reg_residuals")

    gap = r2_train - cvr.mean()
    ans_r = [
        ("R1", "R² sur le jeu de test - le modele explique-t-il bien la variance ?",
         f"R² test = {r2:.3f} : le modele explique {100 * r2:.1f}% de la variance du salaire. Performance {_qual_r2(r2)}."),
        ("R2", "Interpretation du RMSE",
         f"RMSE = {fmt(rmse)} MAD (MAE = {fmt(mae)} MAD) : en moyenne, la prediction s'ecarte d'environ {fmt(rmse)} MAD du salaire reel "
         f"(salaire moyen de l'echantillon : {fmt(sd.salaire_mensuel_mad.mean())} MAD, soit {100 * rmse / sd.salaire_mensuel_mad.mean():.0f}%)."),
        ("R3", "R² test vs R² de la validation croisee - overfitting / underfitting ?",
         f"R² train = {r2_train:.3f}, R² test = {r2:.3f}, R² CV = {cvr.mean():.3f} ± {cvr.std():.3f}. "
         + ("L'ecart train/CV est faible : pas de signe net de surapprentissage. " if gap < 0.1 else "L'ecart train/CV est notable : risque de surapprentissage. ")
         + ("Le R² reste " + _qual_r2(cvr.mean()) + ", donc pas de signe majeur de sous-apprentissage."
            if cvr.mean() >= 0.5 else "Le R² est bas : sous-apprentissage possible (modele lineaire trop simple ou variables insuffisantes).")
         + f" L'ecart-type de la CV ({cvr.std():.3f}) mesure la variabilite selon les folds ; le jeu de test ne contient que {len(X_te)} employes."),
        ("R4", "Variables ayant le plus d'impact sur le salaire",
         "Correlations avec le salaire (EDA) : " + ", ".join(f"{k} ({v:+.2f})" for k, v in top_corr)
         + ". Variables categorielles : le niveau de poste differencie nettement les salaires (" + " | ".join(f"{k} {fmt(v)}" for k, v in med_l.items()) + " MAD, medianes)."),
        ("R5", "Reel vs predit - alignement sur y = x",
         f"Plus les points sont proches de la droite y = x, meilleure est la prediction. Ici R² = {r2:.3f} et MAE = {fmt(mae)} MAD : "
         + ("les points sont globalement bien alignes." if r2 >= 0.75 else "l'alignement est moyen.")),
        ("R6", "Distribution des residus (bonus)",
         f"Moyenne des residus = {fmt(resid.mean())} MAD, asymetrie = {stats.skew(resid):.2f}. Test de Shapiro-Wilk : W = {sh_w:.3f}, p = {sh_p:.3f} "
         + ("(normalite non rejetee a 5% : hypothese de normalite des erreurs plausible)." if sh_p > 0.05
            else "(normalite rejetee a 5% : l'hypothese de normalite des erreurs n'est pas verifiee, les intervalles de confiance classiques sont a manier avec prudence).")
         + f" Test base sur seulement {len(resid)} residus."),
    ]
    res["reg"] = {"mae": _f(mae, 2), "mse": _f(mse, 2), "rmse": _f(rmse, 2), "r2": _f(r2), "r2_train": _f(r2_train),
                  "cv_scores": [_f(s) for s in cvr], "cv_mean": _f(cvr.mean()), "cv_std": _f(cvr.std()),
                  "shapiro_p": _f(sh_p), "answers": [{"id": a, "q": q, "a": t} for a, q, t in ans_r]}

    # ---------------------------------------------------------------- Classification
    models = {"SVM": SVC(kernel="rbf", random_state=RANDOM_STATE), "KNN": KNeighborsClassifier(n_neighbors=5),
              "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
              "Random Forest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)}
    cls, cms, fitted = {}, {}, {}
    for name, m in models.items():
        m.fit(Xtr, yc_tr)
        pred = m.predict(Xte)
        cm = confusion_matrix(yc_te, pred, labels=[0, 1])
        tn, fp, fn, tp = (int(v) for v in cm.ravel())
        rep = classification_report(yc_te, pred, output_dict=True, zero_division=0)
        cv = cross_val_score(cv_pipe(m), X_tr, yc_tr, cv=5, scoring="f1_macro")
        cls[name] = {
            "accuracy": _f(accuracy_score(yc_te, pred)),
            "precision": _f(precision_score(yc_te, pred, average="macro", zero_division=0)),
            "recall": _f(recall_score(yc_te, pred, average="macro", zero_division=0)),
            "f1": _f(f1_score(yc_te, pred, average="macro", zero_division=0)),
            "recall_1": _f(rep["1"]["recall"]), "precision_1": _f(rep["1"]["precision"]), "f1_1": _f(rep["1"]["f1-score"]),
            "f1_0": _f(rep["0"]["f1-score"]), "recall_0": _f(rep["0"]["recall"]),
            "tn": tn, "fp": fp, "fn": fn, "tp": tp,
            "report": [{"label": k, **{kk: _f(vv, 3) for kk, vv in rep[k].items() if kk != "support"}, "support": int(rep[k]["support"])}
                       for k in ["0", "1", "macro avg", "weighted avg"]],
            "cv_scores": [_f(s) for s in cv], "cv_mean": _f(cv.mean()), "cv_std": _f(cv.std()),
            "train_f1": _f(f1_score(yc_tr, m.predict(Xtr), average="macro")),
        }
        cms[name], fitted[name] = cm, m
    names = list(cls)
    best_f1 = max(names, key=lambda n: (cls[n]["f1"], cls[n]["recall_1"]))
    best_rec = max(names, key=lambda n: (cls[n]["recall_1"], cls[n]["f1"]))
    best_acc = max(names, key=lambda n: cls[n]["accuracy"])
    least_fn = min(names, key=lambda n: cls[n]["fn"])
    most_stable = min(names, key=lambda n: cls[n]["cv_std"])

    # graphiques
    fig, axs = plt.subplots(1, 4, figsize=(15, 3.6))
    for ax, n in zip(axs, names):
        sns.heatmap(cms[n], annot=True, fmt="d", cmap=sns.light_palette(GREEN_D, as_cmap=True), cbar=False, ax=ax,
                    xticklabels=["Reste", "Quitte"], yticklabels=["Reste", "Quitte"])
        ax.set_title(n)
        ax.set_xlabel("Predit")
        ax.set_ylabel("Reel")
    _save(fig, "cm_all")
    for key, title, fname in [("f1", "F1-score macro (test)", "f1_compare"), ("recall_1", "Recall classe 1 - employes qui quittent (test)", "recall_compare")]:
        fig, ax = plt.subplots(figsize=(6, 3.8))
        vals = [cls[n][key] for n in names]
        ax.bar(names, vals, color=[ACCENT if n == (best_f1 if key == "f1" else best_rec) else GREEN for n in names])
        _bar_labels(ax, "{:.3f}")
        ax.set_ylim(0, 1.1)
        ax.set_title(title, fontsize=11)
        _save(fig, fname)
    imp = pd.Series(fitted["Random Forest"].feature_importances_, index=FEATURES).sort_values()
    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.barh(imp.index, imp.values, color=GREEN)
    ax.set_title("Feature importances - Random Forest")
    ax.set_xlabel("Importance")
    _save(fig, "feat_importance")
    top_imp = list(imp.sort_values(ascending=False).items())[:3]

    # reponses
    bf = cls[best_f1]
    c1_same = best_f1 == best_rec
    avg_f1_0, avg_f1_1 = np.mean([cls[n]["f1_0"] for n in names]), np.mean([cls[n]["f1_1"] for n in names])
    dt, rf = cls["Decision Tree"], cls["Random Forest"]
    near_random = bf["f1"] < 0.6
    advice = [FEATURE_ADVICE[k] for k, _ in top_imp[:3]]
    prop = lambda s: {str(k): round(100 * v, 1) for k, v in s.value_counts(normalize=True).sort_index().items()}
    ans_c = [
        ("C1", "Meilleur modele selon F1 macro vs Recall classe 1",
         f"F1 macro : {best_f1} ({cls[best_f1]['f1']:.3f}). Recall classe 1 : {best_rec} ({cls[best_rec]['recall_1']:.3f}). "
         + ("Les deux conclusions sont identiques." if c1_same else "Les conclusions different : le meilleur F1 n'est pas le meilleur Recall.")),
        ("C2", "Pourquoi une Accuracy elevee peut etre trompeuse",
         f"Meilleure Accuracy : {best_acc} ({cls[best_acc]['accuracy']:.3f}), mais son Recall classe 1 est de {cls[best_acc]['recall_1']:.3f} avec "
         f"{cls[best_acc]['fn']} faux negatif(s) sur {cls[best_acc]['fn'] + cls[best_acc]['tp']} employes qui quittent. "
         "L'Accuracy melange les deux classes et masque les erreurs sur celle qui coute le plus cher (departs non detectes)."),
        ("C3", "Classe la mieux predite", f"F1 moyen des 4 modeles : classe 0 = {avg_f1_0:.3f}, classe 1 = {avg_f1_1:.3f}. "
         + ("Les modeles sont plus performants sur la classe 0 (reste)." if avg_f1_0 > avg_f1_1 else "Les modeles sont plus performants sur la classe 1 (quitte).")
         + " Le jeu de test est petit : l'ecart peut etre du au hasard."),
        ("C4", "Decision Tree : overfitting vs Random Forest",
         f"F1 macro train / test / CV - Decision Tree : {dt['train_f1']:.3f} / {dt['f1']:.3f} / {dt['cv_mean']:.3f} ; "
         f"Random Forest : {rf['train_f1']:.3f} / {rf['f1']:.3f} / {rf['cv_mean']:.3f}. "
         + ("Ecart train-CV important pour le Decision Tree : signe de surapprentissage" if dt["train_f1"] - dt["cv_mean"] > 0.1 else "Ecart train-CV limite pour le Decision Tree")
         + (" ; idem pour le Random Forest." if rf["train_f1"] - rf["cv_mean"] > 0.1 else " ; le Random Forest presente un ecart train-CV plus faible." if rf["train_f1"] - rf["cv_mean"] <= dt["train_f1"] - dt["cv_mean"] else ".")),
        ("C5", "Variables les plus importantes",
         "Random Forest : " + ", ".join(f"{k} ({v:.3f})" for k, v in top_imp) + f". EDA : correlation satisfaction_travail / attrition = {r_sat:+.2f} ; "
         f"correlation ratio_satisfaction / attrition = {corr_ratio['ratio_satisfaction']:+.2f} (echantillon)."),
        ("C6", "Recommandations RH (a valider avec des experts metier)",
         "Au vu des variables les plus influentes du modele : " + " ; ".join(advice) + ". "
         + ("Attention : les performances etant proches du hasard, ces pistes restent des hypotheses a verifier sur davantage de donnees." if near_random else "")),
        ("C7", "Matrice de confusion du meilleur modele (bonus)",
         f"{best_f1} : {bf['tn']} vrais negatifs, {bf['fp']} faux positifs, {bf['fn']} faux negatifs, {bf['tp']} vrais positifs. "
         + ("Zone de faiblesse principale : les faux negatifs (departs non detectes)." if bf["fn"] >= bf["fp"] else "Zone de faiblesse principale : les faux positifs (alertes inutiles).")),
    ]
    res["clf"] = {
        "models": cls, "order": names, "best_f1": best_f1, "best_recall": best_rec, "best_acc": best_acc,
        "least_fn": least_fn, "most_stable": most_stable, "near_random": bool(near_random),
        "importances": {k: _f(v) for k, v in imp.sort_values(ascending=False).items()},
        "proportions": {"global": prop(sample.attrition), "train": prop(yc_tr), "test": prop(yc_te)},
        "answers": [{"id": a, "q": q, "a": t} for a, q, t in ans_c],
    }

    # ---------------------------------------------------------------- Simulation (references + modeles)
    ref = prepared_all
    res["sim"] = {
        "ranges": {"age": [18, 65], "experience_ans": [0, 47], "anciennete_entreprise": [0, 47], "nb_formations_annee": [0, 20],
                   "heures_sup_mois": [0, 80], "satisfaction_travail": [1, 5], "satisfaction_manager": [1, 5]},
        "ref_ratio_mean": _f(ref.ratio_satisfaction.mean(), 2), "ref_overtime_median": _f(ref.heures_sup_mois.median(), 1),
        "classifier": SIM_MODEL, "has_proba": hasattr(fitted[SIM_MODEL], "predict_proba"),
        "classifier_reason": (f"Random Forest est utilise pour la simulation car il fournit une probabilite d'attrition moyennee sur 100 arbres. "
                              f"Un arbre de decision seul ne renvoie que des probabilites de 0 ou 1 ; le meilleur F1 test est {best_f1} ({cls[best_f1]['f1']:.3f}) "
                              f"contre {cls[SIM_MODEL]['f1']:.3f} pour {SIM_MODEL}."),
    }
    joblib.dump(prep, PREP_PATH)
    joblib.dump(reg, REG_PATH)
    joblib.dump(fitted[SIM_MODEL], CLF_PATH)
    RESULTS_PATH.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    return res


def artifacts_ready():
    return all(p.exists() for p in (RESULTS_PATH, PREP_PATH, REG_PATH, CLF_PATH))


if __name__ == "__main__":
    import ml_pipeline  # evite que les objets soient serialises sous le nom "__main__"
    r = ml_pipeline.train_all()
    print("Entrainement termine.")
    print("Regression R2 test:", r["reg"]["r2"], "| CV:", r["reg"]["cv_mean"], "+/-", r["reg"]["cv_std"])
    for n, m in r["clf"]["models"].items():
        print(f"{n:14s} acc={m['accuracy']:.3f} f1={m['f1']:.3f} recall1={m['recall_1']:.3f} FN={m['fn']} cv={m['cv_mean']:.3f}")
