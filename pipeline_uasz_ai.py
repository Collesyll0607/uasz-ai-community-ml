"""
Projet Machine Learning Collaboratif — UASZ AI Community
Auteure : Collé Syll — Université Assane Seck de Ziguinchor (UASZ)
Technologies : Python, Pandas, NumPy, Scikit-learn
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"


def prepare_student_orientation_dataset(n_samples: int = 600, random_state: int = 42) -> pd.DataFrame:
    """
    Jeu de données d'entraînement collaboratif UASZ AI Community :
    Prétraitement des données académiques et techniques pour l'évaluation de modèles supervisés.
    """
    rng = np.random.default_rng(random_state)
    moyenne_algo = np.clip(rng.normal(13.5, 2.8, n_samples), 5.0, 20.0)
    moyenne_proba_stat = np.clip(rng.normal(13.0, 3.0, n_samples), 4.0, 20.0)
    moyenne_bdd_sql = np.clip(rng.normal(14.0, 2.5, n_samples), 6.0, 20.0)
    projets_realises = rng.integers(1, 8, n_samples)
    heures_pratique_hebdo = np.clip(rng.normal(12.0, 4.5, n_samples), 2.0, 30.0)

    score = (
        0.30 * moyenne_algo
        + 0.30 * moyenne_proba_stat
        + 0.20 * moyenne_bdd_sql
        + 0.85 * projets_realises
        + 0.18 * heures_pratique_hebdo
        + rng.normal(0, 1.4, n_samples)
    )
    validation_projet_ia = (score >= 16.2).astype(int)

    return pd.DataFrame(
        {
            "moyenne_algo": np.round(moyenne_algo, 2),
            "moyenne_proba_stat": np.round(moyenne_proba_stat, 2),
            "moyenne_bdd_sql": np.round(moyenne_bdd_sql, 2),
            "projets_realises": projets_realises,
            "heures_pratique_hebdo": np.round(heures_pratique_hebdo, 1),
            "validation_projet_ia": validation_projet_ia,
        }
    )


def train_and_evaluate(df: pd.DataFrame, random_state: int = 42) -> dict:
    """Prétraitement (StandardScaler), entraînement et évaluation comparée de modèles Scikit-learn."""
    X = df.drop(columns=["validation_projet_ia"])
    y = df["validation_projet_ia"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=random_state
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    rf_model = RandomForestClassifier(n_estimators=150, max_depth=8, random_state=random_state)
    rf_model.fit(X_train_scaled, y_train)
    rf_preds = rf_model.predict(X_test_scaled)

    gb_model = GradientBoostingClassifier(random_state=random_state)
    gb_model.fit(X_train_scaled, y_train)
    gb_preds = gb_model.predict(X_test_scaled)

    return {
        "n_samples": len(df),
        "random_forest": {
            "accuracy": round(float(accuracy_score(y_test, rf_preds)), 4),
            "f1_score": round(float(f1_score(y_test, rf_preds)), 4),
        },
        "gradient_boosting": {
            "accuracy": round(float(accuracy_score(y_test, gb_preds)), 4),
            "f1_score": round(float(f1_score(y_test, gb_preds)), 4),
        },
        "report": classification_report(y_test, rf_preds, output_dict=True),
    }


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dataset = prepare_student_orientation_dataset()
    results = train_and_evaluate(dataset)
    (OUTPUT_DIR / "metrics_uasz_ai.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("=== Projet ML — UASZ AI Community ===")
    print(f"Random Forest Accuracy     : {results['random_forest']['accuracy']:.2%}")
    print(f"Random Forest F1-Score     : {results['random_forest']['f1_score']:.2%}")
    print(f"Gradient Boosting Accuracy : {results['gradient_boosting']['accuracy']:.2%}")
