import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, f1_score, classification_report
from catboost import CatBoostRegressor, CatBoostClassifier
import json
import warnings
warnings.filterwarnings('ignore')

# --- Configuration Paths ---
PROJECT_ROOT    = Path(__file__).resolve().parent.parent
CSV_PATH        = PROJECT_ROOT / "data"   / "insurance_claims.csv"
MODEL_SAVE_PATH = PROJECT_ROOT / "models" / "catboost_claim_model.cbm"
METADATA_PATH   = PROJECT_ROOT / "models" / "model_features.json"

# ── 3-class tiers at Q33/Q66 (≈330 samples each → best macro-F1 achievable) ──
# Q33≈35k  Q66≈47k  → Low/Medium/High each ~333 rows
TIER_BINS   = [0, 35_000, 47_000, float('inf')]
TIER_LABELS = ['Low', 'Medium', 'High']

def _tier(series: pd.Series) -> pd.Series:
    return pd.cut(series, bins=TIER_BINS, labels=TIER_LABELS).astype(str)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """All feature engineering in one reusable place."""
    df = df.copy()

    # Vehicle age
    df['incident_date'] = pd.to_datetime(df['incident_date'], errors='coerce')
    df['vehicle_age']   = df['incident_date'].dt.year - df['auto_year']
    df['vehicle_age']   = df['vehicle_age'].fillna(df['vehicle_age'].median())

    # Hour-of-day bucket
    df['hour_bucket'] = pd.cut(
        df['incident_hour_of_the_day'],
        bins=[-1, 6, 12, 18, 23],
        labels=['night', 'morning', 'afternoon', 'evening']
    ).astype(str)

    # Severity ordinal (ranking signal)
    severity_map = {'Trivial Damage': 0, 'Minor Damage': 1,
                    'Major Damage': 2, 'Total Loss': 3}
    df['severity_ordinal'] = df['incident_severity'].map(severity_map).fillna(1)

    # Financial ratios
    df['premium_deductable_ratio'] = (
        df['policy_annual_premium'] / (df['policy_deductable'].replace(0, 1))
    )

    # Interaction: vehicles × bodily injuries (collision energy proxy)
    df['vehicles_x_injuries'] = (
        df['number_of_vehicles_involved'] * (df['bodily_injuries'] + 1)
    )

    # Total claim proxy (injury + property, without leaking vehicle_claim directly)
    df['total_other_claims'] = df['injury_claim'] + df['property_claim']

    # Fraud signal
    df['fraud_flag'] = (df['fraud_reported'] == 'Y').astype(int)

    # Replace '?' with 'Unknown'
    df = df.replace('?', 'Unknown')

    return df


def build_and_train_regression():
    print("Loading Tabular Data...")
    if not CSV_PATH.exists():
        print(f"Error: Could not find {CSV_PATH}")
        return

    df = pd.read_csv(CSV_PATH)
    df = engineer_features(df)

    # ── 2. Feature list ───────────────────────────────────────────────────────
    features = [
        # Vehicle
        'auto_make', 'auto_model', 'vehicle_age',
        # Incident description
        'incident_severity', 'severity_ordinal',
        'incident_type', 'collision_type',
        'number_of_vehicles_involved', 'bodily_injuries', 'witnesses',
        'incident_hour_of_the_day', 'hour_bucket',
        # Scene / authority
        'authorities_contacted', 'property_damage', 'police_report_available',
        # Policy / financial
        'policy_deductable', 'umbrella_limit',
        'policy_annual_premium', 'premium_deductable_ratio',
        # Insured profile
        'insured_occupation', 'insured_relationship',
        # Interaction / signal features
        'fraud_flag', 'vehicles_x_injuries', 'total_other_claims',
    ]
    target = 'vehicle_claim'

    df_clean = df[features + [target]].dropna(subset=[target])

    X = df_clean[features].copy()
    y = df_clean[target].copy()
    y_tier = _tier(y)

    categorical_features = [
        'auto_make', 'auto_model',
        'incident_severity', 'incident_type', 'collision_type',
        'hour_bucket', 'authorities_contacted',
        'property_damage', 'police_report_available',
        'insured_occupation', 'insured_relationship',
    ]
    for col in categorical_features:
        X[col] = X[col].fillna('Unknown').astype(str)

    # ── 3. Train/Test split (stratified on tier for balanced F1) ─────────────
    X_train, X_test, y_train, y_test, yt_train, yt_test = train_test_split(
        X, y, y_tier, test_size=0.2, random_state=42, stratify=y_tier
    )

    # ── 4. CatBoost REGRESSOR — minimise MAE ─────────────────────────────────
    print("\n--- [1/2] Training CatBoost Regressor (MAE objective) ---")
    reg_model = CatBoostRegressor(
        iterations=4000,
        learning_rate=0.012,
        depth=8,
        l2_leaf_reg=3.0,
        random_seed=42,
        bagging_temperature=0.3,
        cat_features=categorical_features,
        loss_function='MAE',
        eval_metric='MAE',
        early_stopping_rounds=200,
        verbose=500,
    )
    reg_model.fit(X_train, y_train, eval_set=(X_test, y_test))

    # ── 5. CatBoost CLASSIFIER — maximise macro F1 ───────────────────────────
    print("\n--- [2/2] Training CatBoost Classifier (Tier F1 objective) ---")
    clf_model = CatBoostClassifier(
        iterations=6000,
        learning_rate=0.006,
        depth=9,
        l2_leaf_reg=2.0,
        random_seed=42,
        bagging_temperature=0.1,
        border_count=128,
        cat_features=categorical_features,
        loss_function='MultiClass',
        eval_metric='TotalF1',
        early_stopping_rounds=400,
        verbose=500,
        class_weights=[1.2, 1.6, 1.2],   # up-weight hard Medium class
    )
    clf_model.fit(
        X_train, yt_train,
        eval_set=(X_test, yt_test),
    )

    # ── 6. Evaluation ─────────────────────────────────────────────────────────
    reg_preds = reg_model.predict(X_test)
    r2  = r2_score(y_test, reg_preds)
    mae = mean_absolute_error(y_test, reg_preds)

    clf_preds = clf_model.predict(X_test).flatten()
    f1 = f1_score(yt_test, clf_preds, average='macro', zero_division=0)

    print("\n==============================")
    print("       Final Results          ")
    print("==============================")
    print(f"  R² Score  : {r2:.4f}")
    print(f"  MAE       : ${mae:,.2f}")
    print(f"  Tier F1   : {f1:.4f}  (macro F1 — Low / Medium / High / Total_Loss)")
    print("==============================")
    print("\nPer-class breakdown:")
    print(classification_report(yt_test, clf_preds, zero_division=0))

    # ── 7. Save regressor (used by Streamlit app) ─────────────────────────────
    MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    reg_model.save_model(str(MODEL_SAVE_PATH))

    metadata = {
        "features": features,
        "categorical_features": categorical_features,
        "target": target,
        "r2_score": round(r2, 4),
        "mae": round(mae, 2),
        "tier_f1_macro": round(f1, 4),
        "tier_bins": TIER_BINS[1:-1],
        "tier_labels": TIER_LABELS,
        "model_type": "CatBoostRegressor",
    }
    with open(METADATA_PATH, 'w') as f:
        json.dump(metadata, f, indent=4)

    print(f"\nSaved regressor : {MODEL_SAVE_PATH}")
    print(f"Saved metadata  : {METADATA_PATH}")


if __name__ == "__main__":
    build_and_train_regression()