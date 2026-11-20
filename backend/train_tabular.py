from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from backend.app.core.lexical_ml import FEATURE_COLUMNS

logger = logging.getLogger("brandshield.train")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def generate_bootstrap_dataset(
    n_samples: int = 2500, random_state: int = 42
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Generate a deterministic, realistic synthetic dataset covering all 27 canonical FEATURE_COLUMNS
    modeled after real-world distributions for legitimate sites vs. zero-day phishing infrastructure.
    """
    rng = np.random.default_rng(random_state)
    n_half = n_samples // 2

    # 0 = Legitimate, 1 = Phishing
    y = np.array([0] * n_half + [1] * (n_samples - n_half), dtype=int)
    n_phish = n_samples - n_half

    # 1. url_length
    url_len_legit = rng.normal(38.0, 12.0, n_half).clip(14.0, 120.0)
    url_len_phish = rng.normal(88.0, 32.0, n_phish).clip(28.0, 260.0)
    url_length = np.concatenate([url_len_legit, url_len_phish])

    # 2. hostname_length
    host_len_legit = rng.normal(16.0, 4.5, n_half).clip(7.0, 35.0)
    host_len_phish = rng.normal(34.0, 12.0, n_phish).clip(10.0, 85.0)
    hostname_length = np.concatenate([host_len_legit, host_len_phish])

    # 3. path_length
    path_len_legit = rng.normal(15.0, 10.0, n_half).clip(0.0, 80.0)
    path_len_phish = rng.normal(45.0, 24.0, n_phish).clip(5.0, 180.0)
    path_length = np.concatenate([path_len_legit, path_len_phish])

    # 4. subdomain_depth
    sub_depth_legit = rng.choice([0, 1, 2], size=n_half, p=[0.78, 0.19, 0.03])
    sub_depth_phish = rng.choice([0, 1, 2, 3, 4], size=n_phish, p=[0.08, 0.22, 0.42, 0.22, 0.06])
    subdomain_depth = np.concatenate([sub_depth_legit, sub_depth_phish])

    # 5. dot_count
    dot_legit = rng.normal(2.1, 0.5, n_half).clip(1.0, 5.0)
    dot_phish = rng.normal(4.6, 1.4, n_phish).clip(2.0, 12.0)
    dot_count = np.concatenate([dot_legit, dot_phish])

    # 6. hyphen_count
    hyphen_legit = rng.choice([0, 1, 2], size=n_half, p=[0.86, 0.11, 0.03])
    hyphen_phish = rng.normal(3.0, 1.8, n_phish).clip(0.0, 9.0)
    hyphen_count = np.concatenate([hyphen_legit, hyphen_phish])

    # 7. at_symbol_present
    at_legit = np.zeros(n_half, dtype=float)
    at_phish = rng.choice([0.0, 1.0], size=n_phish, p=[0.92, 0.08])
    at_symbol_present = np.concatenate([at_legit, at_phish])

    # 8. ip_in_hostname
    ip_legit = np.zeros(n_half, dtype=float)
    ip_phish = rng.choice([0.0, 1.0], size=n_phish, p=[0.84, 0.16])
    ip_in_hostname = np.concatenate([ip_legit, ip_phish])

    # 9. digit_ratio
    digit_legit = rng.beta(1.0, 45.0, n_half)
    digit_phish = rng.beta(2.5, 12.0, n_phish)
    digit_ratio = np.concatenate([digit_legit, digit_phish])

    # 10. shannon_entropy
    entropy_legit = rng.normal(3.72, 0.32, n_half).clip(2.4, 4.4)
    entropy_phish = rng.normal(4.68, 0.38, n_phish).clip(3.7, 5.9)
    shannon_entropy = np.concatenate([entropy_legit, entropy_phish])

    # 11. suspicious_tld_flag
    tld_legit = np.zeros(n_half, dtype=float)
    tld_phish = rng.choice([0.0, 1.0], size=n_phish, p=[0.52, 0.48])
    suspicious_tld_flag = np.concatenate([tld_legit, tld_phish])

    # 12. brand_keyword_in_subdomain
    brand_legit = np.zeros(n_half, dtype=float)
    brand_phish = rng.choice([0.0, 1.0], size=n_phish, p=[0.40, 0.60])
    brand_keyword_in_subdomain = np.concatenate([brand_legit, brand_phish])

    # 13. punycode_flag
    puny_legit = np.zeros(n_half, dtype=float)
    puny_phish = rng.choice([0.0, 1.0], size=n_phish, p=[0.90, 0.10])
    punycode_flag = np.concatenate([puny_legit, puny_phish])

    # 14. is_https
    https_legit = rng.choice([0.0, 1.0], size=n_half, p=[0.04, 0.96])
    https_phish = rng.choice([0.0, 1.0], size=n_phish, p=[0.42, 0.58])
    is_https = np.concatenate([https_legit, https_phish])

    # 15. query_param_count
    query_legit = rng.choice([0, 1, 2, 3], size=n_half, p=[0.65, 0.22, 0.09, 0.04])
    query_phish = rng.choice([0, 1, 2, 3, 4, 5], size=n_phish, p=[0.20, 0.25, 0.25, 0.15, 0.10, 0.05])
    query_param_count = np.concatenate([query_legit, query_phish])

    # 16. domain_age_days
    age_legit = rng.uniform(365.0, 6500.0, n_half)
    age_phish = []
    for _ in range(n_phish):
        if rng.random() < 0.35:
            age_phish.append(-1.0)
        else:
            age_phish.append(float(rng.uniform(1.0, 60.0)))
    domain_age_days = np.concatenate([age_legit, np.array(age_phish)])

    # 17. ssl_valid
    ssl_v_legit = rng.choice([0.0, 1.0], size=n_half, p=[0.03, 0.97])
    ssl_v_phish = rng.choice([0.0, 1.0], size=n_phish, p=[0.48, 0.52])
    ssl_valid = np.concatenate([ssl_v_legit, ssl_v_phish])

    # 18. ssl_cert_age_days
    cert_legit = rng.uniform(30.0, 750.0, n_half)
    cert_phish = []
    for _ in range(n_phish):
        if rng.random() < 0.40:
            cert_phish.append(-1.0)
        else:
            cert_phish.append(float(rng.uniform(1.0, 25.0)))
    ssl_cert_age_days = np.concatenate([cert_legit, np.array(cert_phish)])

    # 19. ssl_is_self_signed
    self_legit = np.zeros(n_half, dtype=float)
    self_phish = rng.choice([0.0, 1.0], size=n_phish, p=[0.76, 0.24])
    ssl_is_self_signed = np.concatenate([self_legit, self_phish])

    # 20. total_hops
    hops_legit = rng.choice([1, 2], size=n_half, p=[0.92, 0.08])
    hops_phish = rng.choice([1, 2, 3, 4, 5], size=n_phish, p=[0.22, 0.36, 0.22, 0.12, 0.08])
    total_hops = np.concatenate([hops_legit, hops_phish])

    # 21. has_shortener
    short_legit = rng.choice([0.0, 1.0], size=n_half, p=[0.97, 0.03])
    short_phish = rng.choice([0.0, 1.0], size=n_phish, p=[0.62, 0.38])
    has_shortener = np.concatenate([short_legit, short_phish])

    # 22. redirect_loop_detected
    loop_legit = np.zeros(n_half, dtype=float)
    loop_phish = rng.choice([0.0, 1.0], size=n_phish, p=[0.88, 0.12])
    redirect_loop_detected = np.concatenate([loop_legit, loop_phish])

    # 23. has_password_input
    pwd_legit = rng.choice([0.0, 1.0], size=n_half, p=[0.76, 0.24])
    pwd_phish = rng.choice([0.0, 1.0], size=n_phish, p=[0.10, 0.90])
    has_password_input = np.concatenate([pwd_legit, pwd_phish])

    # 24. external_form_action_count
    ext_form_legit = np.zeros(n_half, dtype=float)
    ext_form_phish = rng.choice([0, 1, 2], size=n_phish, p=[0.22, 0.68, 0.10])
    external_form_action_count = np.concatenate([ext_form_legit, ext_form_phish])

    # 25. hidden_iframe_count
    iframe_legit = rng.choice([0, 1], size=n_half, p=[0.98, 0.02])
    iframe_phish = rng.choice([0, 1, 2], size=n_phish, p=[0.58, 0.34, 0.08])
    hidden_iframe_count = np.concatenate([iframe_legit, iframe_phish])

    # 26. external_script_ratio
    ext_script_legit = rng.uniform(0.0, 0.38, n_half)
    ext_script_phish = rng.uniform(0.48, 0.96, n_phish)
    external_script_ratio = np.concatenate([ext_script_legit, ext_script_phish])

    # 27. obfuscated_js_heuristic
    obf_legit = np.zeros(n_half, dtype=float)
    obf_phish = rng.choice([0.0, 1.0], size=n_phish, p=[0.55, 0.45])
    obfuscated_js_heuristic = np.concatenate([obf_legit, obf_phish])

    feature_matrix = np.column_stack([
        url_length,
        hostname_length,
        path_length,
        subdomain_depth,
        dot_count,
        hyphen_count,
        at_symbol_present,
        ip_in_hostname,
        digit_ratio,
        shannon_entropy,
        suspicious_tld_flag,
        brand_keyword_in_subdomain,
        punycode_flag,
        is_https,
        query_param_count,
        domain_age_days,
        ssl_valid,
        ssl_cert_age_days,
        ssl_is_self_signed,
        total_hops,
        has_shortener,
        redirect_loop_detected,
        has_password_input,
        external_form_action_count,
        hidden_iframe_count,
        external_script_ratio,
        obfuscated_js_heuristic,
    ])

    return feature_matrix, y, FEATURE_COLUMNS


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    """Compute standard classification benchmarks, PR-AUC, and False Negative Rate (FNR)."""
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    pr_auc = float(average_precision_score(y_true, y_prob))

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "pr_auc": round(pr_auc, 4),
        "fnr": round(fnr, 4),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def train_and_compare(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    random_state: int = 42,
) -> tuple[Any, str, dict[str, Any], dict[str, Any]]:
    """
    Train and benchmark LightGBM vs. XGBoost (or alternative baseline if XGBoost is not installed).
    Selects the champion model based on lowest FNR with Precision >= 0.85 (tie-breaking on PR-AUC).
    """
    results: dict[str, dict[str, Any]] = {}
    models: dict[str, Any] = {}

    # 1. Train LightGBM (Primary mandated classifier)
    logger.info("Training LightGBM classifier...")
    lgbm = LGBMClassifier(
        n_estimators=120,
        learning_rate=0.07,
        num_leaves=31,
        max_depth=6,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=random_state,
        verbose=-1,
    )
    lgbm.fit(X_train, y_train)
    y_pred_lgbm = lgbm.predict(X_test)
    y_prob_lgbm = lgbm.predict_proba(X_test)[:, 1]
    results["LightGBM"] = compute_metrics(y_test, y_pred_lgbm, y_prob_lgbm)
    models["LightGBM"] = lgbm

    # 2. Train XGBoost if available, else baseline
    xgb_available = False
    try:
        from xgboost import XGBClassifier
        xgb_available = True
    except ImportError:
        xgb_available = False

    if xgb_available:
        logger.info("Training XGBoost classifier...")
        xgb_model = XGBClassifier(
            n_estimators=120,
            learning_rate=0.07,
            max_depth=6,
            subsample=0.85,
            colsample_bytree=0.85,
            eval_metric="logloss",
            random_state=random_state,
        )
        xgb_model.fit(X_train, y_train)
        y_pred_xgb = xgb_model.predict(X_test)
        y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]
        results["XGBoost"] = compute_metrics(y_test, y_pred_xgb, y_prob_xgb)
        models["XGBoost"] = xgb_model
    else:
        from sklearn.ensemble import HistGradientBoostingClassifier
        logger.info("XGBoost not installed; benchmarking against HistGradientBoostingClassifier...")
        hgb = HistGradientBoostingClassifier(
            max_iter=120,
            learning_rate=0.07,
            max_depth=6,
            random_state=random_state,
        )
        hgb.fit(X_train, y_train)
        y_pred_hgb = hgb.predict(X_test)
        y_prob_hgb = hgb.predict_proba(X_test)[:, 1]
        results["HistGradientBoosting"] = compute_metrics(y_test, y_pred_hgb, y_prob_hgb)
        models["HistGradientBoosting"] = hgb

    # Log comparison table
    logger.info("=" * 78)
    logger.info(f"{'Model':<24} | {'FNR':<7} | {'PR-AUC':<7} | {'Prec':<7} | {'Recall':<7} | {'F1':<7}")
    logger.info("-" * 78)
    for name, m in results.items():
        logger.info(
            f"{name:<24} | {m['fnr']:<7.4f} | {m['pr_auc']:<7.4f} | "
            f"{m['precision']:<7.4f} | {m['recall']:<7.4f} | {m['f1_score']:<7.4f}"
        )
    logger.info("=" * 78)

    # Champion selection: Lowest FNR with Precision >= 0.85 (breaking ties by PR-AUC)
    candidates = []
    for name, m in results.items():
        candidates.append((name, m["fnr"], -m["pr_auc"], m["precision"]))

    # Filter for precision >= 0.85
    filtered = [c for c in candidates if c[3] >= 0.85]
    if not filtered:
        filtered = candidates

    # Sort by lowest FNR (primary) then highest PR-AUC (secondary)
    filtered.sort(key=lambda x: (x[1], x[2]))
    champion_name = filtered[0][0]
    champion_model = models[champion_name]

    logger.info("Champion Selected: %s (FNR: %.4f, PR-AUC: %.4f)", champion_name, results[champion_name]["fnr"], results[champion_name]["pr_auc"])

    return champion_model, champion_name, results[champion_name], results


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate Tabular ML Phishing Detection Models")
    parser.add_argument("--dataset", type=str, default=None, help="Path to CSV dataset")
    parser.add_argument("--bootstrap", action="store_true", help="Force synthetic bootstrap dataset generation")
    parser.add_argument("--output-dir", type=str, default="backend/models", help="Directory to save model artifacts")
    parser.add_argument("--random-state", type=int, default=42, help="Seed for reproducibility")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Dataset generation / loading
    if args.dataset and Path(args.dataset).exists():
        import pandas as pd
        logger.info("Loading dataset from %s", args.dataset)
        df = pd.read_csv(args.dataset)
        X = df[FEATURE_COLUMNS].values
        y = df["label"].values
    else:
        logger.info("Generating synthetic bootstrap dataset of 2,500 samples...")
        X, y, feature_names = generate_bootstrap_dataset(n_samples=2500, random_state=args.random_state)

    # 2. Stratified 80/20 train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=args.random_state
    )
    logger.info("Train shape: %s, Test shape: %s", X_train.shape, X_test.shape)

    # 3. Benchmark models
    champion_model, champion_name, champion_metrics, all_results = train_and_compare(
        X_train, X_test, y_train, y_test, random_state=args.random_state
    )

    # 4. Save champion artifact
    artifact_path = output_dir / "tabular_model.joblib"
    artifact = {
        "model": champion_model,
        "feature_columns": FEATURE_COLUMNS,
        "model_version": "tabular_v1.0.0",
        "model_name": champion_name,
        "trained_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "metrics": champion_metrics,
    }
    joblib.dump(artifact, artifact_path)
    logger.info("Saved serialized model artifact to %s", artifact_path)

    # 5. Save metrics JSON
    metrics_path = output_dir / "model_metrics.json"
    metrics_payload = {
        "champion": champion_name,
        "model_version": "tabular_v1.0.0",
        "feature_columns_count": len(FEATURE_COLUMNS),
        "feature_columns": FEATURE_COLUMNS,
        "champion_metrics": champion_metrics,
        "benchmark_comparison": all_results,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)
    logger.info("Saved evaluation report to %s", metrics_path)


if __name__ == "__main__":
    main()
