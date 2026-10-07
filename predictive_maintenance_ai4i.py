"""
Manutenção preditiva com o dataset AI4I 2020 (UCI) - versão com protocolo de validação.

Diferenças em relação aos rascunhos anteriores:
  * hiperparâmetros fixados antes de olhar o teste (nada é ajustado no teste);
  * limiar de decisão escolhido com previsões fora-da-amostra (validação cruzada) no TREINO;
  * o conjunto de teste é usado uma única vez, no final;
  * métricas para classe rara: precisão, revocação, F1, PR-AUC (e ROC-AUC);
  * ablação: variáveis brutas x variáveis derivadas;
  * intervalo de confiança por bootstrap (o teste tem poucas falhas).

Uso:  python predictive_maintenance_ai4i.py ai4i2020.csv saida/
Obs.: o dataset é SINTÉTICO (as falhas são geradas por regras). Veja o README.
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

SEED = 42
COST_RATIO = 10  # premissa: uma falha não detectada custa 10x um alarme falso (ajuste à sua realidade)

csv_path = Path(sys.argv[1])
out = Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- 1. dados
df = pd.read_csv(csv_path, encoding="utf-8-sig")
df = df.rename(
    columns={
        "Air temperature [K]": "Air_Temp",
        "Process temperature [K]": "Process_Temp",
        "Rotational speed [rpm]": "Rot_Speed",
        "Torque [Nm]": "Torque",
        "Tool wear [min]": "Tool_Wear",
        "Machine failure": "Failure",
    }
)

RAW = ["Air_Temp", "Process_Temp", "Rot_Speed", "Torque", "Tool_Wear"]


def build_features(d: pd.DataFrame, engineered: bool) -> pd.DataFrame:
    X = d[RAW].copy()
    X["Type_M"] = (d["Type"] == "M").astype(int)
    X["Type_H"] = (d["Type"] == "H").astype(int)
    if engineered:
        omega = d["Rot_Speed"] * 2 * np.pi / 60  # rad/s
        X["Power_W"] = d["Torque"] * omega  # potência mecânica
        X["Temp_Diff"] = d["Process_Temp"] - d["Air_Temp"]  # dissipação térmica
        X["Torque_x_Wear"] = d["Torque"] * d["Tool_Wear"]  # esforço acumulado
        X["Power_x_Wear"] = X["Power_W"] * d["Tool_Wear"]
        X["TempDiff_x_Wear"] = X["Temp_Diff"] * d["Tool_Wear"]
    return X


y = df["Failure"].values
idx_train, idx_test = train_test_split(
    np.arange(len(df)), test_size=0.2, random_state=SEED, stratify=y
)

# Colunas TWF/HDF/PWF/OSF/RNF (modos de falha) NÃO entram como variáveis: seriam vazamento do alvo.


def make_models():
    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=SEED,
        ),
        "GradientBoosting": HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=300,
            max_depth=3,
            class_weight="balanced",
            random_state=SEED,
        ),
    }
    try:  # XGBoost é opcional: usado se estiver instalado
        from xgboost import XGBClassifier

        n_neg, n_pos = (y[idx_train] == 0).sum(), (y[idx_train] == 1).sum()
        models["XGBoost"] = XGBClassifier(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.05,
            scale_pos_weight=n_neg / n_pos,  # >1 dá MAIS peso à falha
            eval_metric="logloss",
            random_state=SEED,
        )
    except ImportError:
        pass
    return models


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)


def pick_thresholds(y_true, proba):
    """Escolhe limiares só com dados de validação (previsões fora-da-amostra do treino)."""
    prec, rec, thr = precision_recall_curve(y_true, proba)
    f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-12, None)
    thr_f1 = float(thr[np.argmax(f1)])
    # custo: FN * COST_RATIO + FP * 1
    grid = np.unique(np.round(np.linspace(0.01, 0.99, 197), 4))
    costs = []
    for t in grid:
        pred = proba >= t
        fn = int(((~pred) & (y_true == 1)).sum())
        fp = int((pred & (y_true == 0)).sum())
        costs.append(fn * COST_RATIO + fp)
    thr_cost = float(grid[int(np.argmin(costs))])
    return thr_f1, thr_cost


def metrics_at(y_true, proba, thr):
    pred = (proba >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    return {
        "threshold": round(float(thr), 4),
        "precision": round(float(precision_score(y_true, pred, zero_division=0)), 3),
        "recall": round(float(recall_score(y_true, pred)), 3),
        "f1": round(float(f1_score(y_true, pred)), 3),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "cost": int(fn * COST_RATIO + fp),
    }


results = {"cost_ratio_assumption": COST_RATIO, "n_train": int(len(idx_train)),
           "n_test": int(len(idx_test)),
           "failures_train": int(y[idx_train].sum()), "failures_test": int(y[idx_test].sum())}

# ---------------------------------------------- 2. ablação: bruto x derivado (CV no treino)
ablation = {}
for eng in (False, True):
    X = build_features(df, eng)
    Xtr = X.iloc[idx_train]
    oof = cross_val_predict(make_models()["RandomForest"], Xtr, y[idx_train], cv=skf,
                            method="predict_proba")[:, 1]
    ablation["derivadas" if eng else "brutas"] = {
        "cv_pr_auc": round(float(average_precision_score(y[idx_train], oof)), 3),
        "cv_roc_auc": round(float(roc_auc_score(y[idx_train], oof)), 3),
    }
results["ablation_randomforest_cv"] = ablation

# ------------------------------------------- 3. modelos com variáveis derivadas
X = build_features(df, True)
Xtr, Xte = X.iloc[idx_train], X.iloc[idx_test]
ytr, yte = y[idx_train], y[idx_test]

cv_summary, fitted, oof_store = {}, {}, {}
for name, model in make_models().items():
    oof = cross_val_predict(model, Xtr, ytr, cv=skf, method="predict_proba")[:, 1]
    oof_store[name] = oof
    thr_f1, thr_cost = pick_thresholds(ytr, oof)
    cv_summary[name] = {
        "cv_pr_auc": round(float(average_precision_score(ytr, oof)), 3),
        "cv_roc_auc": round(float(roc_auc_score(ytr, oof)), 3),
        "thr_f1": round(thr_f1, 4),
        "thr_cost": round(thr_cost, 4),
        "cv_at_thr_f1": metrics_at(ytr, oof, thr_f1),
        "cv_at_thr_cost": metrics_at(ytr, oof, thr_cost),
    }
    fitted[name] = make_models()[name].fit(Xtr, ytr)
results["cv"] = cv_summary

# modelo final escolhido pela VALIDAÇÃO CRUZADA (PR-AUC), nunca pelo teste
final_name = max(cv_summary, key=lambda k: cv_summary[k]["cv_pr_auc"])
results["final_model_chosen_by_cv"] = final_name

# ---------------------------------------------------- 4. teste (uso único)
test = {}
for name, model in fitted.items():
    p = model.predict_proba(Xte)[:, 1]
    test[name] = {
        "pr_auc": round(float(average_precision_score(yte, p)), 3),
        "roc_auc": round(float(roc_auc_score(yte, p)), 3),
        "at_thr_f1": metrics_at(yte, p, cv_summary[name]["thr_f1"]),
        "at_thr_cost": metrics_at(yte, p, cv_summary[name]["thr_cost"]),
    }
results["test"] = test

# bootstrap (IC 95%) para o modelo final no limiar de F1
p_final = fitted[final_name].predict_proba(Xte)[:, 1]
thr_final = cv_summary[final_name]["thr_f1"]
rng = np.random.default_rng(SEED)
precs, recs = [], []
for _ in range(2000):
    b = rng.integers(0, len(yte), len(yte))
    pred = (p_final[b] >= thr_final).astype(int)
    if yte[b].sum() == 0:
        continue
    precs.append(precision_score(yte[b], pred, zero_division=0))
    recs.append(recall_score(yte[b], pred))
results["bootstrap_95ci_final_at_thr_f1"] = {
    "precision": [round(float(np.percentile(precs, 2.5)), 3), round(float(np.percentile(precs, 97.5)), 3)],
    "recall": [round(float(np.percentile(recs, 2.5)), 3), round(float(np.percentile(recs, 97.5)), 3)],
}

# --------------------------------------------------------------- 5. figuras
sns.set_theme(style="whitegrid")

fig, ax = plt.subplots(figsize=(6, 4.5))
for name, oof in oof_store.items():
    pr, rc, _ = precision_recall_curve(ytr, oof)
    ax.plot(rc, pr, label=f"{name} (PR-AUC CV={cv_summary[name]['cv_pr_auc']})")
ax.axhline(ytr.mean(), ls="--", c="gray", label=f"acaso ({ytr.mean():.3f})")
ax.set_xlabel("Revocação"); ax.set_ylabel("Precisão")
ax.set_title("Curva precisão x revocação (validação cruzada no treino)")
ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(out / "pr_curve_cv.png", dpi=150); plt.close(fig)

cm = confusion_matrix(yte, (p_final >= thr_final).astype(int))
fig, ax = plt.subplots(figsize=(4.5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=["Normal", "Falha"], yticklabels=["Normal", "Falha"], ax=ax)
ax.set_xlabel("Previsto"); ax.set_ylabel("Real")
ax.set_title(f"Teste - {final_name} (limiar {thr_final:.2f})")
fig.tight_layout(); fig.savefig(out / "confusion_matrix_test.png", dpi=150); plt.close(fig)

pi = permutation_importance(fitted[final_name], Xte, yte, scoring="average_precision",
                            n_repeats=10, random_state=SEED, n_jobs=-1)
imp = pd.Series(pi.importances_mean, index=X.columns).sort_values()
results["permutation_importance_test"] = {k: round(float(v), 4) for k, v in imp.sort_values(ascending=False).items()}
fig, ax = plt.subplots(figsize=(6.5, 4.5))
imp.plot(kind="barh", ax=ax, color="#2a6f97")
ax.set_xlabel("Queda de PR-AUC ao embaralhar a variável")
ax.set_title(f"Importância por permutação - {final_name}")
fig.tight_layout(); fig.savefig(out / "permutation_importance.png", dpi=150); plt.close(fig)

(out / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(results, indent=2, ensure_ascii=False))
