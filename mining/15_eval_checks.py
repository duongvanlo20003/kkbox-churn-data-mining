"""Extra evaluation checks asked for in the rewritten report (section 6).

1. Independent-customer check. About 91 % of the mar cohort is also in feb, so "the rules hold in
   feb" (07_stability.py) is mostly the same people one month earlier. Here every churn and
   retention rule is re-measured only on the ~111k users who are in feb but not in mar. These users
   churn far more (most of them left before March), so lifts are compared on the log scale and
   by direction, not by absolute confidence.
2. Stability of the k = 3 (best silhouette) solution over time: K-Means k = 3 fitted on a mar sample
   (same features, scaler and seed as 01_clustering.py) assigns feb users; K-Means k = 3 is then
   refitted on feb from scratch and compared with it (ARI). 07_stability.py does the same for the
   chosen k.

Outputs: report/rule_feb_only.csv, metrics/eval_checks.json
"""
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, REPORT, SEED, load, save_metrics
from transforms import itemise, prepare

N = int(os.environ.get("CLUSTER_SAMPLE", 200_000))
model = joblib.load(DATA / "cluster_model.joblib")
scaler, km, MED = model["scaler"], model["km"], model["median"]
NOLOG = model.get("nolog_id", km.n_clusters)
CLF = model.get("features") or list(MED.index)
out = {}

mar = load("mar").merge(pd.read_parquet(DATA / "clusters_mar.parquet"), on="msno", how="left")
tr, _ = train_test_split(mar, test_size=0.3, stratify=mar.is_churn, random_state=SEED)  # item thresholds
feb = load("feb")
feb["cluster"] = np.where(feb.has_logs == 1, km.predict(scaler.transform(prepare(feb)[CLF].fillna(MED))), NOLOG)

# ---- 1. rules on feb-only users --------------------------------------------------------------
only = feb[~feb.msno.isin(mar.msno)].reset_index(drop=True)
I = itemise(only, tr)
y = only.is_churn.values
rows = []
for kind, f in [("churn", "churn_rules.csv"), ("retain", "retention_rules.csv")]:
    for r in pd.read_csv(REPORT / f).itertuples():
        items = r.rule.split(" & ")
        if not all(i in I.columns for i in items):
            continue
        m = I[items].all(axis=1).values
        conf = y[m].mean() if m.sum() >= 30 else np.nan
        rows.append({"rule": r.rule, "direction": kind, "lift_mar_holdout": r.lift_holdout,
                     "n_feb_only": int(m.sum()), "conf_feb_only": conf, "lift_feb_only": conf / y.mean()})
rt = pd.DataFrame(rows)
rt["same_direction"] = np.where(rt.direction == "churn", rt.lift_feb_only > 1, rt.lift_feb_only < 1)
rt.to_csv(REPORT / "rule_feb_only.csv", index=False)
ok = rt.dropna(subset=["lift_feb_only"])
ok = ok[ok.lift_feb_only > 0]
out["feb_only"] = {"n_users": len(only), "churn_rate": float(y.mean()), "rules_checked": int(len(ok)),
                   "rules_same_direction": int(ok.same_direction.sum()),
                   "log_lift_pearson": float(pearsonr(np.log(ok.lift_mar_holdout), np.log(ok.lift_feb_only)).statistic)}

# ---- 2. k = 3 over time -----------------------------------------------------------------------
q = pd.read_csv(REPORT / "cluster_quality.csv")
KB = int(q.loc[q.chosen == "best silhouette", "k"].iloc[0])
ms = mar[mar.has_logs == 1].sample(min(N, int((mar.has_logs == 1).sum())), random_state=SEED)
km3 = KMeans(KB, n_init=10, random_state=SEED).fit(scaler.transform(prepare(ms)[CLF].fillna(MED)))
fs = feb[feb.has_logs == 1].sample(min(N, int((feb.has_logs == 1).sum())), random_state=SEED)
Rf = prepare(fs)[CLF]
lab_mar_model = km3.predict(scaler.transform(Rf.fillna(MED)))
sc_f = StandardScaler().fit(Rf.fillna(Rf.median()))
lab_refit = KMeans(KB, n_init=10, random_state=SEED).fit_predict(sc_f.transform(Rf.fillna(Rf.median())))
out["k_macro_feb_refit_ari"] = {"k": KB, "ari": float(adjusted_rand_score(lab_mar_model, lab_refit))}
out["k_chosen_feb_refit_ari_from_07"] = "see metrics/stability.json: ari_feb_refit_vs_mar_model"

save_metrics("eval_checks", out)
print(out)
