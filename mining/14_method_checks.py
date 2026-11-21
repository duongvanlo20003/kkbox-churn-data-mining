"""Extra checks on the mining methods asked for in the rewritten report (section 5).

1. Rules without segment and gender items: re-run mining/02_association_rules.py with
   RULE_EXCLUDE="segment=,gender=" in a scratch report folder and compare with the main list.
   Do the segment items (themselves built from the same features) change the main rules?
2. Sequence {PAY_AUTO} -> {PAY_MANUAL}: measured with exactly the sample, 70/30 split and matching
   semantics of mining/05_sequential.py, plus the feb sample. It is not in the selected list
   because the redundancy filter removes it, but the report discusses it as a business trigger.
3. Isolation Forest: overlap of the 2 % anomalies with long plans (> 30 days) and with segment 3,
   churn of anomalies outside long plans (the technique's own added value), and sensitivity to
   contamination 1 % / 2 % / 5 %.
4. Figure 7 v2: k-selection metrics with the SSE (elbow) curve and its per-step decrease.

Outputs: report/churn_rules_nosegment.csv, retention_rules_nosegment.csv,
report/anomaly_contamination.csv, figures/cluster_selection_v2.png, metrics/method_checks.json
"""
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import DATA, REPORT, SEED, load, save_metrics, savefig

out = {}

# ---- 1. rules without segment / gender items ---------------------------------------------------
with tempfile.TemporaryDirectory() as tmp:
    (Path(tmp) / "figures").mkdir(); (Path(tmp) / "metrics").mkdir()
    env = {**os.environ, "REPORT_DIR": tmp, "RULE_EXCLUDE": "segment=,gender=", "PYTHONUTF8": "1"}
    subprocess.run([sys.executable, str(HERE / "02_association_rules.py")], env=env, check=True,
                   stdout=subprocess.DEVNULL)
    for kind in ("churn", "retention"):
        shutil.copy(Path(tmp) / f"{kind}_rules.csv", REPORT / f"{kind}_rules_nosegment.csv")
main = pd.read_csv(REPORT / "churn_rules.csv")
noseg = pd.read_csv(REPORT / "churn_rules_nosegment.csv")
plain = main[~main.rule.str.contains("segment=|gender=")]
kept = plain.merge(noseg, on="rule", suffixes=("_main", "_noseg"))
new = noseg[~noseg.rule.isin(main.rule)]
out["rules_without_segment_gender"] = {
    "n_rules": len(noseg), "main_rules_without_these_items": len(plain), "of_which_found_again": len(kept),
    "max_abs_confidence_change": float((kept.confidence_main - kept.confidence_noseg).abs().max()) if len(kept) else None,
    "new_rules": len(new), "new_rules_lift_range": [float(new.lift.min()), float(new.lift.max())] if len(new) else None,
    "top_rule": noseg.iloc[0][["rule", "confidence", "lift"]].to_dict(),
}

# ---- 2. sequence {PAY_AUTO} -> {PAY_MANUAL} ---------------------------------------------------
spec = importlib.util.spec_from_file_location("seq", HERE / "05_sequential.py")
seq = importlib.util.module_from_spec(spec); spec.loader.exec_module(seq)
rng = np.random.default_rng(42)                       # same generator order as 05_sequential.main
frame, _ = seq.load_cohort(DATA / "sequences_mar.parquet", int(os.environ.get("SEQ_SAMPLE", "300000")), rng)
labels = frame.is_churn.to_numpy(dtype=np.int8)
train, hold = seq.split_indices(labels, rng)
vocab, masks = seq.encode(frame.seq.to_numpy())
feb, _ = seq.load_cohort(DATA / "sequences_feb.parquet", int(os.environ.get("SEQ_SAMPLE", "300000")),
                         np.random.default_rng(42))
feb_labels = feb.is_churn.to_numpy(dtype=np.int8)
_, feb_masks = seq.encode(feb.seq.to_numpy(), vocab)
ix = {t: i for i, t in enumerate(vocab)}


def measure(pattern, m, y):
    hit = seq.match_pattern(pattern, m)
    conf = float(y[hit].mean()) if hit.any() else None
    return {"support": float(hit.mean()), "confidence": conf, "lift": conf / float(y.mean()) if conf is not None else None}


pat = [[ix["PAY_AUTO"]], [ix["PAY_MANUAL"]]]
out["sequence_auto_to_manual"] = {"train": measure(pat, masks[:, train], labels[train]),
                                  "holdout": measure(pat, masks[:, hold], labels[hold]),
                                  "feb": measure(pat, feb_masks, feb_labels)}

# ---- 3. Isolation Forest overlap and contamination sensitivity ----------------------------------
FE = ["n_tx", "n_cancel", "auto_renew_rate", "discount_rate", "last_paid", "last_plan_days",
      "total_paid", "active_days", "total_hours", "avg_unique_songs", "skip_ratio", "complete_ratio"]  # as 04_bonus
df = load("mar").merge(pd.read_parquet(DATA / "clusters_mar.parquet")[["msno", "cluster"]], on="msno", how="left")
Xa = StandardScaler().fit_transform(np.log1p(df[FE].fillna(0).clip(lower=0)))
base = df.is_churn.mean()
rows = []
for c in (0.01, 0.02, 0.05):
    a = IsolationForest(n_estimators=200, contamination=c, random_state=SEED, n_jobs=-1).fit(Xa).predict(Xa) == -1
    long_ = (df.last_plan_days > 30).values
    rows.append({"contamination": c, "n": int(a.sum()), "churn_rate": float(df.is_churn[a].mean()),
                 "lift": float(df.is_churn[a].mean() / base),
                 "share_churners_covered": float(df.is_churn[a].sum() / df.is_churn.sum()),
                 "share_long_plan": float(long_[a].mean()), "share_segment_3": float((df.cluster[a] == 3).mean()),
                 "n_not_long_plan": int((a & ~long_).sum()),
                 "churn_not_long_plan": float(df.is_churn[a & ~long_].mean())})
an = pd.DataFrame(rows)
an.to_csv(REPORT / "anomaly_contamination.csv", index=False)
out["isolation_forest"] = an.round(4).to_dict("records")

# ---- 4. Figure 7 v2 ---------------------------------------------------------------------------
q = pd.read_csv(REPORT / "cluster_quality.csv")
K = int(q.loc[q.chosen == "chosen", "k"].iloc[0]); KB = int(q.loc[q.chosen == "best silhouette", "k"].iloc[0])
fig, axes = plt.subplots(1, 4, figsize=(13, 2.36))
for ax, (col, title, scale) in zip(axes, [("inertia", "SSE (triệu) – thấp tốt", 1e-6),
                                          ("silhouette", "Silhouette – cao tốt", 1),
                                          ("davies_bouldin", "Davies–Bouldin – thấp tốt", 1),
                                          ("stability_ari", "Bootstrap ARI – cao tốt", 1)]):
    ax.plot(q.k, q[col] * scale, marker="o", ms=5, color="#2F5D9C", lw=2)
    ax.axvline(K, color="crimson", ls="--", lw=1.4); ax.axvline(KB, color="grey", ls=":", lw=1.4)
    ax.set_title(title, fontsize=14); ax.set_xticks(q.k); ax.tick_params(labelsize=11, pad=1)
    if col == "inertia":
        for k_, v, p in zip(q.k, q.inertia * scale, -q.inertia.pct_change() * 100):
            if k_ > q.k.min():
                ax.annotate(f"−{p:.0f}%", (k_, v), textcoords="offset points", xytext=(4, 4), fontsize=10)
    if col == "stability_ari":
        ax.axhline(0.8, color="grey", lw=0.8, alpha=0.6)
savefig(fig, "cluster_selection_v2.png")
out["sse_decrease_pct"] = dict(zip(q.k.astype(int).astype(str), (-q.inertia.pct_change() * 100).round(1)))

save_metrics("method_checks", out)
print(out)
