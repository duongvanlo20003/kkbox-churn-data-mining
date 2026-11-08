"""Association rules X -> CHURN (and protective X -> low churn) with FP-Growth, validated on holdout.

Design choices (and why):
* Items come from business thresholds or quantiles learnt on the TRAIN split only (no rank-based
  binning, which split ties arbitrarily and created spurious items in v1).
* FP-Growth enumerates frequent antecedents X (label excluded, max 3 items). For each X we
  compute confidence P(churn|X), lift, leverage and a one-sided binomial z-test against the base
  rate; p-values are corrected with Benjamini-Hochberg (thousands of hypotheses are tested).
* Productive-rule filter (Webb 2007): a rule is kept only if it beats every one of its sub-rules
  by at least MIN_IMPROVE in confidence (churn rules) or is lower by that much (protective).
* Every reported rule is re-measured on a 30 % holdout split never used for mining.
* Sensitivity: rule counts and top-10 stability across min_support values.

Outputs: report/churn_rules.csv, retention_rules.csv, rule_sensitivity.csv, metrics/rules.json,
figures/rules_scatter.png
"""
import os
import sys
import time
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import fpgrowth
from scipy.stats import norm
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, REPORT, SEED, bh_fdr, load, save_metrics, savefig

MIN_SUPPORT = float(os.environ.get("MIN_SUPPORT", 0.01))   # of the antecedent X
MIN_IMPROVE = 0.02
MAX_LEN = 3

df = load("mar")
cl = DATA / "clusters_mar.parquet"
if cl.exists():  # segment label from clustering as an extra item (NaN for users outside the sample)
    df = df.merge(pd.read_parquet(cl), on="msno", how="left")
tr, ho = train_test_split(df, test_size=0.3, stratify=df.is_churn, random_state=SEED)


def itemise(d, ref):
    """Map users to items. Quantile cut-points are taken from ``ref`` (the train split)."""
    act_q = ref.loc[ref.has_logs == 1, "active_days"].quantile([1 / 3, 2 / 3]).values
    skip_q = ref.loc[ref.has_logs == 1, "skip_ratio"].quantile(2 / 3)
    logs = d.has_logs == 1
    it = pd.DataFrame(index=d.index)
    it["renew"] = d.last_auto_renew.map({1: "auto_renew=on", 0: "auto_renew=off"}).fillna("no_tx")
    it["cancel"] = np.where(d.last_is_cancel == 1, "last_tx=cancel",
                   np.where(d.n_cancel > 0, "cancelled_before", "never_cancelled"))
    it["plan"] = pd.cut(d.last_plan_days, [-1, 7, 30, 10_000],
                        labels=["plan<=7d", "plan=30d", "plan>30d"]).astype(str).replace("nan", "plan=none")
    it["price"] = np.select([d.last_paid == 0, d.last_paid < d.last_list_price],
                            ["paid=free", "paid=discount"], "paid=full")
    it["tenure"] = pd.cut(d.tenure_days, [-1, 365, 3 * 365, 6 * 365, 1e6],
                          labels=["tenure<1y", "tenure1-3y", "tenure3-6y", "tenure>6y"]).astype(str)
    it["via"] = "via=" + d.registered_via.astype("Int64").astype(str)
    it["gender"] = "gender=" + d.gender.astype(str)
    it["activity"] = np.where(~logs, "no_logs", pd.cut(d.active_days, [-1, act_q[0], act_q[1], 1e6],
                              labels=["active=low", "active=mid", "active=high"]).astype(str))
    it["idle"] = np.where(~logs, None, pd.cut(d.days_since_last_listen, [-1, 3, 14, 1e6],
                          labels=["idle<=3d", "idle4-14d", "idle>14d"]).astype(str))
    it["trend"] = np.where(~logs, None, pd.cut(d.listen_trend, [-99, -0.5, 0.5, 99],
                           labels=["trend=falling", "trend=flat", "trend=rising"]).astype(str))
    it["skip"] = np.where(logs & (d.skip_ratio > skip_q), "skip=high", None)
    it["expire"] = pd.cut(d.days_to_expire, [-1e6, 0, 31, 1e6],
                          labels=["expired_at_T", "expires<=31d", "expires>31d"]).astype(str).replace("nan", None)
    if "cluster" in d:
        it["segment"] = np.where(d.cluster.isna(), None, "segment=" + d.cluster.astype("Int64").astype(str))
    rows = it.astype(object).where(it.notna(), None).values.tolist()
    vocab = sorted({v for r in rows for v in r if v is not None})
    idx = {v: i for i, v in enumerate(vocab)}
    M = np.zeros((len(rows), len(vocab)), bool)
    for i, r in enumerate(rows):
        for v in r:
            if v is not None:
                M[i, idx[v]] = True
    return pd.DataFrame(M, columns=vocab, index=d.index)


Itr, Iho = itemise(tr, tr), itemise(ho, tr)
Iho = Iho.reindex(columns=Itr.columns, fill_value=False)
ytr, yho = tr.is_churn.values, ho.is_churn.values
base = ytr.mean()


def evaluate(itemsets, I, y):
    out = []
    A = I.values
    col = {c: i for i, c in enumerate(I.columns)}
    for s in itemsets:
        m = A[:, [col[c] for c in s]].all(1)
        n = m.sum()
        k = y[m].sum()
        out.append((n, k))
    n, k = np.array(out).T if out else (np.array([]), np.array([]))
    return n, k


def mine(min_support):
    t = time.time()
    freq = fpgrowth(Itr, min_support=min_support, use_colnames=True, max_len=MAX_LEN)
    sets = [frozenset(s) for s in freq.itemsets]
    n, k = evaluate(sets, Itr, ytr)
    r = pd.DataFrame({"items": sets, "n": n, "churners": k})
    r["len"] = r["items"].map(len)
    r["support"] = r.n / len(Itr)
    r["rule_support"] = r.churners / len(Itr)
    r["confidence"] = r.churners / r.n
    r["lift"] = r.confidence / base
    r["leverage"] = r.rule_support - r.support * base
    r["conviction"] = (1 - base) / (1 - r.confidence).clip(lower=1e-9)
    zstat = (r.confidence - base) / np.sqrt(base * (1 - base) / r.n)
    r["p_value"] = 2 * norm.sf(np.abs(zstat))
    r["q_value"] = bh_fdr(r.p_value)
    return r, time.time() - t


def productive(r, direction):
    conf = dict(zip(r["items"], r.confidence))
    keep = []
    for s, c in zip(r["items"], r.confidence):
        subs = [frozenset(x) for L in range(1, len(s)) for x in combinations(s, L)]
        subs = [conf[x] for x in subs if x in conf]
        if direction == "churn":
            keep.append(all(c >= sc + MIN_IMPROVE for sc in subs))
        else:
            keep.append(all(c <= sc - MIN_IMPROVE for sc in subs))
    return r[keep]


def select(r, direction, top=25):
    sig = r[r.q_value < 0.01]
    sig = sig[sig.lift >= 1.5] if direction == "churn" else sig[sig.lift <= 0.5]
    sig = productive(sig, direction)
    return sig.sort_values("lift", ascending=direction != "churn").head(top)


rules, runtime = mine(MIN_SUPPORT)
print(f"{len(rules):,} frequent antecedents in {runtime:.1f}s, base churn {base:.3%}")
out = {}
for direction in ("churn", "retain"):
    s = select(rules, direction).copy()
    n, k = evaluate(list(s["items"]), Iho, yho)
    s["support_holdout"] = n / len(Iho)
    s["confidence_holdout"] = k / np.maximum(n, 1)
    s["lift_holdout"] = s.confidence_holdout / yho.mean()
    s["rule"] = s["items"].map(lambda x: " & ".join(sorted(x)))
    cols = ["rule", "len", "support", "confidence", "lift", "leverage", "conviction", "q_value",
            "support_holdout", "confidence_holdout", "lift_holdout"]
    out[direction] = s[cols]
    print(f"\n{direction} rules:\n", s[cols].head(12).round(3).to_string(index=False))
out["churn"].to_csv(REPORT / "churn_rules.csv", index=False)
out["retain"].to_csv(REPORT / "retention_rules.csv", index=False)

# ---- sensitivity to min_support -----------------------------------------------------------
top_ref = set(out["churn"].rule.head(10))
sens = []
for ms in [0.005, 0.01, 0.02, 0.05]:
    r, rt = mine(ms)
    s = select(r, "churn", top=10)
    names = {" & ".join(sorted(x)) for x in s["items"]}
    sens.append({"min_support": ms, "n_itemsets": len(r), "n_significant_churn_rules":
                 int(((r.q_value < 0.01) & (r.lift >= 1.5)).sum()), "n_productive_top": len(s),
                 "max_lift": s.lift.max() if len(s) else np.nan,
                 "jaccard_top10_vs_default": len(names & top_ref) / max(1, len(names | top_ref)),
                 "runtime_s": rt})
sens = pd.DataFrame(sens)
sens.to_csv(REPORT / "rule_sensitivity.csv", index=False)
print("\n", sens.round(3).to_string(index=False))

fig, ax = plt.subplots(figsize=(8, 5))
sc = ax.scatter(rules.support, rules.confidence, c=rules.lift, cmap="viridis", s=8, alpha=0.6)
ax.scatter(out["churn"].support, out["churn"].confidence, facecolors="none", edgecolors="red", s=40,
           label="selected churn rules")
ax.axhline(base, color="crimson", ls="--", lw=1, label=f"base churn {base:.1%}")
ax.set_xscale("log"); ax.set_xlabel("support of antecedent (log)"); ax.set_ylabel("P(churn | X)")
fig.colorbar(sc, label="lift"); ax.legend(); ax.set_title("Frequent itemsets: support vs churn confidence")
savefig(fig, "rules_scatter.png")

c = out["churn"]
save_metrics("rules", {
    "n_train": len(Itr), "n_holdout": len(Iho), "n_items": Itr.shape[1], "base_churn": base,
    "min_support": MIN_SUPPORT, "n_itemsets": len(rules), "runtime_s": runtime,
    "n_churn_rules": len(c), "n_retention_rules": len(out["retain"]),
    "holdout_lift_corr": float(np.corrcoef(c.lift, c.lift_holdout)[0, 1]) if len(c) > 2 else None,
    "mean_abs_conf_drift": float((c.confidence - c.confidence_holdout).abs().mean()) if len(c) else None,
    "top_churn": c.head(5)[["rule", "confidence_holdout", "lift_holdout", "support"]].to_dict("records"),
    "top_retain": out["retain"].head(5)[["rule", "confidence_holdout", "lift_holdout", "support"]].to_dict("records"),
})
