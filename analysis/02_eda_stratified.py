"""Stratified EDA: is a factor's effect real, or does it travel through auto-renew?

* Churn by registration channel x auto-renew (confounding check, report table 16).
* Churn by listening level x auto-renew (interaction check, report table 17). Listening levels are
  the same tercile items as the association rules (active=low/mid/high, no_logs), so the cell
  "low / auto-renew off" equals the rule active=low & auto_renew=off.
* Share of users with unknown gender per channel (is missing gender "missing not at random"?).
* Figure 3 v2: churn vs stay distributions, listening features only for users with logs and
  integer features with bins of width 1 (the v1 figure mixed in the 19 % no-log users as zeros and
  aliased integer bins).
Tables use the same 70 % train split as mining/02_association_rules.py, so they match the rules.

Outputs: report/eda_stratified_via.csv, eda_stratified_activity.csv, eda_gender_by_via.csv,
figures/eda_distributions_v2.png, metrics/eda_stratified.json
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mining"))
from common import DATA, REPORT, SEED, load, save_metrics, savefig
from transforms import itemise

df = load("mar")
cl = DATA / "clusters_mar.parquet"
if cl.exists():  # same frame (and therefore the same split) as the association rules
    df = df.merge(pd.read_parquet(cl), on="msno", how="left")
tr, _ = train_test_split(df, test_size=0.3, stratify=df.is_churn, random_state=SEED)
I = itemise(tr, tr)
tr = tr.assign(renew=tr.last_auto_renew.map({1: "on", 0: "off"}))
levels = ["active=low", "active=mid", "active=high", "no_logs"]
tr["activity"] = np.select([I[c].values for c in levels], levels, "other")


def strat(col):
    t = tr.dropna(subset=["renew"]).groupby([col, "renew"]).is_churn.agg(churn_rate="mean", n="size").unstack()
    t.columns = [f"{a}_{b}" for a, b in t.columns]
    t["churn_all"] = tr.groupby(col).is_churn.mean()
    return t


via = strat("registered_via")
via = via[via[["n_off", "n_on"]].sum(axis=1) >= 1000]
via.round(4).to_csv(REPORT / "eda_stratified_via.csv")
act = strat("activity").reindex(levels)
act.round(4).to_csv(REPORT / "eda_stratified_activity.csv")
g = df.assign(unknown=df.gender.isna() | (df.gender.astype(str) == "unknown")).groupby("registered_via") \
      .agg(n=("unknown", "size"), share_unknown=("unknown", "mean"), churn=("is_churn", "mean"))
g = g[g.n >= 1000]
g.round(4).to_csv(REPORT / "eda_gender_by_via.csv")
print("channel x auto-renew:\n", via.round(3).to_string(), "\nlistening x auto-renew:\n", act.round(3).to_string(),
      "\nunknown gender by channel:\n", g.round(3).to_string())

# ---- Figure 3 v2 ----------------------------------------------------------------------------
logs = df.has_logs == 1
panels = [  # column, title, users with logs only, integer bins, upper clip quantile
    ("active_days", "Số ngày có nghe (60 ngày)", True, True, 1.0),
    ("total_hours", "Tổng giờ nghe (60 ngày)", True, False, 0.99),
    ("days_since_last_listen", "Số ngày từ lần nghe cuối", True, True, 1.0),
    ("listen_trend", "Xu hướng nghe (listen_trend)", True, False, 1.0),
    ("n_tx", "Số giao dịch (n_tx)", False, True, 0.99),
    ("tenure_days", "Thâm niên (ngày)", False, False, 0.99),
]
fig, axes = plt.subplots(2, 3, figsize=(11, 6.6))
for ax, (col, title, need_logs, integer, q) in zip(axes.flat, panels):
    d = df[logs] if need_logs else df
    v = d[col].dropna()
    hi, lo = v.quantile(q), v.min()
    bins = np.arange(np.floor(lo) - 0.5, np.ceil(hi) + 1.5, 1) if integer else np.linspace(lo, hi, 41)
    for y, c, lab in [(0, "#2F5D9C", "ở lại"), (1, "#D9622B", "churn")]:
        w = d.loc[d.is_churn == y, col].dropna()
        ax.hist(w[w <= hi], bins=bins, density=True, alpha=0.55, color=c, label=lab, linewidth=0)
    ax.set_title(title + (" – khách có log" if need_logs else ""), fontsize=11)
    ax.set_ylabel("mật độ", fontsize=9)
    ax.tick_params(labelsize=9)
axes.flat[0].legend(frameon=False)
fig.suptitle("Phân phối hành vi của khách churn và khách ở lại (cohort tháng 3; hành vi nghe chỉ tính khách có log; "
             "biến số nguyên dùng bin rộng 1)", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.95))
savefig(fig, "eda_distributions_v2.png")

save_metrics("eda_stratified", {
    "split": "train 70% of mar (same as association rules)",
    "via_x_renew": via.round(4).to_dict("index"),
    "activity_x_renew": act.round(4).to_dict("index"),
    "gender_unknown_by_via": g.share_unknown.round(4).to_dict(),
})
