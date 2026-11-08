"""Behavioural segmentation (unsupervised) on the mar cohort; churn is used only to profile.

1. Features: subscription + listening behaviour, log1p for heavy tails, median imputation,
   standardisation. The label is never an input.
2. Model selection for K-Means over k=2..10 with four internal criteria (inertia/elbow,
   silhouette, Davies-Bouldin, Calinski-Harabasz). k is chosen by silhouette among k>=3
   unless a smaller k is clearly better (a 2-cluster split is rarely actionable).
3. Stability: bootstrap K-Means, predict on a fixed reference set, ARI vs reference labels.
4. Alternatives on a common subsample: Agglomerative (Ward), Gaussian mixture, HDBSCAN.
5. Profiles: size, churn rate with 95 % CI, lift, feature means, z-scores -> automatic names.

Outputs: report/cluster_quality.csv, cluster_algorithms.csv, cluster_profile.csv,
cluster_drilldown.csv, figures/cluster_*.png, metrics/clustering.json,
data/clusters_mar.parquet (msno -> cluster) for the other scripts.
"""
import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN, AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score,
                             normalized_mutual_info_score, silhouette_score)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, REPORT, SEED, churn_table, load, save_metrics, savefig

N = int(os.environ.get("CLUSTER_SAMPLE", 200_000))
N_ALT = int(os.environ.get("CLUSTER_ALT_SAMPLE", 15_000))  # O(n^2) algorithms
K_RANGE = range(2, 11)
FEATS = ["n_tx", "n_cancel", "auto_renew_rate", "discount_rate", "last_plan_days", "last_paid",
         "days_to_expire", "tenure_days", "active_days", "total_hours", "complete_ratio",
         "skip_ratio", "avg_unique_songs", "listen_trend", "days_since_last_listen", "weekly_cv"]
LOGT = ["n_tx", "n_cancel", "last_plan_days", "last_paid", "tenure_days", "active_days",
        "total_hours", "avg_unique_songs", "days_since_last_listen"]

t0 = time.time()
df = load("mar", sample=N)


def prepare(d):
    """Feature matrix before scaling. Users without logs get explicit 'inactive' values."""
    raw = d[FEATS].copy()
    nolog = d.has_logs == 0
    raw.loc[nolog, ["complete_ratio", "skip_ratio", "listen_trend"]] = 0
    raw.loc[nolog, "days_since_last_listen"] = 60
    raw.loc[nolog, "weekly_cv"] = 3.0          # maximal irregularity
    raw["days_to_expire"] = raw.days_to_expire.clip(-60, 400)
    raw[LOGT] = np.log1p(raw[LOGT].clip(lower=0))
    return raw


R = prepare(df)
MED = R.median()
scaler = StandardScaler().fit(R.fillna(MED))
X = scaler.transform(R.fillna(MED))
rng = np.random.default_rng(SEED)
sil_idx = rng.choice(len(X), min(20_000, len(X)), replace=False)

# ---- 1. choose k --------------------------------------------------------------------------
rows = []
for k in K_RANGE:
    km = KMeans(k, n_init=5, random_state=SEED).fit(X)
    lab = km.labels_
    rows.append({"k": k, "inertia": km.inertia_,
                 "silhouette": silhouette_score(X[sil_idx], lab[sil_idx]),
                 "davies_bouldin": davies_bouldin_score(X[sil_idx], lab[sil_idx]),
                 "calinski_harabasz": calinski_harabasz_score(X[sil_idx], lab[sil_idx]),
                 "min_cluster_share": np.bincount(lab).min() / len(lab)})
    print(f"k={k} sil={rows[-1]['silhouette']:.3f} DB={rows[-1]['davies_bouldin']:.3f}")
quality = pd.DataFrame(rows)
cand = quality[(quality.k >= 3) & (quality.min_cluster_share >= 0.02)]
best = cand.loc[cand.silhouette.idxmax()] if len(cand) else quality.loc[quality.silhouette.idxmax()]
k2 = quality.loc[quality.k == 2].iloc[0]
if k2.silhouette > best.silhouette + 0.1:
    best = k2
K = int(best.k)
print(f"chosen k={K}")

# ---- 2. stability (bootstrap ARI on a fixed reference set) ---------------------------------
km = KMeans(K, n_init=10, random_state=SEED).fit(X)
ref_idx = rng.choice(len(X), min(20_000, len(X)), replace=False)
ref = km.predict(X[ref_idx])
aris = []
for b in range(10):
    bi = rng.choice(len(X), len(X), replace=True)
    kb = KMeans(K, n_init=3, random_state=SEED + b + 1).fit(X[bi])
    aris.append(adjusted_rand_score(ref, kb.predict(X[ref_idx])))
quality["chosen"] = quality.k == K
quality.to_csv(REPORT / "cluster_quality.csv", index=False)
print(f"bootstrap ARI {np.mean(aris):.3f} +- {np.std(aris):.3f}")

fig, axes = plt.subplots(1, 4, figsize=(16, 3.6))
for ax, m in zip(axes, ["inertia", "silhouette", "davies_bouldin", "calinski_harabasz"]):
    ax.plot(quality.k, quality[m], marker="o"); ax.axvline(K, color="crimson", ls="--")
    ax.set_title(m + (" (lower better)" if m in ("inertia", "davies_bouldin") else " (higher better)"))
    ax.set_xlabel("k")
savefig(fig, "cluster_selection.png")

# ---- 3. alternative algorithms on a shared subsample -----------------------------------------
alt_idx = rng.choice(len(X), min(N_ALT, len(X)), replace=False)
Xa = X[alt_idx]
algos = {
    "K-Means": lambda: km.predict(Xa),
    "Agglomerative (Ward)": lambda: AgglomerativeClustering(K, linkage="ward").fit_predict(Xa),
    "Gaussian mixture": lambda: GaussianMixture(K, covariance_type="full", random_state=SEED).fit(Xa).predict(Xa),
    "HDBSCAN": lambda: HDBSCAN(min_cluster_size=max(50, len(Xa) // 100)).fit_predict(Xa),
}
alt_rows, alt_labels = [], {}
y_alt = df.is_churn.values[alt_idx]
for name, fn in algos.items():
    s = time.time(); lab = fn(); el = time.time() - s
    alt_labels[name] = lab
    m = lab >= 0                       # HDBSCAN marks noise as -1
    n_cl = len(set(lab[m]))
    ok = n_cl >= 2
    # churn separation: share of label entropy explained by clusters (normalised MI)
    alt_rows.append({"algorithm": name, "n_clusters": n_cl, "noise_share": 1 - m.mean(),
                     "silhouette": silhouette_score(Xa[m], lab[m]) if ok else np.nan,
                     "davies_bouldin": davies_bouldin_score(Xa[m], lab[m]) if ok else np.nan,
                     "ari_vs_kmeans": adjusted_rand_score(alt_labels["K-Means"][m], lab[m]),
                     "churn_rate_spread": pd.Series(y_alt[m]).groupby(lab[m]).mean().agg(lambda s: s.max() - s.min()),
                     "nmi_with_churn": normalized_mutual_info_score(y_alt[m], lab[m]),
                     "runtime_s": el})
algos_df = pd.DataFrame(alt_rows)
algos_df.to_csv(REPORT / "cluster_algorithms.csv", index=False)
print(algos_df.round(3).to_string(index=False))

# ---- 4. profiles and names -----------------------------------------------------------------
df["cluster"] = km.labels_
base = df.is_churn.mean()
prof = churn_table(df, "cluster")
means = df.groupby("cluster")[FEATS + ["has_logs"]].mean()
prof = prof.join(means)
prof["share"] = prof.n / prof.n.sum()
z = ((means[FEATS] - df[FEATS].mean()) / df[FEATS].std().replace(0, np.nan)).fillna(0)

LABELS = {  # (feature, sign) -> short Vietnamese descriptor used in the report
    ("active_days", 1): "nghe nhiều", ("active_days", -1): "ít/không nghe",
    ("auto_renew_rate", 1): "tự gia hạn", ("auto_renew_rate", -1): "trả thủ công",
    ("n_cancel", 1): "từng hủy", ("last_plan_days", 1): "gói dài hạn",
    ("discount_rate", 1): "săn khuyến mãi", ("tenure_days", 1): "lâu năm", ("tenure_days", -1): "mới",
    ("listen_trend", -1): "nghe giảm dần", ("listen_trend", 1): "nghe tăng",
    ("days_since_last_listen", 1): "im lặng gần đây", ("skip_ratio", 1): "hay bỏ qua bài",
    ("n_tx", 1): "giao dịch nhiều", ("days_to_expire", 1): "còn hạn dài",
    ("weekly_cv", 1): "nghe thất thường", ("total_hours", 1): "nghe nhiều giờ",
    ("complete_ratio", 1): "nghe hết bài", ("avg_unique_songs", 1): "đa dạng bài",
    ("last_paid", 1): "trả giá cao", ("last_paid", -1): "trả giá thấp/0đ",
}
names = {}
for c in z.index:
    tags = []
    for f, v in z.loc[c].abs().sort_values(ascending=False).items():
        key = (f, int(np.sign(z.loc[c, f])))
        if key in LABELS and abs(z.loc[c, f]) > 0.35 and LABELS[key] not in tags:
            tags.append(LABELS[key])
        if len(tags) == 3:
            break
    names[c] = ", ".join(tags) if tags else "trung bình"
prof.insert(0, "name", pd.Series(names))
prof.to_csv(REPORT / "cluster_profile.csv")
z.to_csv(REPORT / "cluster_zscores.csv")
print(prof[["name", "n", "share", "churn_rate", "ci_low", "ci_high", "lift"]].round(3).to_string())

# drill-down: churn inside each cluster by auto-renew and recent listening
dd = df.assign(auto_renew=df.last_auto_renew.map({1: "on", 0: "off"}).fillna("no tx"),
               idle=np.where(df.has_logs == 0, "no logs",
                             np.where(df.days_since_last_listen > 7, "idle>7d", "active<=7d")))
drill = (dd.groupby(["cluster", "auto_renew", "idle"])["is_churn"].agg(n="size", churn_rate="mean")
           .reset_index())
drill = drill[drill.n >= 100]
drill.to_csv(REPORT / "cluster_drilldown.csv", index=False)

# ---- 5. figures -------------------------------------------------------------------------
pts = PCA(2, random_state=SEED).fit_transform(X[alt_idx])
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for c in range(K):
    m = km.labels_[alt_idx] == c
    axes[0].scatter(pts[m, 0], pts[m, 1], s=3, alpha=0.5, label=f"{c}: {names[c]}")
axes[0].legend(markerscale=4, fontsize=7); axes[0].set_title(f"K-Means k={K} in PCA space")
axes[1].scatter(pts[:, 0], pts[:, 1], c=y_alt, s=3, cmap="coolwarm", alpha=0.5)
axes[1].set_title("same points coloured by churn (red)")
savefig(fig, "clusters_pca.png")

fig, ax = plt.subplots(figsize=(9, 4.5))
order = prof.sort_values("churn_rate").index
ax.barh([f"{c}: {names[c]}" for c in order], prof.loc[order, "churn_rate"],
        xerr=[prof.loc[order, "churn_rate"] - prof.loc[order, "ci_low"],
              prof.loc[order, "ci_high"] - prof.loc[order, "churn_rate"]], color="#4C72B0")
ax.axvline(base, color="crimson", ls="--", label=f"overall {base:.1%}")
for i, c in enumerate(order):
    ax.text(prof.loc[c, "churn_rate"], i, f"  {prof.loc[c, 'churn_rate']:.1%} (n={prof.loc[c, 'n']:,})", va="center", fontsize=8)
ax.set_xlabel("churn rate (95% CI)"); ax.legend(); ax.set_title("Churn rate per segment")
savefig(fig, "cluster_churn.png")

fig, ax = plt.subplots(figsize=(11, 0.6 * K + 2))
im = ax.imshow(z.values, cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
ax.set_xticks(range(len(FEATS)), FEATS, rotation=60, ha="right", fontsize=8)
ax.set_yticks(range(K), [f"{c}: {names[c]}" for c in z.index], fontsize=8)
fig.colorbar(im); ax.set_title("Cluster centroids (z-score vs population)")
savefig(fig, "cluster_heatmap.png")

full = load("mar")  # assign every user of the cohort to its nearest centroid
full_lab = km.predict(scaler.transform(prepare(full).fillna(MED)))
pd.DataFrame({"msno": full.msno, "cluster": full_lab}).to_parquet(DATA / "clusters_mar.parquet")
full_prof = churn_table(full.assign(cluster=full_lab), "cluster")
full_prof.insert(0, "name", pd.Series(names))
full_prof.to_csv(REPORT / "cluster_profile_full.csv")
print("full cohort:", full_prof.round(3).to_string(), sep="\n")
save_metrics("clustering", {
    "n_users": len(df), "k": K, "silhouette": best.silhouette, "davies_bouldin": best.davies_bouldin,
    "calinski_harabasz": best.calinski_harabasz, "bootstrap_ari_mean": np.mean(aris),
    "bootstrap_ari_std": np.std(aris), "base_churn": base, "names": names,
    "algorithms": algos_df.to_dict("records"), "runtime_s": time.time() - t0,
})
