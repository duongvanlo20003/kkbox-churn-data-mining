"""Mine discriminative transaction-itemset sequences with pure NumPy PrefixSpan.

Patterns contain sorted itemsets: {FREE, PAY_AUTO} -> {CANCEL} requires the
first two facts in one transaction and cancellation in a later transaction.
Length counts tokens, capped at four, not elapsed time or transaction count.
The projected database stores ALL possible ending positions as 12-bit masks;
keeping only the earliest ending would lose valid same-itemset extensions.
Support counts distinct users, including empty histories in the denominator.

A deterministic label-stratified sample (SEQ_SAMPLE=300000; 0 means all) is
split 70/30 with seed 42. Mining, Pearson chi-square tests (1 df), BH correction,
ranking and redundancy pruning use training users only. BH covers every pattern
mined at the lowest sensitivity/requested threshold, including neutral patterns.
Sensitivity reuses that family and its q-values; no holdout-based selection.
Redundancy requires a proper sequential subpattern in the same direction with
at least 95% of the longer pattern's log-lift magnitude. SEQ_TOP=50 limits the
reported patterns after pruning. Holdout is descriptive replication, not a test
used for filtering; zero matches yield null confidence/lift.
"""

import json
import math
import os
import time
from pathlib import Path

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MAX_LENGTH = 4
COLUMNS = ["pattern", "len", "support_train", "conf_train", "lift_train", "q_value",
           "support_holdout", "conf_holdout", "lift_holdout", "direction"]


def stratified_indices(labels, size, rng):
    """Allocate a fixed-size sample proportionally across binary classes."""
    size = min(size, len(labels))
    n_positive = int(round(size * float(labels.mean())))
    chosen = []
    for value, count in ((0, size - n_positive), (1, n_positive)):
        pool = np.flatnonzero(labels == value)
        chosen.append(rng.permutation(pool)[:count])
    return rng.permutation(np.concatenate(chosen))


def split_indices(labels, rng):
    """Keep both classes in each split whenever the input permits it."""
    train, holdout = [], []
    for value in (0, 1):
        pool = rng.permutation(np.flatnonzero(labels == value))
        if len(pool) < 2:
            raise ValueError("Each class needs at least two sampled users for a stratified split")
        n_train = min(len(pool) - 1, max(1, int(round(0.7 * len(pool)))))
        train.extend(pool[:n_train])
        holdout.extend(pool[n_train:])
    return np.asarray(train), np.asarray(holdout)


def encode(sequences):
    """Build one uint16 occurrence mask per user and token (at most 12 bits)."""
    vocabulary = sorted({str(token) for seq in sequences for event in seq for token in event})
    mapping = {token: i for i, token in enumerate(vocabulary)}
    masks = np.zeros((len(vocabulary), len(sequences)), dtype=np.uint16)
    for user, seq in enumerate(sequences):
        if len(seq) > 12:
            raise ValueError("Input has more than 12 events; rebuild sequences first")
        for pos, event in enumerate(seq):
            if isinstance(event, str):
                raise ValueError("Expected nested itemsets, not a flat token sequence")
            for token in event:
                masks[mapping[str(token)], user] |= np.uint16(1 << pos)
    return vocabulary, masks


def suffix_mask(ends):
    """Return positions strictly after the earliest possible pattern ending."""
    bits = ends.astype(np.uint32)
    first = bits & (~bits + np.uint32(1))
    return ((~(2 * first - 1)) & 4095).astype(np.uint16)


def mine(masks, labels, minimum):
    """PrefixSpan with canonical itemset and sequence extensions, user support."""
    n_users = len(labels)
    threshold = max(1, math.ceil(minimum * n_users - 1e-12))
    frequent = [i for i, row in enumerate(masks) if np.count_nonzero(row) >= threshold]
    records = []

    def visit(pattern, users, ends, length):
        records.append({"key": pattern, "len": length, "count": len(users),
                        "churn_count": int(labels[users].sum())})
        if length == MAX_LENGTH:
            return
        after = suffix_mask(ends)
        for token in frequent:
            occurrences = masks[token, users]
            # Lexicographic order gives each within-event itemset exactly one path.
            if token > pattern[-1][-1]:
                new_ends = ends & occurrences
                keep = np.flatnonzero(new_ends)
                if len(keep) >= threshold:
                    extended = pattern[:-1] + (pattern[-1] + (token,),)
                    visit(extended, users[keep], new_ends[keep], length + 1)
            new_ends = after & occurrences
            keep = np.flatnonzero(new_ends)
            if len(keep) >= threshold:
                visit(pattern + ((token,),), users[keep], new_ends[keep], length + 1)

    for token in frequent:
        users = np.flatnonzero(masks[token])
        visit(((token,),), users, masks[token, users], 1)
    return records


def add_statistics(records, labels):
    """Pearson chi-square and monotone Benjamini-Hochberg adjusted p-values."""
    total, positives = len(labels), int(labels.sum())
    base = positives / total
    for row in records:
        matched, a = row["count"], row["churn_count"]
        b, c = matched - a, positives - a
        d = total - matched - c
        denominator = matched * (total - matched) * positives * (total - positives)
        statistic = total * (a * d - b * c) ** 2 / denominator if denominator else 0.0
        row.update(support_train=matched / total,
                   support_churn_train=a / positives,
                   support_stay_train=b / (total - positives), conf_train=a / matched,
                   lift_train=(a / matched) / base,
                   p_value=math.erfc(math.sqrt(statistic / 2)))
    if records:
        order = np.argsort([r["p_value"] for r in records])
        adjusted = np.array([records[i]["p_value"] for i in order])
        adjusted *= len(records) / np.arange(1, len(records) + 1)
        adjusted = np.minimum(1, np.minimum.accumulate(adjusted[::-1])[::-1])
        for index, q in zip(order, adjusted):
            records[index]["q_value"] = float(q)


def subpatterns(pattern):
    """Enumerate all proper nonempty subpatterns by deleting tokens/events."""
    found = set()
    pending = [pattern]
    while pending:
        current = pending.pop()
        for i, event in enumerate(current):
            for j in range(len(event)):
                smaller = event[:j] + event[j + 1:]
                candidate = current[:i] + ((smaller,) if smaller else ()) + current[i + 1:]
                if candidate and candidate not in found:
                    found.add(candidate)
                    pending.append(candidate)
    return found


def select_patterns(records, minimum, n_train):
    """Filter significance/effect and remove longer patterns with marginal gain."""
    threshold = math.ceil(minimum * n_train - 1e-12)
    eligible = {r["key"]: r for r in records if r["count"] >= threshold
                and r["q_value"] < 0.01 and (r["lift_train"] > 1.2 or r["lift_train"] < 0.8)}

    def strength(row):
        return abs(math.log(max(row["lift_train"], 1e-12)))

    selected = []
    for key, row in eligible.items():
        retain = row["lift_train"] < 1
        redundant = any(sub in eligible and (eligible[sub]["lift_train"] < 1) == retain
                        and strength(row) <= 1.05 * strength(eligible[sub])
                        for sub in subpatterns(key))
        if not redundant:
            row["direction"] = "retain" if retain else "churn"
            selected.append(row)
    return sorted(selected, key=lambda r: (-strength(r), r["q_value"], r["key"]))


def match_pattern(pattern, masks):
    """Evaluate a fixed pattern on all holdout users using the same semantics."""
    ends = None
    for event in pattern:
        occurrences = masks[event[0]].copy()
        for token in event[1:]:
            occurrences &= masks[token]
        if ends is None:
            ends = occurrences
        else:
            alive = ends != 0
            ends = occurrences & suffix_mask(ends)
            ends[~alive] = 0
    return ends != 0


def pattern_name(pattern, vocabulary):
    """Render unambiguous simultaneous itemsets and ordered transactions."""
    return " -> ".join("{" + ", ".join(vocabulary[t] for t in event) + "}" for event in pattern)


def main():
    """Write train-selected patterns, holdout estimates, sensitivity and figure."""
    started = time.perf_counter()
    data = Path(os.environ.get("DATA_DIR", ROOT / "data"))
    report = Path(os.environ.get("REPORT_DIR", ROOT / "report"))
    cohort = os.environ.get("COHORT", "mar")
    minimum = float(os.environ.get("SEQ_MIN_SUPPORT", os.environ.get("MIN_SUPPORT", "0.01")))
    sample_size = int(os.environ.get("SEQ_SAMPLE", "300000"))
    top = int(os.environ.get("SEQ_TOP", "50"))
    if not 0 < minimum <= 1 or sample_size < 0 or top < 1:
        raise ValueError("Support must be in (0,1], SEQ_SAMPLE >= 0 and SEQ_TOP >= 1")
    source = data / f"sequences_{cohort}.parquet"
    # Sample the small roster before materializing nested histories for a million users.
    frame = pd.read_parquet(source, columns=["msno", "is_churn"])
    if frame.empty or frame.msno.isna().any() or frame.msno.duplicated().any():
        raise ValueError("Sequences must contain one row per non-null user")
    if not frame.is_churn.isin([0, 1]).all():
        raise ValueError("is_churn must contain binary non-null labels")
    frame = frame.sort_values("msno").reset_index(drop=True)
    n_available = len(frame)
    rng = np.random.default_rng(42)
    if sample_size and sample_size < len(frame):
        frame = frame.iloc[stratified_indices(frame.is_churn.to_numpy(), sample_size, rng)]
    with duckdb.connect() as con:
        con.register("sampled_users", frame[["msno"]])
        histories = con.execute("""SELECT s.msno, s.seq FROM read_parquet(?) s
                                   JOIN sampled_users u USING (msno)""", [str(source)]).df()
    frame = frame.merge(histories, on="msno", how="left", sort=False, validate="one_to_one")
    labels = frame.is_churn.to_numpy(dtype=np.int8)
    train, holdout = split_indices(labels, rng)
    vocabulary, masks = encode(frame.seq.to_numpy())
    train_labels, hold_labels = labels[train], labels[holdout]
    train_masks, hold_masks = masks[:, train], masks[:, holdout]
    del masks
    thresholds = [0.005, 0.01, 0.02, 0.05]
    records = mine(train_masks, train_labels, min(minimum, *thresholds))
    add_statistics(records, train_labels)
    selected = select_patterns(records, minimum, len(train))
    rows = []
    base_holdout = float(hold_labels.mean())
    for row in selected[:top]:
        matched = match_pattern(row["key"], hold_masks)
        count = int(matched.sum())
        confidence = float(hold_labels[matched].mean()) if count else None
        rows.append({**{key: row[key] for key in COLUMNS if key in row},
                     "pattern": pattern_name(row["key"], vocabulary),
                     "support_holdout": count / len(holdout),
                     "conf_holdout": confidence,
                     "lift_holdout": confidence / base_holdout if count else None})
    sensitivity = []
    for threshold in thresholds:
        count_threshold = math.ceil(threshold * len(train) - 1e-12)
        sensitivity.append({"min_support": threshold,
                            "n_frequent": sum(r["count"] >= count_threshold for r in records),
                            "n_patterns": len(select_patterns(records, threshold, len(train)))})
    (report / "figures").mkdir(parents=True, exist_ok=True)
    (report / "metrics").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=COLUMNS).to_csv(report / "sequential_patterns.csv", index=False)
    pd.DataFrame(sensitivity).to_csv(report / "sequential_sensitivity.csv", index=False)
    displayed = rows[:15]
    fig, ax = plt.subplots(figsize=(12, max(3, len(displayed) * 0.5 + 1.5)))
    if displayed:
        y = np.arange(len(displayed))
        lifts = [r["lift_holdout"] if r["lift_holdout"] is not None else np.nan for r in displayed]
        ax.barh(y, lifts, color=["#ce5b51" if r["direction"] == "churn" else "#329985"
                               for r in displayed])
        ax.set_yticks(y, [r["pattern"] for r in displayed], fontsize=8)
        for index, row in enumerate(displayed):
            if row["lift_holdout"] is None:
                ax.text(0.02, index, "No holdout matches", va="center", fontsize=8)
        ax.invert_yaxis()
    else:
        ax.text(0.5, 0.5, "No patterns pass training selection", ha="center", transform=ax.transAxes)
    ax.axvline(1, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Holdout churn lift (selection and order use training data)")
    ax.set_title(f"Sequential patterns: {cohort}, holdout n={len(holdout):,}")
    fig.tight_layout()
    fig.savefig(report / "figures" / "seq_top_patterns.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    metrics = {"cohort": cohort, "n_users": len(frame), "n_available": n_available,
               "n_train": len(train), "n_holdout": len(holdout),
               "n_empty_sequences": int(sum(len(seq) == 0 for seq in frame.seq)),
               "base_rate": float(labels.mean()), "base_rate_train": float(train_labels.mean()),
               "base_rate_holdout": base_holdout, "min_support": minimum,
               "max_pattern_length": MAX_LENGTH, "n_tested": len(records),
               "n_selected": len(selected), "n_patterns": len(rows),
               "seed": 42, "test": "Pearson chi-square (1 df), BH FDR over mined family",
               "sensitivity_fdr": "Shared family at lowest requested/sensitivity support",
               "runtime_s": round(time.perf_counter() - started, 3),
               "top5": [{**row, "support_churn_train": chosen["support_churn_train"],
                         "support_stay_train": chosen["support_stay_train"]}
                        for row, chosen in zip(rows[:5], selected[:5])]}
    with (report / "metrics" / "sequential.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2, allow_nan=False)
    print(f"{len(frame):,} users; {len(records):,} frequent patterns; "
          f"{len(selected):,} selected; {len(rows)} reported; {metrics['runtime_s']:.2f}s")
    print(f"Outputs: {report.resolve()}")


if __name__ == "__main__":
    main()
