"""Build the interactive product demo (demo/kkbox_demo.html) from the pipeline results.

Run after run_all.py:  python demo/build_demo.py
Reads report/*.csv and report/metrics/*.json, embeds them into demo/template.html and
writes a single self-contained page that opens in any browser (no server needed).
"""
import json
import os
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
REPORT = Path(os.environ.get("REPORT_DIR", ROOT / "report"))
DEMO = ROOT / "demo"
# A smoke run with REPORT_DIR=report_synth writes its demo next to those results, not over demo/.
OUT = REPORT / "kkbox_demo.html" if "REPORT_DIR" in os.environ else DEMO / "kkbox_demo.html"


def load_rules(name, direction):
    d = pd.read_csv(REPORT / name)
    return [dict(items=[i.strip() for i in r.rule.split("&")], sup=round(r.support_holdout, 4),
                 conf=round(r.confidence_holdout, 4), lift=round(r.lift_holdout, 3),
                 conf_tr=round(r.confidence, 4), dir=direction) for r in d.itertuples()]


def build_data():
    out = {}
    seg = pd.read_csv(REPORT / "cluster_profile_full.csv")
    names = json.loads((REPORT / "segment_names.json").read_text(encoding="utf-8"))
    seg["name"] = seg.cluster.astype(str).map(names).fillna(seg.name)
    out["segments"] = seg[["cluster", "name", "n", "churn_rate", "ci_low", "ci_high", "lift", "share",
                           "auto_renew_rate", "active_days", "last_plan_days", "n_cancel", "discount_rate",
                           "tenure_days", "days_since_last_listen", "listen_trend"]].round(4).to_dict("records")
    out["rules"] = load_rules("churn_rules.csv", "churn") + load_rules("retention_rules.csv", "retain")
    s = pd.read_csv(REPORT / "sequential_patterns.csv")
    out["seqs"] = [dict(steps=[[t.strip() for t in el.split(",")] for el in re.findall(r"\{([^}]*)\}", r.pattern)],
                        sup=round(r.support_holdout, 4), conf=round(r.conf_holdout, 4), lift=round(r.lift_holdout, 3),
                        conf_feb=round(r.conf_feb, 4), dir=r.direction) for r in s.itertuples()]
    out["coverage"] = pd.read_csv(REPORT / "rule_coverage.csv").round(4).to_dict("records")
    pc = pd.read_csv(REPORT / "profit_curve.csv")
    out["profit"] = {str(k): g[["share_contacted", "net_value_ntd", "random_targeting_ntd", "churners_reached"]]
                     .round(4).values.tolist() for k, g in pc.groupby("success_rate")}
    e = pd.read_csv(REPORT / "eda_churn_by_group.csv")
    out["groups"] = {f: {str(r.level): [int(r.n), round(r.churn_rate, 4)] for r in g.itertuples()}
                     for f, g in e.groupby("feature")}
    c = json.loads((REPORT / "metrics" / "classification.json").read_text(encoding="utf-8"))
    keep = ("model", "roc_auc", "pr_auc", "recall@10%", "precision@10%", "lift@10%")
    out["models"] = [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items() if k in keep}
                     for m in c["out_of_time"]]
    out["prior"] = round(c["churn_test"], 4)
    out["assume"] = json.loads((REPORT / "metrics" / "business_value.json").read_text(encoding="utf-8"))["assumptions"]
    return out


def main():
    data = json.dumps(build_data(), ensure_ascii=False, separators=(",", ":"))
    page = (DEMO / "template.html").read_text(encoding="utf-8").replace("/*DATA*/", data)
    head = '<!doctype html>\n<html lang="vi"><head><meta charset="utf-8">' \
           '<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">\n'
    OUT.write_text(head + page + "\n</html>\n", encoding="utf-8")
    print("written", OUT)


if __name__ == "__main__":
    main()
