"""Build report/report.docx from the pipeline outputs (metrics/*.json, *.csv, figures/).

Every number in the report is read from the result files, so re-running the pipeline and then
this script keeps text and results consistent. Interpretive sentences are written around
those numbers; they are checked against the final run (see README "Kết quả chính").
"""
import json
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from docx_helpers import bullets, figure, new_document, para, table

R = Path(os.environ.get("REPORT_DIR", Path(__file__).resolve().parent))
F = R / "figures"
M = R / "metrics"
DATA = Path(os.environ.get("DATA_DIR", R.parent / "data"))


def j(name):
    p = M / f"{name}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def csv(name, **kw):
    p = R / name
    return pd.read_csv(p, **kw) if p.exists() else pd.DataFrame()


def pct(x, d=1):
    return "n/a" if x is None else f"{x * 100:.{d}f}%".replace(".", ",")


def num(x, d=3):
    return "n/a" if x is None else f"{x:.{d}f}".replace(".", ",")


def thousands(x):
    return f"{int(x):,}".replace(",", ".")


eda, cl, ru, cf, sq, bo, va = (j(n) for n in ["eda", "clustering", "rules", "classification",
                                               "sequential", "bonus", "validation"])
manifest = json.loads((DATA / "manifest.json").read_text()) if (DATA / "manifest.json").exists() else {}
oot = pd.DataFrame(cf.get("out_of_time", []))
best = cf.get("best_model", "LightGBM")
brow = oot.set_index("model").loc[best] if len(oot) and best in set(oot.model) else {}
audit = pd.DataFrame(va.get("leakage_audit", []))
abl = pd.DataFrame(cf.get("ablation", []))

d = new_document()
d.add_heading("Khám phá hành vi người dùng và các mẫu rời bỏ dịch vụ nhạc thuê bao (KKBox)", 0)
para(d, "DAM501 – Data Mining · Final Project", bold=True)
para(d, "Nhóm: Lộc Nguyễn Phúc, Khánh Phạm Duy Văn, Bình Chu Đức, Quân Đoàn Đàm, Hạnh Nguyễn Thị Hồng, "
        "Tùng Trần, Phùng Quý Quốc.")

# ---------------------------------------------------------------------------------------------
d.add_heading("Tóm tắt", 1)
para(d, f"Nghiên cứu áp dụng quy trình KDD trên dữ liệu thật của KKBox ({thousands(eda.get('n_mar', 0))} "
        f"người dùng hết hạn thuê bao tháng 3/2017, tỉ lệ churn {pct(eda.get('churn_mar'))}). Bản phân tích "
        "đầu tiên của nhóm đạt ROC-AUC 0,977 nhưng kiểm tra lại cho thấy kết quả đó dựa vào dữ liệu của chính "
        "tháng nhãn (rò rỉ thông tin). Phiên bản này xây lại toàn bộ pipeline với mốc quan sát (cutoff) nghiêm "
        "ngặt, đánh giá ngoài thời gian (train tháng 2, test tháng 3) và kết hợp bốn kỹ thuật khai phá: "
        f"phân cụm K-Means (k={cl.get('k')}), luật kết hợp FP-Growth, mẫu tuần tự PrefixSpan và phân lớp "
        f"(bổ trợ, {best}: ROC-AUC ngoài thời gian {num(brow.get('roc_auc') if len(brow) else None)}).")

# ---------------------------------------------------------------------------------------------
d.add_heading("1. Introduction & Business Understanding", 1)
para(d, "KKBox là dịch vụ nghe nhạc trực tuyến theo thuê bao lớn nhất Đài Loan. Phần lớn gói cước dài 30 ngày, "
        "nên mỗi tháng doanh nghiệp đối mặt với một quyết định gia hạn của từng người dùng. Giữ chân một "
        "thuê bao rẻ hơn nhiều so với thu hút người mới; vì vậy câu hỏi kinh doanh không chỉ là “ai sẽ rời "
        "đi” mà là “những nhóm hành vi nào, tổ hợp đặc điểm nào và chuỗi sự kiện nào dẫn đến rời bỏ”, để "
        "thiết kế hành động giữ chân phù hợp từng nhóm.")
para(d, "Bài toán data mining:", bold=True)
bullets(d, [
    "Mô tả/phân khúc (unsupervised): người dùng chia thành những nhóm hành vi nào, nhóm nào có rủi ro rời bỏ cao?",
    "Khai phá mẫu (pattern-based): tổ hợp thuộc tính nào (luật kết hợp) và chuỗi giao dịch nào (mẫu tuần tự) "
    "làm tăng/giảm đáng kể xác suất churn?",
    "Phân lớp (bổ trợ): mức độ dự báo được churn từ thông tin có tại thời điểm quyết định, yếu tố nào quan trọng?",
])
para(d, "Định nghĩa và giả định:", bold=True)
bullets(d, [
    "Churn (theo KKBox): không có giao dịch thuê bao hợp lệ mới trong 30 ngày sau ngày hết hạn.",
    "Mốc quan sát T: mọi đặc trưng chỉ dùng dữ liệu có ngày ≤ T (31/01/2017 cho nhóm hết hạn tháng 2, "
    "28/02/2017 cho nhóm tháng 3). Ngày hết hạn đã biết tại T được phép dùng dù nằm sau T.",
    "Hạn chế: nhóm người dùng được KKBox chọn theo tháng hết hạn (dùng thông tin sau T), dữ liệu chỉ "
    "phản ánh một thị trường và một giai đoạn; nhãn tháng 2 chỉ hoàn tất cuối tháng 3, nên trong vận hành "
    "thật mô hình sẽ phải train trên nhóm cũ hơn một tháng.",
])

# ---------------------------------------------------------------------------------------------
d.add_heading("2. Dataset Description", 1)
para(d, "Nguồn: WSDM – KKBox's Churn Prediction Challenge (Kaggle, 2017), dữ liệu thật đã ẩn danh. Các bảng nối "
        "với nhau bằng msno (mã người dùng đã băm).")
ds = pd.DataFrame([
    ("train.csv", "Nhãn churn, nhóm hết hạn 02/2017", "train (mô hình), cohort feb"),
    ("train_v2.csv", "Nhãn churn, nhóm hết hạn 03/2017", "test ngoài thời gian, khai phá mô tả"),
    ("members_v3.csv", "Thành phố, tuổi (bd), giới tính, kênh đăng ký, ngày đăng ký", "hồ sơ"),
    ("transactions.csv + _v2", "Giao dịch: gói, giá, tự gia hạn, hủy, ngày hết hạn", "đặc trưng thuê bao, chuỗi"),
    ("user_logs.csv", "Log nghe hằng ngày tới 28/02/2017 (~28 GB)", "đặc trưng nghe 60 ngày trước T"),
    ("user_logs_v2.csv", "Log nghe tháng 03/2017", "chỉ dùng cho kiểm tra rò rỉ"),
], columns=["Bảng", "Nội dung", "Vai trò trong nghiên cứu"])
table(d, ds, "Bảng 1. Các bảng dữ liệu sử dụng")
if manifest:
    rows = [(c, v["cutoff"], thousands(v["n_users"]), pct(v["churn_rate"]), pct(v["share_has_logs"]),
             pct(v["share_has_tx"])) for c, v in manifest["cohorts"].items()]
    table(d, pd.DataFrame(rows, columns=["Cohort", "Cutoff T", "Số người dùng", "Tỉ lệ churn", "Có log nghe",
                                         "Có giao dịch ≤ T"]), "Bảng 2. Hai cohort sau tiền xử lý")
para(d, "Kiểu thuộc tính: định danh (city, registered_via, payment_method, gender), số liên tục (giá, số giờ nghe, "
        "tỉ lệ nghe hết bài), đếm (số giao dịch, số ngày hoạt động), thời gian (ngày giao dịch, ngày hết hạn, ngày "
        "đăng ký) và chuỗi sự kiện (lịch sử giao dịch).")

# ---------------------------------------------------------------------------------------------
d.add_heading("3. Data Pre-processing & Transformation", 1)
bullets(d, [
    "Xử lý dữ liệu lớn: DuckDB đọc trực tiếp CSV; file log 28 GB được giải nén dạng luồng và lọc ngay trong "
    "lúc giải nén (chỉ giữ 60 ngày trước mỗi cutoff của người dùng thuộc hai cohort), nên không cần 28 GB ổ đĩa.",
    "Hợp nhất transactions.csv và transactions_v2.csv, loại bản ghi trùng; chỉ giữ giao dịch có ngày ≤ T. "
    "Thứ tự giao dịch cùng ngày được cố định (ngày giao dịch, ngày hết hạn, is_cancel, …) và dùng chung cho "
    "mọi bước để “giao dịch cuối” là duy nhất.",
    f"Làm sạch: bd ngoài [10, 80] coi là thiếu (thiếu {pct(eda.get('age_missing'))}); giới tính thiếu gán "
    f"'unknown' ({pct(eda.get('gender_unknown'))}); total_secs chặn trong [0, 86.400] giây/ngày.",
    "Đặc trưng thuê bao (17): số giao dịch (tổng, 180 ngày), số lần hủy, tỉ lệ tự gia hạn, tỉ lệ giảm giá, số "
    "giao dịch 0 đồng, số lần đổi gói, số phương thức thanh toán, tổng tiền, và trạng thái giao dịch cuối "
    "(gói, giá, tự gia hạn, hủy, số ngày từ giao dịch cuối, số ngày còn lại tới hạn).",
    "Đặc trưng nghe (12, cửa sổ 60 ngày): số ngày hoạt động (tổng, 14 ngày cuối), tổng bài, tỉ lệ nghe hết/bỏ "
    "qua, số bài khác nhau, số giờ, phút/ngày hoạt động, xu hướng (log giờ nghe 30 ngày sau − 30 ngày đầu), số "
    "ngày từ lần nghe cuối, độ đều đặn (hệ số biến thiên số ngày hoạt động theo 8 tuần tính lùi từ T).",
    "Giá trị thiếu không được điền trước khi chia dữ liệu; việc điền trung vị/one-hot nằm trong Pipeline "
    "và chỉ học trên tập train. Với clustering, người không có log được gán giá trị “không hoạt động” rõ ràng.",
    "Không lấy mẫu cân bằng: giữ tỉ lệ churn thật (~9%) để tỉ lệ churn, confidence và lift có nghĩa tuyệt đối. "
    "Các thuật toán chi phí O(n²) chạy trên mẫu phân tầng (ghi rõ cỡ mẫu ở từng mục).",
])

# ---------------------------------------------------------------------------------------------
d.add_heading("4. Exploratory & Descriptive Analysis", 1)
eff = csv("eda_effects.csv")
if len(eff):
    e = eff.head(10)[["feature", "median_churn", "median_stay", "auc_effect"]].copy()
    e.columns = ["Đặc trưng", "Trung vị (churn)", "Trung vị (ở lại)", "AUC đơn biến"]
    table(d, e, "Bảng 3. Mười đặc trưng phân biệt churn mạnh nhất (AUC đơn biến; 0,5 = không phân biệt)")
assoc = csv("eda_association.csv", index_col=0)
if len(assoc):
    a = assoc.reset_index().rename(columns={"index": "Nhóm"})[["Nhóm", "cramers_v"]]
    a.columns = ["Biến phân nhóm", "Cramér's V"]
    table(d, a, "Bảng 4. Mức độ liên hệ giữa các biến phân nhóm và churn (kiểm định χ², mọi p < 0,001 trừ khi ghi chú)")
figure(d, F / "eda_churn_by_group.png", "Hình 1. Tỉ lệ churn theo nhóm (thanh lỗi = khoảng tin cậy 95% Wilson)")
figure(d, F / "eda_distributions.png", "Hình 2. Phân phối các hành vi chính: churn (cam) và ở lại (xanh)")
figure(d, F / "eda_churn_by_tenure.png", "Hình 3. Tỉ lệ churn theo thâm niên", width=4.5)
figure(d, F / "eda_correlation.png", "Hình 4. Tương quan Spearman giữa các đặc trưng", width=5.5)
para(d, "EDA_NARRATIVE")

# ---------------------------------------------------------------------------------------------
d.add_heading("5. Data Mining Methods & Results", 1)
d.add_heading("5.1 Phân cụm hành vi (K-Means, so sánh Ward/GMM/HDBSCAN)", 2)
para(d, f"16 đặc trưng thuê bao và nghe được log-transform, chuẩn hóa; nhãn churn không tham gia phân cụm. "
        f"Chọn k trên mẫu {thousands(cl.get('n_users', 0))} người dùng bằng bốn tiêu chí nội tại (hình 5). "
        f"k = {cl.get('k')} được chọn (silhouette {num(cl.get('silhouette'))}, Davies–Bouldin "
        f"{num(cl.get('davies_bouldin'), 2)}); độ ổn định bootstrap ARI {num(cl.get('bootstrap_ari_mean'), 2)} ± "
        f"{num(cl.get('bootstrap_ari_std'), 2)}. Tâm cụm sau đó được dùng để gán toàn bộ cohort.")
figure(d, F / "cluster_selection.png", "Hình 5. Chọn số cụm: inertia, silhouette, Davies–Bouldin, Calinski–Harabasz")
cp = csv("cluster_profile_full.csv")
if len(cp):
    c = cp[["cluster", "name", "n", "churn_rate", "ci_low", "ci_high", "lift"]].copy()
    c.columns = ["Cụm", "Đặc điểm nổi bật", "Số người", "Tỉ lệ churn", "CI thấp", "CI cao", "Lift"]
    table(d, c, "Bảng 5. Các phân khúc trên toàn cohort tháng 3",
          fmt={"Tỉ lệ churn": "{:.1%}", "CI thấp": "{:.1%}", "CI cao": "{:.1%}", "Lift": "{:.2f}", "Số người": "{:,.0f}"})
figure(d, F / "cluster_heatmap.png", "Hình 6. Tâm cụm (z-score so với toàn bộ người dùng)")
figure(d, F / "cluster_churn.png", "Hình 7. Tỉ lệ churn theo phân khúc (CI 95%)")
alg = csv("cluster_algorithms.csv")
if len(alg):
    table(d, alg[["algorithm", "n_clusters", "noise_share", "silhouette", "davies_bouldin", "ari_vs_kmeans",
                  "churn_rate_spread", "runtime_s"]],
          "Bảng 6. So sánh thuật toán phân cụm trên cùng mẫu con", fmt={"noise_share": "{:.1%}", "churn_rate_spread": "{:.1%}"})
para(d, "CLUSTER_NARRATIVE")

d.add_heading("5.2 Luật kết hợp (FP-Growth)", 2)
para(d, f"Mỗi người dùng được mô tả bằng {ru.get('n_items')} item (trạng thái tự gia hạn, lịch sử hủy, loại gói, "
        "mức giá, thâm niên, kênh đăng ký, giới tính, mức hoạt động, số ngày im lặng, xu hướng nghe, tỉ lệ bỏ qua, "
        "hạn còn lại, phân khúc). Ngưỡng định lượng học trên tập train 70%. FP-Growth liệt kê vế trái X phổ biến "
        f"(min_support {ru.get('min_support')}, tối đa 3 item, {thousands(ru.get('n_itemsets', 0))} tập phổ biến). "
        "Với mỗi X: confidence = P(churn | X), lift, leverage, conviction; kiểm định hai tỉ lệ X so với ¬X, hiệu "
        "chỉnh Benjamini–Hochberg; chỉ giữ luật “productive” (tốt hơn mọi luật con ít nhất 2 điểm %). Mọi luật "
        "được đo lại trên 30% holdout chưa dùng để khai phá.")
cr = csv("churn_rules.csv")
if len(cr):
    t = cr.head(12)[["rule", "support", "confidence", "lift", "confidence_holdout", "lift_holdout"]].copy()
    t.columns = ["Vế trái X (→ CHURN)", "Support X", "Conf (train)", "Lift", "Conf (holdout)", "Lift (holdout)"]
    table(d, t, "Bảng 7. Luật dẫn tới churn (q < 0,01, lift ≥ 1,5, productive)",
          fmt={"Support X": "{:.1%}", "Conf (train)": "{:.1%}", "Conf (holdout)": "{:.1%}", "Lift": "{:.2f}",
               "Lift (holdout)": "{:.2f}"})
rr = csv("retention_rules.csv")
if len(rr):
    t = rr.head(8)[["rule", "support", "confidence", "lift", "confidence_holdout"]].copy()
    t.columns = ["Vế trái X (→ ở lại)", "Support X", "Churn (train)", "Lift churn", "Churn (holdout)"]
    table(d, t, "Bảng 8. Luật “bảo vệ”: tổ hợp có tỉ lệ churn thấp hơn hẳn mức chung",
          fmt={"Support X": "{:.1%}", "Churn (train)": "{:.1%}", "Churn (holdout)": "{:.1%}", "Lift churn": "{:.2f}"})
figure(d, F / "rules_scatter.png", "Hình 8. Support và confidence của mọi tập phổ biến (viền đỏ = luật được chọn)")
para(d, "RULES_NARRATIVE")

d.add_heading("5.3 Mẫu tuần tự (PrefixSpan trên chuỗi giao dịch)", 2)
para(d, "Mỗi giao dịch ≤ T được mã hóa thành một itemset sự kiện (PAY_AUTO/PAY_MANUAL, CANCEL, FREE, DISC, "
        "PLAN_30/PLAN_LONG/PLAN_SHORT, PLAN_CHANGE, METHOD_CHANGE, GAP = từng lỡ hạn > 30 ngày); giữ 12 sự kiện "
        f"cuối. PrefixSpan (tự cài đặt bằng bitmask NumPy) khai phá mẫu dài ≤ {sq.get('max_pattern_length', 4)} với "
        f"min_support {sq.get('min_support')} trên {thousands(sq.get('n_train', 0))} chuỗi train; mỗi mẫu được kiểm "
        f"định χ² (churn vs ở lại) và hiệu chỉnh BH ({thousands(sq.get('n_tested', 0))} giả thuyết), loại mẫu dư "
        "thừa rồi đo lại trên holdout.")
sp = csv("sequential_patterns.csv")
if len(sp):
    t = sp.head(12)[["pattern", "support_train", "conf_train", "lift_train", "conf_holdout", "lift_holdout", "direction"]].copy()
    t.columns = ["Mẫu tuần tự", "Support", "Conf (train)", "Lift", "Conf (holdout)", "Lift (holdout)", "Hướng"]
    table(d, t, "Bảng 9. Mẫu tuần tự có ý nghĩa thống kê", fmt={"Support": "{:.1%}", "Conf (train)": "{:.1%}",
          "Conf (holdout)": "{:.1%}", "Lift": "{:.2f}", "Lift (holdout)": "{:.2f}"})
figure(d, F / "seq_top_patterns.png", "Hình 9. Lift trên holdout của các mẫu tuần tự hàng đầu")
para(d, "SEQ_NARRATIVE")

d.add_heading("5.4 Phân lớp (phương pháp bổ trợ)", 2)
para(d, f"Train trên cohort tháng 2 ({thousands(cf.get('n_train', 0))} người, churn {pct(cf.get('churn_train'))}), "
        f"test trên cohort tháng 3 ({thousands(cf.get('n_test', 0))} người, churn {pct(cf.get('churn_test'))}) – "
        "mô hình không thấy tháng kiểm tra. So sánh: baseline theo prior, hồi quy logistic, cây quyết định "
        "(độ sâu 6), random forest, LightGBM. Mô hình chính được chọn bằng 5-fold CV trên tháng 2.")
if len(oot):
    t = oot[["model", "roc_auc", "pr_auc", "log_loss", "brier", "precision@10%", "lift@10%", "fit_s"]].copy()
    t.columns = ["Mô hình", "ROC-AUC", "PR-AUC", "Log-loss", "Brier", "Precision@10%", "Lift@10%", "Thời gian (s)"]
    table(d, t, "Bảng 10. Đánh giá ngoài thời gian (train 02/2017 → test 03/2017)", fmt={"Precision@10%": "{:.1%}", "Thời gian (s)": "{:.0f}"})
cvt = csv("classification_cv.csv")
if len(cvt):
    table(d, cvt, "Bảng 11. 5-fold CV trên cohort tháng 2 (trung bình, độ lệch chuẩn)")
para(d, f"Mô hình chọn theo CV: {best}; ROC-AUC 95% CI (bootstrap) {cf.get('roc_auc_ci')}, PR-AUC 95% CI {cf.get('pr_auc_ci')}.")
figure(d, F / "clf_curves.png", "Hình 10. ROC, Precision–Recall và đường hiệu chuẩn trên tháng 3")
figure(d, F / "clf_gains.png", "Hình 11. Cumulative gains: tỉ lệ người churn bắt được khi liên hệ x% người rủi ro nhất", width=4.5)
if len(abl):
    t = abl[["feature_set", "n_features", "roc_auc", "pr_auc", "lift@10%"]].copy()
    t.columns = ["Nhóm đặc trưng", "Số biến", "ROC-AUC", "PR-AUC", "Lift@10%"]
    table(d, t, "Bảng 12. Ablation theo nhóm đặc trưng (LightGBM, ngoài thời gian)")
figure(d, F / "clf_shap_summary.png", "Hình 12. Giá trị SHAP của LightGBM (5.000 người dùng tháng 3)", width=5.5)
para(d, "CLF_NARRATIVE")

# ---------------------------------------------------------------------------------------------
d.add_heading("6. Evaluation & Interpretation", 1)
d.add_heading("6.1 Kiểm tra rò rỉ thông tin (leakage audit)", 2)
para(d, "Trên cùng tập người dùng tháng 3 và cùng một phép chia 70/30, lần lượt thêm từng nguồn dữ liệu sau "
        "mốc T vào bộ đặc trưng sạch. Khoảng tin cậy của chênh lệch AUC tính bằng bootstrap ghép cặp.")
if len(audit):
    t = pd.DataFrame(audit)[["feature_set", "roc_auc", "pr_auc", "auc_inflation_vs_clean", "delta_ci_low", "delta_ci_high"]]
    t.columns = ["Bộ đặc trưng", "ROC-AUC", "PR-AUC", "ΔAUC so với sạch", "CI thấp", "CI cao"]
    table(d, t, "Bảng 13. Mức thổi phồng do dùng dữ liệu của tháng nhãn")
rule = va.get("leaky_rule_no_march_tx", {})
if rule:
    para(d, f"Luật “không có giao dịch trong tháng 3 → churn” có confidence {pct(rule['confidence'])}, lift "
            f"{num(rule['lift'], 2)} và bao phủ {pct(rule['share_of_churners_covered'])} người churn: đây chính là "
            "định nghĩa nhãn, giải thích vì sao bản phân tích đầu tiên cho các luật confidence 97–99%.")
figure(d, F / "leakage_audit.png", "Hình 13. Leakage audit")
para(d, "AUDIT_NARRATIVE")

d.add_heading("6.2 So sánh các phương pháp", 2)
comp = csv("algorithm_comparison.csv")
if len(comp):
    t = comp.copy()
    t.columns = ["Kỹ thuật", "Loại", "Chất lượng", "Thời gian (s)", "Khả năng diễn giải", "Ứng dụng kinh doanh"]
    table(d, t, "Bảng 14. So sánh kỹ thuật theo chất lượng, chi phí, khả năng diễn giải và tính hành động",
          fmt={"Thời gian (s)": "{:.0f}"})
sens = csv("rule_sensitivity.csv")
if len(sens):
    table(d, sens, "Bảng 15. Độ nhạy của luật kết hợp theo min_support", fmt={"jaccard_top10_vs_default": "{:.2f}"})
para(d, "EVAL_NARRATIVE")

# ---------------------------------------------------------------------------------------------
d.add_heading("7. Business Insights & Recommendations", 1)
para(d, "BUSINESS_NARRATIVE")

# ---------------------------------------------------------------------------------------------
d.add_heading("8. Advanced Techniques (bonus)", 1)
figure(d, F / "bonus_trajectory.png", "Hình 14. Quỹ đạo nghe theo tuần trong 8 tuần trước T")
for name, cap in [("bonus_trend.csv", "Bảng 16. Churn theo xu hướng nghe"), ("bonus_idle.csv", "Bảng 17. Churn theo số ngày im lặng")]:
    t = csv(name)
    if len(t):
        t = t.rename(columns={t.columns[0]: "Nhóm"})[["Nhóm", "n", "churn_rate", "ci_low", "ci_high", "lift"]]
        table(d, t, cap, fmt={"churn_rate": "{:.1%}", "ci_low": "{:.1%}", "ci_high": "{:.1%}", "lift": "{:.2f}", "n": "{:,.0f}"})
an = csv("bonus_anomaly.csv")
if len(an):
    table(d, an, "Bảng 18. Isolation Forest: churn ở nhóm bất thường (2%) và bình thường",
          fmt={"churn_rate": "{:.1%}", "ci_low": "{:.1%}", "ci_high": "{:.1%}", "lift": "{:.2f}"})
sub = csv("bonus_subsets.csv")
if len(sub):
    table(d, sub, "Bảng 19. So sánh tập con: mô hình ngoài thời gian hoạt động thế nào trong từng nhóm",
          fmt={"share": "{:.1%}", "churn_rate": "{:.1%}", "mean_pred": "{:.1%}", "n": "{:,.0f}"})
figure(d, F / "umap_churn.png", "Hình 15. UMAP của đặc trưng hành vi: theo churn và theo phân khúc")
figure(d, F / "bonus_pca_scree.png", "Hình 16. PCA scree plot", width=4.5)
para(d, "BONUS_NARRATIVE")

d.add_heading("9. Kết luận, hạn chế và hướng phát triển", 1)
para(d, "CONCLUSION_NARRATIVE")

d.add_heading("Phụ lục: tái lập kết quả", 1)
para(d, "pip install -r requirements.txt; python run_all.py (tải dữ liệu bằng Kaggle CLI hoặc truyền file zip "
        "“Download All”: python data_prep/00_download.py <zip>). Kiểm thử nhanh trên dữ liệu giả: "
        "python tests/make_synthetic_data.py data_synth; DATA_DIR=data_synth REPORT_DIR=report_synth python "
        "run_all.py --skip-download. Seed cố định 42; mọi cutoff/cỡ mẫu nằm trong data_prep/config.py và biến môi trường.")

# narratives written after inspecting the final results live in narratives.json
narr = json.loads((R / "narratives.json").read_text(encoding="utf-8")) if (R / "narratives.json").exists() else {}
for p in d.paragraphs:
    key = p.text.strip()
    if key.endswith("_NARRATIVE"):
        text = narr.get(key, "")
        p.text = ""
        if isinstance(text, list):
            p.text = text[0] if text else ""
            anchor = p
            for extra in text[1:]:
                new = anchor.insert_paragraph_before("")  # keep order: insert after via XML move
                anchor._p.addnext(new._p)
                new.text = extra
                anchor = new
        else:
            p.text = text
d.save(R / "report.docx")
print("report written:", R / "report.docx")
