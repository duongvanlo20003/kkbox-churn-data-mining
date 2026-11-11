"""Build report/report.docx from the pipeline outputs (metrics/*.json, *.csv, figures/).

The text is written for a reader who does not know the code: short sentences, every technical
term explained the first time, rule/pattern codes translated to Vietnamese (report/vi_labels.py).
The main numbers are read from the result files; run `python report/make_diagrams.py` for the
two explanatory diagrams. Tables and figures are numbered automatically.
"""
import json
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vi_labels as vi
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
    return "–" if x is None else f"{x * 100:.{d}f}%".replace(".", ",")


def num(x, d=3):
    return "–" if x is None else f"{x:.{d}f}".replace(".", ",")


def k(x):
    return f"{int(round(x)):,}".replace(",", ".")


eda, cl, ru, cf, sq, va, st, bv, mi_m, vol = (j(n) for n in [
    "eda", "clustering", "rules", "classification", "sequential", "validation", "stability",
    "business_value", "model_improvement", "data_volume"])
manifest = json.loads((DATA / "manifest.json").read_text()) if (DATA / "manifest.json").exists() else {}
oot = pd.DataFrame(cf.get("out_of_time", []))
lg = oot.set_index("model").loc["LightGBM"] if len(oot) else {}
grp = csv("eda_churn_by_group.csv")


def g(feature, level):
    r = grp[(grp.feature == feature) & (grp.level.astype(str) == level)]
    return float(r.churn_rate.iloc[0]) if len(r) else None


seg = csv("cluster_profile_full.csv").set_index("cluster") if len(csv("cluster_profile_full.csv")) else pd.DataFrame()
base = eda.get("churn_mar", 0.09)

# ---- numbering ------------------------------------------------------------------------------
n_tab, n_fig = [0], [0]


def T(df, caption, **kw):
    n_tab[0] += 1
    return table(d, df, f"Bảng {n_tab[0]}. {caption}", **kw)


def FIG(path, caption, width=6.0):
    n_fig[0] += 1
    figure(d, path, f"Hình {n_fig[0]}. {caption}", width=width)
    return n_fig[0]


def P(text):
    return para(d, text)


def H(text, level):
    d.add_heading(text, level)


d = new_document()
d.add_heading("Khám phá hành vi người dùng và các mẫu rời bỏ dịch vụ nghe nhạc thuê bao KKBox", 0)
para(d, "Môn DAM501 – Data Mining · Đồ án cuối kỳ", bold=True)
para(d, "Nhóm thực hiện: Lộc Nguyễn Phúc, Khánh Phạm Duy Văn, Bình Chu Đức, Quân Đoàn Đàm, Hạnh Nguyễn Thị Hồng, "
        "Tùng Trần, Phùng Quý Quốc.")

# =============================================================================================
H("Tóm tắt", 1)
P(f"KKBox là dịch vụ nghe nhạc trả phí theo tháng ở Đài Loan. Nhóm dùng dữ liệu thật của KKBox (khoảng "
  f"{num(vol.get('unique_users_both_cohorts', 1_082_190) / 1e6, 2)} triệu người dùng, {k(vol.get('user_logs_rows_scanned', 392e6) / 1e6)} "
  "triệu dòng nhật ký nghe nhạc) để trả lời câu hỏi: nhóm khách hàng nào hay bỏ dịch vụ, và những dấu hiệu nào "
  "báo trước việc đó.")
P("Kết quả chính:")
bullets(d, [
    f"Việc khách có rời đi hay không gắn với cách họ trả tiền nhiều hơn là với việc họ nghe nhạc nhiều hay ít. "
    f"Khách đã tắt tự gia hạn rời đi {pct(g('auto_renew', 'off'), 0)}, trong khi khách bật tự gia hạn chỉ "
    f"{pct(g('auto_renew', 'on'))}.",
    f"Phân cụm chia khách hàng thành 7 nhóm. Tỉ lệ rời bỏ của các nhóm chênh nhau rất xa, từ "
    f"{pct(seg.churn_rate.min())} đến {pct(seg.churn_rate.max(), 0)}.",
    "Nhóm rủi ro nhất là khách mua gói dài hạn và trả tiền thủ công: phần lớn không gia hạn khi hết gói.",
    f"Mô hình dự báo xếp hạng khách theo rủi ro khá tốt (ROC-AUC {num(lg.get('roc_auc'))}). Nếu chỉ liên hệ 10% "
    f"khách rủi ro nhất, danh sách này đã chứa {pct(lg.get('recall@10%'), 0)} số khách sẽ rời đi.",
    "Phiên bản trước của nhóm có ROC-AUC 0,977 nhưng con số này bị thổi phồng do dùng nhầm dữ liệu của tháng "
    "cần dự báo. Bản này sửa lỗi đó và đo lại mức thổi phồng.",
])

# =============================================================================================
H("Giải thích một số thuật ngữ", 1)
T(pd.DataFrame([
    ("Rời bỏ (churn)", "Khách không mua gói mới trong vòng 30 ngày sau khi gói cũ hết hạn (định nghĩa của KKBox)."),
    ("Mốc quan sát", "Ngày mà ta “đứng” để nhìn về quá khứ. Mọi thông tin dùng để phân tích phải xảy ra trước ngày này."),
    ("Đặc trưng", "Một con số mô tả khách hàng, ví dụ số lần hủy gói hay số ngày nghe nhạc trong 60 ngày."),
    ("Rò rỉ dữ liệu", "Vô tình dùng thông tin của tương lai (sau mốc quan sát) khiến kết quả trông tốt hơn thực tế."),
    ("Support", "Tỉ lệ khách hàng thỏa mãn một điều kiện. Support 2% nghĩa là 2 trong 100 khách."),
    ("Confidence", "Trong số khách thỏa điều kiện, bao nhiêu phần trăm rời bỏ."),
    ("Lift", f"Confidence chia cho tỉ lệ rời bỏ chung ({pct(base)}). Lift 3 nghĩa là rủi ro gấp 3 lần bình thường; "
             "lift dưới 1 nghĩa là nhóm đó ít rời bỏ hơn bình thường."),
    ("Tập holdout", "Một phần dữ liệu (30%) được giữ lại, không dùng khi tìm mẫu, chỉ dùng để kiểm tra lại."),
    ("Silhouette", "Điểm đo các cụm tách nhau rõ đến đâu, từ −1 đến 1. Trên 0,25 là chấp nhận được."),
    ("ROC-AUC", "Xác suất mô hình xếp một khách sẽ rời đi cao hơn một khách ở lại. 0,5 tương đương dự báo ngẫu nhiên, 1 là hoàn hảo."),
    ("PR-AUC", "Độ chính xác trung bình khi tìm khách sẽ rời đi; dự báo ngẫu nhiên chỉ đạt bằng tỉ lệ rời bỏ chung."),
], columns=["Thuật ngữ", "Ý nghĩa"]), "Các thuật ngữ dùng trong báo cáo")

# =============================================================================================
H("1. Giới thiệu và bài toán kinh doanh", 1)
P("Phần lớn gói thuê bao của KKBox có thời hạn 30 ngày, nên việc duy trì khách hàng phụ thuộc vào quyết định gia "
  "hạn hằng tháng. Giữ một khách cũ "
  "rẻ hơn nhiều so với tìm một khách mới. Vì vậy công ty cần biết không chỉ ai sắp rời đi, mà còn vì sao, để chọn "
  "cách giữ chân phù hợp cho từng nhóm.")
P("Nhóm đặt ra ba câu hỏi:")
bullets(d, [
    "Khách hàng chia thành những nhóm hành vi nào, và nhóm nào hay rời bỏ? (phân cụm)",
    "Những tổ hợp đặc điểm nào, và những chuỗi giao dịch nào, đi kèm với việc rời bỏ? (luật kết hợp, mẫu tuần tự)",
    "Có thể dự báo trước ai sẽ rời bỏ không, và yếu tố nào quan trọng nhất? (phân lớp – phương pháp bổ trợ)",
])
P("Đề bài coi trọng việc khám phá mẫu hơn việc dự báo, nên phân lớp chỉ đóng vai trò hỗ trợ giải thích.")
P("Cách làm quan trọng nhất của đồ án là chọn một mốc quan sát cho mỗi nhóm khách và chỉ dùng dữ liệu trước "
  "mốc đó (Hình 2). Làm vậy thì kết quả phản ánh đúng tình huống thật: công ty phải đoán trước khi khách quyết định.")
P("Báo cáo sử dụng định nghĩa rời bỏ và danh sách khách hàng hằng tháng do KKBox cung cấp. Dữ liệu chỉ thuộc thị "
  "trường Đài Loan; thông tin khách nào rời bỏ, khách nào ở lại chỉ có cho tháng 2 và tháng 3 năm 2017.")
FIG(F / "diagram_pipeline.png", "Các bước thực hiện dự án, từ dữ liệu gốc đến khuyến nghị")
FIG(F / "diagram_timeline.png", "Cách cắt dữ liệu theo mốc quan sát cho hai nhóm khách hàng")

# =============================================================================================
H("2. Mô tả dữ liệu", 1)
P("Dữ liệu lấy từ cuộc thi “WSDM – KKBox's Churn Prediction Challenge” trên Kaggle (2017). Đây là dữ liệu thật, "
  "mã khách hàng đã được ẩn danh. Các bảng nối với nhau qua mã khách hàng.")
T(pd.DataFrame([
    ("train.csv", "Khách hết hạn tháng 2/2017 và nhãn rời bỏ", "Huấn luyện mô hình"),
    ("train_v2.csv", "Khách hết hạn tháng 3/2017 và nhãn rời bỏ", "Khai phá mẫu, kiểm tra mô hình"),
    ("members_v3.csv", "Thành phố, tuổi, giới tính, kênh đăng ký, ngày đăng ký", "Thông tin hồ sơ"),
    ("transactions.csv và _v2", "Từng lần mua gói: độ dài gói, giá, tự gia hạn, hủy, ngày hết hạn", "Đặc trưng thuê bao, chuỗi giao dịch"),
    ("user_logs.csv (28 GB)", "Mỗi ngày nghe bao nhiêu bài, nghe hết hay bỏ qua, bao nhiêu giây", "Đặc trưng nghe nhạc"),
    ("user_logs_v2.csv", "Nhật ký nghe tháng 3/2017", "Chỉ dùng để đo mức rò rỉ dữ liệu"),
], columns=["Bảng", "Nội dung", "Dùng để"]), "Các bảng dữ liệu")
if manifest:
    c = manifest["cohorts"]
    T(pd.DataFrame([
        ("Tháng 2 (huấn luyện)", "31/01/2017", k(c["feb"]["n_users"]), pct(c["feb"]["churn_rate"]), pct(c["feb"]["share_has_logs"], 0)),
        ("Tháng 3 (khai phá, kiểm tra)", "28/02/2017", k(c["mar"]["n_users"]), pct(c["mar"]["churn_rate"]), pct(c["mar"]["share_has_logs"], 0)),
    ], columns=["Nhóm khách", "Mốc quan sát", "Số khách", "Tỉ lệ rời bỏ", "Có nhật ký nghe"]), "Hai nhóm khách hàng sau xử lý")
P("Dữ liệu có nhiều kiểu thuộc tính: dạng phân loại (thành phố, kênh đăng ký, giới tính), dạng số (giá, số giờ "
  "nghe), dạng đếm (số giao dịch, số ngày nghe), dạng ngày tháng và dạng chuỗi sự kiện (lịch sử giao dịch).")

# =============================================================================================
H("3. Tiền xử lý và biến đổi dữ liệu", 1)
bullets(d, [
    f"Dữ liệu lớn: file nhật ký 28 GB không được giải nén ra ổ đĩa. Chương trình đọc từng phần trong lúc giải nén "
    f"và chỉ giữ những ngày cần dùng, từ {k(vol.get('user_logs_rows_scanned', 392e6) / 1e6)} triệu dòng còn "
    f"{num(vol.get('log_rows_kept_in_window', 40.9e6) / 1e6, 1)} triệu dòng.",
    "Gộp hai file giao dịch, bỏ dòng trùng và bỏ mọi giao dịch sau mốc quan sát. Khi một khách có nhiều giao dịch "
    "cùng ngày, thứ tự được cố định theo một quy tắc chung để “giao dịch cuối” luôn xác định.",
    f"Làm sạch: tuổi ngoài khoảng 10–80 bị coi là thiếu ({pct(eda.get('age_missing'), 0)} khách không có tuổi hợp "
    f"lệ); giới tính bỏ trống ghi là “không rõ” ({pct(eda.get('gender_unknown'), 0)}); tổng thời gian nghe của mỗi "
    "khách trong một ngày được giới hạn ở mức 24 giờ.",
    "Tạo 29 đặc trưng cho mỗi khách. Nhóm thuê bao (17): số giao dịch, số lần hủy, tỉ lệ tự gia hạn, giảm giá, "
    "đổi gói và thông tin giao dịch cuối (độ dài gói, giá, có tự gia hạn không, còn bao nhiêu ngày tới hạn). Nhóm "
    "nghe nhạc (12, trong 60 ngày trước mốc): số ngày nghe, số giờ, tỉ lệ nghe hết bài, tỉ lệ bỏ qua, nghe tăng "
    "hay giảm, bao lâu chưa nghe, nghe đều hay thất thường. Ngoài 29 đặc trưng này, mô hình dùng thêm 6 biến: thành "
    "phố, kênh đăng ký, giới tính, tuổi, thời gian sử dụng dịch vụ và phương thức thanh toán, tổng cộng 35 biến đầu vào.",
    "Tạo chuỗi giao dịch cho mỗi khách (tối đa 12 giao dịch gần nhất), mỗi giao dịch được mô tả bằng các sự kiện "
    "như “trả tự động”, “trả thủ công”, “hủy”, “đổi gói”.",
    f"Giữ nguyên tỉ lệ rời bỏ thật (khoảng {pct(base, 0)}), không lấy mẫu cân bằng, để các tỉ lệ trong báo cáo "
    "đúng với thực tế.",
    "Các giá trị thiếu chỉ được điền sau khi chia dữ liệu, và chỉ dựa trên phần huấn luyện.",
])

# =============================================================================================
H("4. Phân tích khám phá dữ liệu", 1)
eff = csv("eda_effects.csv")
if len(eff):
    e = eff.head(8)[["feature", "median_churn", "median_stay", "auc_effect"]].copy()
    e["feature"] = e.feature.map(vi.feature)
    e["auc_effect"] = (e.auc_effect - 0.5).abs() + 0.5
    e.columns = ["Đặc trưng", "Trung vị (rời bỏ)", "Trung vị (ở lại)", "Khả năng phân biệt (0,5–1)"]
    T(e, "Tám đặc trưng phân biệt khách rời bỏ và khách ở lại rõ nhất")
assoc = csv("eda_association.csv", index_col=0)
if len(assoc):
    a = assoc.reset_index()
    a = pd.DataFrame({"Yếu tố": a.iloc[:, 0].map(lambda x: vi.GROUPS.get(x, x)), "Mức liên hệ (Cramér's V, 0–1)": a.cramers_v})
    T(a, "Mức liên hệ giữa từng yếu tố và việc rời bỏ (càng gần 1 càng mạnh)")
FIG(F / "eda_churn_by_group.png", "Tỉ lệ rời bỏ theo từng nhóm; vạch đứt là mức chung, thanh nhỏ là khoảng tin cậy 95%")
FIG(F / "eda_distributions.png", "So sánh phân bố một số hành vi giữa khách rời bỏ và khách ở lại")
FIG(F / "eda_churn_by_tenure.png", "Tỉ lệ rời bỏ theo số năm sử dụng", width=4.5)
FIG(F / "eda_correlation.png", "Mức tương quan giữa các đặc trưng", width=5.5)
P("Những gì dữ liệu cho thấy:")
bullets(d, [
    f"Cách trả tiền là yếu tố mạnh nhất. Khách tắt tự gia hạn ở lần mua cuối rời đi {pct(g('auto_renew', 'off'))}, "
    f"khách bật tự gia hạn chỉ {pct(g('auto_renew', 'on'))}. Khách có lần giao dịch cuối là hủy gói rời đi "
    f"{pct(g('cancelled_last', 'yes'), 0)}.",
    f"Độ dài gói cũng quan trọng. Gói thông thường (8–30 ngày) có tỉ lệ rời bỏ {pct(g('plan', '8-30'))}; gói rất ngắn "
    f"(1–7 ngày) là {pct(g('plan', '1-7'), 0)} và gói dài hơn 30 ngày là {pct(g('plan', '>30'), 0)}.",
    f"Kênh đăng ký tạo khác biệt lớn: kênh 7, chiếm gần một nửa số khách, có tỉ lệ rời bỏ {pct(g('registered_via', '7.0'))}, "
    f"còn kênh 4 là {pct(g('registered_via', '4.0'), 0)} và kênh 3 là {pct(g('registered_via', '3.0'), 0)}.",
    "Khi xét riêng từng yếu tố, số ngày không nghe gần đây, xu hướng nghe tăng hay giảm và thời gian sử dụng dịch "
    "vụ đều có mức liên hệ thấp với việc rời bỏ (Cramér's V dưới 0,05). Đây là điểm khác lớn nhất so với phiên bản trước, vì phiên bản "
    "trước đã dùng nhầm nhật ký nghe của tháng 3, lúc nhiều khách đã ngừng dịch vụ.",
    f"Dữ liệu có chỗ thiếu: {pct(eda.get('age_missing'), 0)} khách không có tuổi hợp lệ, gần một nửa không khai giới tính; "
    f"{pct(eda.get('share_no_logs'), 0)} khách vẫn trả tiền nhưng không có nhật ký nghe nào trong 60 ngày. Nhóm này "
    "được giữ lại vì việc không nghe cũng là một thông tin.",
])
P("Từ đó nhóm chọn kỹ thuật: luật kết hợp để tìm các tổ hợp giữa cách trả tiền, loại gói và kênh; mẫu tuần tự "
  "vì thứ tự các lần mua có thể chứa tín hiệu; phân cụm để kết hợp cả thông tin thuê bao lẫn thói quen nghe.")

# =============================================================================================
H("5. Các phương pháp khai phá và kết quả", 1)

H("5.1 Phân cụm khách hàng (K-Means)", 2)
P("Mục tiêu là chia khách thành các nhóm có thói quen giống nhau mà không cho thuật toán biết ai đã rời đi. "
  "Nhóm dùng 16 đặc trưng về thuê bao và nghe nhạc, đưa về cùng thang đo, rồi chạy K-Means với số cụm từ 2 đến 10 "
  f"trên mẫu {k(cl.get('n_users', 200000))} khách.")
P(f"Số cụm có điểm silhouette cao nhất là {cl.get('k_macro', 3)} ({num(cl.get('silhouette_macro'))}), nhưng cách chia này chỉ "
  f"tách riêng được một nhóm rủi ro; hai cụm lớn còn lại có cùng tỉ lệ rời bỏ khoảng 7%, nên khó dùng để ra quyết "
  f"định. Vì vậy nhóm chọn số cụm lớn nhất thỏa mãn hai điều kiện: điểm silhouette đạt ít nhất 75% điểm "
  f"cao nhất, và mỗi cụm chiếm ít nhất 2% số khách. Kết quả là {cl.get('k')} cụm, silhouette {num(cl.get('silhouette'))}.")
P(f"Kiểm tra độ ổn định: chạy lại toàn bộ quy trình trên 10 mẫu khác nhau, các lần phân cụm cho kết quả tương đồng cao "
  f"(ARI {num(cl.get('bootstrap_ari_mean'), 2)}; ARI càng gần 1 thì hai cách chia càng giống nhau). Sau đó mô hình được dùng để xếp toàn bộ "
  f"{k(eda.get('n_mar', 970960))} khách tháng 3 vào nhóm.")
FIG(F / "cluster_selection.png", "Các chỉ số dùng để chọn số cụm (vạch đỏ: số cụm được chọn)")
if len(seg):
    s = seg.reset_index()[["cluster", "name", "n", "churn_rate", "lift"]].copy()
    s["share"] = s.n / s.n.sum()
    s = s[["cluster", "name", "n", "share", "churn_rate", "lift"]].sort_values("churn_rate")
    s.columns = ["Nhóm", "Đặc điểm", "Số khách", "Tỉ trọng", "Tỉ lệ rời bỏ", "Lift"]
    T(s, "Bảy nhóm khách hàng tháng 3, xếp theo tỉ lệ rời bỏ tăng dần",
      fmt={"Số khách": "{:,.0f}", "Tỉ trọng": "{:.1%}", "Tỉ lệ rời bỏ": "{:.1%}", "Lift": "{:.2f}"})
FIG(F / "cluster_churn.png", "Tỉ lệ rời bỏ của từng nhóm")
FIG(F / "cluster_heatmap.png", "Đặc điểm từng nhóm (đỏ: cao hơn trung bình, xanh: thấp hơn trung bình)")
if len(seg):
    P("Cách đọc các nhóm:")
    bullets(d, [
        f"Nhóm 6 – trung thành ({pct(seg.loc[6, 'share'] if 'share' in seg else seg.loc[6, 'n'] / seg.n.sum(), 0)} khách): "
        f"bật tự gia hạn, nghe đều. Chỉ {pct(seg.loc[6, 'churn_rate'])} rời bỏ.",
        f"Nhóm 0 và nhóm 2: vẫn tự gia hạn nhưng nghe ít hoặc gần như không nghe. Tỉ lệ rời bỏ thấp "
        f"({pct(seg.loc[0, 'churn_rate'])} và {pct(seg.loc[2, 'churn_rate'])}), nhưng đây là khách đang trả tiền mà ít dùng.",
        f"Nhóm 5: khách lâu năm, mua nhiều lần, từng hủy hoặc đổi gói. Tỉ lệ rời bỏ {pct(seg.loc[5, 'churn_rate'])}, "
        "thấp hơn mức chung.",
        f"Nhóm 1: ít giao dịch, mức nghe tăng vọt (thường là khách mới hoặc khách quay lại). Rời bỏ "
        f"{pct(seg.loc[1, 'churn_rate'])}, gấp đôi mức chung.",
        f"Nhóm 3: nghe nhiều nhưng tự tay trả tiền mỗi lần. Rời bỏ {pct(seg.loc[3, 'churn_rate'])}, gấp 2,5 lần mức "
        "chung dù đang dùng dịch vụ nhiều.",
        f"Nhóm 4: mua gói dài hạn và trả thủ công. Rời bỏ {pct(seg.loc[4, 'churn_rate'], 0)}, cao nhất.",
    ])
    P("Điểm đáng chú ý là mức độ nghe không quyết định rủi ro: nhóm 3 nghe nhiều mà vẫn hay rời đi, còn nhóm 2 "
      "gần như không nghe mà ít rời đi. Điều khác biệt giữa hai nhóm là cách trả tiền. Đây là quan hệ đi kèm, "
      "chưa phải bằng chứng rằng trả thủ công làm khách rời đi.")
alg = csv("cluster_algorithms.csv")
if len(alg):
    a = alg[["algorithm", "n_clusters", "noise_share", "silhouette", "ari_vs_kmeans", "churn_rate_spread"]].copy()
    a["algorithm"] = a.algorithm.replace({"Agglomerative (Ward)": "Phân cụm phân cấp (Ward)",
                                          "Gaussian mixture": "Hỗn hợp Gauss (GMM)"})
    a.columns = ["Thuật toán", "Số cụm", "Điểm bị coi là nhiễu", "Silhouette", "Giống K-Means (ARI)",
                 "Chênh lệch tỉ lệ rời bỏ giữa các cụm"]
    T(a, "So sánh bốn thuật toán phân cụm trên cùng 15.000 khách",
      fmt={"Điểm bị coi là nhiễu": "{:.0%}", "Silhouette": "{:.3f}", "Giống K-Means (ARI)": "{:.2f}",
           "Chênh lệch tỉ lệ rời bỏ giữa các cụm": "{:.1%}"})
    P("Phân cụm phân cấp (Ward) cho kết quả gần giống K-Means, củng cố các nhóm tìm được. HDBSCAN, thuật toán tìm "
      "các vùng dày đặc, chỉ thấy 3 vùng, coi 27% khách là nhiễu và các vùng này không khác nhau về tỉ lệ rời bỏ. "
      "Với thiết lập đã thử, HDBSCAN không tìm được các nhóm tách biệt rõ và có ý nghĩa kinh doanh, nên nhóm chọn "
      "K-Means.")

H("5.2 Luật kết hợp (FP-Growth)", 2)
P(f"Mỗi khách được mô tả bằng một tập đặc điểm dạng có/không, ví dụ “tắt tự gia hạn”, “kênh đăng ký 9”, “ít "
  f"ngày nghe” ({ru.get('n_items', 45)} đặc điểm). FP-Growth tìm mọi tổ hợp tối đa 3 đặc điểm xuất hiện ở ít nhất 1% "
  f"khách ({k(ru.get('n_itemsets', 3620))} tổ hợp). Với mỗi tổ hợp, nhóm tính tỉ lệ rời bỏ và lift.")
P("Để tránh chọn nhầm những luật chỉ đúng do may mắn, nhóm làm thêm ba bước: (1) kiểm định thống kê có hiệu chỉnh "
  "cho việc thử hàng nghìn luật cùng lúc; (2) bỏ những luật dài mà không tốt hơn luật ngắn hơn chứa trong nó; (3) "
  "tìm luật trên 70% dữ liệu rồi đo lại trên 30% còn lại.")
cr = csv("churn_rules.csv")
if len(cr):
    t = cr.head(12)[["rule", "support", "confidence", "lift", "confidence_holdout"]].copy()
    t["rule"] = t.rule.map(vi.rule)
    t.columns = ["Nếu khách có đặc điểm", "Support", "Tỉ lệ rời bỏ", "Lift", "Tỉ lệ rời bỏ khi kiểm tra lại"]
    T(t, "Các tổ hợp đặc điểm đi kèm tỉ lệ rời bỏ cao",
      fmt={"Support": "{:.1%}", "Tỉ lệ rời bỏ": "{:.1%}", "Lift": "{:.1f}", "Tỉ lệ rời bỏ khi kiểm tra lại": "{:.1%}"})
rr = csv("retention_rules.csv")
if len(rr):
    t = rr.head(8)[["rule", "support", "confidence", "lift", "confidence_holdout"]].copy()
    t["rule"] = t.rule.map(vi.rule)
    t.columns = ["Nếu khách có đặc điểm", "Support", "Tỉ lệ rời bỏ", "Lift", "Tỉ lệ rời bỏ khi kiểm tra lại"]
    T(t, "Các tổ hợp đặc điểm đi kèm tỉ lệ rời bỏ thấp",
      fmt={"Support": "{:.1%}", "Tỉ lệ rời bỏ": "{:.1%}", "Lift": "{:.2f}", "Tỉ lệ rời bỏ khi kiểm tra lại": "{:.1%}"})
FIG(F / "rules_scatter.png", "Mỗi điểm là một tổ hợp đặc điểm; điểm viền đỏ là luật được chọn")
P(f"Kết quả: có {ru.get('n_churn_rules')} luật gắn với tỉ lệ rời bỏ cao và {ru.get('n_retention_rules')} luật gắn "
  f"với tỉ lệ rời bỏ thấp đáp ứng mọi tiêu chí kiểm tra. Khi đo lại trên 30% dữ liệu giữ riêng, tỉ lệ rời bỏ của các luật lệch trung bình chỉ "
  f"{num(ru.get('mean_abs_conf_drift', 0.01) * 100, 1)} điểm phần trăm, nên các luật này đáng tin.")
bullets(d, [
    "Rủi ro nhất là khách mua gói dài hơn 30 ngày: khoảng 83% không gia hạn. Đây cũng chính là nhóm 4 ở phần phân cụm.",
    "Nhóm khách tắt tự gia hạn có tỉ lệ rời bỏ cao gấp khoảng 4,5 lần mức chung. Nếu khách vừa tắt tự gia hạn vừa ít nghe, tỉ lệ rời bỏ vượt 50%. "
    "Như vậy thói quen nghe có ích khi đi kèm thông tin thanh toán, dù riêng nó thì yếu.",
    "Tỉ lệ rời bỏ thấp nhất thuộc về khách mới (dưới 1 năm) đăng ký qua kênh 7: chỉ 2,2%.",
])
sens = csv("rule_sensitivity.csv")
if len(sens):
    P("Khi đổi ngưỡng support tối thiểu từ 0,5% lên 5%, số luật đạt yêu cầu giảm từ "
      f"{k(sens.n_significant_churn_rules.iloc[0])} xuống {k(sens.n_significant_churn_rules.iloc[-1])}. Các chủ đề "
      "chính (tắt tự gia hạn, gói dài hạn, kênh 9) vẫn xuất hiện, nhưng thứ tự từng luật thay đổi theo ngưỡng. "
      "Ngưỡng 1% được chọn để mỗi luật dựa trên ít nhất khoảng 6.800 khách.")

H("5.3 Mẫu tuần tự (PrefixSpan)", 2)
P("Luật kết hợp chỉ nhìn trạng thái hiện tại. Mẫu tuần tự xét thêm thứ tự: điều gì xảy ra trước, điều gì xảy ra "
  "sau trong lịch sử mua gói. Mỗi lần giao dịch được mô tả bằng các sự kiện như trả tự động, trả thủ công, hủy, "
  "đổi gói, gói dài. PrefixSpan tìm các chuỗi sự kiện xuất hiện ở ít nhất 1% khách, sau đó các chuỗi được kiểm "
  f"định thống kê và đo lại trên phần dữ liệu giữ riêng giống như luật kết hợp ({k(sq.get('n_users', 300000))} khách, "
  f"{k(sq.get('n_tested', 4945))} chuỗi được kiểm tra).")
sp = csv("sequential_patterns.csv")
if len(sp):
    t = sp.groupby("direction", group_keys=False).head(7)[["pattern", "direction", "support_train", "conf_train",
                                                            "lift_train", "conf_holdout"]].copy()
    t["pattern"] = t.pattern.map(vi.pattern)
    t["direction"] = t.direction.map({"churn": "rời bỏ cao", "retain": "rời bỏ thấp"})
    t.columns = ["Chuỗi sự kiện (→ là “sau đó”)", "Loại", "Support", "Tỉ lệ rời bỏ", "Lift", "Khi kiểm tra lại"]
    T(t, "Các chuỗi giao dịch nổi bật",
      fmt={"Support": "{:.1%}", "Tỉ lệ rời bỏ": "{:.1%}", "Lift": "{:.1f}", "Khi kiểm tra lại": "{:.1%}"})
FIG(F / "seq_top_patterns.png", "Lift của các chuỗi rủi ro nhất trên tháng 3 và tháng 2")
bullets(d, [
    "Chuỗi gắn với tỉ lệ rời bỏ cao nhất gồm các giao dịch theo thứ tự: trả tiền thủ công, sau đó mua gói dài, "
    "rồi tiếp tục mua gói dài. Khoảng 70% khách có chuỗi này rời bỏ.",
    "Đổi gói nhiều lần, hoặc hủy rồi đổi gói, đi kèm tỉ lệ rời bỏ khoảng 40%.",
    "Chuyển từ trả tự động sang trả thủ công kèm đổi gói cũng là dấu hiệu xấu (khoảng 44%).",
    "Ngược lại, khách trả tự động liên tục chỉ rời bỏ khoảng 5%.",
])
P("Ưu điểm của mẫu tuần tự là các sự kiện này xảy ra ngay lúc khách giao dịch, nên có thể dùng làm tín hiệu cảnh "
  "báo sớm. Tuy nhiên nhóm chưa đo được các sự kiện đó xảy ra trước ngày hết hạn bao lâu.")

H("5.4 Dự báo khách rời bỏ (phương pháp bổ trợ)", 2)
P(f"Mô hình được huấn luyện bằng dữ liệu của {k(cf.get('n_train', 0))} khách tháng 2, sau đó được đánh giá trên "
  f"{k(cf.get('n_test', 0))} khách tháng 3 để mô phỏng việc dự báo cho tháng tiếp theo. Nhóm so sánh năm cách: "
  "đoán theo tỉ lệ chung, hồi quy logistic, cây quyết định, rừng ngẫu nhiên và LightGBM.")
if len(oot):
    t = oot[["model", "roc_auc", "pr_auc", "precision@10%", "recall@10%"]].copy()
    t["model"] = t.model.map(vi.MODELS)
    t.columns = ["Mô hình", "ROC-AUC", "PR-AUC", "Độ chính xác ở top 10%", "Bắt được (top 10%)"]
    T(t, "Kết quả dự báo trên tháng 3", fmt={"ROC-AUC": "{:.3f}", "PR-AUC": "{:.3f}",
                                             "Độ chính xác ở top 10%": "{:.1%}", "Bắt được (top 10%)": "{:.1%}"})
FIG(F / "clf_curves.png", "Đường ROC, đường độ chính xác – độ phủ và đường hiệu chuẩn trên tháng 3")
FIG(F / "clf_gains.png", "Liên hệ x% khách rủi ro nhất thì bắt được bao nhiêu phần trăm khách sẽ rời đi", width=4.5)
P(f"LightGBM đạt kết quả tốt nhất, với ROC-AUC {num(lg.get('roc_auc'))} và PR-AUC {num(lg.get('pr_auc'))}. Để so "
  f"sánh, một cách dự báo ngẫu nhiên chỉ đạt PR-AUC khoảng {num(base, 3)}. Trong 10% khách bị mô hình đánh giá rủi ro nhất có {pct(lg.get('precision@10%'), 0)} thật sự rời "
  f"đi, và nhóm 10% này chứa {pct(lg.get('recall@10%'), 0)} tổng số khách rời bỏ.")
cvt = csv("classification_cv.csv")
if len(cvt):
    cv_lgb = cvt.set_index(cvt.columns[0]).loc["LightGBM", "roc_auc_mean"]
    P(f"Một điểm cần lưu ý: khi chỉ kiểm tra trong cùng tháng 2, ROC-AUC là {num(cv_lgb)}; sang tháng 3 giảm còn "
      f"{num(lg.get('roc_auc'))}. Như vậy đánh giá trong cùng tháng cho kết quả cao hơn đánh giá trên tháng kế "
      f"tiếp khoảng {num(cv_lgb - lg.get('roc_auc'))} ROC-AUC.")
abl = pd.DataFrame(cf.get("ablation", []))
if len(abl):
    t = abl[["feature_set", "n_features", "roc_auc", "pr_auc"]].copy()
    t["feature_set"] = t.feature_set.map(vi.ABLATION)
    t.columns = ["Nhóm đặc trưng được dùng", "Số biến", "ROC-AUC", "PR-AUC"]
    T(t, "Mô hình dùng từng nhóm đặc trưng riêng lẻ", fmt={"ROC-AUC": "{:.3f}", "PR-AUC": "{:.3f}"})
    P("Chỉ dùng thông tin thuê bao đã đạt gần như toàn bộ kết quả; chỉ dùng thói quen nghe thì chỉ nhỉnh hơn dự báo ngẫu nhiên một chút. "
      "Điều này khớp với phần khám phá dữ liệu.")
FIG(F / "clf_shap_summary.png", "Mức ảnh hưởng của từng đặc trưng lên dự báo của LightGBM (phương pháp SHAP)", width=5.5)
P("Bốn yếu tố ảnh hưởng nhất là: lần mua cuối có tự gia hạn không, cách thanh toán cuối, giá gói cuối, và khách "
  "đã giao dịch bao lâu. Thông tin nghe nhạc đứng sau. Cây quyết định nhỏ (độ sâu 4) cũng chia theo đúng logic đó: "
  "câu hỏi đầu tiên là “có tự gia hạn không”.")

mi = csv("model_improvement.csv")
if len(mi):
    H("5.5 Thử cải thiện mô hình", 2)
    P("Nhóm thử ba phương án: (1) thêm 13 đặc trưng về lịch sử thanh toán, như số lần bật/tắt tự gia "
      "hạn, số lần lỡ hạn, khoảng cách giữa các lần mua, trạng thái 3 lần mua gần nhất; (2) tinh chỉnh tham số "
      "LightGBM bằng công cụ Optuna (30 lần thử); (3) kết hợp cả hai. Việc tinh chỉnh chỉ dùng dữ liệu tháng 2; tháng 3 chỉ dùng để chấm "
      "điểm cuối cùng.")
    t = mi[["configuration", "n_features", "roc_auc", "pr_auc", "recall@10%", "delta_auc"]].copy()
    t["configuration"] = t.configuration.map(vi.CONFIGS)
    t.columns = ["Cách làm", "Số biến", "ROC-AUC", "PR-AUC", "Bắt được (top 10%)", "Tăng so với A"]
    T(t, "Kết quả thử cải thiện mô hình trên tháng 3",
      fmt={"ROC-AUC": "{:.4f}", "PR-AUC": "{:.4f}", "Bắt được (top 10%)": "{:.1%}", "Tăng so với A": "{:+.4f}"})
    FIG(F / "model_improvement.png", "Mức tăng ROC-AUC và mức đóng góp của các đặc trưng mới (màu cam)")
    best_mi = mi.iloc[-1]
    P(f"Cả hai cách đều giúp tăng điểm, kết hợp lại đạt ROC-AUC {num(best_mi.roc_auc, 4)} (tăng "
      f"{num(best_mi.delta_auc, 4)}). Mức tăng này có thật về mặt thống kê nhưng rất nhỏ: số khách rời bỏ bắt được "
      "trong top 10% gần như không đổi. Đặc trưng mới hữu ích nhất là tỉ lệ tự gia hạn trong 3 lần mua gần nhất, "
      "khớp với chuỗi “trả tự động → trả thủ công” ở phần mẫu tuần tự. Trong các thử nghiệm này, bổ sung đặc trưng và "
      "tinh chỉnh tham số chỉ cải thiện một phần nhỏ; hướng tiếp theo nên là bổ sung dữ liệu mới, như lý do khách "
      "tắt tự gia hạn và lịch sử chăm sóc khách hàng. Các phần sau vẫn dùng mô hình LightGBM ban đầu (trước khi bổ "
      "sung đặc trưng và tinh chỉnh) để thống nhất số liệu.")

# =============================================================================================
H("6. Đánh giá và diễn giải", 1)
H("6.1 Kiểm tra rò rỉ dữ liệu", 2)
P("Để biết phiên bản trước bị thổi phồng bao nhiêu, nhóm giữ nguyên cùng một nhóm khách tháng 3 và cùng cách chia "
  "dữ liệu, rồi lần lượt thêm vào các thông tin xảy ra sau mốc quan sát.")
audit = pd.DataFrame(va.get("leakage_audit", []))
if len(audit):
    t = audit[["feature_set", "roc_auc", "auc_inflation_vs_clean"]].copy()
    t["feature_set"] = t.feature_set.map(vi.AUDIT)
    t.columns = ["Dữ liệu dùng", "ROC-AUC", "Tăng ảo so với dữ liệu sạch"]
    T(t, "Kết quả tăng lên bao nhiêu khi dùng nhầm dữ liệu sau mốc quan sát", fmt={"ROC-AUC": "{:.3f}",
                                                                                  "Tăng ảo so với dữ liệu sạch": "{:+.3f}"})
FIG(F / "leakage_audit.png", "Xanh: chỉ dùng thông tin có trước mốc; đỏ: có dùng thông tin của tháng cần dự báo")
rule_v1 = va.get("leaky_rule_no_march_tx", {})
P("Thêm giao dịch tháng 3 vào làm ROC-AUC tăng từ 0,922 lên 0,990. Lý do rất đơn giản: khách không mua gói trong "
  "tháng 3 thì phần lớn chính là khách rời bỏ, tức mô hình đang “nhìn thấy” đáp án. Cách làm của phiên bản 1 "
  "cũng được lợi theo kiểu này (0,976). Riêng với thói quen nghe, dùng nhật ký tháng 3 làm điểm tăng từ 0,575 lên "
  "0,657, nên phần lớn tín hiệu nghe nhạc mà phiên bản 1 tìm được thực ra xuất hiện sau khi khách đã rời đi.")
P("Bài học: khi nhãn được xác định bằng một sự kiện trong tương lai, phải cắt dữ liệu tại mốc quan sát, và phải "
  "kiểm tra mô hình trên một tháng khác chứ không chỉ trong cùng tháng.")

H("6.2 Kết quả có lặp lại ở tháng khác không?", 2)
P(f"Một mẫu chỉ có giá trị nếu nó không phải chuyện ngẫu nhiên của một tháng. Nhóm lấy đúng các nhóm khách, luật và "
  f"chuỗi tìm được trên tháng 3, áp lên khách tháng 2 rồi so sánh. Vì tỉ lệ rời bỏ chung của hai tháng khác nhau "
  f"({pct(st.get('base_feb'))} và {pct(st.get('base_mar'))}), nhóm so sánh bằng lift.")
ctemp = csv("cluster_temporal.csv")
if len(ctemp):
    t = ctemp[["cluster", "name", "share_mar", "share_feb", "churn_mar", "churn_feb"]].copy()
    t.columns = ["Nhóm", "Đặc điểm", "Tỉ trọng T3", "Tỉ trọng T2", "Rời bỏ T3", "Rời bỏ T2"]
    T(t, "Bảy nhóm khách hàng ở tháng 3 và tháng 2",
      fmt={c: "{:.1%}" for c in ["Tỉ trọng T3", "Tỉ trọng T2", "Rời bỏ T3", "Rời bỏ T2"]})
FIG(F / "stability.png", "Trái: tỉ lệ rời bỏ của từng nhóm ở hai tháng. Phải: lift của các luật ở hai tháng (càng gần đường chéo càng ổn định)")
bullets(d, [
    "Nhóm khách hàng: tỉ trọng mỗi nhóm gần như không đổi và thứ tự rủi ro giữ nguyên hoàn toàn. Chạy lại phân "
    f"cụm từ đầu trên tháng 2 cũng ra gần đúng các nhóm cũ (ARI {num(st.get('ari_feb_refit_vs_mar_model'), 2)}).",
    f"Luật kết hợp: {round(st.get('rule_share_same_direction', 0) * st.get('rules_checked', 0))} trên "
    f"{st.get('rules_checked')} luật vẫn gắn với mức rủi ro cao hoặc thấp như ở tháng 3. Luật duy nhất đổi chiều có "
    "điều kiện “còn hạn > 31 ngày”, một điều kiện phụ thuộc lịch hết hạn của từng tháng; một luật khác cùng điều "
    "kiện này cũng yếu đi rõ rệt.",
    "Chuỗi giao dịch: cả 30 chuỗi được báo cáo đều giữ nguyên ý nghĩa ở tháng 2.",
    "Mức rủi ro tuyệt đối thì thay đổi: nhóm 4 rời bỏ 84% ở tháng 3 nhưng 49% ở tháng 2, dù vẫn là nhóm rủi ro "
    "nhất. Vì vậy khi áp dụng nên dựa vào thứ hạng rủi ro, không nên dựa vào một ngưỡng phần trăm cố định.",
])

H("6.3 Xác suất dự báo và giá trị kinh doanh", 2)
calt = csv("calibration.csv")
P("Ngoài việc xếp hạng, nhóm kiểm tra xác suất mô hình đưa ra có khớp thực tế không. Mô hình học ở tháng 2 (6,4% "
  "rời bỏ) nên dự báo trung bình 6,4%, trong khi tháng 3 thực tế là 9,0%. Nhóm thử điều chỉnh bằng thuật toán EM, "
  "một cách ước lượng lại tỉ lệ chung mà không cần biết đáp án, nhưng thuật toán vẫn ước lượng khoảng 6,5% nên "
  "gần như không cải thiện. Nếu biết trước tỉ lệ thật thì sai số giảm hơn một nửa, nhưng điều này không làm được "
  "trong thực tế. Kết luận: mô hình dùng tốt để xếp hạng khách theo rủi ro; còn con số phần trăm thì cần điều "
  "chỉnh lại khi đã biết tỉ lệ rời bỏ thật của tháng gần nhất.")
if len(calt):
    t = calt[["version", "mean_pred", "actual_rate", "ece"]].copy()
    t["version"] = t.version.map(vi.CALIB)
    t.columns = ["Cách tính", "Dự báo trung bình", "Thực tế", "Sai số hiệu chuẩn (ECE)"]
    T(t, "Xác suất dự báo so với thực tế trên tháng 3",
      fmt={"Dự báo trung bình": "{:.1%}", "Thực tế": "{:.1%}", "Sai số hiệu chuẩn (ECE)": "{:.3f}"})
P(f"Giá trị kinh doanh được ước tính bằng một ví dụ với giả định rõ ràng: mỗi lần liên hệ tốn 5 NTD; khách được "
  f"liên hệ và ở lại nhận ưu đãi giảm 20% phí một tháng; giữ được một khách đem lại 6 tháng doanh thu (phí tháng "
  f"phổ biến là {k(bv.get('median_arpu_ntd', 149))} NTD); tỉ lệ cứu được khách là 10%, 20% hoặc 30%.")
pcv = csv("profit_curve.csv")
if len(pcv):
    b = pcv.loc[pcv.groupby("success_rate").net_value_ntd.idxmax()]
    t = b[["success_rate", "share_contacted", "churners_reached", "net_value_ntd", "random_targeting_ntd"]].copy()
    t.columns = ["Tỉ lệ cứu được", "Nên liên hệ", "Khách rời bỏ được tiếp cận", "Lãi ròng (NTD)", "Nếu chọn ngẫu nhiên (NTD)"]
    T(t, "Số khách nên liên hệ để có lãi cao nhất, theo từng giả định",
      fmt={"Tỉ lệ cứu được": "{:.0%}", "Nên liên hệ": "{:.0%}", "Khách rời bỏ được tiếp cận": "{:.0%}",
           "Lãi ròng (NTD)": "{:,.0f}", "Nếu chọn ngẫu nhiên (NTD)": "{:,.0f}"})
FIG(F / "profit_curve.png", "Lãi ròng theo phần trăm khách được liên hệ")
P("Theo các giả định này, chỉ nên liên hệ khoảng 8–16% khách rủi ro nhất. Liên hệ nhiều hơn thì lãi giảm, vì ưu đãi "
  "bị dùng bởi cả những khách vốn không định rời đi; liên hệ tất cả thì lỗ. Chọn khách ngẫu nhiên thì lỗ ở mọi "
  "mức. Các con số tiền chỉ để minh họa cách ra quyết định; tỉ lệ cứu được thật cần đo bằng thử nghiệm A/B.")

H("6.4 So sánh các kỹ thuật", 2)
comp = pd.DataFrame([
    ("Phân cụm (K-Means)", f"silhouette {num(cl.get('silhouette'), 2)}, ổn định cao (ARI {num(cl.get('bootstrap_ari_mean'), 2)})",
     "Cao", "Lặp lại ở tháng 2, thứ tự rủi ro giữ nguyên", "Chia ngân sách, thiết kế chương trình theo nhóm"),
    ("Luật kết hợp (FP-Growth)", f"{ru.get('n_churn_rules')} luật rủi ro, kiểm tra lại gần như khớp",
     "Rất cao (dạng nếu… thì…)", f"{round(st.get('rule_share_same_direction', 0) * 100)}% giữ ý nghĩa", "Đặt điều kiện kích hoạt hành động"),
    ("Mẫu tuần tự (PrefixSpan)", "30 chuỗi nổi bật, kiểm tra lại gần như khớp", "Cao",
     "100% giữ ý nghĩa", "Cảnh báo sớm ngay khi khách giao dịch"),
    ("Phân lớp (LightGBM)", f"ROC-AUC {num(lg.get('roc_auc'))}, bắt {pct(lg.get('recall@10%'), 0)} ở top 10%",
     "Trung bình (cần SHAP)", "Xếp hạng tốt, xác suất lệch", "Chọn khách để liên hệ khi ngân sách có hạn"),
], columns=["Kỹ thuật", "Chất lượng", "Dễ hiểu", "Ổn định theo thời gian", "Dùng để"])
T(comp, "So sánh bốn kỹ thuật")
P("Bốn kỹ thuật đi đến cùng một kết luận: nhóm mua gói dài hạn và trả thủ công là rủi ro nhất; tắt tự gia hạn là "
  "dấu hiệu rủi ro chính; thói quen nghe nhạc chỉ bổ sung. Vì các kỹ thuật dùng chung một bộ dữ liệu nên đây là sự "
  "nhất quán, chưa phải bằng chứng độc lập. Mỗi kỹ thuật có điểm yếu riêng: phân cụm cần người phân tích đặt tên và "
  "chọn số cụm; luật kết hợp phụ thuộc cách chia ngưỡng; mẫu tuần tự phải thử rất nhiều chuỗi nên cần kiểm định "
  "chặt; mô hình dự báo khó giải thích và nhạy với thay đổi giữa các tháng.")
P(f"Về tốc độ, toàn bộ quy trình chạy trên một máy tính cá nhân 16 GB RAM. Bước lâu nhất là huấn luyện mô hình "
  f"(khoảng 12 phút); phân cụm khoảng 1 phút; luật kết hợp và mẫu tuần tự mỗi bước dưới 30 giây.")

# =============================================================================================
H("7. Gợi ý cho doanh nghiệp", 1)
P("Các gợi ý dưới đây dựa trên mối liên hệ tìm được trong dữ liệu. Trước khi áp dụng rộng, mỗi gợi ý nên được thử "
  "nghiệm có nhóm đối chứng (A/B test).")
bullets(d, [
    f"Khuyến khích bật tự gia hạn. Khách tắt tự gia hạn rời đi {pct(g('auto_renew', 'off'), 0)}, khách bật chỉ "
    f"{pct(g('auto_renew', 'on'))}. Nên thử trước với nhóm 3 (khoảng {k(seg.loc[3, 'n'] / 1000) if len(seg) else '81'} "
    "nghìn khách nghe nhiều nhưng trả thủ công), ví dụ tặng ưu đãi nhỏ khi bật tự gia hạn.",
    "Chăm sóc riêng khách mua gói dài hạn (nhóm 4). Nhóm này rất hiếm khi gia hạn. Có thể nhắc gia hạn trước ngày "
    "hết hạn kèm ưu đãi, hoặc mời chuyển sang gói tháng tự gia hạn. Nếu vẫn không cải thiện, nên xem lại hiệu quả "
    "của loại gói này.",
    "Cảnh báo theo sự kiện: khi khách tắt tự gia hạn, chuyển sang trả thủ công hoặc đổi gói nhiều lần, hệ thống nên "
    "báo cho bộ phận chăm sóc khách hàng ngay, không đợi đến ngày hết hạn.",
    f"Theo dõi khách trả tiền nhưng không nghe (nhóm 2, khoảng {k(seg.loc[2, 'n'] / 1000) if len(seg) else '190'} nghìn "
    "khách). Hiện họ ít rời bỏ, nhưng có thể hủy khi để ý đến khoản phí. Có thể gợi ý nội dung phù hợp để họ quay lại nghe.",
    "Xem lại kênh đăng ký 3 và 4 (tỉ lệ rời bỏ 17–23%) so với kênh 7 (4,5%) trước khi quyết định đầu tư thêm cho kênh nào.",
    "Dùng mô hình để chọn khách cần liên hệ mỗi tháng: liên hệ 10% khách rủi ro nhất đã tiếp cận khoảng 61% số khách "
    "sắp rời đi. Mô hình cần được huấn luyện lại mỗi tháng.",
    "Không dùng việc nghe ít đi làm tín hiệu cảnh báo chính: trong 8 tuần trước mốc quan sát, khách rời bỏ không hề "
    "nghe ít đi so với khách ở lại.",
])

# =============================================================================================
H("8. Kỹ thuật nâng cao (phần cộng điểm)", 1)
P("Ngoài các kỹ thuật chính, nhóm dùng thêm: tạo đặc trưng theo thời gian (xu hướng nghe, số ngày ngưng nghe, độ "
  "đều đặn theo tuần); xử lý dữ liệu lớn bằng cách đọc dữ liệu nén theo luồng; phát hiện bất thường; so sánh các "
  "nhóm con; giảm chiều dữ liệu bằng PCA và UMAP; và các kiểm tra như dữ liệu giữ riêng, lặp lại trên tháng khác, "
  "đo rò rỉ dữ liệu.")
FIG(F / "bonus_trajectory.png", "Số ngày nghe và số giờ nghe mỗi tuần trong 8 tuần trước mốc quan sát")
P("Khách rời bỏ không nghe ít đi trước khi rời đi; số ngày nghe của họ còn nhỉnh hơn khách ở lại và tăng nhẹ ở "
  "những tuần cuối. Nhóm có mức nghe tăng mạnh lại có tỉ lệ rời bỏ cao nhất (12,7%), phù hợp với nhóm 1 (khách "
  "mới hoặc quay lại dùng thử).")
tr = csv("bonus_trend.csv")
if len(tr):
    t = tr.rename(columns={tr.columns[0]: "g"})
    t = pd.DataFrame({"Xu hướng nghe": t.g.map(vi.TREND), "Số khách": t.n, "Tỉ lệ rời bỏ": t.churn_rate})
    T(t, "Tỉ lệ rời bỏ theo xu hướng nghe (khách có nhật ký nghe)", fmt={"Số khách": "{:,.0f}", "Tỉ lệ rời bỏ": "{:.1%}"})
an = csv("bonus_anomaly.csv")
if len(an):
    t = pd.DataFrame({"Nhóm": an.iloc[:, 0].map({"anomaly": "bất thường", "normal": "bình thường"}),
                      "Số khách": an.n, "Tỉ lệ rời bỏ": an.churn_rate})
    T(t, "Phát hiện bất thường (Isolation Forest): 2% khách khác thường nhất", fmt={"Số khách": "{:,.0f}", "Tỉ lệ rời bỏ": "{:.1%}"})
    P("Isolation Forest không biết ai rời đi, chỉ tìm những khách khác thường nhất. 2% khách bị đánh dấu có tỉ lệ rời "
      "bỏ 69%. Phần lớn họ chỉ có một lần mua, trả thủ công cho gói khoảng 195 ngày, tức cùng kiểu khách với nhóm 4.")
sub = csv("bonus_subsets.csv")
if len(sub):
    t = sub[~sub.subset.str.startswith("segment")][["subset", "n", "churn_rate", "roc_auc"]].copy()
    t["subset"] = t.subset.map(lambda x: vi.SUBSETS.get(x, x))
    t.columns = ["Nhóm con", "Số khách", "Tỉ lệ rời bỏ", "ROC-AUC trong nhóm"]
    T(t, "Mô hình dự báo hoạt động thế nào trong từng nhóm con",
      fmt={"Số khách": "{:,.0f}", "Tỉ lệ rời bỏ": "{:.1%}", "ROC-AUC trong nhóm": "{:.3f}"})
    P("Mô hình phân biệt tốt nhất ở khách mới dưới 1 năm và kém hơn ở khách đang bật tự gia hạn, vì với nhóm này dữ "
      "liệu trước mốc quan sát ít cho biết ai sẽ hủy.")
FIG(F / "umap_churn.png", "UMAP: chiếu dữ liệu xuống 2 chiều; trái tô màu theo rời bỏ, phải theo nhóm khách")
FIG(F / "bonus_pca_scree.png", "PCA: phần thông tin mà mỗi thành phần chính giữ lại", width=4.5)
P("Khi chiếu xuống 2 chiều bằng UMAP, khách hàng tách thành các “đảo” trùng với các nhóm của K-Means, xác nhận thêm "
  "kết quả phân cụm. Khách rời bỏ tập trung ở đảo của nhóm 3 và các đảo nhỏ của nhóm 4. Với PCA, ba thành phần "
  "chính đầu tiên giải thích khoảng 67% tổng phương sai; chúng lần lượt liên quan chủ yếu đến mức độ nghe nhạc, "
  "số lượng giao dịch trong lịch sử, và giá cùng thời hạn gói.")

# =============================================================================================
H("9. Kết luận, hạn chế và hướng phát triển", 1)
P("Kết luận: ở KKBox, rủi ro khách rời bỏ gắn chặt nhất với cách khách trả tiền (tự gia hạn hay trả tay, gói tháng "
  "hay gói dài), còn thói quen nghe nhạc chỉ bổ sung. Bốn kỹ thuật cho kết quả nhất quán; các mẫu chính đều được "
  "kiểm tra lại trên dữ liệu giữ riêng và lặp lại ở tháng 2.")
P("Về phương pháp, đồ án phát hiện và đo được lỗi rò rỉ dữ liệu của phiên bản trước (ROC-AUC 0,977 so với 0,869 "
  "khi làm đúng), và xây dựng quy trình có thể chạy lại từ đầu với cùng kết quả.")
P("Hạn chế:")
bullets(d, [
    "Danh sách khách mỗi tháng do KKBox chọn sẵn; chỉ có hai tháng có nhãn.",
    "Không có dữ liệu về lý do khách tắt tự gia hạn, giá của đối thủ hay khiếu nại.",
    "Các luật và nhóm khách cho thấy mối liên hệ, chưa chứng minh nguyên nhân.",
    "Tuổi và giới tính thiếu nhiều.",
    "Xác suất dự báo bị lệch khi tỉ lệ rời bỏ thay đổi giữa các tháng; giá trị kinh doanh dựa trên giả định.",
])
P("Hướng phát triển: kiểm tra trên nhiều tháng hơn; thử nghiệm A/B cho từng biện pháp giữ chân; đo khoảng thời gian "
  "từ tín hiệu cảnh báo đến ngày hết hạn; tìm mẫu tuần tự trong nhật ký nghe hằng ngày; bổ sung dữ liệu về lý do "
  "khách tắt tự gia hạn.")

H("Phụ lục: cách chạy lại", 1)
bullets(d, [
    "Cài thư viện: pip install -r requirements.txt",
    "Lấy dữ liệu: tải file “Download All” trên trang Kaggle của cuộc thi, rồi chạy "
    "python data_prep/00_download.py <đường dẫn file zip>",
    "Chạy toàn bộ: python run_all.py --skip-download (khoảng 45 phút trên máy 16 GB RAM)",
    "Dựng lại báo cáo: python report/make_diagrams.py rồi python report/build_report.py",
    "Thử nhanh không cần dữ liệu thật: python tests/make_synthetic_data.py data_synth, sau đó đặt DATA_DIR=data_synth "
    "REPORT_DIR=report_synth và chạy python run_all.py --skip-download",
])

OUT = Path(os.environ.get("REPORT_OUT", R / "report.docx"))
d.save(OUT)
print(f"report written: {OUT} ({n_tab[0]} tables, {n_fig[0]} figures)")
