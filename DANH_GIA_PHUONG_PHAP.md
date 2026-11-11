# Đánh giá phương pháp và tổng hợp cải tiến

Tài liệu này dùng để tự đánh giá đồ án theo góc nhìn người chấm: mỗi bước làm gì, vì sao, đã kiểm chứng bằng gì, và còn điểm yếu nào. Phần đánh giá được Claude và Codex rà soát độc lập trên code và dữ liệu thật (ngày 11/11/2026).

Ký hiệu trạng thái: **Đạt** – ổn ở mức thạc sĩ · **Cần sửa** – ảnh hưởng tới kết quả, nên sửa trước khi nộp · **Nên bổ sung** – làm bài mạnh hơn nhưng không bắt buộc.

---

## Phần A. Tổng hợp cải tiến theo từng vòng

| Vòng | Nội dung | Lý do | Kết quả chính |
|---|---|---|---|
| 1. Rà soát bản cũ | Codex đọc code bản cũ | Tìm lỗi trước khi làm tiếp | Phát hiện rò rỉ dữ liệu tháng 3, lỗi `rank + qcut` tạo đặc điểm giả, báo cáo ghi số tay |
| 2. Thiết kế lại | Mốc quan sát cho từng nhóm khách; học tháng 2, kiểm tra tháng 3 | Bỏ rò rỉ, mô phỏng đúng cách dùng thật | ROC-AUC từ 0,977 (ảo) xuống 0,869 (thật) |
| 3. Dữ liệu lớn | Đọc file nhật ký 28 GB theo luồng | Máy không đủ ổ đĩa để giải nén | 392 triệu dòng → 40,9 triệu dòng cần dùng, 13 phút |
| 4. Khai phá | Phân cụm, luật kết hợp, mẫu tuần tự (Codex), phân lớp, phát hiện bất thường | Đề yêu cầu ≥ 2 kỹ thuật, ≥ 1 không giám sát | 7 nhóm khách; luật "gói > 30 ngày → 83%"; chuỗi "trả thủ công → gói dài" |
| 5. Kiểm chứng | Holdout 30%, kiểm định + FDR, bootstrap, đo rò rỉ | Tránh mẫu ngẫu nhiên, đo mức thổi phồng | Lift train/holdout tương quan 0,998; rò rỉ thổi AUC thêm 0,05–0,07 |
| 6. Ổn định theo thời gian | Áp kết quả tháng 3 lên tháng 2 | Mẫu thật phải lặp lại | 34/35 luật, 30/30 chuỗi giữ ý nghĩa; thứ tự rủi ro nhóm giữ nguyên |
| 7. Giá trị kinh doanh | Hiệu chỉnh xác suất (EM), đường lợi nhuận | Nối kết quả với quyết định | Nên liên hệ 8–16% khách rủi ro nhất; EM không sửa được lệch xác suất |
| 8. Cải thiện mô hình | 13 đặc trưng thanh toán + Optuna | Thử nâng độ chính xác | +0,005 ROC-AUC, không đổi ở top 10% |
| 9. Trình bày | Viết lại lời văn, sơ đồ, thuật ngữ, Việt hoá | Báo cáo dễ đọc | 21 bảng, 21 hình, có sơ đồ quy trình và trục thời gian |
| 10. Rà soát phương pháp (tài liệu này) | Kiểm tra lại dữ liệu và phương pháp | Tự đánh giá trước khi nộp | Phát hiện lỗi gói 0 ngày và các điểm yếu ở mục B, C |

Chi tiết thay đổi từng file: [THAY_DOI.md](THAY_DOI.md).

---

## Phần B. Tiền xử lý và làm sạch dữ liệu

### B1. Những gì đã làm tốt

| Vấn đề | Cách xử lý | Trạng thái |
|---|---|---|
| Rò rỉ dữ liệu tháng nhãn | Cắt mọi nguồn tại mốc quan sát; `manifest.json` tự kiểm tra | **Đạt** |
| Giao dịch trùng giữa 2 file | Gộp và bỏ dòng trùng | **Đạt** |
| Thứ tự giao dịch cùng ngày | Một quy tắc sắp xếp chung cho mọi bước | **Đạt** (xem B2.5) |
| Tuổi sai (âm, 0, > 100) | Ngoài 10–80 coi là thiếu (60% thiếu); không dùng tuổi trong phân cụm | **Đạt** |
| Giới tính trống | Nhóm "không rõ" | **Đạt** |
| Giờ nghe phi lý (> 24 giờ/ngày, âm) | Chặn trong [0, 24 giờ] | **Đạt** |
| Khách không có nhật ký nghe (19%) | Giữ lại, có cờ `has_logs`, số đếm = 0 | **Đạt** |
| Điền giá trị thiếu | Chỉ học trên phần huấn luyện (trong Pipeline) | **Đạt** |
| Mất cân bằng lớp (9% rời bỏ) | Giữ tỉ lệ thật, dùng PR-AUC và lift thay vì accuracy | **Đạt** |
| Đuôi dài (giờ nghe, số giao dịch) | log1p trước khi chuẩn hoá cho phân cụm; mô hình cây không cần | **Đạt** |

### B2. Vấn đề phát hiện khi rà soát lại

| # | Vấn đề (số liệu trên dữ liệu thật) | Mức | Ảnh hưởng | Cách sửa đề xuất |
|---|---|---|---|---|
| 1 | **Giao dịch gói 0 ngày**: 581.132 giao dịch (3,8%) có độ dài gói = 0 và giá niêm yết = 0 nhưng vẫn trả 149 NTD, 97% tự gia hạn. Đây là bản ghi cũ thiếu thông tin gói, không phải gói thật. | **Cao** | 78% số lần "đổi gói" là đổi giả (30 → 0 → 30); 282 nghìn khách bị ghi "có đổi gói" chỉ vì lỗi này. Làm sai `n_plan_changes`, `discount_rate`, sự kiện `PLAN_CHANGE`/`PLAN_SHORT` trong mẫu tuần tự. Chỉ 6 khách có lỗi này ở giao dịch cuối nên các kết quả chính (tự gia hạn, gói dài) ít bị ảnh hưởng. | Coi gói 0 ngày là "không rõ"; chỉ đếm đổi gói giữa các gói hợp lệ; tính giảm giá trên giao dịch có giá niêm yết > 0. Chạy lại toàn bộ. |
| 2 | **Ngày hết hạn trước ngày giao dịch**: 61.098 giao dịch; 6.912 có ngày hết hạn trước 2015 (hỏng). Ở giao dịch cuối: 153 khách. | Trung bình | `days_to_expire` có giá trị cực trị (−1156 đến 2113 ngày). | Ngày hỏng chuyển thành thiếu + cờ lỗi; không xoá giao dịch hủy (có thể hợp lệ). |
| 3 | **Ngày đăng ký sau mốc quan sát**: 1.057 khách (thâm niên âm), tỉ lệ rời bỏ 48%. | Trung bình (ít khách) | Bất nhất thời gian; có thể là đăng ký lại sau khi rời bỏ. | Chuyển thâm niên thành thiếu, thêm cờ. |
| 4 | **Thành phố 1**: 51% khách, 97% trong số đó không khai giới tính. | Thấp | Có thể là mã "không rõ"; chưa đủ bằng chứng khẳng định. | Ghi chú khi diễn giải; không dùng thành phố để kết luận. |
| 5 | **Thứ tự giao dịch cùng ngày**: quy tắc sắp xếp giúp tái lập nhưng không chứng minh thứ tự thật. | Thấp | Có thể ảnh hưởng "giao dịch cuối" khi có nhiều giao dịch cùng ngày. | Kiểm tra độ nhạy (đổi quy tắc, xem kết quả có đổi không). |
| 6 | **11,3% khách không có bản ghi hồ sơ** (không có kênh đăng ký). | Thấp | Đang để thiếu; mô hình cây xử lý được. | Thêm cờ riêng "thiếu hồ sơ". |
| 7 | **Ngưỡng tuổi 10–80, chặn `days_to_expire` [−60, 400]** chưa giải thích trong báo cáo. | Thấp | Người chấm có thể hỏi. | Ghi lý do (phân phối, ý nghĩa nghiệp vụ). |

---

## Phần C. Phân cụm

### C1. Điểm mạnh

- Không đưa nhãn rời bỏ vào dữ liệu phân cụm; nhãn chỉ dùng để mô tả nhóm sau khi đã chia.
- Biến đổi log1p và chuẩn hoá phù hợp với dữ liệu đuôi dài.
- Chọn k bằng 4 chỉ số (elbow, silhouette, Davies–Bouldin, Calinski–Harabasz), có hình minh hoạ.
- Kiểm tra độ ổn định bằng bootstrap, **fit lại cả bước điền thiếu và chuẩn hoá** ở mỗi lần (ARI 0,88).
- So sánh với 3 thuật toán khác (Ward, GMM, HDBSCAN) trên cùng mẫu.
- Kiểm tra trên tháng khác: thứ tự rủi ro các nhóm giữ nguyên, fit lại trên tháng 2 ra gần cùng cấu trúc (ARI 0,98).
- Có xác nhận chéo: UMAP và Isolation Forest tìm lại được các nhóm tương tự.

### C2. Điểm yếu

| # | Điểm yếu | Mức | Đề xuất |
|---|---|---|---|
| 1 | **Khách không có nhật ký nghe được gán hằng số** (60 ngày không nghe, độ thất thường = 3, tỉ lệ = 0). Đã kiểm chứng: 99,1% khách trong nhóm 2 không có nhật ký, và 99,8% số khách không có nhật ký nằm trong nhóm 2. Như vậy nhóm 2 là **cụm do cách gán giá trị tạo ra** (thuật toán chỉ tìm lại cờ "không có nhật ký"), không phải cấu trúc mà thuật toán tự phát hiện. Đây vẫn là một nhóm kinh doanh có thật, nhưng nên tách bằng quy tắc thay vì để K-Means tìm. | **Cao** | Tách khách không có nhật ký thành một nhóm theo quy tắc, chỉ phân cụm khách có nhật ký; so sánh với cách hiện tại. |
| 2 | **Lý do loại k = 3 có dùng thông tin rời bỏ** ("hai cụm lớn có cùng tỉ lệ rời bỏ"). Như vậy nhãn đã gián tiếp ảnh hưởng tới việc chọn mô hình không giám sát. | **Cao** | Trình bày k = 3 (tốt nhất theo chỉ số nội tại) và k = 7 (chi tiết hơn) song song; lý do chọn k = 7 dựa trên tiêu chí nội tại và nhu cầu nghiệp vụ, không dựa trên tỉ lệ rời bỏ. |
| 3 | **Quy tắc "75% silhouette tốt nhất" là quy ước** của nhóm, không phải tiêu chí khách quan. Silhouette 0,27 cho thấy các cụm tách nhau ở mức vừa phải. | Trung bình | Bổ sung chọn k theo độ ổn định (ARI cho từng k) hoặc gap statistic. |
| 4 | **Nhiều biến nghe nhạc đo cùng một thứ** (số ngày, số giờ, số bài): chuẩn hoá không loại được tương quan, nên chiều "mức độ nghe" bị tính nhiều lần. | Trung bình | Thử bỏ biến trùng hoặc cân trọng số theo nhóm biến (thuê bao / nghe / thâm niên), so sánh kết quả. |
| 5 | **ARI 0,88 ± 0,15**: độ lệch khá lớn, có lần bootstrap kém ổn định. Chưa đo ổn định của **từng cụm**. | Trung bình | Đo Jaccard từng cụm qua bootstrap để biết cụm nào chắc, cụm nào kém. |
| 6 | Bỏ các biến phân loại (kênh đăng ký, cách thanh toán) khỏi phân cụm. | Thấp | Chấp nhận được vì mục tiêu là phân cụm theo hành vi; các biến này đã dùng để mô tả nhóm sau đó. Có thể thử k-prototypes. |
| 7 | So sánh thuật toán chưa công bằng: HDBSCAN và GMM chưa tinh chỉnh tham số. | Thấp | Ghi rõ là so sánh với thiết lập mặc định. |
| 8 | Bảng nhiệt đặc điểm nhóm dùng z-score trên thang gốc, không phải thang đã biến đổi mà K-Means dùng. | Thấp | Ghi chú trong chú thích hình. |

---

## Phần D. Kết luận đánh giá

**Khung phương pháp đã đạt mức thạc sĩ**: có quy trình KDD rõ ràng, chống rò rỉ dữ liệu, nhiều kỹ thuật bổ trợ nhau, kiểm chứng trên dữ liệu giữ riêng và trên tháng khác, có đánh giá giá trị kinh doanh và tái lập được.

**Kết luận chính (rời bỏ gắn với cách trả tiền) không bị ảnh hưởng** bởi các vấn đề ở trên, vì chúng chủ yếu dựa trên thông tin giao dịch cuối (tự gia hạn, độ dài gói), vốn rất ít bị lỗi.

**Nên sửa trước khi nộp:**

1. Lỗi gói 0 ngày (B2.1), sau đó chạy lại. Phần mẫu tuần tự về "đổi gói" có thể thay đổi.
2. Cách xử lý khách không có nhật ký nghe trong phân cụm (C2.1).
3. Bỏ thông tin rời bỏ khỏi lý do chọn k, trình bày k = 3 và k = 7 song song (C2.2).
4. Ngày hỏng và thâm niên âm (B2.2, B2.3).

**Nên bổ sung nếu còn thời gian:** chọn k theo độ ổn định, ổn định từng cụm, thử trọng số nhóm biến, kiểm tra độ nhạy của thứ tự giao dịch cùng ngày.
