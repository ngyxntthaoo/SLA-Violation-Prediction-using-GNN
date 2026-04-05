# ĐỀ XUẤT NGHIÊN CỨU

**Tiêu đề:** Dự đoán Vi phạm SLA trong Quy trình Nghiệp vụ Sử dụng Mạng Nơ-ron Đồ thị (Graph Neural Networks)

---

## Tóm tắt

Chúng tôi đề xuất một phương pháp dự đoán vi phạm Thỏa thuận Mức dịch vụ (SLA) trong các quy trình nghiệp vụ bằng cách mô hình hóa từng hồ sơ (case) thành một đồ thị có hướng và phân loại chúng bằng Mạng Nơ-ron Đồ thị Tích chập (Graph Convolutional Network – GCN). Khác với các phương pháp truyền thống xử lý nhật ký sự kiện như véc-tơ phẳng hay chuỗi tuần tự, phương pháp này bảo toàn đặc trưng cấu trúc và thời gian của từng hồ sơ: mỗi sự kiện hoạt động được mã hóa thành một nút đồ thị, và quan hệ "trực tiếp theo sau" (directly follows) được biểu diễn qua các cạnh có hướng. Mô hình được huấn luyện trên các đồ thị tiền tố (prefix graphs) ở nhiều mức độ hoàn thành khác nhau, từ đó đóng vai trò như một hệ thống cảnh báo sớm có khả năng đánh dấu các hồ sơ có nguy cơ vi phạm trước khi chúng kết thúc. Được đánh giá trên tập dữ liệu BPI Challenge 2012 (13.087 hồ sơ đơn vay), mô hình đạt F1 = 0,87 và ROC-AUC = 0,997 trên toàn bộ hồ sơ hoàn chỉnh, đồng thời xuất hiện tín hiệu dự đoán có ý nghĩa từ khi hồ sơ mới hoàn thành 40% (F1 = 0,42, ROC-AUC = 0,88).

---

## 1. Giới thiệu Vấn đề

Các tổ chức trong lĩnh vực ngân hàng, y tế, logistics và dịch vụ công nghệ thông tin thường vận hành dưới các cam kết SLA — xác định thời gian tối đa được phép để hoàn thành một hồ sơ. Vi phạm SLA dẫn đến các hậu quả nghiêm trọng: phạt hợp đồng, mất khách hàng, ảnh hưởng đến uy tín thương hiệu và giảm hiệu quả vận hành.

Thực trạng hiện tại của nhiều tổ chức là chỉ phát hiện vi phạm **sau khi chúng đã xảy ra** — khi không còn khả năng can thiệp. Điều này tạo ra một khoảng trống lớn giữa nhu cầu quản lý rủi ro thực tế và năng lực công nghệ hiện tại.

Đề xuất này giải quyết bài toán **giám sát SLA dự đoán (predictive SLA monitoring)**: dựa vào nhật ký thực thi một phần của một hồ sơ đang diễn ra, ước tính xác suất hồ sơ đó sẽ vượt quá hạn SLA. Đây là bài toán phân loại đồ thị nhị phân, trong đó mỗi hồ sơ quy trình được biểu diễn như một đồ thị có hướng và GCN được huấn luyện để phân biệt hồ sơ đúng hạn với hồ sơ vi phạm.

---

## 2. Phương pháp Đề xuất

### 2.1 Phát biểu Bài toán

Cho một nhật ký sự kiện (event log) trong đó mỗi hồ sơ gồm chuỗi các sự kiện hoạt động có dấu thời gian, và một ngưỡng SLA định sẵn (thời gian xử lý tối đa cho phép), ta gán nhãn nhị phân cho mỗi hồ sơ:

- **0 (đúng hạn):** tổng thời gian thực hiện ≤ ngưỡng SLA
- **1 (vi phạm):** tổng thời gian thực hiện > ngưỡng SLA

Nhiệm vụ là dự đoán nhãn này từ dữ liệu thực thi một phần hoặc toàn bộ của hồ sơ.

### 2.2 Biểu diễn Đồ thị

Mỗi hồ sơ được chuyển đổi thành đồ thị có hướng *G = (V, E, X)* như sau:

- **Nút** *V*: Mỗi nút đại diện cho một sự kiện hoạt động trong luồng thực thi của hồ sơ.
- **Cạnh** *E*: Cạnh có hướng *(i, i+1)* kết nối mỗi sự kiện với sự kiện kế tiếp trực tiếp, tạo ra một chuỗi bảo toàn thứ tự thời gian.
- **Đặc trưng nút** *X*: Mỗi nút mang một véc-tơ đặc trưng gồm |A| + 2 chiều, trong đó |A| là số loại hoạt động phân biệt. Cụ thể:
  - |A| chiều đầu: mã hóa one-hot của loại hoạt động
  - `time_since_start`: thời gian trôi qua (ngày) kể từ sự kiện đầu tiên trong hồ sơ
  - `time_since_prev`: thời gian trôi qua (ngày) kể từ sự kiện ngay trước đó

Hai đặc trưng thời gian được chuẩn hóa z-score dựa trên thống kê tính **chỉ từ tập huấn luyện**, tránh rò rỉ thông tin (data leakage).

### 2.3 Tăng cường Dữ liệu Dạng Tiền tố

Để huấn luyện mô hình có khả năng dự đoán sớm từ dữ liệu không đầy đủ, chúng tôi sinh ra nhiều đồ thị tiền tố (prefix graphs) cho mỗi hồ sơ tại các mức độ hoàn thành cố định: **20%, 40%, 60%, 80%, 100%**. Mỗi đồ thị tiền tố giữ nguyên nhãn ground-truth của hồ sơ gốc.

Một hồ sơ có N sự kiện sinh ra tối đa 5 đồ thị huấn luyện có độ dài tăng dần. Kỹ thuật này buộc mô hình học các mẫu dự đoán từ thông tin một phần, thay vì chỉ dựa vào đặc trưng có sẵn khi hồ sơ đã kết thúc.

### 2.4 Kiến trúc Mô hình

Chúng tôi sử dụng mạng GCN hai lớp với global mean pooling:

```
Input (|A|+2 chiều)
  → GCNConv(|A|+2, 128) → ReLU → Dropout(0.3)
  → GCNConv(128, 64) → ReLU
  → Global Mean Pool (embedding cấp đồ thị, 64 chiều)
  → Linear(64, 1) → Sigmoid
```

Các lớp GCN thực hiện tổng hợp vùng lân cận (neighbourhood aggregation): biểu diễn của mỗi nút được cập nhật bằng cách kết hợp đặc trưng của bản thân với các nút lân cận. Sau hai vòng truyền thông điệp (message passing), mỗi nút nắm bắt được thông tin về ngữ cảnh cục bộ của nó. Global mean pooling nén toàn bộ embedding nút thành một véc-tơ 64 chiều đại diện cho toàn bộ hồ sơ. Lớp tuyến tính cuối cùng ánh xạ véc-tơ này thành xác suất vi phạm.

### 2.5 Huấn luyện


| Thông số     | Giá trị                                                                            |
| ------------ | ---------------------------------------------------------------------------------- |
| Hàm mất mát  | Binary cross-entropy, trọng số lớp dương √(neg/pos) ≈ 2,9 (xử lý mất cân bằng lớp) |
| Bộ tối ưu    | Adam (lr = 5×10⁻⁴, weight decay = 1×10⁻⁴)                                          |
| Lịch học     | ReduceLROnPlateau (patience=5, factor=0,5) theo validation F1                      |
| Dừng sớm     | Patience 20 epoch theo validation F1                                               |
| Tính tái lập | Tất cả random seed cố định (Python, NumPy, PyTorch)                                |


### 2.6 Giao thức Đánh giá

Dữ liệu được chia ở **cấp độ hồ sơ** (70% huấn luyện, 10% validation, 20% kiểm thử) với stratified sampling. Các đồ thị tiền tố được sinh trong từng phân vùng độc lập — không có hồ sơ nào xuất hiện trong nhiều hơn một phân vùng, ngăn ngừa rò rỉ thông tin.

Kết quả chính được báo cáo trên **hồ sơ hoàn chỉnh (100%)** từ tập kiểm thử. Bảng đánh giá theo tiền tố báo cáo F1, Precision, Recall, ROC-AUC ở từng mức độ hoàn thành (20%–100%) để đặc tả hiệu năng cảnh báo sớm.

---

## 3. Dữ liệu

Chúng tôi sử dụng bộ dữ liệu chuẩn **BPI Challenge 2012** — nhật ký sự kiện thực tế từ quy trình xét duyệt hồ sơ vay vốn tại một tổ chức tài chính Hà Lan.


| Thuộc tính          | Giá trị                     |
| ------------------- | --------------------------- |
| Số sự kiện          | 262.200                     |
| Số hồ sơ            | 13.087                      |
| Số loại hoạt động   | 24                          |
| Giai đoạn thời gian | Tháng 9/2011 – Tháng 3/2012 |
| Ngưỡng SLA          | 30 ngày                     |
| Hồ sơ đúng hạn      | 11.693 (89,3%)              |
| Hồ sơ vi phạm       | 1.394 (10,7%)               |


Tập dữ liệu có tính mất cân bằng lớp đáng kể và độ phức tạp hồ sơ cao (từ 3 đến 175 sự kiện mỗi hồ sơ, trung vị là 11).

---

## 4. Kết quả

### 4.1 Đánh giá Hồ sơ Hoàn chỉnh (100%)


| Chỉ số    | Giá trị |
| --------- | ------- |
| Accuracy  | 0,969   |
| Precision | 0,780   |
| Recall    | 0,989   |
| F1-Score  | 0,872   |
| ROC-AUC   | 0,997   |
| PR-AUC    | 0,967   |


Mô hình phát hiện chính xác 99% hồ sơ vi phạm (recall) với độ chính xác 78% (precision). ROC-AUC cao cho thấy khả năng phân biệt mạnh mẽ trên mọi ngưỡng quyết định.

### 4.2 Đánh giá Theo Mức Hoàn thành (Cảnh báo Sớm)


| Mức hoàn thành | F1    | Precision | Recall | ROC-AUC |
| -------------- | ----- | --------- | ------ | ------- |
| 20%            | 0,085 | 0,464     | 0,047  | 0,826   |
| 40%            | 0,419 | 0,564     | 0,333  | 0,877   |
| 60%            | 0,647 | 0,625     | 0,670  | 0,938   |
| 80%            | 0,687 | 0,557     | 0,896  | 0,970   |
| 100%           | 0,872 | 0,780     | 0,989  | 0,997   |


Hiệu suất cải thiện đơn điệu theo mức độ hoàn thành hồ sơ. Tại mức 20%, mô hình có khả năng phân loại hạn chế (F1 = 0,085) nhưng đã đạt xếp hạng tương đối tốt (ROC-AUC = 0,826). Đến mức 60%, mô hình đạt F1 = 0,647 với precision và recall cân bằng — đây là ngưỡng cảnh báo sớm thực tế. Tại mức 80%, recall đạt 0,896 — mô hình phát hiện 90% hồ sơ vi phạm trong khi vẫn còn 20% hồ sơ chưa hoàn thành.

### 4.3 Nghiên cứu Điển hình (Early Warning)


| Hồ sơ  | Nhãn     | Số sự kiện | Rủi ro @25% | Rủi ro @50% | Rủi ro @75% | Rủi ro @100% |
| ------ | -------- | ---------- | ----------- | ----------- | ----------- | ------------ |
| 173928 | Vi phạm  | 115        | 0,405       | 0,995       | 1,000       | 1,000        |
| 174000 | Vi phạm  | 40         | 0,191       | 0,363       | 0,784       | 0,948        |
| 173691 | Đúng hạn | 39         | 0,256       | 0,269       | 0,029       | 0,001        |


Hồ sơ 173928 bị đánh dấu với xác suất gần tuyệt đối chỉ khi mới hoàn thành 50%. Hồ sơ 174000 thể hiện mức độ rủi ro leo thang dần, vượt ngưỡng 0,5 tại khoảng 75% hoàn thành. Hồ sơ đúng hạn duy trì xác suất thấp xuyên suốt.

### 4.4 Phát hiện Nút Thắt Cổ chai (Bottleneck)

Thông qua phân tích độ lớn embedding nút trong các hồ sơ vi phạm, các hoạt động liên quan nhiều nhất đến vi phạm SLA được xác định là: **O_DECLINED**, **A_CANCELLED**, **A_DECLINED** — các sự kiện quyết định và hủy bỏ, phù hợp với kiến thức chuyên môn rằng các vòng lặp xử lý lại và từ chối là nguyên nhân chính gây chậm trễ quy trình.

---

### 4.5 Minh giải Kết quả Vi phạm theo Hồ sơ

#### Cơ sở xác định nhãn vi phạm

Nhãn ground-truth được gán dựa trên quy tắc tất định:

\[
\text{label} = \begin{cases} 1\ (\text{vi phạm}) & \text{nếu } \Delta t = t_{\text{cuối}} - t_{\text{đầu}} > \theta_{\text{SLA}} \\ 0\ (\text{đúng hạn}) & \text{nếu } \Delta t \leq \theta_{\text{SLA}} \end{cases}
\]

trong đó \(\theta_{\text{SLA}} = 30\) ngày. Đây là sự thật khách quan từ nhật ký sự kiện, không phụ thuộc vào bất kỳ mô hình nào.

#### Ba tầng bằng chứng giải thích vi phạm

Hệ thống cung cấp ba lớp bằng chứng bổ sung nhau để trả lời câu hỏi *"tại sao hồ sơ này vi phạm?"*:

**Tầng 1 — Chỉ số rủi ro theo hoạt động (Process Mining)**

Mỗi hoạt động trong quy trình được đánh giá qua chỉ số SLA risk rate:

\[
\text{SLA Risk Rate}(a) = \frac{\text{số hồ sơ vi phạm có hoạt động } a}{\text{tổng số hồ sơ có hoạt động } a}
\]

Hoạt động có SLA risk rate cao là nút thắt cổ chai tiềm năng. Ví dụ, nếu `Request Additional Info` có SLA risk rate = 85%, thì 85% hồ sơ đi qua bước này đều bị vi phạm — đây là tín hiệu cần ưu tiên xử lý vận hành.

**Tầng 2 — Đặc trưng thời gian tại từng sự kiện**

Mỗi nút trong đồ thị mang hai đặc trưng thời gian cho phép định vị bước chậm:

| Đặc trưng | Ý nghĩa |
|---|---|
| `time_since_start` | Tổng thời gian đã trôi qua kể từ sự kiện đầu tiên (ngày) |
| `time_since_prev` | Khoảng chờ so với bước ngay trước đó (ngày) |

Giá trị `time_since_prev` bất thường lớn tại một bước cụ thể là dấu hiệu trực tiếp của việc hồ sơ bị tắc nghẽn tại đó.

**Tầng 3 — Phân tích embedding GCN theo từng hồ sơ**

Khi một hồ sơ đi qua GCN, mỗi sự kiện (nút) được biến đổi thành một **vector số** gọi là embedding:

```
Nút "Request Additional Info"  →  [0.82, -1.3, 0.45, 2.1, ..., 0.67]  (64 số)
```

**GCN Norm** là độ dài (magnitude) của vector đó — đo mức độ bất thường mà mô hình nhận thấy tại từng sự kiện:

\[
\text{norm}(v) = \| h_v^{(2)} \|_2 = \sqrt{\sum_k \left(h_{v,k}^{(2)}\right)^2}
\]

trong đó \(h_v^{(2)}\) là embedding của nút \(v\) sau lớp GCN thứ hai. Trực giác:

- **Norm thấp** → nút hoạt động bình thường, GCN không cần đẩy vector xa gốc để phân biệt
- **Norm cao** → nút mang tín hiệu bất thường, GCN kéo vector ra xa để tạo tín hiệu phân loại

Phân tích này được thực hiện **ở cấp độ từng hồ sơ** (per-case), không chỉ tổng hợp toàn cục:

\[
\text{Bottleneck}_{\text{GCN}}(case) = \arg\max_{v \in V} \| h_v^{(2)} \|_2
\]

Nút có norm cao nhất là bước mà mô hình "chú ý" nhiều nhất — có thể là bước chờ lâu, hoặc bước mang tín hiệu cấu trúc bất thường (ví dụ: vòng lặp xử lý lại, quyết định từ chối).

**Ví dụ minh họa GCN Norm trên một hồ sơ vi phạm:**

| Bước | Hoạt động | Chờ (ngày) | GCN Norm | Nhận xét |
|:---:|---|:---:|:---:|---|
| 1 | Submit Application | 0 | 0,31 | Bình thường |
| 2 | Verify Documents | 2 | 0,28 | Bình thường |
| 3 | Credit Check | 5 | 0,35 | Bình thường |
| 4 | Assess Risk | 3 | 0,29 | Bình thường |
| 5 | **Request Additional Info** | **18** | **1,87** | ⚠️ Nút thắt — cả hai chỉ số cao nhất |
| 6 | Provide Additional Info | 12 | 0,94 | Hệ quả của bước 5 |
| 7 | Assess Risk *(lặp lại)* | 4 | 0,71 | Vòng lặp đáng ngờ |
| 8 | Approve | 3 | 0,33 | Bình thường |

Ở hồ sơ này, cả hai phương pháp đồng thuận: bước 5 vừa có thời gian chờ cao nhất (18 ngày) vừa có GCN norm cao nhất (1,87) → kết luận chắc chắn.

#### Ví dụ minh họa — Hồ sơ vi phạm điển hình

Xét hồ sơ vay vốn có tổng thời gian xử lý **47 ngày** (vượt ngưỡng SLA 30 ngày). Dưới đây là chuỗi sự kiện và thời gian chờ tại từng bước:

| Bước | Hoạt động | Thời gian chờ (ngày) | Ghi chú |
|:---:|---|:---:|---|
| 1 | Submit Application | 0 | Khởi đầu hồ sơ |
| 2 | Verify Documents | 2 | Bình thường |
| 3 | Credit Check | 5 | Bình thường |
| 4 | Assess Risk | 3 | Bình thường |
| 5 | **Request Additional Info** | **18** | ⚠️ Nút thắt chính |
| 6 | Provide Additional Info | 12 | Hệ quả của bước 5 |
| 7 | Assess Risk *(lặp lại)* | 4 | Vòng lặp xử lý lại |
| 8 | Approve | 3 | Kết thúc |
| | **Tổng** | **47 ngày** | **Vi phạm SLA** |

**Phân tích:** Khoảng chờ 18 ngày tại bước *Request Additional Info* là nguyên nhân trực tiếp. Đây là bước yêu cầu khách hàng bổ sung tài liệu — thời gian phụ thuộc hoàn toàn vào phản hồi của khách hàng và không được kiểm soát nội bộ. Nếu không có cơ chế nhắc nhở tự động, bước này trở thành điểm mù trong giám sát SLA.

**Tín hiệu cảnh báo sớm của mô hình:**

| Mức hoàn thành | Sự kiện đã thấy | Xác suất vi phạm |
|:---:|---|:---:|
| 25% | Bước 1–2 | 0,21 |
| 50% | Bước 1–4 | 0,58 → **vượt ngưỡng 0,5** |
| 75% | Bước 1–6 | 0,94 |
| 100% | Toàn bộ | 0,99 |

Mô hình đã vượt ngưỡng quyết định (0,5) tại mức **50% hoàn thành** — tức là khi hồ sơ vừa hoàn thành bước kiểm tra rủi ro lần đầu, hệ thống đã có thể cảnh báo để đội vận hành chủ động liên hệ khách hàng, tiết kiệm ít nhất 12–18 ngày xử lý.

#### So sánh hai phương pháp phát hiện nút thắt per-case

Hai phương pháp bổ sung nhau — mỗi phương pháp có điểm mạnh riêng:

| | Thời gian chờ (`time_since_prev`) | GCN Norm |
|---|---|---|
| **Nguồn** | Dữ liệu thực, tính trực tiếp | Mô hình học được |
| **Đo gì** | Chờ bao lâu (ngày) | Bất thường đến mức nào |
| **Hạn chế** | Không phân biệt được *tại sao* chờ lâu | Khó diễn giải nếu thiếu domain knowledge |
| **Mạnh nhất khi** | Bottleneck rõ ràng do thời gian | Bottleneck do cấu trúc quy trình (vòng lặp, từ chối) |

Khi hai phương pháp đồng thuận, kết luận rất chắc chắn. Khi bất đồng, thông tin bổ sung quan trọng:

| Tình huống | Ý nghĩa |
|---|---|
| Cả hai cùng chỉ một bước | Bước đó vừa chờ lâu, vừa bất thường về cấu trúc → nút thắt rõ ràng |
| Chờ lâu nhưng GCN norm thấp | Chờ lâu là do yếu tố ngoại cảnh (khách hàng), không bất thường trong mô hình |
| GCN norm cao nhưng chờ không lâu | Bước này tuy nhanh nhưng xảy ra trong ngữ cảnh đáng ngờ (vòng lặp, lặp lại) |

#### Tóm tắt cơ sở xác định vi phạm

| Câu hỏi | Câu trả lời |
|---|---|
| **Dựa vào con số nào?** | `duration_days > θ_SLA` — sự thật từ nhật ký sự kiện |
| **Tại sao vi phạm?** | Một hoặc vài bước có `time_since_prev` bất thường lớn |
| **Bước nào nguy hiểm nhất?** | Phát hiện per-case: so sánh thời gian chờ thực tế vs. GCN norm từng nút |
| **Mô hình học được điều gì?** | Pattern cấu trúc đồ thị + thời gian → embedding → xác suất |
| **Cảnh báo từ khi nào?** | Từ 40–60% hoàn thành (F1 = 0,42–0,65, ROC-AUC = 0,88–0,94) |

---

## 5. Thảo luận

### Điểm mạnh

- Biểu diễn đồ thị mã hóa tự nhiên cả cấu trúc (hoạt động nào đã xảy ra) lẫn thời gian (tiến độ thực hiện).
- Tăng cường tiền tố cho phép dự đoán sớm thực sự mà không có rò rỉ dữ liệu.
- Mô hình đạt hiệu năng cao trên benchmark nổi tiếng với kiến trúc đơn giản hai lớp.
- Toàn bộ pipeline chạy trong khoảng 10 phút trên CPU (notebook Kaggle).

### Hạn chế và Hướng Phát triển

- Cấu trúc đồ thị hiện tại là chuỗi tuần tự đơn giản. Quy trình thực tế có thể chứa các nhánh song song, cổng quyết định và vòng lặp xử lý lại không được thể hiện qua quan hệ "directly follows" đơn thuần. Tích hợp cạnh từ mô hình quy trình (BPMN) có thể làm giàu thêm cấu trúc đồ thị.
- Hiệu suất tại mức 20% còn yếu (F1 = 0,085), cho thấy dự đoán cực sớm vẫn là thách thức nếu thiếu đặc trưng bổ sung như tải lượng nhân sự hoặc thông tin lịch.
- **Hướng mở rộng:** áp dụng GNN có cơ chế attention (GAT) để tăng tính giải thích được; bổ sung đặc trưng tài nguyên và lịch; đánh giá trên nhiều tập dữ liệu benchmark khác nhau.

---

## 6. Kết luận

Chúng tôi chứng minh rằng việc mô hình hóa hồ sơ quy trình nghiệp vụ dưới dạng đồ thị có hướng và áp dụng mạng nơ-ron đồ thị tạo ra một hệ thống dự đoán vi phạm SLA hiệu quả. Phương pháp huấn luyện dựa trên tiền tố loại bỏ rò rỉ đặc trưng thời gian và cho phép đánh giá cảnh báo sớm trung thực. Trên tập dữ liệu BPI Challenge 2012, mô hình đạt F1 = 0,87 trên hồ sơ hoàn chỉnh và cung cấp điểm rủi ro có giá trị thực tiễn từ mức 40% hoàn thành trở lên (F1 = 0,42, ROC-AUC = 0,88), với khả năng cảnh báo sớm thực tế xuất hiện từ mức 60% (F1 = 0,65).

Công trình này mở ra tiềm năng ứng dụng rộng rãi trong các hệ thống giám sát quy trình thông minh, đặc biệt trong ngành ngân hàng và tài chính nơi SLA đóng vai trò quan trọng trong cam kết dịch vụ và tuân thủ quy định.

---

*Tài liệu này được soạn dựa trên mã nguồn notebook `thao-nguyen.ipynb` và báo cáo phương pháp `approach-report.md`.*