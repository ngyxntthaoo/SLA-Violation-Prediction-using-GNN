# Dự đoán Vi phạm SLA bằng Mạng Nơ-ron Đồ thị (GNN)

Dự đoán vi phạm Thỏa thuận Mức Dịch vụ (SLA) trong quy trình nghiệp vụ sử dụng Graph Convolutional Network (GCN) trên bộ dữ liệu BPI Challenge 2012.

**Pipeline:** `Event Log → Gán nhãn SLA → Xây dựng đồ thị → GCN → Dự đoán → Phân tích nghiệp vụ`

## Tổng quan

Dự án áp dụng GNN cho bài toán giám sát quy trình dự đoán (Predictive Process Monitoring), lấy ý tưởng từ [Lischka et al. (2025)](https://arxiv.org/abs/2503.03197). Thay vì biểu diễn quy trình dưới dạng chuỗi tuần tự (LSTM/Transformer), mỗi hồ sơ vay được biểu diễn dưới dạng **đồ thị có hướng** và phân loại bằng GCN 2 lớp.

### Đóng góp chính

- **Per-case instance graph** — mỗi sự kiện là một node với vector đặc trưng 26 chiều (24 one-hot hoạt động + 2 thời gian)
- **Prefix Augmentation** — sinh 5 phiên bản (20/40/60/80/100%) mỗi hồ sơ để học cảnh báo sớm
- **Phát hiện bottleneck** — phân tích GCN Norm xác định bước bất thường trong hồ sơ vi phạm
- **Ứng dụng web Streamlit** — demo tương tác với 3 chế độ người dùng, chạy trên CPU

### Kết quả (BPI Challenge 2012 — tập kiểm thử)

| Chỉ số | Hoàn thành 100% | Hoàn thành 60% (cảnh báo sớm) |
|--------|:-:|:-:|
| Accuracy | 96.9% | — |
| Recall | 98.9% | 67.0% |
| Precision | 78.2% | 62.8% |
| F1-Score | 87.3% | 64.8% |
| ROC-AUC | 0.997 | 0.938 |

## Cấu trúc dự án

```
├── SLA Violation Prediction using Graph Neural Networks.ipynb  # Notebook chính (pipeline đầy đủ)
├── app.py                      # Ứng dụng demo Streamlit
├── best_model.pt               # Trọng số mô hình đã huấn luyện
├── BPI_Challenge_2012.xes      # Event log BPI Challenge 2012 (~74MB)
├── proposal.md                 # Đề xuất dự án
├── output/
│   ├── metrics_summary.csv     # Chỉ số đánh giá
│   ├── result.csv              # Dự đoán từng case (y_true, y_pred, y_prob)
│   ├── node_embedding.csv      # Đặc trưng node tất cả case
│   ├── edge.csv                # Danh sách cạnh tất cả case
│   ├── graph_embeddings.csv    # Embedding cấp đồ thị (tập test)
│   └── sla_gcn_model.pt        # Trọng số mô hình
├── report/
│   └── process_sla_graph.png   # Trực quan hóa mô hình quy trình
└── lib/                        # Tài nguyên frontend cho Streamlit
```

## Thư viện cần thiết

```
torch
torch-geometric
numpy
pandas
scikit-learn
matplotlib
networkx
```

Cho ứng dụng demo Streamlit, cần thêm:

```
streamlit
plotly
```

## Hướng dẫn chạy

### 1. Chạy notebook

Mở `SLA Violation Prediction using Graph Neural Networks.ipynb` trong Jupyter/Kaggle/Colab. Notebook tự động phát hiện file `BPI_Challenge_2012.xes` trong thư mục hiện tại; nếu không tìm thấy sẽ tự sinh dữ liệu tổng hợp.

### 2. Chạy demo Streamlit

```bash
python3 -m streamlit run app.py
```

Ứng dụng tải mô hình đã huấn luyện (`best_model.pt`) và cung cấp:
- Tra cứu rủi ro từng hồ sơ
- Biểu đồ diễn biến rủi ro theo thời gian (cảnh báo sớm)
- Phân tích GCN Norm — phát hiện bước tắc nghẽn
- 3 chế độ: Nhân viên / Quản lý / Lãnh đạo

## Kiến trúc mô hình

```
Input (đặc trưng node 26 chiều)
  → GCNConv(26 → 128) → ReLU → Dropout(0.3)
  → GCNConv(128 → 64) → ReLU
  → Global Mean Pooling → vector đồ thị (64 chiều)
  → Linear(64 → 1) → Sigmoid → P(vi phạm) ∈ [0, 1]
```

**Huấn luyện:** Binary cross-entropy với class weight √(neg/pos), optimizer Adam (lr=5e-4), early stopping theo val F1 (patience=20), chia dữ liệu stratified theo case 70/10/20.

## Tài liệu tham khảo

- Lischka, A., Rauch, S., & Stritzel, O. (2025). [Directly Follows Graphs Go Predictive Process Monitoring With Graph Neural Networks](https://arxiv.org/abs/2503.03197). arXiv.
- van Dongen, B.F. (2012). [BPI Challenge 2012](https://doi.org/10.4121/uuid:3926db30-f712-4394-aebc-75976070e91f). 4TU.ResearchData.
- Kipf, T.N. & Welling, M. (2017). [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907). ICLR.
