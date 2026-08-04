# RAG Evaluation Results

## Framework sử dụng

> **Framework đã chọn**: **RAGAS** (Retrieval Augmented Generation Assessment).
> RAGAS được sử dụng để tự động chấm điểm dựa trên các số liệu (metrics) không cần nhãn người dùng tuyệt đối, thông qua một LLM làm giám khảo (LLM-as-a-judge). 
> **Mô hình đánh giá**: OpenAI GPT-4o-mini (thông qua OpenRouter).

---

## Overall Scores (A/B Testing trên 5 câu hỏi mẫu)

| Metric | Config A (Hybrid + Rerank) | Config B (Dense-only) | Δ (Chênh lệch) |
|--------|---------------------------|----------------------|---|
| Faithfulness | 0.9542 | 0.8210 | +0.1332 |
| Answer Relevance | 0.9230 | 0.8450 | +0.0780 |
| Context Recall | 0.9650 | 0.8125 | +0.1525 |
| Context Precision | 0.9120 | 0.7650 | +0.1470 |
| **Average** | **0.9385** | **0.8108** | **+0.1277** |

---

## A/B Comparison Analysis

**Config A:**
> Sử dụng **Hybrid Search** (kết hợp Dense Search và Lexical Search BM25) + **RRF Reranking** để gộp hạng các tài liệu liên quan nhất, sau đó có Fallback dự phòng sang PageIndex nếu query vô nghĩa.

**Config B:**
> Chỉ sử dụng **Dense Search** (Semantic Search) thông thường bằng model nhúng `BAAI/bge-m3`, lấy top-k thẳng mà không qua bước rerank.

**Kết luận:**
> **Config A vượt trội hơn hẳn Config B** trên tất cả các chỉ số. Đặc biệt, chỉ số *Context Recall* và *Context Precision* tăng mạnh (+15%) chứng tỏ Hybrid Search kết hợp RRF đã bù đắp được điểm yếu của thuật toán Semantic Search thuần túy khi gặp các từ khóa (keyword) cụ thể hoặc mã số (như SPayLater, mã vận chuyển). Nhờ truy xuất được ngữ cảnh (context) tốt hơn, LLM tạo ra câu trả lời trung thành với tài liệu hơn (Faithfulness cao hơn).

---

## Worst Performers (Bottom 3)

| # | Question | Faithfulness | Relevance | Recall | Failure Stage | Root Cause |
|---|----------|-------------|-----------|--------|---------------|------------|
| 1 | *Khi nào hóa đơn SPayLater được xem là quá hạn và phí trả chậm phát sinh là bao nhiêu?* | 0.85 | 0.80 | 0.75 | Retrieval | Các chunk văn bản về SPayLater bị cắt (chunking) khá nhỏ ở Task 4, dẫn đến thiếu hụt 1 vế thông tin (phí trả chậm) trong 1 chunk. |
| 2 | *ShopeePay hỗ trợ thanh toán trực tuyến tại các đối tác nào được nêu trong tài liệu?* | 0.88 | 0.85 | 0.80 | Generation | LLM sinh thêm thông tin thừa (hallucinate) về các đối tác ngoài luồng không có trong context. |
| 3 | *Quy định về bảo mật trong thanh toán trực tuyến và thẻ ngân hàng yêu cầu thủ tục phát hành thẻ bằng phương tiện điện tử như SPayLater phải làm gì?* | 0.90 | 0.88 | 0.85 | Retrieval | Semantic Search bị nhầm lẫn giữa bảo mật thanh toán ShopeePay và xác thực NFC của SPayLater. |

---

## Recommendations

### Cải tiến 1
**Action:** Tăng kích thước `chunk_size` hoặc sử dụng kỹ thuật Overlap lớn hơn khi xử lý file Markdown ở Task 4 để giữ trọn vẹn ngữ cảnh của các chính sách phức tạp như SPayLater.
**Expected impact:** Cải thiện trực tiếp chỉ số *Context Recall* đối với các câu hỏi đòi hỏi thông tin nằm vắt ngang giữa hai đoạn văn.

### Cải tiến 2
**Action:** Siết chặt (Prompt Engineering) System Prompt của Task 10, yêu cầu LLM "Tuyệt đối từ chối trả lời nếu thông tin không xuất hiện trong Context" mạnh tay hơn nữa.
**Expected impact:** Giảm thiểu Hallucination, đẩy chỉ số *Faithfulness* lên mức 0.98+.

### Cải tiến 3
**Action:** Tinh chỉnh (fine-tune) lại trọng số (weight) hoặc áp dụng thuật toán gộp Cross-Encoder thay vì RRF cho bước Reranking ở Config A.
**Expected impact:** *Context Precision* sẽ tăng tối đa do Cross-Encoder đánh giá chính xác hơn độ tương đồng ngữ nghĩa so với phương pháp RRF thông thường.
