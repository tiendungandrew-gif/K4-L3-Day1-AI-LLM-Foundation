# K4 — Ngày 1: Bài Tập & Phản Ánh

## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 4 tiếng
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature

Gọi `call_openai` với temperature 0.0, 0.5, 1.0 và 1.5 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)

> Ở temperature 0.0, câu trả lời thường ổn định và ít biến thể nhất; khi tăng lên 0.5 và 1.0, cách diễn đạt và chi tiết trở nên đa dạng hơn. Với 1.5, phản hồi có xu hướng sáng tạo hoặc bất ngờ hơn nhưng cũng dễ thêm chi tiết kém chắc chắn. Temperature thay đổi cách lấy mẫu, không bảo đảm mỗi mức luôn dài hơn hay đúng hơn.

### Câu 1.2 — Chọn temperature cho sản phẩm

**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**

> Tôi sẽ đặt khoảng 0.2--0.3 cho chatbot hỗ trợ khách hàng. Các câu trả lời về chính sách, đơn hàng và hướng dẫn cần nhất quán, chính xác và dễ kiểm tra; vẫn giữ một ít độ linh hoạt để lời văn không quá máy móc.

### Câu 1.3 — Đánh đổi chi phí

Kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 3 lần,
mỗi lần trung bình ~350 token đầu ra.

**Ước tính GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này? Nêu một
trường hợp GPT-4o xứng đáng với chi phí và một trường hợp nên dùng mini:**

> Workload tạo 10,000 x 3 x 350 = 10.5 triệu output token/ngày. Theo bảng giá trong bài, GPT-4o ($0.010/1K output token) đắt khoảng 16.7 lần GPT-4o-mini ($0.0006/1K), tương ứng khoảng $105 so với $6.30 mỗi ngày, chưa tính input. GPT-4o đáng dùng cho phân tích tài liệu phức tạp hoặc tư vấn có rủi ro cao cần chất lượng suy luận; mini phù hợp cho phân loại ticket, FAQ và tác vụ số lượng lớn, cấu trúc rõ ràng.

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona

Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích blockchain là gì?"** nhưng hai system prompt khác nhau:

- "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi."
- "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật."

**Hai phản hồi khác nhau như thế nào (độ dài, từ vựng, ví dụ)? System prompt
ảnh hưởng đến hành vi model ra sao?** (3–4 câu)

> Persona giáo viên tiểu học thường tạo câu trả lời ngắn hơn, dùng từ quen thuộc như "cuốn sổ chung" và ví dụ chia sẻ giữa các bạn. Persona chuyên gia tài chính thường dài hơn, dùng các thuật ngữ như sổ cái phân tán, cơ chế đồng thuận, hàm băm và bất biến dữ liệu. Vì cùng một câu hỏi nhưng mục tiêu người đọc khác nhau, system prompt định hướng mức độ chi tiết, giọng điệu, vốn từ và kiểu ví dụ của model. Nó là chỉ dẫn ưu tiên cho cách trả lời trong toàn bộ phiên, dù không thay thế được việc kiểm chứng tính đúng đắn.

### Câu 2.2 — tiktoken vs đếm từ

Chọn một đoạn văn tiếng Việt ~100 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Vì sao tiếng Việt thường tốn
nhiều token hơn tiếng Anh cùng độ dài?**

> Tôi dùng đoạn mẫu 113 từ về chuyển đổi số và an toàn dữ liệu. `count_tokens` với encoding của gpt-4o đếm 134 token, trong khi ước lượng số từ / 0.75 là khoảng 151 token; chênh 17 token, tức khoảng 11.3% so với ước lượng. Tiếng Việt thường tốn nhiều token hơn tiếng Anh cùng độ dài vì dấu thanh, ký tự có dấu và các mảnh từ phổ biến của tiếng Việt có thể bị tokenizer tách nhỏ hơn; vì vậy đếm từ chỉ nên dùng để ước lượng thô.

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming

**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì
non-streaming lại phù hợp hơn?** (1 đoạn văn)

> Streaming quan trọng nhất khi câu trả lời có thể dài hoặc thời gian tạo không chắc chắn, như chatbot tư vấn, trợ lý viết và giao diện hỏi đáp: người dùng thấy phản hồi bắt đầu ngay nên cảm nhận độ trễ thấp hơn và có thể dừng khi đã đủ thông tin. Non-streaming phù hợp khi ứng dụng cần nhận kết quả hoàn chỉnh trước khi hiển thị hoặc xử lý, chẳng hạn parse JSON có cấu trúc, kiểm tra an toàn, lưu vào cơ sở dữ liệu hay một câu trả lời rất ngắn.

### Câu 3.2 — Vì sao backoff theo cấp số nhân?

**So với delay cố định (ví dụ luôn chờ 1 giây), exponential backoff có lợi
thế gì khi API bị quá tải? Điều gì xảy ra nếu hàng nghìn client cùng retry
với delay cố định giống nhau?**

> Exponential backoff giảm tần suất gửi lại theo từng lần lỗi, cho máy chủ có thời gian hồi phục và tránh làm tình trạng quá tải nặng hơn. Delay cố định khiến nhiều client thất bại cùng lúc lại retry cùng lúc, tạo các đợt request đồng bộ lặp lại (retry storm). Thực tế nên thêm jitter ngẫu nhiên vào thời gian chờ để giảm khả năng các client lại trùng nhịp.

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona

**Bạn chọn persona gì cho trợ lý của mình? Viết lại system prompt đó và giải
thích 1–2 lựa chọn từ ngữ quan trọng trong prompt (ví dụ: vì sao yêu cầu
"trả lời ngắn gọn", vì sao chỉ định ngôn ngữ...):**

> Tôi chọn persona: "Bạn là trợ giảng AI thân thiện. Trả lời bằng tiếng Việt rõ ràng, ngắn gọn, nêu ví dụ khi cần và nói rõ khi không chắc chắn." Cụm "bằng tiếng Việt" phù hợp với người học mục tiêu và tránh lẫn ngôn ngữ; "ngắn gọn" giúp câu trả lời dễ đọc trong terminal và kiểm soát chi phí. Yêu cầu thừa nhận điều không chắc chắn giúp giảm cảm giác khẳng định quá mức khi thiếu ngữ cảnh.

### Câu 4.2 — Hạn chế & cải thiện

**Trợ lý của bạn hiện có hạn chế lớn nhất là gì (ví dụ: history chỉ 3 lượt,
không có bộ nhớ dài hạn, không kiểm duyệt nội dung...)? Đề xuất một cải
thiện cụ thể và mô tả ngắn cách triển khai:**

> Hạn chế lớn nhất là history chỉ giữ ba lượt gần nhất, nên trợ lý sẽ quên mục tiêu hoặc thông tin quan trọng ở các lượt cũ. Tôi sẽ thêm bộ nhớ tóm tắt: khi history sắp bị cắt, gọi một bước tóm tắt các quyết định, sở thích và dữ kiện cần giữ, rồi đưa bản tóm tắt đó vào system/context của các lượt sau. Cần giới hạn độ dài và chỉ lưu dữ kiện được người dùng cho phép để vừa kiểm soát token vừa bảo vệ riêng tư.

---

## Danh Sách Kiểm Tra Nộp Bài

- [ ] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [ ] Cả 4 checkpoint pytest đều pass
- [ ] Tất cả 9 câu trong file này đã được trả lời
- [ ] Đã copy bài làm vào folder `solution/`, push lên fork và dán link trên trang bài Lab ở VLearn trước 23:59 ngày 11/09/2026
