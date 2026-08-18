/**
 * Các hàm toán học của phần hậu xử lý (post-processing).
 *
 * Cố ý viết tay thay vì gọi thư viện: đây chính là phần bài giảng muốn sinh viên
 * nhìn thấy. Thư viện `pipeline()` làm sẵn những bước này và che nó đi.
 */

/**
 * Softmax: biến vector logits (số thực bất kỳ) thành phân phối xác suất
 * (mọi phần tử trong [0, 1], tổng bằng 1).
 *
 *     p_i = exp(z_i) / Σ_j exp(z_j)
 *
 * Vì sao phải trừ max trước khi exp?
 *   exp(1000) = Infinity trong số học dấu phẩy động -> ra NaN.
 *   Trừ đi hằng số C bất kỳ không làm đổi kết quả:
 *     exp(z_i - C) / Σ exp(z_j - C) = exp(z_i)/exp(C) / (Σ exp(z_j)/exp(C))
 *   nên chọn C = max(z) để mọi số hạng exp đều ≤ 1. Đây là mẹo "numerical
 *   stability" mà mọi thư viện ML đều dùng.
 */
export function softmax(logits: number[]): number[] {
  if (logits.length === 0) return []
  const max = Math.max(...logits)
  const exps = logits.map((z) => Math.exp(z - max))
  const sum = exps.reduce((acc, v) => acc + v, 0)
  return exps.map((e) => e / sum)
}

/**
 * Sigmoid: bóp một số thực bất kỳ về khoảng (0, 1).
 *
 *     σ(z) = 1 / (1 + e^{-z})
 *
 * Dùng khi model chỉ có MỘT đầu ra (num_labels = 1), ví dụ cross-encoder
 * reranker: không có nhãn nào để softmax với nhau, nên softmax vô nghĩa.
 *
 * ⚠️ σ(z) trông giống xác suất nhưng chỉ thật sự là xác suất nếu model được
 * huấn luyện bằng binary cross-entropy. Reranker MS MARCO được huấn luyện để xếp
 * hạng, nên hãy đọc nó như một điểm số có thứ tự, đừng đọc như "khả năng đúng".
 */
export function sigmoid(z: number): number {
  return 1 / (1 + Math.exp(-z))
}

/** Chỉ số của phần tử lớn nhất — tương ứng nhãn được dự đoán. */
export function argmax(values: number[]): number {
  let best = 0
  for (let i = 1; i < values.length; i++) {
    if (values[i] > values[best]) best = i
  }
  return best
}

/**
 * Cosine similarity giữa hai vector — dùng cho demo semantic search.
 *
 *     cos(a, b) = (a · b) / (‖a‖ · ‖b‖)
 *
 * Đo GÓC chứ không đo khoảng cách, nên độ dài vector không ảnh hưởng. Đó là lý
 * do nó phù hợp để so sánh embedding.
 */
export function cosineSimilarity(a: number[], b: number[]): number {
  if (a.length !== b.length) {
    throw new Error(`Hai vector khác chiều: ${a.length} vs ${b.length}`)
  }
  let dot = 0
  let normA = 0
  let normB = 0
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i]
    normA += a[i] * a[i]
    normB += b[i] * b[i]
  }
  const denom = Math.sqrt(normA) * Math.sqrt(normB)
  return denom === 0 ? 0 : dot / denom
}

/**
 * Entropy của phân phối dự đoán (đơn vị: bit) — đo độ "không chắc chắn".
 *   0 bit          = model rất chắc chắn
 *   log2(n) bit    = model đoán bừa hoàn toàn (n nhãn đều nhau)
 * Hữu ích để dạy khái niệm confidence calibration.
 */
export function entropyBits(probabilities: number[]): number {
  return -probabilities.reduce((acc, p) => (p > 0 ? acc + p * Math.log2(p) : acc), 0)
}
