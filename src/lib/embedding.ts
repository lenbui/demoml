/**
 * Pooling: biến ma trận [seq_len, hidden] thành MỘT vector cho cả câu.
 *
 * Đây là bước mà `pipeline('feature-extraction')` làm sẵn và che đi, nhưng nó
 * là một lựa chọn thiết kế thật sự — không có "embedding của câu" duy nhất.
 *
 * File này được worker import, nên không được chạm vào React/DOM.
 */

export type Pooling = 'mean' | 'cls'

export const POOLING_LABELS: Record<Pooling, string> = {
  mean: 'Mean pooling',
  cls: 'CLS pooling',
}

/**
 * Gộp last_hidden_state thành một vector cho mỗi câu trong batch.
 *
 * @param hidden        dữ liệu phẳng của tensor [batch, seq, dim]
 * @param dims          [batch, seq, dim]
 * @param attentionMask độ dài batch*seq; 1 = token thật, 0 = padding
 *
 * Vì sao attention_mask là bắt buộc với mean pooling?
 *   Khi encode nhiều câu cùng lúc, các câu ngắn được đệm thêm token [PAD] cho
 *   bằng câu dài nhất. Nếu lấy trung bình cả phần đệm thì câu ngắn bị pha loãng
 *   bởi vector vô nghĩa — càng ngắn càng sai. Đây là lỗi im lặng rất phổ biến.
 *
 * mean vs cls:
 *   mean : trung bình mọi token thật. Bền, là mặc định của hầu hết model
 *          sentence-transformers (chúng được huấn luyện với mean pooling).
 *   cls  : lấy đúng vector của token [CLS] ở vị trí 0. Chỉ tốt nếu model được
 *          huấn luyện để dồn thông tin câu vào [CLS]. Dùng sai sẽ cho kết quả
 *          tệ hơn rõ rệt — hãy thử đổi trên demo để thấy.
 */
export function poolHiddenStates(
  hidden: ArrayLike<number>,
  dims: readonly [number, number, number],
  attentionMask: number[],
  mode: Pooling,
): number[][] {
  const [batch, seqLen, dim] = dims
  const out: number[][] = []

  for (let b = 0; b < batch; b++) {
    const vector = new Array<number>(dim).fill(0)

    if (mode === 'cls') {
      // Token đầu tiên của câu — offset của (b, t=0, d) là b*seq*dim + d.
      const base = b * seqLen * dim
      for (let d = 0; d < dim; d++) vector[d] = hidden[base + d]
    } else {
      let realTokens = 0
      for (let t = 0; t < seqLen; t++) {
        if (attentionMask[b * seqLen + t] === 0) continue // bỏ qua [PAD]
        realTokens++
        const base = (b * seqLen + t) * dim
        for (let d = 0; d < dim; d++) vector[d] += hidden[base + d]
      }
      const divisor = realTokens || 1
      for (let d = 0; d < dim; d++) vector[d] /= divisor
    }

    out.push(vector)
  }

  return out
}

/**
 * Chuẩn hoá về độ dài 1.
 *
 * Sau khi normalize, cosine similarity chỉ còn là phép nhân vô hướng:
 *     cos(a, b) = a · b        (vì ‖a‖ = ‖b‖ = 1)
 * Đó là lý do mọi vector database đều lưu vector đã normalize — tìm kiếm chỉ
 * còn là một phép nhân ma trận.
 */
export function l2Normalize(vector: number[]): number[] {
  let sumOfSquares = 0
  for (const v of vector) sumOfSquares += v * v
  const norm = Math.sqrt(sumOfSquares)
  return norm === 0 ? vector.slice() : vector.map((v) => v / norm)
}

/** Nhân vô hướng. Với vector đã normalize thì đây chính là cosine similarity. */
export function dot(a: number[], b: number[]): number {
  let sum = 0
  for (let i = 0; i < a.length; i++) sum += a[i] * b[i]
  return sum
}
