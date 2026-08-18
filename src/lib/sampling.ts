/**
 * SAMPLING — cách một LLM chọn token kế tiếp.
 *
 * Đây là phần bài giảng chính của demo LLM, nên viết tay thay vì gọi thư viện.
 *
 * ── Điều quan trọng nhất cần hiểu ─────────────────────────────────────────
 * Model KHÔNG chọn token. Model chỉ trả về một vector logits — một con số cho
 * mỗi token trong vocabulary. Toàn bộ những gì dưới đây là HẬU XỬ LÝ chạy bên
 * ngoài model:
 *
 *     logits ──chia nhiệt độ──▶ softmax ──cắt top-k──▶ cắt top-p ──▶ bốc thăm
 *
 * Nhờ vậy demo kéo được các slider mà không phải chạy lại model: chỉ cần logits
 * là tính lại được toàn bộ. Đó cũng là lý do temperature/top-k/top-p là tham số
 * của lần *gọi*, không phải thuộc tính của model.
 */
import { softmax } from './math'

export interface SamplingParams {
  /**
   * Nhiệt độ. Chia logits trước khi softmax:
   *   T → 0   : phân phối nhọn hoắt, luôn chọn token cao nhất (greedy)
   *   T = 1   : giữ nguyên phân phối model học được
   *   T > 1   : san phẳng, token hiếm có cơ hội hơn — "sáng tạo" và cũng lảm nhảm hơn
   */
  temperature: number
  /** Chỉ giữ k token cao nhất. 0 = không cắt. */
  topK: number
  /** Nucleus: giữ các token cao nhất cho tới khi xác suất dồn đủ p. 1 = không cắt. */
  topP: number
}

export interface ScoredToken {
  id: number
  token: string
  logit: number
  /** Xác suất sau khi chia nhiệt độ và softmax, TRƯỚC khi cắt. */
  probability: number
  /** Xác suất dồn tính từ token cao nhất — dùng để hiểu top-p. */
  cumulative: number
  /** Còn sống sót sau cả top-k lẫn top-p không. */
  kept: boolean
  /** Xác suất sau khi cắt và chuẩn hoá lại. 0 nếu bị loại. */
  finalProbability: number
  /** Bị loại bởi bước nào — để UI giải thích được lý do. */
  cutBy?: 'top-k' | 'top-p'
}

export interface SamplingResult {
  tokens: ScoredToken[]
  keptCount: number
  /**
   * Entropy (bit) của phân phối SAU khi cắt và chuẩn hoá.
   * Đo mức "còn bao nhiêu lựa chọn thật sự" ở bước này.
   */
  entropy: number
  /**
   * Phần xác suất mà danh sách ứng viên này bao phủ được, tính trên phân phối
   * ở nhiệt độ hiện tại. Nhỏ hơn 1 vì worker chỉ gửi về phần đỉnh của
   * vocabulary — nói rõ ra thay vì giả vờ đây là toàn bộ phân phối.
   */
  coverage: number
}

/**
 * Áp nhiệt độ, top-k, top-p lên một danh sách ứng viên đã sắp theo logit giảm.
 *
 * ⚠️ Lưu ý về tính chính xác: `candidates` chỉ là phần đỉnh của vocabulary
 * (worker gửi về top-60 trong ~50.000 token). Ở nhiệt độ thấp, phần đuôi bị bỏ
 * là không đáng kể nên kết quả gần như chính xác tuyệt đối. Ở nhiệt độ rất cao
 * phần đuôi mới đáng kể — `coverage` cho biết đang bỏ sót bao nhiêu.
 */
export function applySampling(
  candidates: Array<{ id: number; token: string; logit: number }>,
  { temperature, topK, topP }: SamplingParams,
): SamplingResult {
  if (candidates.length === 0) {
    return { tokens: [], keptCount: 0, entropy: 0, coverage: 0 }
  }

  // Nhiệt độ 0 nghĩa là greedy. Chia cho 0 ra Infinity, nên xử lý riêng bằng
  // cách dùng một nhiệt độ rất nhỏ — kết quả hội tụ về đúng "luôn chọn top-1".
  const t = temperature <= 0 ? 1e-6 : temperature

  const probabilities = softmax(candidates.map((c) => c.logit / t))

  // Ứng viên đã được worker sắp theo logit giảm dần; chia cho một hằng số dương
  // không làm đổi thứ tự, nên xác suất cũng đã giảm dần.
  const tokens: ScoredToken[] = []
  let cumulative = 0

  for (let i = 0; i < candidates.length; i++) {
    const probability = probabilities[i]
    const before = cumulative
    cumulative += probability

    // top-k: chỉ giữ k phần tử đầu.
    const survivesK = topK <= 0 || i < topK

    // top-p (nucleus): giữ tới khi xác suất DỒN vượt p. Token làm cumulative
    // vượt ngưỡng vẫn được giữ — nếu không, p nhỏ hơn xác suất của token đầu
    // sẽ loại sạch mọi thứ và không còn gì để bốc.
    const survivesP = topP >= 1 || before < topP

    const kept = survivesK && survivesP

    tokens.push({
      id: candidates[i].id,
      token: candidates[i].token,
      logit: candidates[i].logit,
      probability,
      cumulative,
      kept,
      finalProbability: 0,
      cutBy: kept ? undefined : !survivesK ? 'top-k' : 'top-p',
    })
  }

  // Chuẩn hoá lại trên các token còn sống — đây mới là phân phối thật sự được
  // bốc thăm. Phần bị cắt coi như xác suất 0.
  const keptMass = tokens.reduce((sum, item) => (item.kept ? sum + item.probability : sum), 0)
  if (keptMass > 0) {
    for (const item of tokens) {
      if (item.kept) item.finalProbability = item.probability / keptMass
    }
  }

  const entropy = -tokens.reduce(
    (sum, item) =>
      item.finalProbability > 0 ? sum + item.finalProbability * Math.log2(item.finalProbability) : sum,
    0,
  )

  return {
    tokens,
    keptCount: tokens.filter((item) => item.kept).length,
    entropy,
    coverage: cumulative,
  }
}

/** Các cấu hình hay gặp, để bấm một cái là thấy khác biệt. */
export const SAMPLING_PRESETS: Array<{ label: string; params: SamplingParams; note: string }> = [
  {
    label: 'Greedy',
    params: { temperature: 0, topK: 0, topP: 1 },
    note: 'Luôn chọn token cao nhất. Tất định, và rất dễ lặp vòng.',
  },
  {
    label: 'Cân bằng',
    params: { temperature: 0.7, topK: 50, topP: 0.9 },
    note: 'Mặc định của phần lớn API. Đủ đa dạng mà vẫn mạch lạc.',
  },
  {
    label: 'Sáng tạo',
    params: { temperature: 1.2, topK: 0, topP: 0.95 },
    note: 'Nhiệt độ cao, ít cắt. Bất ngờ hơn, cũng dễ lạc đề hơn.',
  },
  {
    label: 'Hỗn loạn',
    params: { temperature: 2, topK: 0, topP: 1 },
    note: 'Gần như bốc ngẫu nhiên. Cho thấy vì sao phải cắt bớt phần đuôi.',
  },
]
