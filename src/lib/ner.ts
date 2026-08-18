/**
 * GỘP NHÃN THEO TOKEN THÀNH ENTITY (BIO aggregation).
 *
 * Model NER trả về một nhãn cho MỖI token, nhưng thứ người dùng cần là các đoạn
 * văn bản. Hai việc phải làm bằng tay:
 *
 *  1. Gộp theo sơ đồ BIO — `B-PER` mở một entity mới, `I-PER` nối tiếp entity
 *     đang mở, `O` đóng entity lại.
 *  2. Dán các subword lại — tokenizer cắt "Nguyễn" thành `Nguy` + `##ễn`, mỗi
 *     mảnh nhận một nhãn riêng. Không dán lại thì kết quả là mảnh vụn.
 *
 * Trong thư viện Python việc này nằm sau tham số `aggregation_strategy`.
 * Ở đây viết ra để thấy nó thật sự làm gì — và để thấy nó là HEURISTIC, không
 * phải một phép biến đổi hiển nhiên: cùng một chuỗi logits có thể gộp ra kết quả
 * khác nhau tuỳ quy tắc bạn chọn.
 *
 * File này không chạm React/DOM, nhưng cũng không được worker import — nó chạy ở
 * phía UI, trên logits thô mà runner 'token-classification' trả về.
 */
import { argmax, softmax } from './math'

/** Một token sau khi đã argmax, trước khi gộp. */
export interface LabeledToken {
  index: number
  token: string
  id: number
  /** Nhãn BIO đầy đủ, ví dụ 'B-PER' hoặc 'O'. */
  label: string
  /** Xác suất của nhãn thắng, sau softmax trên chiều nhãn. */
  score: number
  /** Loại entity đã bỏ tiền tố B-/I-. `null` nếu nhãn là 'O'. */
  type: string | null
  /** true = subword dán vào token trước (marker '##' của WordPiece). */
  continuation: boolean
  /** true = [CLS] / [SEP] / [PAD] — không phải chữ của người dùng. */
  special: boolean
}

export interface Entity {
  type: string
  /** Văn bản đã dán lại các subword. */
  text: string
  /** Vị trí các token cấu thành, để đối chiếu với hàng token phía trên. */
  tokenIndices: number[]
  /**
   * Điểm của entity = TRUNG BÌNH điểm các token cấu thành.
   *
   * Đây là một lựa chọn, không phải chân lý: thư viện Python có cả `first`
   * (lấy điểm token đầu), `max` và `average`. Với entity dài, average làm
   * điểm thấp đi khi có một mảnh subword không chắc chắn.
   */
  score: number
}

const SPECIAL_TOKEN = /^\[(CLS|SEP|PAD|MASK|UNK)\]$/

/**
 * Từ logits thô (một hàng cho mỗi token) ra danh sách token đã gán nhãn.
 *
 * Softmax được tính TRÊN TỪNG TOKEN, tức trên chiều nhãn — không phải trên chiều
 * độ dài câu. Nhầm chiều ở đây là lỗi rất dễ mắc và không có gì báo.
 */
export function labelTokens(
  tokens: string[],
  ids: number[],
  logits: number[][],
  labels: string[],
): LabeledToken[] {
  return tokens.map((token, index) => {
    const probabilities = softmax(logits[index] ?? [])
    const best = argmax(probabilities)
    const label = labels[best] ?? 'O'
    const continuation = token.startsWith('##')

    return {
      index,
      token,
      id: ids[index],
      label,
      score: probabilities[best] ?? 0,
      type: label === 'O' ? null : label.slice(2),
      continuation,
      special: SPECIAL_TOKEN.test(token),
    }
  })
}

/** Dán một dãy subword thành chữ đọc được. */
function joinTokens(tokens: string[]): string {
  return tokens.reduce((text, token, i) => {
    if (token.startsWith('##')) return text + token.slice(2)
    return i === 0 ? token : `${text} ${token}`
  }, '')
}

/**
 * Gộp các token đã gán nhãn thành entity.
 *
 * Quy tắc, theo đúng thứ tự kiểm tra:
 *   'O'                      → đóng entity đang mở
 *   'B-X'                    → đóng entity đang mở, mở entity mới loại X
 *   'I-X' khi đang mở X      → nối tiếp
 *   'I-X' khi KHÔNG mở X     → mở entity mới loại X
 *   token '##…'  khi đang mở → nối tiếp bất kể nhãn của nó là gì
 *
 * Hai dòng cuối là phần đáng chú ý nhất:
 *
 *  • `I-X` không có `B-X` đứng trước vẫn phải mở entity mới. Bộ nhãn CoNLL-2003
 *    mà model này học dùng quy ước IOB1, ở đó `B-` chỉ xuất hiện khi hai entity
 *    cùng loại nằm sát nhau — nên phần lớn entity thật sự BẮT ĐẦU bằng `I-`.
 *    Ai áp dụng đúng lý thuyết IOB2 ("phải có B- mới mở") sẽ mất gần hết entity.
 *
 *  • Subword được nối vô điều kiện. Model có thể gán `Nguy` = B-PER nhưng `##ễn`
 *    = O; cắt entity ở giữa một từ thì ra chữ vô nghĩa, nên ranh giới từ được ưu
 *    tiên hơn nhãn của từng mảnh.
 */
export function mergeEntities(labeled: LabeledToken[]): Entity[] {
  const entities: Entity[] = []

  let current: { type: string; tokens: string[]; indices: number[]; scores: number[] } | null = null

  const close = () => {
    if (!current) return
    entities.push({
      type: current.type,
      text: joinTokens(current.tokens),
      tokenIndices: current.indices,
      score: current.scores.reduce((a, b) => a + b, 0) / current.scores.length,
    })
    current = null
  }

  for (const item of labeled) {
    if (item.special) {
      close()
      continue
    }

    const extend = () => {
      current!.tokens.push(item.token)
      current!.indices.push(item.index)
      current!.scores.push(item.score)
    }

    // Subword luôn dính vào entity đang mở, kể cả khi nhãn riêng của nó là 'O'.
    if (item.continuation && current) {
      extend()
      continue
    }

    if (!item.type) {
      close()
      continue
    }

    const isBegin = item.label.startsWith('B-')
    if (current && current.type === item.type && !isBegin) {
      extend()
      continue
    }

    close()
    current = { type: item.type, tokens: [item.token], indices: [item.index], scores: [item.score] }
  }

  close()
  return entities
}

/** Màu theo loại entity — dùng chung cho hàng token và danh sách entity. */
export const ENTITY_COLORS: Record<string, string> = {
  PER: '#e5734a',
  ORG: '#7c9aff',
  LOC: '#45c48c',
  MISC: '#c77dd6',
}

export const ENTITY_LABELS: Record<string, string> = {
  PER: 'Người',
  ORG: 'Tổ chức',
  LOC: 'Địa điểm',
  MISC: 'Khác',
}

export function entityColor(type: string): string {
  return ENTITY_COLORS[type] ?? '#dfa945'
}
