/**
 * BM25 — tìm kiếm theo TỪ KHOÁ, để đặt cạnh tìm kiếm ngữ nghĩa.
 *
 * Viết tay (thay vì dùng thư viện) vì đây chính là phần bài giảng: sinh viên cần
 * thấy BM25 chỉ đếm từ trùng nhau, nên khi truy vấn không dùng đúng từ của tài
 * liệu thì nó *không thể* tìm ra — dù ý nghĩa giống hệt. Đó là lý do embedding
 * tồn tại.
 *
 * Công thức:
 *
 *   score(D, Q) = Σ_{t∈Q}  IDF(t) · ( f(t,D) · (k₁+1) ) / ( f(t,D) + k₁·(1 − b + b·|D|/avgdl) )
 *
 *   IDF(t) = ln( 1 + (N − n(t) + 0.5) / (n(t) + 0.5) )
 *
 * trong đó f(t,D) = số lần từ t xuất hiện trong tài liệu D, n(t) = số tài liệu
 * chứa t, N = tổng số tài liệu, |D| = độ dài tài liệu, avgdl = độ dài trung bình.
 */

/** k₁ điều khiển mức bão hoà tần số: từ xuất hiện 10 lần không "gấp 10" 1 lần. */
const K1 = 1.5
/** b điều khiển mức chuẩn hoá theo độ dài: 0 = bỏ qua, 1 = chuẩn hoá hoàn toàn. */
const B = 0.75

export interface Bm25Index {
  docTerms: string[][]
  docLengths: number[]
  averageLength: number
  /** Số tài liệu chứa mỗi từ. */
  documentFrequency: Map<string, number>
  documentCount: number
}

export interface Bm25Hit {
  docIndex: number
  score: number
  /** Các từ của truy vấn thực sự xuất hiện trong tài liệu — dùng để highlight. */
  matchedTerms: string[]
}

/**
 * Tách từ rất đơn giản: hạ chữ thường rồi cắt ở mọi thứ không phải chữ/số.
 *
 * Cố ý GIỮ dấu tiếng Việt: "hoc" và "học" là hai từ khác nhau, bỏ dấu sẽ làm
 * BM25 trông có vẻ tốt hơn thực tế. Đây cũng là điểm để thảo luận trong lớp —
 * tách từ tiếng Việt thật sự cần tách từ ghép ("máy học" là một khái niệm).
 */
export function tokenizeForBm25(text: string): string[] {
  return text
    .toLowerCase()
    .normalize('NFC')
    .split(/[^\p{L}\p{N}]+/u)
    .filter((term) => term.length > 0)
}

export function buildBm25Index(documents: string[]): Bm25Index {
  const docTerms = documents.map(tokenizeForBm25)
  const docLengths = docTerms.map((terms) => terms.length)
  const totalLength = docLengths.reduce((a, b) => a + b, 0)

  const documentFrequency = new Map<string, number>()
  for (const terms of docTerms) {
    // Set: mỗi tài liệu chỉ đóng góp 1 vào document frequency của một từ.
    for (const term of new Set(terms)) {
      documentFrequency.set(term, (documentFrequency.get(term) ?? 0) + 1)
    }
  }

  return {
    docTerms,
    docLengths,
    averageLength: documents.length ? totalLength / documents.length : 0,
    documentFrequency,
    documentCount: documents.length,
  }
}

function inverseDocumentFrequency(index: Bm25Index, term: string): number {
  const n = index.documentFrequency.get(term) ?? 0
  // Từ xuất hiện ở mọi tài liệu -> IDF gần 0 -> gần như không mang thông tin.
  return Math.log(1 + (index.documentCount - n + 0.5) / (n + 0.5))
}

export function bm25Search(index: Bm25Index, query: string): Bm25Hit[] {
  const queryTerms = tokenizeForBm25(query)

  const hits: Bm25Hit[] = index.docTerms.map((terms, docIndex) => {
    const frequency = new Map<string, number>()
    for (const term of terms) frequency.set(term, (frequency.get(term) ?? 0) + 1)

    let score = 0
    const matchedTerms: string[] = []

    for (const term of new Set(queryTerms)) {
      const f = frequency.get(term) ?? 0
      if (f === 0) continue
      matchedTerms.push(term)

      const lengthNorm = 1 - B + (B * index.docLengths[docIndex]) / (index.averageLength || 1)
      score += inverseDocumentFrequency(index, term) * ((f * (K1 + 1)) / (f + K1 * lengthNorm))
    }

    return { docIndex, score, matchedTerms }
  })

  return hits.sort((a, b) => b.score - a.score)
}
