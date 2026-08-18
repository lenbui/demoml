/**
 * PCA giảm chiều 384 → 2 để vẽ được embedding lên mặt phẳng.
 *
 * Viết tay bằng power iteration, không dùng thư viện, vì thuật toán đủ ngắn để
 * đọc hiểu trong một tiết và nó cho thấy PCA thực chất là gì: tìm hướng mà dữ
 * liệu biến thiên mạnh nhất.
 *
 * ⚠️ Điều cần nói rõ với sinh viên: bản đồ 2D là một PHÉP CHIẾU, có mất mát.
 * Hai điểm trông gần nhau trên hình chưa chắc gần nhau trong không gian 384
 * chiều. Xếp hạng tìm kiếm luôn tính bằng cosine trên vector đầy đủ, không bao
 * giờ tính trên toạ độ 2D. Tỉ lệ phương sai giữ lại (explainedRatio) cho biết
 * bức tranh 2D đáng tin đến đâu.
 */

export interface Pca2dResult {
  points: Array<[number, number]>
  /** Phần phương sai mà 2 trục giữ lại được, trong [0, 1]. */
  explainedRatio: number
}

/** Trừ đi vector trung bình — PCA luôn làm việc trên dữ liệu đã căn giữa. */
function center(vectors: number[][]): { centered: number[][]; mean: number[] } {
  const n = vectors.length
  const dim = vectors[0].length
  const mean = new Array<number>(dim).fill(0)

  for (const v of vectors) for (let d = 0; d < dim; d++) mean[d] += v[d]
  for (let d = 0; d < dim; d++) mean[d] /= n

  return { centered: vectors.map((v) => v.map((value, d) => value - mean[d])), mean }
}

function norm(v: number[]): number {
  let sum = 0
  for (const x of v) sum += x * x
  return Math.sqrt(sum)
}

/**
 * Tìm hướng biến thiên mạnh nhất của X bằng power iteration.
 *
 * Ý tưởng: lặp lại v ← Xᵀ(Xv) rồi chuẩn hoá. Mỗi lần lặp, thành phần của v theo
 * hướng có phương sai lớn nhất được nhân lên nhiều nhất, nên v hội tụ về đúng
 * hướng đó (eigenvector ứng với eigenvalue lớn nhất của Xᵀ X).
 *
 * Không cần dựng ma trận covariance 384×384 — chỉ cần hai phép nhân vector.
 */
function topDirection(X: number[][], iterations = 64): number[] {
  const dim = X[0].length
  // Khởi tạo tất định (không dùng random) để cùng dữ liệu luôn cho cùng hình.
  let v = Array.from({ length: dim }, (_, i) => Math.sin(i + 1))
  const initialNorm = norm(v)
  v = v.map((x) => x / initialNorm)

  for (let iter = 0; iter < iterations; iter++) {
    // projections = X v  (độ dài n)
    const projections = X.map((row) => {
      let sum = 0
      for (let d = 0; d < dim; d++) sum += row[d] * v[d]
      return sum
    })

    // next = Xᵀ projections  (độ dài dim)
    const next = new Array<number>(dim).fill(0)
    for (let i = 0; i < X.length; i++) {
      const p = projections[i]
      const row = X[i]
      for (let d = 0; d < dim; d++) next[d] += row[d] * p
    }

    const length = norm(next)
    if (length < 1e-12) break
    v = next.map((x) => x / length)
  }

  return v
}

/** Bỏ thành phần theo hướng `direction` khỏi mọi hàng (deflation). */
function removeComponent(X: number[][], direction: number[]): number[][] {
  const dim = direction.length
  return X.map((row) => {
    let projection = 0
    for (let d = 0; d < dim; d++) projection += row[d] * direction[d]
    return row.map((value, d) => value - projection * direction[d])
  })
}

function totalVariance(X: number[][]): number {
  let sum = 0
  for (const row of X) for (const value of row) sum += value * value
  return sum
}

export function pca2d(vectors: number[][]): Pca2dResult {
  if (vectors.length < 2 || vectors[0].length < 2) {
    return { points: vectors.map(() => [0, 0]), explainedRatio: 0 }
  }

  const { centered } = center(vectors)
  const variance = totalVariance(centered)

  // Trục 1: hướng phương sai lớn nhất.
  const axis1 = topDirection(centered)
  // Trục 2: làm lại trên phần dư sau khi đã bỏ trục 1 -> vuông góc với trục 1.
  const axis2 = topDirection(removeComponent(centered, axis1))

  const points = centered.map((row) => {
    let x = 0
    let y = 0
    for (let d = 0; d < row.length; d++) {
      x += row[d] * axis1[d]
      y += row[d] * axis2[d]
    }
    return [x, y] as [number, number]
  })

  const kept = points.reduce((sum, [x, y]) => sum + x * x + y * y, 0)

  return { points, explainedRatio: variance > 0 ? kept / variance : 0 }
}
