export interface ScatterPoint {
  x: number
  y: number
  color: string
  /** Nội dung tooltip khi hover. */
  label: string
  /** Vẽ viền nổi bật — dùng cho các kết quả top-k của truy vấn. */
  highlighted?: boolean
}

const WIDTH = 460
const HEIGHT = 280
const PADDING = 18

/**
 * Scatter SVG cho bản đồ 2D của embedding.
 *
 * Không dùng thư viện chart: dữ liệu chỉ là vài chục điểm, và tự vẽ thì kiểm
 * soát được hoàn toàn màu theme và không thêm 100 KB vào bundle.
 *
 * `queryPoint` được vẽ thành dấu ✕ để phân biệt rõ với các đoạn văn — nó là một
 * loại đối tượng khác (truy vấn), không phải một tài liệu trong corpus.
 */
export function ScatterPlot({
  points,
  queryPoint,
}: {
  points: ScatterPoint[]
  queryPoint?: { x: number; y: number; label: string }
}) {
  const all = queryPoint ? [...points, queryPoint] : points
  if (all.length === 0) return null

  const xs = all.map((p) => p.x)
  const ys = all.map((p) => p.y)
  const minX = Math.min(...xs)
  const maxX = Math.max(...xs)
  const minY = Math.min(...ys)
  const maxY = Math.max(...ys)

  // Chia cho khoảng biến thiên; `|| 1` chặn trường hợp mọi điểm trùng nhau.
  const scaleX = (x: number) =>
    PADDING + ((x - minX) / (maxX - minX || 1)) * (WIDTH - PADDING * 2)
  const scaleY = (y: number) =>
    // Trục y của SVG hướng xuống, nên đảo lại cho giống hệ toạ độ toán học.
    HEIGHT - PADDING - ((y - minY) / (maxY - minY || 1)) * (HEIGHT - PADDING * 2)

  return (
    <div className="scatter">
      <svg viewBox={`0 0 ${WIDTH} ${HEIGHT}`} role="img" aria-label="Bản đồ 2D các embedding">
        <rect
          x={0.5}
          y={0.5}
          width={WIDTH - 1}
          height={HEIGHT - 1}
          rx={7}
          className="scatter-frame"
        />

        {points.map((point, i) => (
          <circle
            key={i}
            cx={scaleX(point.x)}
            cy={scaleY(point.y)}
            r={point.highlighted ? 6.5 : 4.5}
            fill={point.color}
            fillOpacity={point.highlighted ? 1 : 0.72}
            stroke={point.highlighted ? 'currentColor' : 'none'}
            strokeWidth={point.highlighted ? 1.6 : 0}
          >
            <title>{point.label}</title>
          </circle>
        ))}

        {queryPoint && (
          <g className="scatter-query">
            <line
              x1={scaleX(queryPoint.x) - 7}
              y1={scaleY(queryPoint.y) - 7}
              x2={scaleX(queryPoint.x) + 7}
              y2={scaleY(queryPoint.y) + 7}
            />
            <line
              x1={scaleX(queryPoint.x) - 7}
              y1={scaleY(queryPoint.y) + 7}
              x2={scaleX(queryPoint.x) + 7}
              y2={scaleY(queryPoint.y) - 7}
            />
            <title>{queryPoint.label}</title>
          </g>
        )}
      </svg>
    </div>
  )
}
