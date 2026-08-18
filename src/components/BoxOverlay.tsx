/**
 * Lớp phủ bounding box cho demo phát hiện vật thể.
 *
 * Toạ độ được nhận ở dạng TỈ LỆ (0–1) nhờ tuỳ chọn `percentage: true` của
 * pipeline, nên hộp tự co giãn đúng theo ảnh mà không cần biết ảnh đang được
 * hiển thị ở kích thước nào. Đây là lý do nên dùng `percentage: true` thay vì
 * toạ độ pixel: pixel là của ảnh đã qua tiền xử lý, không phải của ảnh gốc.
 */
export interface DetectionBox {
  label: string
  score: number
  /** Tất cả trong [0, 1], tính theo bề rộng/chiều cao ảnh. */
  xmin: number
  ymin: number
  xmax: number
  ymax: number
  color: string
}

export function BoxOverlay({ boxes }: { boxes: DetectionBox[] }) {
  return (
    <div className="box-overlay">
      {boxes.map((box, i) => {
        const left = box.xmin * 100
        const top = box.ymin * 100
        const width = (box.xmax - box.xmin) * 100
        const height = (box.ymax - box.ymin) * 100

        return (
          <div
            key={`${box.label}-${i}`}
            className="box"
            style={{
              left: `${left}%`,
              top: `${top}%`,
              width: `${width}%`,
              height: `${height}%`,
              borderColor: box.color,
            }}
          >
            {/* Hộp sát mép trên thì nhãn phải nằm bên TRONG, nếu không nó bị
                đẩy ra ngoài ảnh và mất luôn. */}
            <span
              className={top < 8 ? 'box-label box-label--inside' : 'box-label'}
              style={{ background: box.color }}
            >
              {box.label} {(box.score * 100).toFixed(0)}%
            </span>
          </div>
        )
      })}
    </div>
  )
}

/**
 * Màu ổn định theo TÊN nhãn.
 *
 * Băm tên thay vì lấy theo thứ tự trong danh sách: chỉnh ngưỡng làm danh sách
 * đổi độ dài, nếu lấy theo thứ tự thì "person" đang xanh sẽ nhảy sang đỏ, trông
 * như model vừa đổi ý.
 */
export function boxColor(label: string): string {
  let hash = 0
  for (let i = 0; i < label.length; i++) hash = (hash * 31 + label.charCodeAt(i)) % 360
  return `hsl(${hash} 70% 58%)`
}
