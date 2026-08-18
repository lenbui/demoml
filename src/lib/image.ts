/**
 * Chuẩn bị ảnh do người dùng chọn trước khi đưa vào model.
 *
 * Vì sao không đưa thẳng File vào worker? Vì ảnh chụp từ điện thoại thường
 * 4000×3000 và nặng vài MB. postMessage sẽ copy nguyên khối đó sang worker, còn
 * model thì sẽ thu nhỏ ngay lập tức về 224×224 (ViT) hoặc cạnh ngắn 800 (DETR).
 * Thu nhỏ trước ở main thread rẻ hơn nhiều và không làm đổi kết quả.
 *
 * Vẫn giữ lại kích thước GỐC để demo hiện được "3024×4032 → 224×224" — con số đó
 * là phần dạy học: tiền xử lý ảnh vứt đi rất nhiều thông tin trước khi model kịp
 * nhìn thấy gì.
 */

/** Cạnh dài tối đa sau khi thu nhỏ để truyền sang worker. */
const MAX_EDGE = 800

export interface PreparedImage {
  /** data URL dùng làm input cho pipeline() và làm src cho thẻ <img> xem trước. */
  src: string
  /** Tên hiển thị (tên file, hoặc nhãn của ảnh mẫu). */
  label: string
  /** Kích thước GỐC trước khi thu nhỏ. */
  naturalWidth: number
  naturalHeight: number
  /** Kích thước thực sự được gửi sang worker. */
  width: number
  height: number
  /** Cỡ xấp xỉ của data URL (byte) — để thấy chi phí truyền. */
  bytes: number
}

/** Đọc kích thước thật của một ảnh từ URL bất kỳ. */
export function readImageSize(src: string): Promise<{ width: number; height: number }> {
  return new Promise((resolve, reject) => {
    const image = new Image()
    // Ảnh mẫu nằm ở origin khác; xin CORS để còn vẽ được lên canvas nếu cần.
    image.crossOrigin = 'anonymous'
    image.onload = () => resolve({ width: image.naturalWidth, height: image.naturalHeight })
    image.onerror = () => reject(new Error(`Không tải được ảnh: ${src}`))
    image.src = src
  })
}

/** Đọc một File người dùng chọn thành data URL, có thu nhỏ nếu quá lớn. */
export function prepareImageFile(file: File): Promise<PreparedImage> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()

    reader.onerror = () => reject(new Error('Không đọc được file ảnh.'))
    reader.onload = () => {
      const original = String(reader.result)
      const image = new Image()

      image.onerror = () => reject(new Error('File không phải là ảnh hợp lệ.'))
      image.onload = () => {
        const { naturalWidth, naturalHeight } = image
        const scale = Math.min(1, MAX_EDGE / Math.max(naturalWidth, naturalHeight))

        // Ảnh đã đủ nhỏ: giữ nguyên bản gốc, không encode lại (encode lại một
        // ảnh PNG thành JPEG là làm giảm chất lượng mà chẳng được gì).
        if (scale === 1) {
          resolve({
            src: original,
            label: file.name,
            naturalWidth,
            naturalHeight,
            width: naturalWidth,
            height: naturalHeight,
            bytes: original.length,
          })
          return
        }

        const width = Math.round(naturalWidth * scale)
        const height = Math.round(naturalHeight * scale)

        const canvas = document.createElement('canvas')
        canvas.width = width
        canvas.height = height

        const context = canvas.getContext('2d')
        if (!context) {
          reject(new Error('Trình duyệt không tạo được canvas 2D.'))
          return
        }
        context.drawImage(image, 0, 0, width, height)

        const src = canvas.toDataURL('image/jpeg', 0.9)
        resolve({
          src,
          label: file.name,
          naturalWidth,
          naturalHeight,
          width,
          height,
          bytes: src.length,
        })
      }

      image.src = original
    }

    reader.readAsDataURL(file)
  })
}
