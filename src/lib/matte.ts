/**
 * HẬU XỬ LÝ CHO DEMO TÁCH NỀN — biến mặt nạ alpha thành ảnh.
 *
 * File này tồn tại vì một lý do dạy học duy nhất: **model không hề "xoá nền"**.
 * Thứ nó trả về là một MA TRẬN ALPHA — mỗi pixel một số trong [0, 1] trả lời câu
 * hỏi "pixel này thuộc chủ thể bao nhiêu phần". Tấm ảnh nền trong suốt mà người
 * dùng nhìn thấy là do đoạn code dưới đây nhân ma trận đó vào ảnh gốc, chứ
 * không phải do model sinh ra.
 *
 * Đặt ở đây thay vì gọi task 'background-removal' của Transformers.js (task đó
 * làm đúng việc này nhưng làm ở bên trong thư viện) vì cùng lý do mà bước gộp
 * BIO nằm ở lib/ner.ts và softmax nằm ở lib/math.ts: bước biến output thô thành
 * kết quả người đọc được là bước sinh viên phải nhìn thấy, không phải bước để
 * giấu đi.
 *
 * Không import '@huggingface/transformers' — chỉ toàn canvas 2D thuần.
 */

/**
 * Mặt nạ alpha do worker gửi về.
 *
 * Bên worker đây là một `RawImage` của Transformers.js, nhưng postMessage chỉ
 * copy các thuộc tính (prototype không qua được structured clone), nên phía này
 * nhận được đúng bốn trường dưới đây. `data` là uint8: 0 = nền, 255 = chủ thể.
 *
 * Mặt nạ đã được pipeline resize về ĐÚNG kích thước ảnh gốc, nên ghép thẳng
 * được, không phải nội suy lại.
 */
export interface AlphaMask {
  data: Uint8ClampedArray
  width: number
  height: number
  /** Luôn bằng 1: mặt nạ là ảnh xám một kênh, không phải ảnh màu. */
  channels: number
}

/** Nền để đặt chủ thể lên sau khi tách. */
export type Background =
  /** Giữ kênh alpha — đây là thứ duy nhất xuất ra PNG được mà không mất thông tin. */
  | { kind: 'transparent' }
  /** Tô một màu đặc phía sau. Dùng để thấy rìa cắt có sạch hay không. */
  | { kind: 'color'; color: string }

export interface CompositeOptions {
  /**
   * Nhị phân hoá mặt nạ trước khi ghép: alpha < ngưỡng -> 0, còn lại -> 255.
   *
   * `null` = giữ nguyên alpha mềm (mặc định, và là cách đúng).
   *
   * Có tuỳ chọn này để so sánh trực tiếp: bật lên thì tóc và lông biến thành
   * rìa răng cưa ngay lập tức. Đó là cách nhanh nhất để thấy vì sao bài toán
   * này gọi là *matting* chứ không phải *segmentation nhị phân*.
   */
  hardThreshold?: number | null
}

/** Đọc một ảnh (data URL hoặc URL) thành HTMLImageElement đã load xong. */
function loadImage(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const image = new Image()
    // Ảnh mẫu nằm ở huggingface.co. Không xin CORS thì canvas bị "tainted" và
    // toDataURL() ném SecurityError — tức là nút Tải ảnh về sẽ hỏng, mà lỗi thì
    // chỉ hiện lúc bấm nút chứ không phải lúc vẽ.
    image.crossOrigin = 'anonymous'
    image.onload = () => resolve(image)
    image.onerror = () => reject(new Error(`Không tải được ảnh: ${src}`))
    image.src = src
  })
}

function createCanvas(width: number, height: number): [HTMLCanvasElement, CanvasRenderingContext2D] {
  const canvas = document.createElement('canvas')
  canvas.width = width
  canvas.height = height
  const context = canvas.getContext('2d')
  if (!context) throw new Error('Trình duyệt không tạo được canvas 2D.')
  return [canvas, context]
}

/** Lấy giá trị alpha (0–255) của pixel thứ i, đã áp ngưỡng nếu có. */
function alphaAt(mask: AlphaMask, i: number, hardThreshold: number | null): number {
  const raw = mask.data[i]
  if (hardThreshold == null) return raw
  return raw >= hardThreshold * 255 ? 255 : 0
}

/**
 * Ghép ảnh gốc với mặt nạ alpha thành ảnh đã tách nền.
 *
 * Đây là TOÀN BỘ phép "xoá nền", và nó chỉ là một vòng lặp gán kênh alpha:
 *
 *     RGB  giữ nguyên từ ảnh gốc
 *     A    lấy từ mặt nạ
 *
 * Không có mô hình nào chạy ở bước này.
 */
export async function compositeCutout(
  imageSrc: string,
  mask: AlphaMask,
  background: Background,
  options: CompositeOptions = {},
): Promise<string> {
  const hard = options.hardThreshold ?? null
  const image = await loadImage(imageSrc)

  // Vẽ ảnh gốc ở đúng độ phân giải của mặt nạ. Hai bên vốn đã cùng kích thước;
  // drawImage có tham số kích thước chỉ để phòng trường hợp lệch một pixel do
  // làm tròn khi pipeline resize.
  const [canvas, context] = createCanvas(mask.width, mask.height)
  context.drawImage(image, 0, 0, mask.width, mask.height)

  const frame = context.getImageData(0, 0, mask.width, mask.height)
  const pixels = frame.data

  for (let i = 0; i < mask.width * mask.height; i++) {
    // pixels là RGBA phẳng nên kênh alpha của pixel thứ i nằm ở 4i + 3.
    pixels[i * 4 + 3] = alphaAt(mask, i, hard)
  }
  context.putImageData(frame, 0, 0)

  if (background.kind === 'transparent') return canvas.toDataURL('image/png')

  // Tô nền: phải vẽ lên một canvas KHÁC rồi đặt ảnh đã tách lên trên.
  // Tô thẳng lên canvas hiện tại bằng globalCompositeOperation 'destination-over'
  // cũng ra kết quả đúng, nhưng làm hỏng luôn bản trong suốt nếu sau đó muốn
  // dùng lại — tách canvas ra thì hàm này không có tác dụng phụ.
  const [output, outputContext] = createCanvas(mask.width, mask.height)
  outputContext.fillStyle = background.color
  outputContext.fillRect(0, 0, mask.width, mask.height)
  outputContext.drawImage(canvas, 0, 0)
  return output.toDataURL('image/png')
}

/**
 * Vẽ chính mặt nạ ra thành ảnh xám để xem trực tiếp.
 *
 * Đây là output THẬT của model — thứ mà task 'background-removal' dựng sẵn của
 * thư viện sẽ không bao giờ cho bạn nhìn thấy.
 */
export function maskToDataUrl(mask: AlphaMask, hardThreshold: number | null = null): string {
  const [canvas, context] = createCanvas(mask.width, mask.height)
  const frame = context.createImageData(mask.width, mask.height)

  for (let i = 0; i < mask.width * mask.height; i++) {
    const value = alphaAt(mask, i, hardThreshold)
    frame.data[i * 4] = value
    frame.data[i * 4 + 1] = value
    frame.data[i * 4 + 2] = value
    frame.data[i * 4 + 3] = 255
  }

  context.putImageData(frame, 0, 0)
  return canvas.toDataURL('image/png')
}

/**
 * Dung sai khi xếp một pixel vào nhóm "chắc chắn".
 *
 * ⚠️ Đừng đổi thành so sánh bằng đúng 0 và đúng 255. Alpha sinh ra từ hàm
 * sigmoid, mà sigmoid KHÔNG BAO GIỜ trả về đúng 0 hay đúng 1 — nó chỉ tiệm cận.
 * Nền của một tấm ảnh chân dung thường ra alpha = 1/255 hoặc 2/255 chứ không
 * phải 0. Đã kiểm chứng trên MODNet + ảnh portrait-of-woman: đếm theo "khác
 * chính xác 0 và 255" cho ra 59% pixel "bán trong suốt", trong khi biểu đồ phân
 * bố cho thấy 94% pixel nằm gọn ở hai đầu. Con số 59% đó vô nghĩa và dạy sai.
 *
 * 2/255 ≈ 0.008 — nhỏ hơn cả một bậc của uint8 mà mắt phân biệt được.
 */
const CERTAIN_TOLERANCE = 2

export interface AlphaStats {
  /** Tổng số pixel của mặt nạ. */
  total: number
  /** alpha ≈ 0 (≤ 2/255) — coi như chắc chắn là nền. */
  transparent: number
  /** alpha ≈ 1 (≥ 253/255) — coi như chắc chắn là chủ thể. */
  opaque: number
  /** Phần còn lại: rìa chủ thể — vùng model không chắc, hoặc bán trong suốt thật. */
  partial: number
  /** Phân bố alpha theo 16 khoảng đều nhau, dùng để vẽ biểu đồ cột. */
  histogram: number[]
  /** Tỉ lệ diện tích chủ thể chiếm trong ảnh (tính theo tổng alpha). */
  coverage: number
}

/**
 * Thống kê phân bố alpha.
 *
 * Con số đáng nhìn nhất là `partial`: nếu mặt nạ chỉ có 0 và 1 thì bài toán này
 * đúng là phân đoạn nhị phân. Thực tế với ảnh chân dung có tóc, vài phần trăm
 * pixel nằm lưng chừng — và chính vài phần trăm đó quyết định ảnh cắt ra trông
 * thật hay trông như dán giấy.
 */
export function alphaStats(mask: AlphaMask): AlphaStats {
  const total = mask.width * mask.height
  const histogram = new Array(16).fill(0)
  let transparent = 0
  let opaque = 0
  let sum = 0

  for (let i = 0; i < total; i++) {
    const value = mask.data[i]
    sum += value
    if (value <= CERTAIN_TOLERANCE) transparent++
    else if (value >= 255 - CERTAIN_TOLERANCE) opaque++
    // 256/16 = 16 giá trị mỗi khoảng; value = 255 rơi vào khoảng cuối (index 15).
    histogram[value >> 4]++
  }

  return {
    total,
    transparent,
    opaque,
    partial: total - transparent - opaque,
    histogram,
    coverage: sum / (total * 255),
  }
}
