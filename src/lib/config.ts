/**
 * Cấu hình đọc từ biến môi trường.
 *
 * File này CỐ Ý không import '@huggingface/transformers'. UI cần biết đang ở chế
 * độ offline hay không, nhưng nếu UI import env.ts (file có import transformers)
 * thì cả thư viện ~1 MB bị kéo vào bundle chính thay vì chỉ nằm trong worker.
 */

/** true = đọc model từ public/models/ thay vì tải từ huggingface.co. */
export const USE_LOCAL_MODELS = import.meta.env.VITE_LOCAL_MODELS === 'true'

/** Đường dẫn tới model cục bộ, tính theo `base` trong vite.config.ts. */
export const LOCAL_MODEL_PATH = `${import.meta.env.BASE_URL}models/`

/** Đường dẫn tới ảnh mẫu cục bộ cho các demo thị giác. Xem lib/samples.ts. */
export const LOCAL_SAMPLE_PATH = `${import.meta.env.BASE_URL}samples/`
