/**
 * Cấu hình môi trường Transformers.js.
 *
 * Gọi configureTransformersEnv() MỘT LẦN ở đầu worker, trước khi load model.
 *
 * ── Vì sao file này quan trọng với lớp học ──
 * Mặc định model được tải từ huggingface.co. 30 máy trong phòng lab cùng tải
 * một model 67MB là 2GB qua đường truyền chung -> buổi thực hành đứng.
 * Đặt VITE_LOCAL_MODELS=true trong file .env.local để đọc model từ
 * `public/models/` (chạy `npm run fetch-models` để tải sẵn về đó).
 */
import { env } from '@huggingface/transformers'

import { LOCAL_MODEL_PATH, USE_LOCAL_MODELS } from './config'

let configured = false

export function configureTransformersEnv(): void {
  if (configured) return
  configured = true

  if (USE_LOCAL_MODELS) {
    // Chế độ offline: chỉ đọc từ public/models/<org>/<model>/...
    env.allowRemoteModels = false
    env.allowLocalModels = true
    env.localModelPath = LOCAL_MODEL_PATH
  } else {
    // Chế độ mặc định: tải từ Hub, browser cache lại vào Cache Storage
    // nên lần chạy thứ hai gần như tức thì.
    env.allowRemoteModels = true
    env.useBrowserCache = true
  }

  // Không cần cấu hình wasmPaths: Vite tự bundle file .wasm của onnxruntime-web
  // vào assets của app (kiểm chứng bằng cách mở DevTools > Network và xác nhận
  // mọi request đều cùng origin). Nhờ vậy chế độ offline ở trên là offline thật.
}
