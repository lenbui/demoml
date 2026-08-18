import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  // Deploy lên GitHub Pages dưới subfolder thì đổi thành '/ten-repo/'.
  // Giữ dạng đường dẫn tuyệt đối (không dùng './') để env.localModelPath còn đúng.
  base: '/',
  plugins: [react()],
  worker: {
    // Worker viết bằng ES module (dùng `import` bên trong worker).
    format: 'es',
  },
  server: {
    // Hai header này bật SharedArrayBuffer -> onnxruntime-web dùng được WASM multi-thread
    // (nhanh hơn đáng kể khi không có WebGPU, tức là trên phần lớn máy phòng lab).
    // 'credentialless' cho phép vẫn tải model từ huggingface.co mà không cần CORP header.
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'credentialless',
    },
  },
})
