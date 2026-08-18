/** Máy có WebGPU không? Quyết định demo nào chạy được. */
export const HAS_WEBGPU = typeof navigator !== 'undefined' && 'gpu' in navigator

/**
 * WASM multi-thread chỉ bật được khi trang có COOP/COEP (xem vite.config.ts).
 * Không có nó thì ORT chạy đơn luồng, chậm hơn khoảng 2–4 lần.
 */
export const HAS_SHARED_ARRAY_BUFFER = typeof SharedArrayBuffer !== 'undefined'

export const CPU_THREADS = typeof navigator !== 'undefined' ? (navigator.hardwareConcurrency ?? 0) : 0
