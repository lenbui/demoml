import type { DemoDefinition } from '../types'
import { AsrDemo } from './AsrDemo'

const SNIPPET = `import { pipeline } from '@huggingface/transformers'

const transcribe = await pipeline(
  'automatic-speech-recognition',
  'Xenova/whisper-tiny',
  { dtype: 'q8' },   // tải HAI file: encoder + decoder_model_merged
)

// ── Âm thanh phải được giải mã TRƯỚC, ở main thread ───────────────────
// Web Audio API không tồn tại trong Web Worker, nên pipeline audio nhận
// thẳng Float32Array chứ không nhận URL như pipeline ảnh.
const context = new AudioContext()
const decoded = await context.decodeAudioData(await blob.arrayBuffer())

// Resample về 16 kHz mono — Whisper KHÔNG tự kiểm tra tần số lấy mẫu.
// Đưa vào 44.1 kHz thì model vẫn chạy, chỉ là nghe nhanh gấp 2,75 lần.
const offline = new OfflineAudioContext(1, Math.ceil(decoded.duration * 16000), 16000)
const source = offline.createBufferSource()
source.buffer = decoded
source.connect(offline.destination)
source.start()
const samples = (await offline.startRendering()).getChannelData(0)

await transcribe(samples)
// { text: ' And so my fellow Americans, ask not what your country can do for you...' }

// ── Ngôn ngữ và tác vụ chỉ là TOKEN điều khiển ────────────────────────
// Chuỗi decoder sinh ra bắt đầu bằng:
//   <|startoftranscript|> <|fr|> <|translate|> <|notimestamps|> ...
// Nên cùng một trọng số làm được 99 ngôn ngữ lẫn dịch, không cần đổi model.
await transcribe(samples, { language: 'french', task: 'translate' })

// ── ⚠️ Cửa sổ encoder cố định 30 giây ─────────────────────────────────
// Đoạn dài hơn bị CẮT CỤT và bỏ phần thừa, KHÔNG có cảnh báo nào.
// Văn bản trả về vẫn trông bình thường — chỉ là thiếu mất nửa sau.
await transcribe(longSamples, {
  chunk_length_s: 30,   // cắt thành nhiều cửa sổ 30 giây rồi ghép lại
  stride_length_s: 5,   // chồng lấn 5 giây để từ ở chỗ cắt không bị mất
  return_timestamps: true,
})
// { text: '...', chunks: [ { timestamp: [0, 5.2], text: '...' }, ... ] }

// ── Khác mọi card còn lại: đây là model SINH ──────────────────────────
// Các card khác chọn một nhãn có sẵn. Whisper đẻ ra văn bản từng token,
// mỗi token phụ thuộc mọi token trước đó:
//     P(y) = Π P(y_t | y_<t, audio)
// Hệ quả: một token sai ở đầu kéo lệch cả câu, và model có thể BỊA ra chữ
// không hề có trong âm thanh — nhất là ở đoạn im lặng hoặc nhiễu.`

export const asrDemo: DemoDefinition = {
  id: 'asr',
  title: 'Nhận dạng giọng nói (Whisper)',
  subtitle:
    'Chép lời nói thành văn bản ngay trong trình duyệt, dịch sang tiếng Anh, và xem điều gì xảy ra khi đoạn âm thanh dài hơn cửa sổ 30 giây của model.',
  tagline: 'Model sinh duy nhất của dashboard',
  concepts: [
    'Log-mel spectrogram',
    'Kiến trúc encoder–decoder',
    'Sinh tự hồi quy',
    'Token điều khiển & đa ngữ',
    'Cửa sổ 30 giây & chunking',
    'Tần số lấy mẫu 16 kHz',
  ],
  group: 'audio',
  Component: AsrDemo,
  snippet: SNIPPET,
  order: 40,
}
