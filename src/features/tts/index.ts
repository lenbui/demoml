import type { DemoDefinition } from '../types'
import { TtsDemo } from './TtsDemo'

const SNIPPET = `import { pipeline } from '@huggingface/transformers'

// MMS-TTS của Meta: mỗi ngôn ngữ một model VITS riêng, mỗi cái ~38 MB.
// Hiếm khi có model TTS tiếng Việt chạy được trong browser — đây là một.
const speak = await pipeline('text-to-speech', 'Xenova/mms-tts-vie', {
  dtype: 'q8',
})

const output = await speak('Xin chào, đây là mô hình chạy trong trình duyệt.')
// { audio: Float32Array(38400), sampling_rate: 16000 }

// ── Model trả về DÃY SỐ, không trả về file ────────────────────────────
// Đúng thứ mà card Whisper NHẬN VÀO: biên độ [-1, 1], 16.000 số mỗi giây.
// Hai card là hai chiều ngược nhau của cùng một biểu diễn.
//
// Thẻ <audio> không phát được dãy số, nên phải tự đóng gói. WAV là lựa
// chọn hiển nhiên: 44 byte header rồi tới mẫu, không nén, không thư viện.
// Xem encodeWav() trong src/lib/wav.ts — toàn bộ đặc tả nằm trong 44 byte đó.

// ── VITS gộp hai tầng cổ điển vào một mạng ────────────────────────────
// Hệ TTS cổ điển tách rời:
//   acoustic model : văn bản -> mel spectrogram
//   vocoder        : mel spectrogram -> sóng âm
//
// Đó là lý do SpeechT5 cần BA file onnx (encoder + decoder + postnet/vocoder,
// ~178 MB) và thêm một file speaker embedding rời:
//
//   const tts = await pipeline('text-to-speech', 'Xenova/speecht5_tts')
//   await tts('Hello', { speaker_embeddings: '<url>/speaker_embeddings.bin' })
//
// VITS huấn luyện cả hai tầng cùng lúc, đối kháng với discriminator, nên
// chỉ còn MỘT mạng đi thẳng từ chữ ra sóng.

// ── Tokenize mức KÝ TỰ, không phải subword ────────────────────────────
// Vì đầu ra là âm thanh nên thứ model cần là chuỗi âm vị, và ký tự là xấp
// xỉ gần nhất. Với chính tả đều đặn như tiếng Việt thì xấp xỉ đó khá tốt.
//
// Cứ hai ký tự lại có một TOKEN TRỐNG xen vào — bộ dự đoán trường độ dùng
// các ô đó để đặt khoảng chuyển tiếp giữa các âm. Nên số token ≈ 2× số ký tự.

// ── ⚠️ Tokenizer BỎ dấu câu và chữ hoa. Đã kiểm chứng ─────────────────
const ids = Array.from(speak.tokenizer('Xin chào, đây là mô hình.').input_ids.data, Number)
const toks = ids.map((id) => speak.tokenizer.decode([id]))
toks.filter((_, i) => i % 2 === 1).join('')
// 'xin chào đây là mô hình'   <- mất dấu phẩy, dấu chấm, và chữ hoa
//
// Nghĩa là model KHÔNG có cách nào biết chỗ nào nên ngắt nhịp: mọi khoảng
// nghỉ nghe thấy đều do nó tự suy ra từ chuỗi ký tự. Đúng bài học của card
// Tokenizer Explorer, lần này thể hiện ở đầu ra âm thanh.

// ── ⚠️ Model KHÔNG chuẩn hoá văn bản ──────────────────────────────────
// '1.250.000' và '15%' sẽ không được đọc thành chữ. Chuẩn hoá số, ngày
// tháng, viết tắt là một bước RIÊNG phải làm trước khi đưa vào model.`

export const ttsDemo: DemoDefinition = {
  id: 'tts',
  title: 'Tổng hợp giọng nói (TTS)',
  subtitle:
    'Biến văn bản thành sóng âm ngay trong trình duyệt, kể cả tiếng Việt — chiều ngược lại của card Whisper, trên đúng cùng một dạng dữ liệu.',
  tagline: 'Chữ thành sóng âm, có tiếng Việt',
  concepts: [
    'Tổng hợp tiếng nói',
    'VITS end-to-end vs vocoder rời',
    'Tokenize mức ký tự',
    'Sóng âm & định dạng WAV',
    'Chuẩn hoá văn bản',
    'Một model cho mỗi ngôn ngữ',
  ],
  group: 'audio',
  Component: TtsDemo,
  snippet: SNIPPET,
  order: 45,
}
