import type { DemoDefinition } from '../types'
import { AudioClassificationDemo } from './AudioClassificationDemo'

const SNIPPET = `import { pipeline } from '@huggingface/transformers'

const classify = await pipeline(
  'audio-classification',
  'Xenova/ast-finetuned-audioset-10-10-0.4593',
  { dtype: 'q8' },
)

// Cũng như Whisper: phải tự giải mã + resample về 16 kHz mono ở main thread,
// vì Web Audio API không tồn tại trong Web Worker. Xem src/lib/audio.ts.
await classify(samples, { top_k: 5 })
// [ { label: 'Speech',       score: 0.9421 },
//   { label: 'Male speech',  score: 0.8933 },
//   { label: 'Narration',    score: 0.7712 }, ... ]

// ── AST = Audio Spectrogram Transformer, hiểu theo nghĩa đen ──────────
//   âm thanh -> log-mel spectrogram, ma trận [128, 1024]
//              (trục dọc = tần số theo thang mel, trục ngang = thời gian)
//   ma trận đó -> cắt thành ô 16x16 -> mỗi ô là một token
//   -> chạy đúng kiến trúc ViT của card Phân loại ảnh
//
// AST còn khởi tạo từ trọng số một ViT đã học ImageNet: model "nhìn ảnh"
// được dùng lại để "nghe". Transformer là kiến trúc chung, không gắn với
// riêng loại dữ liệu nào.

// ── ⚠️ Chỗ vênh đã ĐO ĐƯỢC giữa cách huấn luyện và cách hậu xử lý ─────
// AudioSet là cây phân cấp, các nhãn KHÔNG loại trừ nhau. Tiếng mèo kêu
// đồng thời là Meow, Cat, Domestic animals, Animal — cả bốn cùng đúng.
// Model cũng được huấn luyện đúng vậy: binary cross-entropy, sigmoid từng
// nhãn độc lập.
//
// Nhưng đo thực tế trên cat_meow.wav:
//   Meow 56.1% · Cat 22.8% · Domestic animals 11.1% · Animal 9.0% · ...
//   tổng = 1.00  ->  pipeline đã áp SOFTMAX lên cả 527 nhãn.
//
// Hậu quả: bốn nhãn cùng đúng phải CHIA NHAU một phần trăm duy nhất, nên
// "Meow" chỉ được 56% — không phải vì model lưỡng lự, mà vì bước hậu xử lý
// buộc chúng giành nhau.
//
// Muốn đúng với cách model được huấn luyện thì phải lấy logits thô rồi tự
// sigmoid từng nhãn:
//
//   import { AutoModelForAudioClassification, AutoProcessor } from '@huggingface/transformers'
//   const { logits } = await model(await processor(samples))
//   const probs = logits.tolist()[0].map((z) => 1 / (1 + Math.exp(-z)))
//   // giờ mỗi nhãn độc lập, tổng KHÔNG bằng 1
//
// Bài học: hậu xử lý là một LỰA CHỌN nằm ngoài model, và chọn sai thì
// không có gì báo lỗi — kết quả vẫn là những con số trông rất hợp lý.`

export const audioClassificationDemo: DemoDefinition = {
  id: 'audio-classification',
  title: 'Nhận diện âm thanh (AST)',
  subtitle:
    'Biến âm thanh thành ảnh spectrogram rồi chạy đúng kiến trúc Vision Transformer lên trên — và phơi ra chỗ vênh giữa cách model được huấn luyện với cách pipeline hậu xử lý.',
  tagline: 'Nghe bằng cách nhìn spectrogram',
  concepts: [
    'Log-mel spectrogram',
    'AST = ViT cho âm thanh',
    'Đa nhãn vs softmax',
    'Hậu xử lý là lựa chọn ngoài model',
    'AudioSet 527 nhãn',
    'Chuyển giao kiến trúc',
  ],
  group: 'audio',
  Component: AudioClassificationDemo,
  snippet: SNIPPET,
  order: 42,
}
