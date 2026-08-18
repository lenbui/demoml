import type { DemoDefinition } from '../types'
import { ImageClassificationDemo } from './ImageClassificationDemo'

const SNIPPET = `import { pipeline } from '@huggingface/transformers'

const classify = await pipeline(
  'image-classification',
  'Xenova/vit-base-patch16-224',
  { dtype: 'q8' },
)

// pipeline() nhận URL, data URL, hoặc Blob — ảnh không cần rời khỏi máy.
await classify(imageUrl, { top_k: 5 })
// [ { label: 'tiger, Panthera tigris', score: 0.9384 },
//   { label: 'tiger cat',              score: 0.0537 }, ... ]

// ── Bước bị che: tiền xử lý ảnh ───────────────────────────────────────
// preprocessor_config.json quy định model nhận ĐÚNG 224x224. Mọi ảnh đều bị:
//   1. thu nhỏ cạnh ngắn về 224 rồi cắt giữa  -> mất phần rìa
//   2. chuẩn hoá từng kênh màu: x' = (x - mean) / std
//   3. cắt thành (224/16)^2 = 196 ô vuông 16x16
//
// 196 ô + 1 token [CLS] = chuỗi 197 token đi vào Transformer.
// Đúng cơ chế của BERT, chỉ khác cách cắt đầu vào: ô ảnh thay vì subword.

// ── Giới hạn quan trọng: tập nhãn ĐÓNG ────────────────────────────────
// Softmax chạy trên đúng 1000 lớp ImageNet và luôn cộng thành 1.
// Đưa ảnh Pikachu vào -> vẫn ra một nhãn kèm một con số, vì không tồn tại
// lớp "không biết". Card CLIP giải đúng vấn đề này: nhãn do bạn gõ lúc chạy.`

export const imageClassificationDemo: DemoDefinition = {
  id: 'image-classification',
  title: 'Phân loại ảnh (ViT)',
  subtitle:
    'Vision Transformer cắt ảnh thành 196 ô vuông rồi xử lý y như một câu văn — và chỉ được chọn trong đúng 1000 nhãn có sẵn.',
  tagline: 'Ảnh cũng bị cắt thành token',
  concepts: [
    'Vision Transformer',
    'Patch = token của ảnh',
    'Tiền xử lý ảnh',
    'ImageNet-1k',
    'Tập nhãn đóng',
    'Không có lớp “không biết”',
  ],
  group: 'vision',
  Component: ImageClassificationDemo,
  snippet: SNIPPET,
  order: 30,
}
