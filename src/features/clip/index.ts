import type { DemoDefinition } from '../types'
import { ClipDemo } from './ClipDemo'

const SNIPPET = `import { pipeline } from '@huggingface/transformers'

const classify = await pipeline(
  'zero-shot-image-classification',
  'Xenova/clip-vit-base-patch32',
  { dtype: 'q8' },
)

// candidate_labels là tham số VỊ TRÍ thứ hai, không nằm trong options.
await classify(imageUrl, ['a cat', 'a dog', 'Pikachu'], {
  hypothesis_template: 'a photo of a {}',
})
// [ { label: 'Pikachu', score: 0.9820 }, { label: 'a cat', score: 0.0121 }, ... ]

// ── Vì sao gõ nhãn nào cũng được ──────────────────────────────────────
// CLIP có HAI encoder được huấn luyện cùng nhau trên 400 triệu cặp
// (ảnh, chú thích), với mục tiêu contrastive: kéo cặp đúng lại gần, đẩy
// cặp sai ra xa. Cả ảnh lẫn chữ cùng rơi vào MỘT không gian vector.
//
// Nên phân loại ảnh thu về đúng phép tính ở card Embedding:
//     score_k = cos( f_ảnh(I), f_chữ(nhãn_k) )
//
// Nhãn được mã hoá LÚC CHẠY, không khoá cứng lúc huấn luyện như ViT.

// ── Làm tay để thấy rõ hai nhánh ──────────────────────────────────────
// import { AutoProcessor, AutoTokenizer, CLIPModel, RawImage } from '@huggingface/transformers'
// const image = await RawImage.read(imageUrl)
// const { pixel_values } = await processor(image)
// const text = tokenizer(labels.map((l) => \`a photo of a \${l}\`), { padding: true })
// const { logits_per_image } = await model({ ...text, pixel_values })
// -> logits_per_image là cosine đã nhân với nhiệt độ học được, chỉ còn softmax.

// ── Giới hạn: softmax chỉ chạy trên nhãn BẠN đưa vào ──────────────────
// Điểm luôn cộng thành 1, kể cả khi không nhãn nào đúng. Bỏ nhãn đúng ra
// khỏi danh sách -> CLIP vẫn chọn một cái với một con số trông rất tự tin.`

export const clipDemo: DemoDefinition = {
  id: 'clip',
  title: 'CLIP — nhãn ảnh tuỳ ý',
  subtitle:
    'Mã hoá ảnh và chữ vào cùng một không gian vector, rồi phân loại bằng cosine — nên nhãn do bạn gõ ra lúc chạy, không bị khoá cứng lúc huấn luyện.',
  tagline: 'Ảnh và chữ trong cùng một không gian',
  concepts: [
    'Không gian nhúng chung ảnh–chữ',
    'Contrastive learning',
    'Zero-shot nhãn mở',
    'Cosine similarity',
    'Prompt cho ảnh',
    'Tìm kiếm ảnh bằng ngôn ngữ',
  ],
  group: 'vision',
  Component: ClipDemo,
  snippet: SNIPPET,
  order: 32,
}
