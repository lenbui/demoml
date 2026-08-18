import type { DemoDefinition } from '../types'
import { EmbeddingDemo } from './EmbeddingDemo'

const SNIPPET = `import { AutoTokenizer, AutoModel } from '@huggingface/transformers'

const model_id = 'Xenova/paraphrase-multilingual-MiniLM-L12-v2'
const tokenizer = await AutoTokenizer.from_pretrained(model_id)
const model = await AutoModel.from_pretrained(model_id, { dtype: 'q8' })

const passages = [
  'Khi mô hình ghi nhớ từng mẫu trong tập huấn luyện...',
  'Thuật toán đi ngược hướng đạo hàm để hạ dần hàm mất mát...',
]

// padding: true -> đệm cho bằng câu dài nhất, encode cả batch trong 1 lần
const inputs = tokenizer(passages, { padding: true, truncation: true })
const { last_hidden_state } = await model(inputs)   // [2, seq, 384]

// ── Mean pooling CÓ mask. Bỏ mask là lỗi im lặng: câu ngắn bị pha loãng
//    bởi các token [PAD] ─────────────────────────────────────────────────
const [batch, seqLen, dim] = last_hidden_state.dims
const h = last_hidden_state.data
const mask = Array.from(inputs.attention_mask.data, Number)

const vectors = []
for (let b = 0; b < batch; b++) {
  const v = new Array(dim).fill(0)
  let n = 0
  for (let t = 0; t < seqLen; t++) {
    if (mask[b * seqLen + t] === 0) continue      // bỏ qua [PAD]
    n++
    for (let d = 0; d < dim; d++) v[d] += h[(b * seqLen + t) * dim + d]
  }
  // L2 normalize -> cosine similarity thu về phép nhân vô hướng
  const pooled = v.map((x) => x / n)
  const norm = Math.hypot(...pooled)
  vectors.push(pooled.map((x) => x / norm))
}

const cosine = (a, b) => a.reduce((s, x, i) => s + x * b[i], 0)
console.log(cosine(vectors[0], vectors[1]))

// ── Cách ngắn: pipeline() pooling + normalize sẵn ─────────────────────
// const extract = await pipeline('feature-extraction', model_id)
// const out = await extract(passages, { pooling: 'mean', normalize: true })
//
// Bản dài ở trên được dùng trong demo này để đổi được mean <-> CLS pooling
// và để thấy attention_mask thực sự tham gia vào phép tính.`

export const embeddingsDemo: DemoDefinition = {
  id: 'embeddings',
  title: 'Embedding & Semantic Search',
  subtitle:
    'Biến câu thành vector 384 chiều, tìm theo ý nghĩa thay vì theo từ khoá — và đặt cạnh BM25 để thấy khác biệt.',
  tagline: 'Tìm theo ý nghĩa, so trực tiếp với BM25',
  concepts: [
    'Không gian vector',
    'Cosine similarity',
    'Pooling (mean vs CLS)',
    'BM25 vs semantic search',
    'PCA & giảm chiều',
    'Vocabulary đa ngữ',
  ],
  group: 'vector',
  Component: EmbeddingDemo,
  snippet: SNIPPET,
  order: 20,
}
