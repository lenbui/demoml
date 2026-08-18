import type { DemoDefinition } from '../types'
import { SentimentDemo } from './SentimentDemo'

const SNIPPET = `import {
  AutoTokenizer,
  AutoModelForSequenceClassification,
} from '@huggingface/transformers'

const model_id = 'Xenova/distilbert-base-uncased-finetuned-sst-2-english'

const tokenizer = await AutoTokenizer.from_pretrained(model_id)
const model = await AutoModelForSequenceClassification.from_pretrained(
  model_id,
  { dtype: 'q8' },   // lượng tử hoá 8-bit: 268MB -> 67MB
)

// 1. Text -> số. inputs.input_ids là Tensor int64 shape [1, seq_len]
const inputs = tokenizer('The lab sessions were a waste of time.')

// 2. Forward pass. logits là Tensor shape [1, num_labels]
const { logits } = await model(inputs)
const z = logits.tolist()[0]          // ví dụ: [ 3.4162, -2.8975 ]

// 3. Softmax (model KHÔNG tự làm bước này)
const max = Math.max(...z)
const exps = z.map((v) => Math.exp(v - max))
const sum = exps.reduce((a, b) => a + b, 0)
const probs = exps.map((e) => e / sum)  // [ 0.9982, 0.0018 ]

console.log(model.config.id2label)      // { '0': 'NEGATIVE', '1': 'POSITIVE' }

// ── Cách ngắn: pipeline() gói cả 3 bước trên ──────────────────────────
// import { pipeline } from '@huggingface/transformers'
// const classify = await pipeline('text-classification', model_id)
// await classify('The lab sessions were a waste of time.')
// -> [ { label: 'NEGATIVE', score: 0.9982 } ]
//
// Bản dài ở trên được dùng trong demo này vì nó phơi ra logits thô —
// thứ pipeline() đã softmax sẵn và che đi.`

export const sentimentDemo: DemoDefinition = {
  id: 'sentiment',
  title: 'Phân loại cảm xúc',
  subtitle:
    'Chạy DistilBERT fine-tune trên SST-2 ngay trong browser, và phơi ra toàn bộ chuỗi biến đổi: text → token → logits → softmax → quyết định.',
  tagline: 'Từ logits đến xác suất, từng bước một',
  concepts: [
    'Logits vs xác suất',
    'Softmax',
    'Ngưỡng & Precision–Recall',
    'Forward pass',
    'Argmax',
    'Entropy / độ tin cậy',
    'Overconfidence & calibration',
    'Quantization (q8)',
    'Giới hạn của dữ liệu huấn luyện',
  ],
  group: 'text',
  Component: SentimentDemo,
  snippet: SNIPPET,
  order: 10,
}
