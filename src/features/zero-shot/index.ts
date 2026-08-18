import type { DemoDefinition } from '../types'
import { ZeroShotDemo } from './ZeroShotDemo'

const SNIPPET = `import {
  AutoTokenizer,
  AutoModelForSequenceClassification,
} from '@huggingface/transformers'

// Model chỉ biết MỘT task: Natural Language Inference (NLI).
// Nó chưa từng thấy nhãn của bạn.
const model_id = 'Xenova/mobilebert-uncased-mnli'
const tokenizer = await AutoTokenizer.from_pretrained(model_id)
const model = await AutoModelForSequenceClassification.from_pretrained(model_id, {
  dtype: 'q8',
})

const text = 'The new graphics card doubles the frame rate.'
const candidates = ['technology', 'cooking', 'politics']

// 1. Mỗi nhãn -> một giả thuyết. Đây là toàn bộ "phép màu" của zero-shot.
const hypotheses = candidates.map((label) => \`This example is \${label}.\`)

// 2. Ghép cặp: [CLS] text [SEP] hypothesis [SEP]
const inputs = tokenizer(
  candidates.map(() => text),          // premise, lặp lại
  { text_pair: hypotheses, padding: true, truncation: true },
)
const { logits } = await model(inputs)   // [3 nhãn, 3 lớp NLI]

// 3. Tìm cột ENTAILMENT theo TÊN — thứ tự nhãn tuỳ từng model!
//    mobilebert-mnli: [ENTAILMENT, NEUTRAL, CONTRADICTION]
//    bart-large-mnli: thứ tự NGƯỢC LẠI. Hardcode index = sai thầm lặng.
const id2label = model.config.id2label
const entail = Object.keys(id2label).find((k) => /entail/i.test(id2label[k]))

// 4. Softmax logit entailment NGANG các nhãn (chế độ single-label)
const z = logits.tolist().map((row) => row[Number(entail)])
const max = Math.max(...z)
const exps = z.map((v) => Math.exp(v - max))
const sum = exps.reduce((a, b) => a + b, 0)
console.log(candidates.map((c, i) => [c, exps[i] / sum]))
// [ ['technology', 0.94], ['cooking', 0.03], ['politics', 0.03] ]

// ── Cách ngắn: pipeline() gói cả 4 bước ───────────────────────────────
// const classify = await pipeline('zero-shot-classification', model_id)
// await classify(text, candidates, { hypothesis_template: 'This example is {}.' })
//
// Bản dài ở trên được dùng trong demo này để thấy các cặp câu THẬT SỰ được
// đưa vào model — thứ mà pipeline() dựng ngầm rồi xoá đi.`

export const zeroShotDemo: DemoDefinition = {
  id: 'zero-shot',
  title: 'Zero-shot Classification',
  subtitle:
    'Phân loại theo bộ nhãn bạn tự nghĩ ra, không huấn luyện gì — bằng cách biến mỗi nhãn thành một câu giả thuyết và hỏi model NLI xem nó có suy ra được hay không.',
  tagline: 'Nhãn tuỳ ý, không cần fine-tune',
  concepts: [
    'NLI (entailment)',
    'Hypothesis template',
    'Zero-shot vs fine-tuning',
    'Cặp câu & token_type_ids',
    'single-label vs multi-label',
    'Chi phí N nhãn = N forward pass',
  ],
  group: 'text',
  Component: ZeroShotDemo,
  snippet: SNIPPET,
  order: 12,
}
