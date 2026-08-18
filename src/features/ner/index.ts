import type { DemoDefinition } from '../types'
import { NerDemo } from './NerDemo'

const SNIPPET = `import {
  AutoTokenizer,
  AutoModelForTokenClassification,
} from '@huggingface/transformers'

const model_id = 'Xenova/bert-base-NER'
const tokenizer = await AutoTokenizer.from_pretrained(model_id)
const model = await AutoModelForTokenClassification.from_pretrained(model_id, {
  dtype: 'q8',
})

const inputs = tokenizer('Tim Cook will visit Ho Chi Minh City.')
const { logits } = await model(inputs)

// Khác sequence-classification: logits có THÊM chiều độ dài câu
console.log(logits.dims)          // [1, 11, 9] = [batch, seq_len, num_labels]
console.log(model.config.id2label)
// { 0:'O', 1:'B-MISC', 2:'I-MISC', 3:'B-PER', 4:'I-PER',
//   5:'B-ORG', 6:'I-ORG', 7:'B-LOC', 8:'I-LOC' }

// 1. Softmax trên chiều CUỐI — 9 nhãn của MỘT token cạnh tranh nhau.
//    Chạy sai chiều vẫn ra số hợp lệ, tổng vẫn bằng 1, và không báo lỗi.
const rows = logits.tolist()[0]
const labels = rows.map((z) => {
  const max = Math.max(...z)
  const exps = z.map((v) => Math.exp(v - max))
  const sum = exps.reduce((a, b) => a + b, 0)
  const probs = exps.map((e) => e / sum)
  const best = probs.indexOf(Math.max(...probs))
  return { label: model.config.id2label[best], score: probs[best] }
})

const tokens = tokenizer.model.convert_ids_to_tokens(
  Array.from(inputs.input_ids.data, Number),
)
// [ '[CLS]', 'Tim', 'Cook', ..., 'Ho', 'Chi', 'Minh', 'City', '.', '[SEP]' ]
// [  —      I-PER   I-PER          I-LOC I-LOC I-LOC  I-LOC   O     —     ]

// 2. Gộp BIO + dán subword lại — xem mergeEntities() trong src/lib/ner.ts.
//    Chú ý: CoNLL-2003 dùng quy ước IOB1, phần lớn entity BẮT ĐẦU bằng 'I-'.
//    Áp dụng đúng lý thuyết IOB2 ("phải có B- mới mở") sẽ mất gần hết entity.

// ── Cách ngắn: pipeline() ─────────────────────────────────────────────
// const ner = await pipeline('token-classification', model_id)
// await ner('Tim Cook will visit Ho Chi Minh City.')
//
// Bản dài ở trên được dùng trong demo này vì bước gộp là một HEURISTIC —
// đổi quy tắc thì kết quả đổi theo, nên nó đáng được đọc thay vì bị chôn
// trong tham số aggregation_strategy.`

export const nerDemo: DemoDefinition = {
  id: 'ner',
  title: 'Nhận diện thực thể (NER)',
  subtitle:
    'Gán nhãn cho từng token thay vì cho cả câu, rồi tự gộp các mảnh subword lại thành tên người, tổ chức và địa điểm.',
  tagline: 'Một nhãn cho mỗi token, rồi gộp lại',
  concepts: [
    'Token classification',
    'Sơ đồ BIO / IOB1',
    'Gộp subword thành entity',
    'Softmax đúng chiều tensor',
    'Aggregation là heuristic',
    'Ngữ cảnh vs từ điển',
  ],
  group: 'text',
  Component: NerDemo,
  snippet: SNIPPET,
  order: 15,
}
