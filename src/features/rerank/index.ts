import type { DemoDefinition } from '../types'
import { RerankDemo } from './RerankDemo'

const SNIPPET = `import {
  AutoTokenizer,
  AutoModelForSequenceClassification,
} from '@huggingface/transformers'

const model_id = 'Xenova/ms-marco-MiniLM-L-6-v2'
const tokenizer = await AutoTokenizer.from_pretrained(model_id)
const model = await AutoModelForSequenceClassification.from_pretrained(model_id, {
  dtype: 'q8',
})

const query = 'how do I know my model just memorised?'
const candidates = [
  'Dropout randomly switches off a fraction of units at each step...',
  'When a model commits every training example to memory...',
]

// Điểm mấu chốt: truy vấn và tài liệu được GHÉP thành một chuỗi
//   [CLS] query [SEP] document [SEP]
// nên mọi token của truy vấn attend được tới mọi token của tài liệu.
const inputs = tokenizer(
  candidates.map(() => query),
  { text_pair: candidates, padding: true, truncation: true },
)
const { logits } = await model(inputs)

// num_labels = 1 -> KHÔNG softmax. Một logit duy nhất là điểm liên quan,
// là số thực bất kỳ (thường −11..+11), số âm là bình thường.
console.log(model.config.id2label)   // { '0': 'LABEL_0' }
const scores = logits.tolist().map((row) => row[0])

const ranked = candidates
  .map((text, i) => ({ text, score: scores[i] }))
  .sort((a, b) => b.score - a.score)

// ── Vì sao KHÔNG dùng cái này cho cả corpus ───────────────────────────
// Điểm phụ thuộc cả CẶP, nên không precompute được gì: đổi truy vấn là
// chạy lại toàn bộ N forward pass. Chi phí tuyến tính theo số tài liệu.
//
// Bi-encoder (card Embedding) thì f(document) không phụ thuộc truy vấn:
//   score = cos(f(query), f(document))
// -> mã hoá tài liệu MỘT lần, lưu vào vector database, tìm bằng nhân ma trận.
//
// Nên hệ thống thật xếp hai tầng:
//   retrieve (bi-encoder / BM25): 1 triệu -> 50 ứng viên, vài ms
//   rerank   (cross-encoder)    : xếp lại 50 ứng viên đó, vài trăm ms
//
// Giới hạn phải nhớ: reranker KHÔNG THỂ tìm ra tài liệu mà tầng retrieve
// không đưa cho nó. Recall của tầng đầu là trần của cả hệ thống.`

export const rerankDemo: DemoDefinition = {
  id: 'rerank',
  title: 'Reranker (Cross-encoder)',
  subtitle:
    'Chấm điểm lại từng cặp (truy vấn, tài liệu) bằng một model đọc cả hai cùng lúc — chính xác hơn embedding, nhưng phải trả giá bằng một forward pass cho mỗi tài liệu.',
  tagline: 'Xếp lại kết quả, và cái giá phải trả',
  concepts: [
    'Bi-encoder vs cross-encoder',
    'Kiến trúc retrieve → rerank',
    'Cặp câu & token_type_ids',
    'num_labels = 1 (hồi quy)',
    'Chi phí tuyến tính',
    'Trần recall',
  ],
  group: 'vector',
  Component: RerankDemo,
  snippet: SNIPPET,
  order: 25,
}
