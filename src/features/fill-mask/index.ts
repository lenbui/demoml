import type { DemoDefinition } from '../types'
import { FillMaskDemo } from './FillMaskDemo'

const SNIPPET = `import { pipeline } from '@huggingface/transformers'

// bert-base-uncased là BERT GỐC, chưa fine-tune cho task nào.
// Điền chỗ trống chính là mục tiêu pre-training của nó.
const unmask = await pipeline('fill-mask', 'Xenova/bert-base-uncased', {
  dtype: 'q8',
})

await unmask('She works as a [MASK] at the hospital.', { top_k: 5 })
// [ { score: 0.2634, token: 6821, token_str: 'nurse',  sequence: '...' },
//   { score: 0.1287, token: 3460, token_str: 'doctor', sequence: '...' }, ... ]

// ── Vì sao tổng xác suất top-5 không bằng 1 ───────────────────────────
// Softmax chạy trên TOÀN BỘ vocabulary 30.522 token, không phải trên 5.
// Xác suất 8% ở đây là rất cao; 8% trong bài toán 2 nhãn là gần như loại bỏ.

// ── Kiểm tra thiên lệch: đổi đúng một từ, giữ nguyên ngữ cảnh ─────────
await unmask('The man worked as a [MASK].')
await unmask('The woman worked as a [MASK].')
// Chênh lệch không giải thích được bằng ngữ pháp thì đến từ dữ liệu huấn luyện.

// ── ⚠️ Nhiều [MASK]: Transformers.js chỉ điền chỗ trống ĐẦU TIÊN ──────
await unmask('The [MASK] ate the [MASK].')
// -> [ { token_str: 'dog', sequence: 'the dog ate the.' }, ... ]
//    Chỉ MỘT danh sách, và dấu [MASK] thứ hai bị bỏ luôn khỏi câu.
//    Không có lỗi nào được báo. Bản Python điền được mọi vị trí trong cùng
//    một forward pass; bản JS thì không. Đã kiểm chứng, không phải suy đoán.`

export const fillMaskDemo: DemoDefinition = {
  id: 'fill-mask',
  title: 'Điền chỗ trống (Masked LM)',
  subtitle:
    'Chạy BERT gốc — chưa fine-tune — trên đúng bài tập mà nó được huấn luyện: đoán token bị che. Kèm phép thử thiên lệch bằng cách đổi một từ.',
  tagline: 'Bài tập mà BERT thực sự được học',
  concepts: [
    'Masked Language Modeling',
    'Pre-training vs fine-tuning',
    'Softmax trên 30k lớp',
    'Ngữ cảnh hai chiều',
    'Weight tying',
    'Thiên lệch từ dữ liệu',
  ],
  group: 'text',
  Component: FillMaskDemo,
  snippet: SNIPPET,
  order: 8,
}
