import type { DemoDefinition } from '../types'
import { LlmDemo } from './LlmDemo'

const SNIPPET = `import { pipeline, TextStreamer } from '@huggingface/transformers'

const generate = await pipeline('text-generation', 'Xenova/gpt2', {
  dtype: 'q8',
})

// ── Streaming: chữ hiện dần vì model THẬT SỰ sinh từng token một ──────
const streamer = new TextStreamer(generate.tokenizer, {
  skip_prompt: true,
  skip_special_tokens: true,
  callback_function: (text) => process.stdout.write(text),
})

await generate('The capital of France is', {
  max_new_tokens: 40,
  do_sample: true,
  temperature: 0.7,
  top_k: 50,
  top_p: 0.9,
  streamer,
})

// ── Model KHÔNG chọn token. Nó chỉ trả logits ─────────────────────────
// Chạy một forward pass rồi tự xem phân phối:
const inputs = generate.tokenizer('The capital of France is')
const { logits } = await generate.model(inputs)   // [1, seq_len, 50257]

// Chỉ hàng CUỐI có nghĩa — đó là dự đoán cho token tiếp theo.
// Các hàng trước là dự đoán cho những token đã biết rồi.
const [, seqLen, vocab] = logits.dims
const row = Array.from({ length: vocab }, (_, i) =>
  logits.data[(seqLen - 1) * vocab + i],
)

// ── Toàn bộ temperature / top-k / top-p là HẬU XỬ LÝ ──────────────────
// Chúng biến đổi chính dãy logits trên, bên ngoài model:
//
//   logits --chia T--> softmax --cắt top-k--> cắt top-p --> bốc thăm
//
// Nên đổi tham số KHÔNG cần chạy lại model — đó là điều demo này chứng
// minh bằng cách cho kéo slider trên một lần chạy duy nhất.
// Xem applySampling() trong src/lib/sampling.ts.

const T = 0.7
const scaled = row.map((z) => z / T)          // nhiệt độ tác động TRƯỚC softmax
const max = Math.max(...scaled)
const exps = scaled.map((z) => Math.exp(z - max))
const sum = exps.reduce((a, b) => a + b, 0)
const probs = exps.map((e) => e / sum)

// top-p (nucleus): giữ token cao nhất tới khi xác suất DỒN đủ p.
// Thích ứng được — model chắc chắn thì giữ 1-2 token, lưỡng lự thì giữ hàng chục.
// top-k thì cứng nhắc: luôn đúng k token bất kể phân phối nhọn hay phẳng.

// ── GPT-2 là BASE model, chưa instruction-tune ────────────────────────
// Nó NỐI TIẾP văn bản chứ không trả lời. Đưa vào "Question: ... Answer:"
// nó sẽ viết thêm câu hỏi khác. Chính chỗ đó giải thích vì sao cần bước
// instruction tuning — thứ mà một model chat đã che mất.`

export const llmDemo: DemoDefinition = {
  id: 'llm',
  title: 'Sinh văn bản & Sampling',
  subtitle:
    'Chạy GPT-2 ngay trong trình duyệt, xem phân phối xác suất cho token kế tiếp, và kéo temperature / top-k / top-p để thấy chúng biến đổi phân phối đó — mà không phải chạy lại model.',
  tagline: 'Model không chọn token, nó chỉ trả logits',
  concepts: [
    'Dự đoán token kế tiếp',
    'Temperature',
    'top-k vs top-p (nucleus)',
    'Sinh tự hồi quy & streaming',
    'Base model vs instruction tuning',
    'Chưng cất (DistilGPT-2)',
  ],
  group: 'llm',
  Component: LlmDemo,
  snippet: SNIPPET,
  order: 50,
}
