/**
 * WEB WORKER DÙNG CHUNG cho mọi demo.
 *
 * Mỗi card trên dashboard tạo một worker riêng (xem hook useModel). Worker đọc
 * ModelSpec từ MODEL_REGISTRY, load model, rồi phục vụ các lần `run`.
 *
 * Sinh viên thường KHÔNG cần sửa file này: chỉ cần khai báo model trong
 * modelRegistry.ts với mode 'pipeline'. Chỉ khi cần output thô đặc biệt (như
 * logits) thì mới viết thêm một runner ở phần RUNNERS bên dưới.
 */
import {
  pipeline,
  AutoTokenizer,
  AutoModel,
  AutoModelForSequenceClassification,
  AutoModelForTokenClassification,
  TextStreamer,
} from '@huggingface/transformers'

import { configureTransformersEnv } from '../lib/env'
import { l2Normalize, poolHiddenStates, type Pooling } from '../lib/embedding'
import {
  getModelSpec,
  resolveDtype,
  resolveVariant,
  type Device,
  type ModelSpec,
  type ModelVariant,
} from '../lib/modelRegistry'
import type { DebugInfo, PairInput, WorkerRequest, WorkerResponse } from './protocol'

configureTransformersEnv()

/**
 * Trong module worker, `self` là DedicatedWorkerGlobalScope. Ta không bật lib
 * "WebWorker" trong tsconfig (nó xung đột với lib "DOM"), nên cast một lần ở đây
 * để có postMessage đúng kiểu.
 */
const ctx = self as unknown as { postMessage(message: WorkerResponse): void }
const post = (message: WorkerResponse) => ctx.postMessage(message)

// ─────────────────────────────────────────────────────────────────────────────
// Tiện ích
// ─────────────────────────────────────────────────────────────────────────────

/** Các dtype lượng tử hoá số nguyên. */
const INTEGER_QUANT: ReadonlyArray<NonNullable<ModelSpec['dtype']>> = ['q8', 'int8', 'uint8']

/**
 * Chọn backend thật sự sẽ chạy.
 *
 * ⚠️ CÁI BẪY QUAN TRỌNG NHẤT CỦA SCAFFOLD NÀY:
 * Trọng số lượng tử hoá số nguyên (q8/int8/uint8) chạy trên WebGPU execution
 * provider cho ra kết quả SAI mà KHÔNG báo lỗi — model vẫn trả về logits, chỉ là
 * logits rác. Đã kiểm chứng với distilbert-sst-2 dtype q8:
 *     WASM  : POSITIVE 99.89%   (đúng)
 *     WebGPU: NEGATIVE 97.60%   (sai)
 * Nguyên nhân: WebGPU EP chưa hỗ trợ đầy đủ các toán tử QuantizeLinear /
 * MatMulInteger, và nó lặng lẽ tính sai thay vì báo lỗi.
 *
 * Vì một lỗi âm thầm là thứ tệ nhất trong công cụ dạy học (sinh viên tin vào kết
 * quả sai), scaffold tự hạ về WASM trong trường hợp này và báo cho UI biết.
 * Muốn dùng WebGPU thì phải đổi dtype sang 'fp32' hoặc 'fp16'.
 */
function resolveDevice(
  spec: ModelSpec,
  variant: ModelVariant,
): { device: 'wasm' | 'webgpu'; note?: string } {
  const hasWebGPU = 'gpu' in navigator && Boolean((navigator as { gpu?: unknown }).gpu)
  const want: Device = spec.device ?? 'auto'
  // Phải là dtype của VARIANT, không phải của spec: demo Tách nền chạy MODNet ở
  // fp32 và RMBG ở q8 trong cùng một spec, nên hai variant phải ra hai kết luận
  // khác nhau về WebGPU.
  const dtype = resolveDtype(spec, variant)

  const device: 'wasm' | 'webgpu' =
    want === 'wasm' ? 'wasm' : want === 'webgpu' ? 'webgpu' : hasWebGPU ? 'webgpu' : 'wasm'

  if (device === 'webgpu' && !hasWebGPU) {
    return { device: 'wasm', note: 'Máy không có WebGPU nên đã chuyển sang WASM (CPU).' }
  }

  if (device === 'webgpu' && dtype && INTEGER_QUANT.includes(dtype)) {
    return {
      device: 'wasm',
      note:
        `dtype '${dtype}' (lượng tử hoá số nguyên) cho kết quả SAI trên WebGPU mà không ` +
        'báo lỗi, nên đã tự chuyển sang WASM. Muốn chạy WebGPU thì đổi dtype sang fp32 hoặc fp16.',
    }
  }

  return { device }
}

/**
 * Tensor của Transformers.js: `.data` là TypedArray, `.dims` là shape, `.type` là dtype.
 * input_ids có dtype int64 -> `.data` là BigInt64Array, phải Number() lại.
 */
function tensorToNumbers(tensor: { data: ArrayLike<number | bigint> }): number[] {
  return Array.from(tensor.data, (v) => Number(v))
}

/** Đổi id token về dạng chuỗi đọc được ('##ing', '[CLS]', ...). */
function idsToTokens(tokenizer: any, ids: number[]): string[] {
  try {
    return tokenizer.model.convert_ids_to_tokens(ids)
  } catch {
    // Fallback: decode từng id một. Chậm hơn nhưng luôn hoạt động.
    return ids.map((id) => tokenizer.decode([id], { skip_special_tokens: false }))
  }
}

/**
 * Input có phải là một tấm ảnh không?
 *
 * Cần vì task ảnh cũng nhận input dạng string (URL hoặc data URL), mà model như
 * CLIP thì vẫn CÓ tokenizer — nếu không phân biệt, phần Under the hood sẽ đem
 * chính chuỗi URL đi tokenize và hiện ra một dãy token vô nghĩa.
 */
function isImageInput(input: string): boolean {
  return /^(data:image\/|blob:|https?:\/\/)/i.test(input)
}

/** Sinh phần "Under the hood" cho input dạng văn bản. */
function inspectText(tokenizer: any, text: string): DebugInfo {
  const encoded = tokenizer(text)
  const inputIds = tensorToNumbers(encoded.input_ids)
  return {
    tokens: idsToTokens(tokenizer, inputIds),
    inputIds,
    attentionMask: encoded.attention_mask ? tensorToNumbers(encoded.attention_mask) : undefined,
  }
}

/**
 * Đọc danh sách nhãn từ `config.id2label`, sắp đúng theo index.
 *
 * Phải đọc từ config chứ KHÔNG được hardcode: thứ tự nhãn là tuỳ từng model.
 * Ví dụ mobilebert-uncased-mnli dùng [ENTAILMENT, NEUTRAL, CONTRADICTION], còn
 * bart-large-mnli dùng thứ tự NGƯỢC LẠI. Hardcode index sẽ cho kết quả sai mà
 * không báo lỗi.
 */
function readLabels(net: any): string[] {
  const id2label: Record<string, string> = net.config?.id2label ?? { 0: 'LABEL_0' }
  return Object.keys(id2label)
    .sort((a, b) => Number(a) - Number(b))
    .map((k) => id2label[k])
}

/** Cắt một mảng phẳng thành từng hàng độ dài `width`. */
function reshapeRows(flat: number[], width: number): number[][] {
  const rows: number[][] = []
  for (let i = 0; i < flat.length; i += width) rows.push(flat.slice(i, i + width))
  return rows
}

/** Đọc kích thước vocabulary — cấu trúc khác nhau tuỳ thuật toán tokenize. */
function readVocabSize(tokenizer: any): number | undefined {
  const vocab = tokenizer.model?.vocab
  if (Array.isArray(vocab)) return vocab.length
  if (vocab instanceof Map) return vocab.size
  if (vocab && typeof vocab === 'object') return Object.keys(vocab).length
  return undefined
}

// ─────────────────────────────────────────────────────────────────────────────
// RUNNERS — mỗi mode trong ModelSpec ứng với một hàm tạo runner
// ─────────────────────────────────────────────────────────────────────────────

interface Runner {
  run(
    input: unknown,
    options?: Record<string, unknown>,
    /** Tham số vị trí chèn giữa input và options — chỉ mode 'pipeline' dùng. */
    args?: unknown[],
    /** Gửi một mẩu văn bản về UI ngay lập tức. Chỉ có khi request bật `stream`. */
    emit?: (text: string) => void,
  ): Promise<{ output: unknown; debug?: DebugInfo }>
}

type ProgressCallback = (info: any) => void

/** Cách phổ biến: để pipeline() lo hết. Dùng cho gần như mọi task. */
async function createPipelineRunner(
  spec: ModelSpec,
  variant: ModelVariant,
  device: 'wasm' | 'webgpu',
  onProgress: ProgressCallback,
): Promise<Runner> {
  // Cast `any`: chữ ký của pipeline() là union rất lớn theo từng task, còn ở đây
  // task chỉ biết lúc chạy. Chỗ code sinh viên đọc (snippet trong mỗi demo) vẫn
  // gọi pipeline() với task cụ thể nên vẫn có type đầy đủ.
  const pipe: any = await (pipeline as any)(spec.task, variant.model, {
    dtype: resolveDtype(spec, variant),
    device,
    progress_callback: onProgress,
  })

  const tokenizer = pipe.tokenizer

  return {
    async run(input, options, args) {
      // Một số task nhận tham số thứ hai theo VỊ TRÍ (candidate_labels của CLIP,
      // context của question-answering), không nằm trong object options.
      const output = args?.length
        ? await pipe(input, ...args, options ?? {})
        : await pipe(input, options ?? {})

      const debug: DebugInfo = {
        notes: [`pipeline('${spec.task}') tự làm 3 việc: tiền xử lý → forward → hậu xử lý.`],
      }
      // Với task văn bản, phơi thêm bước tokenize cho sinh viên xem.
      // Task ảnh không có tokenizer cho input, nên nhánh này tự bỏ qua.
      if (tokenizer && typeof input === 'string' && !isImageInput(input)) {
        Object.assign(debug, inspectText(tokenizer, input))
      }
      return { output, debug }
    },
  }
}

/**
 * Mode 'sequence-classification': tự tokenize + tự forward.
 *
 * Dài hơn pipeline() nhưng đổi lại lấy được LOGITS THÔ — nhờ đó card mẫu dạy
 * được chuỗi: text → input_ids → logits → softmax → xác suất → quyết định.
 */
async function createSequenceClassificationRunner(
  spec: ModelSpec,
  variant: ModelVariant,
  device: 'wasm' | 'webgpu',
  onProgress: ProgressCallback,
): Promise<Runner> {
  const [tokenizer, model] = await Promise.all([
    AutoTokenizer.from_pretrained(variant.model, { progress_callback: onProgress }),
    AutoModelForSequenceClassification.from_pretrained(variant.model, {
      dtype: resolveDtype(spec, variant),
      device,
      progress_callback: onProgress,
    }),
  ])

  const tok = tokenizer as any
  const net = model as any

  // id2label nằm trong config.json của model, ví dụ {0: 'NEGATIVE', 1: 'POSITIVE'}.
  const id2label: Record<string, string> = net.config?.id2label ?? { 0: 'LABEL_0', 1: 'LABEL_1' }
  const labels = Object.keys(id2label)
    .sort((a, b) => Number(a) - Number(b))
    .map((k) => id2label[k])

  return {
    async run(input) {
      const text = String(input ?? '')
      const encoded = tok(text)
      const { logits } = await net(encoded)

      const logitValues = tensorToNumbers(logits)
      const inputIds = tensorToNumbers(encoded.input_ids)

      return {
        // Chú ý: trả về LOGITS, không phải xác suất. Softmax được tính ở phía UI
        // (src/lib/math.ts) để sinh viên đọc được công thức.
        output: { labels, logits: logitValues },
        debug: {
          tokens: idsToTokens(tok, inputIds),
          inputIds,
          attentionMask: encoded.attention_mask ? tensorToNumbers(encoded.attention_mask) : undefined,
          tensors: [
            { name: 'input_ids', dims: encoded.input_ids.dims, dtype: encoded.input_ids.type },
            ...(encoded.attention_mask
              ? [
                  {
                    name: 'attention_mask',
                    dims: encoded.attention_mask.dims,
                    dtype: encoded.attention_mask.type,
                  },
                ]
              : []),
            { name: 'logits', dims: logits.dims, dtype: logits.type },
          ],
          raw: { logits: logitValues, id2label },
          notes: [
            `Tokenizer cắt câu thành ${inputIds.length} token (đã gồm [CLS] và [SEP]).`,
            `logits có shape [${logits.dims.join(', ')}] = [batch_size, num_labels].`,
            'Logits là số thực bất kỳ, chưa phải xác suất — cần softmax.',
          ],
        },
      }
    },
  }
}

/**
 * Mode 'pair-classification': chấm điểm cho từng CẶP câu (a, b).
 *
 * Khác biệt cốt lõi so với mode 'embedding': ở đây hai câu được GHÉP thành một
 * chuỗi `[CLS] a [SEP] b [SEP]` rồi đi qua model cùng nhau, nên mọi token của a
 * đều attend được tới mọi token của b. Đó là lý do cross-encoder chính xác hơn
 * bi-encoder — và cũng là lý do nó không cache được: đổi truy vấn là phải chạy
 * lại toàn bộ N cặp.
 *
 * Một runner phục vụ hai demo khác nhau, chỉ khác số nhãn của model:
 *   • reranker (num_labels = 1) : logits là điểm liên quan, không phải xác suất
 *   • NLI      (num_labels = 3) : entailment / neutral / contradiction
 *                                 -> đây chính là cách zero-shot classification
 *                                    hoạt động, xem demo zero-shot.
 */
async function createPairClassificationRunner(
  spec: ModelSpec,
  variant: ModelVariant,
  device: 'wasm' | 'webgpu',
  onProgress: ProgressCallback,
): Promise<Runner> {
  const [tokenizer, model] = await Promise.all([
    AutoTokenizer.from_pretrained(variant.model, { progress_callback: onProgress }),
    AutoModelForSequenceClassification.from_pretrained(variant.model, {
      dtype: resolveDtype(spec, variant),
      device,
      progress_callback: onProgress,
    }),
  ])

  const tok = tokenizer as any
  const net = model as any
  const labels = readLabels(net)

  return {
    async run(input) {
      const { a, b } = (input ?? {}) as PairInput
      if (!Array.isArray(a) || !Array.isArray(b) || a.length !== b.length) {
        throw new Error(
          "Runner 'pair-classification' cần input dạng { a: string[], b: string[] } với hai mảng cùng độ dài.",
        )
      }
      if (a.length === 0) {
        return { output: { labels, logits: [], tokenCounts: [] } }
      }

      // text_pair là tham số làm nên chuỗi ghép [CLS] a [SEP] b [SEP].
      const encoded = tok(a, { text_pair: b, padding: true, truncation: true })
      const { logits } = await net(encoded)

      const [batch, numLabels] = logits.dims as [number, number]
      const rows = reshapeRows(tensorToNumbers(logits), numLabels)

      const seqLen = encoded.input_ids.dims[1] as number
      const mask = tensorToNumbers(encoded.attention_mask)
      const tokenCounts = Array.from({ length: batch }, (_, i) => {
        let count = 0
        for (let t = 0; t < seqLen; t++) if (mask[i * seqLen + t] === 1) count++
        return count
      })

      // Phơi cặp ĐẦU TIÊN ra Under the hood để thấy chuỗi ghép trông thế nào.
      const firstIds = tensorToNumbers(encoded.input_ids).slice(0, tokenCounts[0])

      return {
        output: { labels, logits: rows, tokenCounts },
        debug: {
          tokens: idsToTokens(tok, firstIds),
          inputIds: firstIds,
          tensors: [
            { name: 'input_ids', dims: encoded.input_ids.dims, dtype: encoded.input_ids.type },
            ...(encoded.token_type_ids
              ? [
                  {
                    name: 'token_type_ids',
                    dims: encoded.token_type_ids.dims,
                    dtype: encoded.token_type_ids.type,
                  },
                ]
              : []),
            { name: 'logits', dims: logits.dims, dtype: logits.type },
          ],
          notes: [
            `${batch} cặp câu được ghép thành ${batch} chuỗi rồi chạy trong MỘT batch; mọi chuỗi bị đệm cho bằng ${seqLen} token.`,
            `logits có shape [${logits.dims.join(', ')}] = [số cặp, num_labels]. Nhãn: ${labels.join(' · ')}.`,
            'Chuỗi hiện ở phần Tokenize phía trên là cặp thứ nhất — chú ý hai dấu [SEP].',
            'token_type_ids (segment embedding) là thứ cho model biết token nào thuộc đoạn a, token nào thuộc đoạn b.',
          ],
        },
      }
    },
  }
}

/**
 * Mode 'token-classification': một dự đoán cho MỖI token, thay vì một dự đoán
 * cho cả câu.
 *
 * Trả về logits thô theo từng token (không dùng pipeline) vì hai lý do dạy học:
 *  • thấy được rằng nhãn được gán ở mức token, nên một tên riêng bị tokenizer
 *    cắt thành 3 mảnh sẽ nhận 3 nhãn và phải tự gộp lại;
 *  • bước gộp BIO nằm ở src/lib/ner.ts, đọc được, thay vì bị chôn trong
 *    `aggregation_strategy` của thư viện.
 */
async function createTokenClassificationRunner(
  spec: ModelSpec,
  variant: ModelVariant,
  device: 'wasm' | 'webgpu',
  onProgress: ProgressCallback,
): Promise<Runner> {
  const [tokenizer, model] = await Promise.all([
    AutoTokenizer.from_pretrained(variant.model, { progress_callback: onProgress }),
    AutoModelForTokenClassification.from_pretrained(variant.model, {
      dtype: resolveDtype(spec, variant),
      device,
      progress_callback: onProgress,
    }),
  ])

  const tok = tokenizer as any
  const net = model as any
  const labels = readLabels(net)

  return {
    async run(input) {
      const text = String(input ?? '')
      const encoded = tok(text)
      const { logits } = await net(encoded)

      // [1, seq_len, num_labels] — chiều giữa là thứ phân biệt task này với
      // sequence-classification (chỉ có [1, num_labels]).
      const [, seqLen, numLabels] = logits.dims as [number, number, number]
      const rows = reshapeRows(tensorToNumbers(logits), numLabels)
      const ids = tensorToNumbers(encoded.input_ids)
      const tokens = idsToTokens(tok, ids)

      return {
        output: { labels, tokens, ids, logits: rows },
        debug: {
          tokens,
          inputIds: ids,
          attentionMask: encoded.attention_mask
            ? tensorToNumbers(encoded.attention_mask)
            : undefined,
          tensors: [
            { name: 'input_ids', dims: encoded.input_ids.dims, dtype: encoded.input_ids.type },
            { name: 'logits', dims: logits.dims, dtype: logits.type },
          ],
          notes: [
            `logits có shape [${logits.dims.join(', ')}] = [batch, seq_len, num_labels] — MỘT vector ${numLabels} nhãn cho mỗi token trong ${seqLen} token.`,
            `Sơ đồ nhãn BIO: ${labels.join(' · ')}.`,
            'Softmax được tính theo từng token (trên chiều cuối), rồi các token liền nhau được gộp thành entity ở src/lib/ner.ts.',
          ],
        },
      }
    },
  }
}

/**
 * Mode 'tokenizer': chỉ tải tokenizer, KHÔNG tải trọng số.
 *
 * Nhờ vậy demo nặng 1–9 MB thay vì hàng trăm MB, tải trong 1–2 giây — phù hợp
 * làm demo mở đầu bài giảng.
 */
async function createTokenizerRunner(
  _spec: ModelSpec,
  variant: ModelVariant,
  _device: 'wasm' | 'webgpu',
  onProgress: ProgressCallback,
): Promise<Runner> {
  const tokenizer: any = await AutoTokenizer.from_pretrained(variant.model, {
    progress_callback: onProgress,
  })

  const tokenizerClass = tokenizer.constructor?.name ?? 'unknown'
  const algorithm = tokenizer.model?.constructor?.name ?? 'unknown'
  const vocabSize = readVocabSize(tokenizer)

  const rawSpecials = tokenizer.special_tokens ?? tokenizer.all_special_tokens
  const specialTokens: string[] = Array.isArray(rawSpecials) ? rawSpecials.filter(Boolean) : []

  return {
    async run(input) {
      const text = String(input ?? '')
      const encoded = tokenizer(text)
      const ids = tensorToNumbers(encoded.input_ids)
      const tokens = idsToTokens(tokenizer, ids)

      return {
        output: {
          tokens,
          ids,
          // Decode ngược lại để so với input gốc. Chênh nhau ở đâu chính là
          // thông tin mà tokenizer đã làm mất (chữ hoa, dấu tiếng Việt…).
          decoded: tokenizer.decode(ids, { skip_special_tokens: true }),
          tokenizerClass,
          algorithm,
          vocabSize,
          specialTokens,
          unkToken: tokenizer.unk_token ?? undefined,
        },
        debug: {
          tensors: [
            { name: 'input_ids', dims: encoded.input_ids.dims, dtype: encoded.input_ids.type },
            ...(encoded.attention_mask
              ? [
                  {
                    name: 'attention_mask',
                    dims: encoded.attention_mask.dims,
                    dtype: encoded.attention_mask.type,
                  },
                ]
              : []),
          ],
          notes: [
            `${tokenizerClass} dùng thuật toán ${algorithm}.`,
            'Không có trọng số model nào được tải — đây thuần là bước tiền xử lý.',
          ],
        },
      }
    },
  }
}

/**
 * Mode 'embedding': encode một MẢNG câu trong đúng một forward pass.
 *
 * Vì sao không dùng pipeline('feature-extraction')? Vì nó pooling sẵn bên trong.
 * Ở đây ta lấy last_hidden_state thô rồi gọi poolHiddenStates() từ
 * src/lib/embedding.ts — nhờ vậy dạy được mean vs CLS pooling, và sinh viên đọc
 * được đúng đoạn code làm việc đó.
 */
async function createEmbeddingRunner(
  spec: ModelSpec,
  variant: ModelVariant,
  device: 'wasm' | 'webgpu',
  onProgress: ProgressCallback,
): Promise<Runner> {
  const [tokenizer, model] = await Promise.all([
    AutoTokenizer.from_pretrained(variant.model, { progress_callback: onProgress }),
    AutoModel.from_pretrained(variant.model, {
      dtype: resolveDtype(spec, variant),
      device,
      progress_callback: onProgress,
    }),
  ])

  const tok = tokenizer as any
  const net = model as any

  return {
    async run(input, options) {
      const texts = Array.isArray(input) ? (input as string[]) : [String(input ?? '')]
      const pooling: Pooling = (options?.pooling as Pooling) ?? 'mean'

      // padding: true đệm mọi câu cho bằng câu dài nhất -> một forward pass duy
      // nhất cho cả batch. truncation: true chặn câu vượt max_position_embeddings.
      const encoded = tok(texts, { padding: true, truncation: true })
      const output = await net(encoded)

      const hidden = output.last_hidden_state
      const dims = hidden.dims as [number, number, number]
      const mask = Array.from(encoded.attention_mask.data as ArrayLike<number | bigint>, (v) =>
        Number(v),
      )

      const pooled = poolHiddenStates(hidden.data, dims, mask, pooling)
      // Normalize để cosine similarity thu về đúng một phép nhân vô hướng.
      const vectors = pooled.map(l2Normalize)

      const [batch, seqLen] = dims
      const tokenCounts = Array.from({ length: batch }, (_, b) => {
        let count = 0
        for (let t = 0; t < seqLen; t++) if (mask[b * seqLen + t] === 1) count++
        return count
      })

      return {
        output: { vectors, dim: dims[2], tokenCounts, hiddenDims: dims, pooling },
        debug: {
          tensors: [
            { name: 'input_ids', dims: encoded.input_ids.dims, dtype: encoded.input_ids.type },
            {
              name: 'attention_mask',
              dims: encoded.attention_mask.dims,
              dtype: encoded.attention_mask.type,
            },
            { name: 'last_hidden_state', dims: hidden.dims, dtype: hidden.type },
          ],
          notes: [
            `Encode ${batch} câu trong MỘT forward pass; mọi câu bị đệm cho bằng ${seqLen} token.`,
            `last_hidden_state [${dims.join(', ')}] → pooling (${pooling}) → ${batch} vector ${dims[2]} chiều.`,
            'Vector đã L2-normalize, nên cosine similarity = nhân vô hướng.',
          ],
        },
      }
    },
  }
}

/**
 * Mode 'text-generation': model SINH văn bản, khác mọi mode còn lại vốn chỉ chọn
 * một nhãn có sẵn.
 *
 * Một runner phục vụ HAI chế độ vì cả hai dùng chung một model đã load (model
 * này nặng 85–128 MB, load hai lần là không chấp nhận được):
 *
 *   mode 'distribution' : chạy MỘT forward pass rồi trả logits thô của vị trí
 *                         cuối. UI dùng chúng để dạy temperature/top-k/top-p mà
 *                         không phải chạy lại model — xem src/lib/sampling.ts.
 *   mode 'generate'     : sinh nhiều token, đẩy từng mẩu về UI qua TextStreamer.
 */
async function createTextGenerationRunner(
  spec: ModelSpec,
  variant: ModelVariant,
  device: 'wasm' | 'webgpu',
  onProgress: ProgressCallback,
): Promise<Runner> {
  const pipe: any = await (pipeline as any)('text-generation', variant.model, {
    dtype: resolveDtype(spec, variant),
    device,
    progress_callback: onProgress,
    // Repo xuất từ thời v2 dùng tên file khác — xem modelFileName trong
    // modelRegistry.ts. Hậu tố dtype vẫn được ghép vào phía sau.
    ...(spec.modelFileName ? { model_file_name: spec.modelFileName } : {}),
  })

  const tok = pipe.tokenizer
  const net = pipe.model
  const vocabSize: number = net.config?.vocab_size ?? readVocabSize(tok) ?? 0

  /** Số ứng viên trả về cho bảng phân phối. */
  const TOP_N = 60

  return {
    async run(input, options, _args, emit) {
      const prompt = String(input ?? '')
      const encoded = tok(prompt)
      const promptTokens = encoded.input_ids.dims[1] as number

      // ── Chế độ 1: chỉ lấy phân phối của token kế tiếp ──────────────────
      if (options?.mode === 'distribution') {
        const { logits } = await net(encoded)

        // logits: [1, seq_len, vocab]. Chỉ hàng CUỐI có nghĩa — đó là dự đoán
        // cho token đứng sau prompt. Các hàng trước là dự đoán cho token đã biết.
        const [, seqLen, vocab] = logits.dims as [number, number, number]
        const flat = logits.data as ArrayLike<number>
        const offset = (seqLen - 1) * vocab

        // Chọn top-N bằng một lượt quét, không sort cả 50k phần tử.
        const top: Array<{ id: number; logit: number }> = []
        let smallestKept = -Infinity
        for (let id = 0; id < vocab; id++) {
          const logit = flat[offset + id]
          if (top.length < TOP_N) {
            top.push({ id, logit })
            if (top.length === TOP_N) {
              top.sort((a, b) => b.logit - a.logit)
              smallestKept = top[TOP_N - 1].logit
            }
          } else if (logit > smallestKept) {
            top[TOP_N - 1] = { id, logit }
            top.sort((a, b) => b.logit - a.logit)
            smallestKept = top[TOP_N - 1].logit
          }
        }
        top.sort((a, b) => b.logit - a.logit)

        const candidates = top.map(({ id, logit }) => ({
          id,
          // Không skip special token: nếu model định kết thúc bằng <|endoftext|>
          // thì đó là thông tin đáng thấy, không phải thứ nên giấu đi.
          token: tok.decode([id], { skip_special_tokens: false }),
          logit,
        }))

        return {
          output: { candidates, vocabSize: vocab, promptTokens },
          debug: {
            tokens: idsToTokens(tok, tensorToNumbers(encoded.input_ids)),
            inputIds: tensorToNumbers(encoded.input_ids),
            tensors: [
              { name: 'input_ids', dims: encoded.input_ids.dims, dtype: encoded.input_ids.type },
              { name: 'logits', dims: logits.dims, dtype: logits.type },
            ],
            notes: [
              `logits có shape [${logits.dims.join(', ')}] = [batch, seq_len, vocab_size].`,
              `Chỉ hàng cuối (vị trí ${seqLen - 1}) được dùng — đó là dự đoán cho token TIẾP THEO.`,
              `Bảng chỉ hiện ${TOP_N} ứng viên đầu trong ${vocab.toLocaleString('vi-VN')} token của vocabulary.`,
              'Temperature / top-k / top-p KHÔNG chạy trong model — chúng biến đổi chính dãy logits này, ở phía UI.',
            ],
          },
        }
      }

      // ── Chế độ 2: sinh văn bản, đẩy từng token về UI ───────────────────
      const streamer = emit
        ? new TextStreamer(tok, {
            // Bỏ prompt: UI đã hiện sẵn, đẩy lại là hiện hai lần.
            skip_prompt: true,
            skip_special_tokens: true,
            callback_function: (text: string) => emit(text),
          })
        : undefined

      const output = await pipe(prompt, { ...(options ?? {}), mode: undefined, streamer })
      const generated: string = output?.[0]?.generated_text ?? ''
      // pipeline trả về prompt + phần sinh thêm; UI chỉ cần phần sinh thêm.
      const text = generated.startsWith(prompt) ? generated.slice(prompt.length) : generated
      const newTokens = (tok(text).input_ids.dims[1] as number) ?? 0

      return {
        output: { text, newTokens, promptTokens },
        debug: {
          notes: [
            `Prompt ${promptTokens} token → sinh thêm ~${newTokens} token.`,
            'Mỗi token được sinh ra cần MỘT forward pass, và phụ thuộc mọi token trước đó.',
            `vocab_size = ${vocabSize.toLocaleString('vi-VN')} — mỗi bước là một lần chọn trong ngần ấy khả năng.`,
          ],
        },
      }
    },
  }
}

function createRunner(
  spec: ModelSpec,
  variant: ModelVariant,
  device: 'wasm' | 'webgpu',
  onProgress: ProgressCallback,
): Promise<Runner> {
  switch (spec.mode ?? 'pipeline') {
    case 'sequence-classification':
      return createSequenceClassificationRunner(spec, variant, device, onProgress)
    case 'pair-classification':
      return createPairClassificationRunner(spec, variant, device, onProgress)
    case 'token-classification':
      return createTokenClassificationRunner(spec, variant, device, onProgress)
    case 'text-generation':
      return createTextGenerationRunner(spec, variant, device, onProgress)
    case 'tokenizer':
      return createTokenizerRunner(spec, variant, device, onProgress)
    case 'embedding':
      return createEmbeddingRunner(spec, variant, device, onProgress)
    case 'pipeline':
      return createPipelineRunner(spec, variant, device, onProgress)
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Vòng lặp thông điệp
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Cache theo (demoId, variantId): load một lần, chạy nhiều lần. Nhờ key có
 * variant nên demo so sánh nhiều model đổi qua lại không phải tải lại.
 */
const runners = new Map<string, Promise<Runner>>()

function loadRunner(demoId: string, variantId?: string): Promise<Runner> {
  const spec = getModelSpec(demoId)
  const variant = resolveVariant(spec, variantId)
  const cacheKey = `${demoId}::${variant.id}`

  const cached = runners.get(cacheKey)
  if (cached) return cached

  const { device, note: deviceNote } = resolveDevice(spec, variant)
  const startedAt = performance.now()

  const promise = createRunner(spec, variant, device, (info: any) => {
    post({
      kind: 'progress',
      demoId,
      variantId: variant.id,
      phase: info?.status ?? 'progress',
      file: info?.file,
      progress: typeof info?.progress === 'number' ? info.progress : undefined,
      loaded: info?.loaded,
      total: info?.total,
    })
  })
    .then((runner) => {
      post({
        kind: 'loaded',
        demoId,
        variantId: variant.id,
        ms: Math.round(performance.now() - startedAt),
        device,
        deviceNote,
      })
      return runner
    })
    .catch((err) => {
      // Xoá khỏi cache để lần bấm sau thử lại được (ví dụ mạng vừa hỏng).
      runners.delete(cacheKey)
      throw err
    })

  runners.set(cacheKey, promise)
  return promise
}

self.addEventListener('message', async (event: MessageEvent<WorkerRequest>) => {
  const request = event.data

  try {
    if (request.kind === 'load') {
      await loadRunner(request.demoId, request.variantId)
      return
    }

    const runner = await loadRunner(request.demoId, request.variantId)
    const startedAt = performance.now()

    const emit = request.stream
      ? (text: string) =>
          post({
            kind: 'token',
            demoId: request.demoId,
            variantId: request.variantId,
            requestId: request.requestId,
            text,
          })
      : undefined

    const { output, debug } = await runner.run(
      request.input,
      request.options,
      request.args,
      emit,
    )

    post({
      kind: 'result',
      demoId: request.demoId,
      variantId: request.variantId,
      requestId: request.requestId,
      ms: Math.round(performance.now() - startedAt),
      output,
      debug,
    })
  } catch (err) {
    post({
      kind: 'error',
      demoId: request.demoId,
      variantId: request.variantId,
      requestId: request.kind === 'run' ? request.requestId : undefined,
      message: err instanceof Error ? err.message : String(err),
    })
  }
})
