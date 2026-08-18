/**
 * Giao thức thông điệp giữa UI (main thread) và Web Worker.
 *
 * Vì sao phải có worker? Inference một model BERT mất vài trăm ms đến vài giây
 * và là tính toán ĐỒNG BỘ trong WASM. Nếu chạy trên main thread thì cả trang
 * đứng cứng (không scroll, không bấm được). Worker chạy ở thread riêng nên UI
 * vẫn mượt và ta hiển thị được progress bar.
 */

/** Thông tin "mở hộp đen" — phần quan trọng nhất về mặt dạy học. */
export interface DebugInfo {
  /** Chuỗi đã được tokenizer cắt thành các subword token. */
  tokens?: string[]
  /** Chỉ số của từng token trong vocabulary. Đây là thứ model thực sự nhận. */
  inputIds?: number[]
  /** 1 = token thật, 0 = padding. */
  attentionMask?: number[]
  /** Shape/dtype của các tensor vào–ra, để sinh viên hiểu [batch, seq_len, hidden]. */
  tensors?: Array<{ name: string; dims: number[]; dtype: string }>
  /** Output thô TRƯỚC hậu xử lý (ví dụ: logits chưa qua softmax). */
  raw?: unknown
  /** Ghi chú do runner sinh ra, hiện dưới dạng bullet list. */
  notes?: string[]
}

export type WorkerRequest =
  | { kind: 'load'; demoId: string; variantId?: string }
  | {
      kind: 'run'
      demoId: string
      requestId: number
      input: unknown
      variantId?: string
      options?: Record<string, unknown>
      /**
       * Tham số VỊ TRÍ chèn giữa `input` và `options` khi gọi pipeline().
       *
       * Cần vì một số task của Transformers.js nhận tham số thứ hai theo vị trí
       * chứ không nằm trong object options:
       *   zero-shot-image-classification : pipe(image, candidate_labels, opts)
       *   zero-shot-classification       : pipe(text,  candidate_labels, opts)
       *   question-answering             : pipe(question, context, opts)
       *
       * Chỉ runner mode 'pipeline' dùng tới. Bỏ trống với mọi mode khác.
       */
      args?: unknown[]
      /**
       * Bật để worker gửi từng token ngay khi model sinh ra, thay vì chờ xong
       * cả đoạn (xem WorkerResponse kind 'token').
       *
       * Với model sinh văn bản, chờ xong mới hiện là trải nghiệm rất tệ: sinh
       * 80 token trên WASM mất hàng chục giây, trong lúc đó màn hình đứng im và
       * người dùng tưởng máy treo. Streaming còn dạy được một điều đúng về LLM:
       * văn bản được đẻ ra TỪNG TOKEN MỘT, không phải xuất hiện cùng lúc.
       */
      stream?: boolean
    }

export type LoadPhase = 'initiate' | 'download' | 'progress' | 'done' | 'ready'

export type WorkerResponse =
  | {
      kind: 'progress'
      demoId: string
      variantId?: string
      phase: LoadPhase
      /** Tên file đang tải, ví dụ 'onnx/model_quantized.onnx'. */
      file?: string
      /** 0–100. */
      progress?: number
      loaded?: number
      total?: number
    }
  | {
      kind: 'loaded'
      demoId: string
      variantId?: string
      /** Thời gian tải + khởi tạo model (ms). */
      ms: number
      /** Backend thực tế đã dùng, sau khi resolve 'auto'. */
      device: 'wasm' | 'webgpu'
      /** Có nội dung khi scaffold tự đổi backend so với khai báo (xem resolveDevice). */
      deviceNote?: string
    }
  | {
      kind: 'result'
      demoId: string
      variantId?: string
      requestId: number
      /** Thời gian inference (ms) — số liệu để sinh viên đưa vào báo cáo. */
      ms: number
      output: unknown
      debug?: DebugInfo
    }
  | {
      /**
       * Một mẩu văn bản vừa được sinh ra. Gửi nhiều lần trước khi 'result' về.
       * Chỉ xuất hiện khi request bật `stream`.
       */
      kind: 'token'
      demoId: string
      variantId?: string
      requestId: number
      /** Phần chữ mới — nối thêm vào cuối, không thay thế. */
      text: string
    }
  | { kind: 'error'; demoId: string; variantId?: string; requestId?: number; message: string }

/** Output của runner mode 'embedding'. */
export interface EmbedOutput {
  /** Một vector đã L2-normalize cho mỗi câu đầu vào. */
  vectors: number[][]
  /** Số chiều của vector (hidden_size của model). */
  dim: number
  /** Số token thật (không tính padding) của từng câu. */
  tokenCounts: number[]
  /** Shape của last_hidden_state, để hiện trong Under the hood. */
  hiddenDims: [number, number, number]
  pooling: 'mean' | 'cls'
}

/**
 * Input của runner mode 'pair-classification'.
 *
 * Model nhận vào MỘT chuỗi ghép: `[CLS] a [SEP] b [SEP]`, kèm token_type_ids để
 * phân biệt đoạn nào là đoạn nào. Hai mảng phải cùng độ dài — phần tử thứ i của
 * `a` ghép với phần tử thứ i của `b` thành cặp thứ i.
 */
export interface PairInput {
  /** Đoạn thứ nhất của mỗi cặp (truy vấn, hoặc premise trong NLI). */
  a: string[]
  /** Đoạn thứ hai của mỗi cặp (tài liệu, hoặc hypothesis trong NLI). */
  b: string[]
}

/** Output của runner mode 'pair-classification'. */
export interface PairScoreOutput {
  /** Nhãn đọc từ `config.id2label`, đã sắp theo index. */
  labels: string[]
  /** `logits[i]` là vector logits THÔ của cặp thứ i. Chưa softmax. */
  logits: number[][]
  /** Số token thật của từng cặp sau khi ghép (đã gồm [CLS] và 2 [SEP]). */
  tokenCounts: number[]
}

/** Output của runner mode 'token-classification'. */
export interface TokenLabelOutput {
  /** Nhãn đọc từ `config.id2label` — với NER là sơ đồ BIO: O, B-PER, I-PER… */
  labels: string[]
  tokens: string[]
  ids: number[]
  /** `logits[t]` là vector logits của token thứ t. Một dự đoán cho MỖI token. */
  logits: number[][]
}

/**
 * Một ứng viên cho token kế tiếp, kèm logit THÔ.
 *
 * Cố ý trả về logit chứ không phải xác suất: nhiệt độ (temperature) tác động
 * lên logit *trước* khi softmax, nên UI phải có logit mới tính lại được mà không
 * phải chạy lại model. Xem src/lib/sampling.ts.
 */
export interface NextTokenCandidate {
  id: number
  /** Chuỗi đã decode, giữ nguyên khoảng trắng đầu token. */
  token: string
  logit: number
}

/** Output của runner mode 'text-generation' khi chạy ở chế độ phân phối. */
export interface NextTokenOutput {
  candidates: NextTokenCandidate[]
  /** Kích thước vocabulary — để nói rõ danh sách trên chỉ là phần đỉnh. */
  vocabSize: number
  /** Số token của prompt sau khi tokenize. */
  promptTokens: number
}

/** Output của runner mode 'text-generation' khi chạy ở chế độ sinh văn bản. */
export interface GenerateOutput {
  /** Phần model sinh thêm, KHÔNG gồm prompt. */
  text: string
  /** Số token thực sự được sinh — để tính token/giây. */
  newTokens: number
  promptTokens: number
}

/** Output của runner mode 'tokenizer'. */
export interface TokenizeOutput {
  tokens: string[]
  ids: number[]
  /** Kết quả decode ngược từ ids — so với input để thấy tokenizer làm mất gì. */
  decoded: string
  /** Tên class tokenizer, ví dụ 'BertTokenizer'. */
  tokenizerClass: string
  /** Thuật toán tokenize, ví dụ 'WordPieceTokenizer' / 'BPE' / 'Unigram'. */
  algorithm: string
  vocabSize?: number
  /** Các token đặc biệt mà tokenizer này khai báo. */
  specialTokens: string[]
  unkToken?: string
}
