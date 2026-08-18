/**
 * REGISTRY MODEL — nguồn sự thật duy nhất về việc mỗi demo dùng model nào.
 *
 * File này được import bởi CẢ UI (React) và Web Worker, nên tuyệt đối không
 * import React / không chạm vào DOM ở đây.
 *
 * Sinh viên thêm demo mới = thêm một entry vào MODEL_REGISTRY bên dưới.
 */

/** Backend thực thi. 'auto' = dùng WebGPU nếu máy hỗ trợ, ngược lại rơi về WASM (CPU). */
export type Device = 'auto' | 'wasm' | 'webgpu'

/**
 * Độ chính xác của trọng số. Đây là một khái niệm ML cần dạy:
 *  - fp32 : gốc, chính xác nhất, nặng nhất (4 byte/tham số)
 *  - fp16 : nửa dung lượng, cần WebGPU
 *  - q8   : lượng tử hoá 8-bit, ~1/4 dung lượng, sai số nhỏ  <-- mặc định tốt cho web
 *  - q4   : 4-bit, ~1/8 dung lượng, sai số rõ hơn, thường chỉ dùng cho LLM
 *
 * ⚠️ q8/int8/uint8 KHÔNG dùng được với WebGPU (kết quả sai mà không báo lỗi).
 *    Worker tự phát hiện và hạ về WASM — chi tiết ở resolveDevice() trong
 *    src/workers/pipeline.worker.ts. Muốn chạy WebGPU thì dùng fp32 hoặc fp16.
 */
export type DType = 'fp32' | 'fp16' | 'q8' | 'int8' | 'uint8' | 'q4' | 'q4f16' | 'bnb4'

/**
 * Cách worker chạy model.
 *  - 'pipeline'               : dùng pipeline() có sẵn của Transformers.js. Ngắn, dùng cho hầu hết demo.
 *  - 'sequence-classification': tự tokenize + tự gọi model để LẤY ĐƯỢC LOGITS THÔ.
 *                               Chọn mode này khi muốn dạy logits -> softmax -> xác suất.
 *  - 'tokenizer'              : CHỈ tải tokenizer, không tải trọng số model.
 *                               Nhẹ (~1–5 MB) vì không có phần nặng nhất.
 *  - 'embedding'              : encode nhiều câu một lượt, trả về last_hidden_state
 *                               đã pooling. Pooling làm ở lib/embedding.ts để
 *                               dạy được mean vs CLS.
 *  - 'pair-classification'    : ghép từng CẶP câu thành [CLS] a [SEP] b [SEP] rồi
 *                               chấm điểm. Dùng cho cross-encoder (reranker) và
 *                               cho NLI (nền tảng của zero-shot classification).
 *  - 'token-classification'   : một dự đoán cho MỖI token thay vì cho cả câu.
 *                               Dùng cho NER; trả logits thô để bước gộp BIO
 *                               nằm ở lib/ner.ts, đọc được.
 *  - 'text-generation'        : model SINH văn bản. Hai chế độ dùng chung một
 *                               model đã load: lấy phân phối token kế tiếp
 *                               (options.mode = 'distribution'), hoặc sinh văn
 *                               bản có streaming từng token.
 */
export type RunnerMode =
  | 'pipeline'
  | 'sequence-classification'
  | 'tokenizer'
  | 'embedding'
  | 'pair-classification'
  | 'token-classification'
  | 'text-generation'

/**
 * Một model mà demo có thể chuyển qua lại. Dùng khi demo cần SO SÁNH nhiều
 * model cùng task — ví dụ Tokenizer Explorer so 5 thuật toán tokenize, hoặc một
 * card A/B so distilbert với bert-base để dạy về distillation.
 *
 * Mọi variant của cùng một demo dùng CHUNG một worker; worker cache riêng theo
 * từng variant nên đổi qua lại không phải tải lại.
 */
export interface ModelVariant {
  id: string
  /** Tên ngắn hiện trên nút chọn. */
  label: string
  model: string
  approxSizeMB: number
  /** Ghi chú một dòng, hiện dưới nút chọn (ví dụ: thuật toán tokenize). */
  note?: string
  /**
   * Ghi đè `dtype` của ModelSpec cho RIÊNG variant này.
   *
   * Cần khi hai model cùng task nhưng không chịu chung một mức lượng tử hoá.
   * Ví dụ ở demo Tách nền: RMBG-1.4 bản fp32 nặng 168 MB nên buộc phải dùng q8,
   * còn MODNet chỉ 24.7 MB ở fp32 — mà mặt nạ alpha của nó lại là thứ q8 làm hỏng
   * rõ nhất (rìa tóc bị vỡ thành bậc thang). Ép cả hai về cùng một dtype thì
   * hoặc phải tải 168 MB, hoặc phải chấp nhận mặt nạ xấu.
   */
  dtype?: DType
  /**
   * Model retrieval họ E5 được huấn luyện với hai tiền tố KHÁC NHAU cho truy vấn
   * và tài liệu (asymmetric retrieval). Quên tiền tố thì chất lượng tụt rõ rệt
   * mà không có cảnh báo nào — nên khai báo ở đây để demo tự thêm.
   */
  prefixes?: { query: string; passage: string }
}

export interface ModelSpec {
  /** Phải trùng với `id` của DemoDefinition. */
  id: string
  /** Tên task của Hugging Face, ví dụ 'text-classification', 'object-detection'. */
  task: string
  /** Repo id trên Hugging Face Hub. Bỏ trống nếu dùng `variants`. */
  model?: string
  /** Nhiều model cùng task để so sánh. Phần tử đầu là mặc định. */
  variants?: ModelVariant[]
  /** Mặc định cho mọi variant; từng variant ghi đè được bằng `ModelVariant.dtype`. */
  dtype?: DType
  device?: Device
  mode?: RunnerMode
  /**
   * Tên file .onnx (KHÔNG kèm hậu tố dtype và đuôi .onnx).
   *
   * Mặc định Transformers.js v3 tìm `onnx/model<hậu-tố-dtype>.onnx`, ví dụ
   * `onnx/model_quantized.onnx` với dtype 'q8'. Các repo xuất từ thời v2 lại
   * dùng tên khác — `Xenova/gpt2` chỉ có `onnx/decoder_model_merged_quantized.onnx`
   * chứ không có `model_quantized.onnx`, và lỗi báo ra là "Could not locate file".
   *
   * Khai `modelFileName: 'decoder_model_merged'` thì hậu tố dtype vẫn được ghép
   * vào phía sau như thường.
   */
  modelFileName?: string
  /** Dung lượng xấp xỉ phải tải (MB). Với `variants` thì khai báo trong từng variant. */
  approxSizeMB?: number
  /** true = demo chỉ chạy được khi có WebGPU (thường là các LLM). */
  requiresWebGPU?: boolean
}

export const MODEL_REGISTRY: Record<string, ModelSpec> = {
  tokenizer: {
    id: 'tokenizer',
    task: 'tokenization',
    mode: 'tokenizer',
    // Năm thuật toán tokenize khác nhau. Cho cùng một câu chạy qua cả năm là
    // cách nhanh nhất để thấy tokenizer là một phần của model, không phải một
    // bước tiền xử lý trung tính.
    variants: [
      {
        id: 'bert-en',
        label: 'BERT (en)',
        model: 'Xenova/bert-base-uncased',
        approxSizeMB: 1,
        note: 'WordPiece · uncased · bỏ dấu',
      },
      {
        id: 'bert-multi',
        label: 'BERT (đa ngữ)',
        model: 'Xenova/bert-base-multilingual-cased',
        approxSizeMB: 3,
        note: 'WordPiece · cased · 104 ngôn ngữ',
      },
      {
        id: 'gpt2',
        label: 'GPT-2',
        model: 'Xenova/gpt2',
        approxSizeMB: 2,
        note: 'BPE mức byte · không có token đặc biệt',
      },
      {
        id: 'xlm-r',
        label: 'XLM-RoBERTa',
        model: 'Xenova/xlm-roberta-base',
        // Đo thật: tokenizer.json 16.3 MB — vocabulary 250k token của 100 ngôn ngữ.
        approxSizeMB: 17,
        note: 'SentencePiece Unigram · 100 ngôn ngữ',
      },
      {
        id: 't5',
        label: 'T5',
        model: 'Xenova/t5-small',
        approxSizeMB: 3,
        note: 'SentencePiece · không có [CLS]/[SEP]',
      },
    ],
  },

  embeddings: {
    id: 'embeddings',
    task: 'feature-extraction',
    mode: 'embedding',
    // Đã đo: q8 vs fp32 trên all-MiniLM-L6-v2 cho cosine 0.430 vs 0.450 và thứ
    // hạng y hệt. Lượng tử hoá gần như không ảnh hưởng chất lượng embedding ở
    // đây, nên chọn q8 để tải 22 MB thay vì 86 MB.
    dtype: 'q8',
    device: 'auto',
    // Hai model cùng 384 chiều nên toàn bộ UI/PCA dùng chung được. Chênh lệch
    // dung lượng 22 MB vs 113 MB gần như nằm hết ở ma trận embedding: vocab
    // 30k token (chỉ tiếng Anh) so với 250k token (50 ngôn ngữ).
    variants: [
      {
        id: 'multi',
        label: 'Đa ngữ (E5)',
        model: 'Xenova/multilingual-e5-small',
        approxSizeMB: 113,
        note: '100 ngôn ngữ · 384 chiều',
        prefixes: { query: 'query: ', passage: 'passage: ' },
      },
      {
        id: 'en',
        label: 'Chỉ tiếng Anh',
        model: 'Xenova/all-MiniLM-L6-v2',
        approxSizeMB: 22,
        note: 'tiếng Anh · 384 chiều',
      },
    ],
  },

  sentiment: {
    id: 'sentiment',
    task: 'text-classification',
    model: 'Xenova/distilbert-base-uncased-finetuned-sst-2-english',
    dtype: 'q8',
    // 'auto' + q8: worker sẽ tự hạ về WASM và hiện cảnh báo, vì q8 chạy trên
    // WebGPU cho kết quả sai mà không báo lỗi. Xem resolveDevice() trong
    // src/workers/pipeline.worker.ts.
    device: 'auto',
    // Dùng mode thủ công để phơi ra logits thô cho panel "Under the hood".
    mode: 'sequence-classification',
    approxSizeMB: 67,
  },

  'fill-mask': {
    id: 'fill-mask',
    task: 'fill-mask',
    // BERT gốc, CHƯA fine-tune cho task nào. Đây đúng là thứ cần dùng: mục tiêu
    // pre-training của nó (đoán từ bị che) chính là nội dung demo.
    model: 'Xenova/bert-base-uncased',
    dtype: 'q8',
    device: 'auto',
    // Đo thật: onnx/model_quantized.onnx = 110.8 MB.
    approxSizeMB: 111,
  },

  rerank: {
    id: 'rerank',
    // Cross-encoder được đóng gói như một model phân loại chuỗi, nhưng chỉ có
    // MỘT nhãn (num_labels = 1) nên logit là điểm hồi quy, không phải xác suất.
    task: 'text-classification',
    model: 'Xenova/ms-marco-MiniLM-L-6-v2',
    dtype: 'q8',
    device: 'auto',
    mode: 'pair-classification',
    // Đo thật: 23.1 MB — nhẹ hơn cả model embedding, nhưng chi phí lúc CHẠY thì
    // tuyến tính theo số tài liệu, xem demo.
    approxSizeMB: 23,
  },

  'zero-shot': {
    id: 'zero-shot',
    task: 'zero-shot-classification',
    // Model NLI thuần, không biết gì về nhãn của bạn. Demo tự dựng cặp
    // (câu, "This example is {nhãn}.") rồi đọc xác suất entailment.
    model: 'Xenova/mobilebert-uncased-mnli',
    dtype: 'q8',
    device: 'auto',
    mode: 'pair-classification',
    // Đo thật: 27.0 MB. MobileBERT chỉ 25M tham số.
    approxSizeMB: 27,
  },

  ner: {
    id: 'ner',
    task: 'token-classification',
    model: 'Xenova/bert-base-NER',
    dtype: 'q8',
    device: 'auto',
    mode: 'token-classification',
    // Đo thật: 109.0 MB.
    approxSizeMB: 109,
  },

  'image-classification': {
    id: 'image-classification',
    task: 'image-classification',
    // ViT gốc, huấn luyện trên ImageNet-1k. Tập nhãn ĐÓNG đúng 1000 lớp —
    // chính giới hạn đó là nội dung demo, và là cái cớ để so với CLIP.
    model: 'Xenova/vit-base-patch16-224',
    dtype: 'q8',
    device: 'auto',
    // Đo thật: onnx/model_quantized.onnx = 88.3 MB.
    approxSizeMB: 88,
  },

  clip: {
    id: 'clip',
    task: 'zero-shot-image-classification',
    model: 'Xenova/clip-vit-base-patch32',
    dtype: 'q8',
    device: 'auto',
    // Đo thật: 153.7 MB. Nặng vì repo dùng model_quantized.onnx GỘP cả nhánh
    // ảnh (89 MB) lẫn nhánh văn bản (64 MB) — pipeline cần cả hai để so vector
    // ảnh với vector chữ.
    approxSizeMB: 154,
  },

  detection: {
    id: 'detection',
    task: 'object-detection',
    dtype: 'q8',
    device: 'auto',
    // Hai model cùng task, chênh nhau 4 lần dung lượng — và kết quả đo NGƯỢC
    // với trực giác: model nhỏ hơn lại CHẬM hơn.
    //
    // Đo thật trên WASM, ảnh football-match, cả hai đã warm:
    //     YOLOS-tiny (10 MB) : 7.7 s, 16 hộp ≥ 0.5
    //     DETR-R50   (43 MB) : 4.4 s,  8 hộp ≥ 0.5
    //
    // Lý do: YOLOS là ViT thuần, attention chạy trên rất nhiều patch token nên
    // FLOPs cao dù ít tham số. DETR dùng backbone CNN (ResNet-50) vốn rất hiệu
    // quả, phần transformer phía sau thì nhỏ. Dung lượng file đo SỐ THAM SỐ,
    // không đo KHỐI LƯỢNG TÍNH TOÁN — đây chính là bài học của demo.
    variants: [
      {
        id: 'yolos-tiny',
        label: 'YOLOS-tiny',
        model: 'Xenova/yolos-tiny',
        // Đo thật: 9.7 MB.
        approxSizeMB: 10,
        note: 'ViT thuần · nhiều hộp hơn · chậm hơn',
      },
      {
        id: 'detr',
        label: 'DETR ResNet-50',
        model: 'Xenova/detr-resnet-50',
        // Đo thật: 43.1 MB.
        approxSizeMB: 43,
        note: 'backbone CNN · ít hộp hơn · nhanh hơn',
      },
    ],
  },

  'background-removal': {
    id: 'background-removal',
    // CHÚ Ý: task là 'image-segmentation', KHÔNG phải 'background-removal'.
    //
    // Transformers.js có sẵn task 'background-removal', nhưng nó tự nhân mặt nạ
    // vào ảnh rồi chỉ trả về tấm ảnh đã trong suốt — tức là giấu đi đúng thứ
    // demo này muốn dạy. Dùng 'image-segmentation' để nhận MẶT NẠ ALPHA thô,
    // rồi tự ghép ảnh ở src/lib/matte.ts (đọc được, sinh viên sửa được).
    //
    // Với các model matting này, processor không có post_process_*_segmentation
    // nào nên pipeline chạy nhánh "không subtask": sigmoid → ×255 → resize về
    // đúng kích thước ảnh gốc → RawImage 1 kênh. Đó chính là mặt nạ ta cần.
    task: 'image-segmentation',
    device: 'auto',
    // Hai cách tiếp cận khác nhau cho cùng một việc, và chênh nhau 7 lần dung lượng.
    //
    // MODNet CHUYÊN cho chân dung người: nó không biết "vật thể nổi bật" là gì,
    // nó chỉ biết tìm người. Đổi lại chỉ 24.7 MB và giữ nguyên tỉ lệ ảnh.
    // RMBG-1.4 TỔNG QUÁT: tách bất kỳ chủ thể nổi bật nào, nhưng ép mọi ảnh về
    // đúng 1024×1024 (bóp méo tỉ lệ) và tính toán nặng hơn hẳn.
    //
    // Đo thật trên ảnh portrait-of-woman 360×450, cùng một máy:
    //     MODNet  (fp32, WebGPU) : 1.6 s · 12.2% pixel có alpha lưng chừng
    //     RMBG-1.4 (q8,  WASM)   : 6.0 s ·  2.8% pixel có alpha lưng chừng
    //
    // Hai con số cuối là điểm đáng dạy nhất: MODNet trả về dải chuyển tiếp RỘNG
    // hơn 4 lần quanh rìa tóc, còn RMBG cho rìa gần như nhị phân. Đó không phải
    // "model nào tốt hơn" — đó là hai mục tiêu huấn luyện khác nhau (matting
    // chân dung vs phân đoạn vật thể nổi bật) hiện thẳng ra trong output.
    //
    // Cho ảnh con bướm chạy qua cả hai để thấy MODNet gãy — và gãy im lặng.
    variants: [
      {
        id: 'modnet',
        label: 'MODNet',
        model: 'Xenova/modnet',
        // fp32 chứ không q8: repo tự khai `transformers.js_config.dtype = fp32`,
        // và ở đây đầu ra là ẢNH chứ không phải một nhãn — sai số lượng tử hoá
        // hiện thẳng ra thành rìa răng cưa quanh sợi tóc. 24.7 MB vẫn là model
        // nhẹ nhất trong nhóm thị giác của dashboard.
        dtype: 'fp32',
        approxSizeMB: 25,
        note: 'chuyên chân dung · cạnh ngắn 512 · giữ tỉ lệ',
      },
      {
        id: 'rmbg',
        label: 'RMBG-1.4',
        model: 'briaai/RMBG-1.4',
        // q8 vì bản fp32 nặng 168 MB — quá sức một buổi thực hành. Đo thật:
        // onnx/model_quantized.onnx = 42.3 MB.
        dtype: 'q8',
        approxSizeMB: 42,
        note: 'tổng quát · ép 1024×1024 · nặng hơn',
      },
    ],
  },

  asr: {
    id: 'asr',
    task: 'automatic-speech-recognition',
    // Model ENCODER-DECODER duy nhất của dashboard, và cũng là model SINH duy
    // nhất: nó đẻ ra text từng token một chứ không phân loại. Vì thế nó tải HAI
    // file onnx (encoder + decoder_merged) thay vì một — chi tiết ở
    // scripts/fetch-models.mjs.
    model: 'Xenova/whisper-tiny',
    dtype: 'q8',
    device: 'auto',
    // Đo thật: encoder_model_quantized 10.1 MB + decoder_model_merged_quantized
    // 30.7 MB + tokenizer ≈ 43 MB.
    approxSizeMB: 43,
  },

  'audio-classification': {
    id: 'audio-classification',
    task: 'audio-classification',
    // AST = Audio Spectrogram Transformer. Đúng nghĩa đen là ViT chạy trên ảnh
    // spectrogram — nối thẳng với card Phân loại ảnh.
    model: 'Xenova/ast-finetuned-audioset-10-10-0.4593',
    dtype: 'q8',
    device: 'auto',
    // Đo thật: onnx/model_quantized.onnx = 90.8 MB.
    approxSizeMB: 91,
  },

  tts: {
    id: 'tts',
    task: 'text-to-speech',
    dtype: 'q8',
    device: 'auto',
    // MMS-TTS của Meta: một model VITS riêng cho mỗi ngôn ngữ, nhưng nhẹ nên
    // đổi qua lại được. Chọn họ này thay vì SpeechT5 vì hai lý do:
    //  • CÓ bản tiếng Việt — hiếm, và đúng ngôn ngữ của lớp học;
    //  • một file .onnx duy nhất, trong khi SpeechT5 cần ba (encoder, decoder,
    //    postnet+vocoder ≈ 178 MB) cộng thêm file speaker embedding rời.
    variants: [
      {
        id: 'vie',
        label: 'Tiếng Việt',
        model: 'Xenova/mms-tts-vie',
        // Đo thật: onnx/model_quantized.onnx = 38.4 MB.
        approxSizeMB: 38,
        note: 'VITS · MMS của Meta',
      },
      {
        id: 'eng',
        label: 'Tiếng Anh',
        model: 'Xenova/mms-tts-eng',
        // Đo thật: 38.4 MB.
        approxSizeMB: 38,
        note: 'VITS · cùng kiến trúc',
      },
    ],
  },

  llm: {
    id: 'llm',
    task: 'text-generation',
    mode: 'text-generation',
    dtype: 'q8',
    device: 'auto',
    // Hai repo này xuất từ thời Transformers.js v2 nên không có
    // 'onnx/model_quantized.onnx'. Đã kiểm chứng: thiếu dòng này thì lỗi
    // "Could not locate file: .../onnx/model_quantized.onnx".
    modelFileName: 'decoder_model_merged',
    // GPT-2 chứ không phải một model instruct đời mới, vì ba lý do:
    //  • chạy được trên WASM của máy phòng lab — không cần WebGPU;
    //  • tokenizer đã nằm sẵn trong public/models từ card Tokenizer Explorer;
    //  • nó là BASE model, chưa instruction-tune, nên nó NỐI TIẾP văn bản chứ
    //    không trả lời câu hỏi. Chính điều đó giải thích được vì sao phải có
    //    bước instruction tuning — thứ mà một model đã instruct che mất.
    variants: [
      {
        id: 'gpt2',
        label: 'GPT-2',
        model: 'Xenova/gpt2',
        // Đo thật: onnx/decoder_model_merged_quantized.onnx = 128.3 MB.
        approxSizeMB: 128,
        note: '124M tham số · 12 lớp',
      },
      {
        id: 'distilgpt2',
        label: 'DistilGPT-2',
        model: 'Xenova/distilgpt2',
        // Đo thật: 84.9 MB.
        approxSizeMB: 85,
        note: '82M · 6 lớp · chưng cất từ GPT-2',
      },
    ],
  },

  // ── Sinh viên thêm entry mới ở đây ─────────────────────────────────────────
  // Ví dụ (đang comment, bỏ comment khi làm demo tương ứng):
  //
  // 'zero-shot': {
  //   id: 'zero-shot',
  //   task: 'zero-shot-classification',
  //   model: 'Xenova/mobilebert-uncased-mnli',
  //   dtype: 'q8',
  //   approxSizeMB: 26,
  // },
  //
  // embeddings: {
  //   id: 'embeddings',
  //   task: 'feature-extraction',
  //   model: 'Xenova/all-MiniLM-L6-v2',
  //   dtype: 'q8',
  //   approxSizeMB: 23,
  // },
}

export function getModelSpec(id: string): ModelSpec {
  const spec = MODEL_REGISTRY[id]
  if (!spec) {
    throw new Error(
      `Không tìm thấy ModelSpec cho demo "${id}". Bạn đã thêm entry vào MODEL_REGISTRY chưa?`,
    )
  }
  return spec
}

/**
 * Trả về variant đang dùng. Với spec không có `variants`, gói `spec.model` lại
 * thành một variant giả để phần còn lại của code chỉ cần xử lý một trường hợp.
 */
export function resolveVariant(spec: ModelSpec, variantId?: string): ModelVariant {
  if (spec.variants?.length) {
    return (variantId && spec.variants.find((v) => v.id === variantId)) || spec.variants[0]
  }
  if (!spec.model) {
    throw new Error(`ModelSpec "${spec.id}" phải có 'model' hoặc 'variants'.`)
  }
  return {
    id: 'default',
    label: spec.model,
    model: spec.model,
    approxSizeMB: spec.approxSizeMB ?? 0,
  }
}

/**
 * dtype thực sự dùng cho một variant: variant ghi đè spec.
 *
 * Mọi nơi cần dtype (worker khi load model, và cả bước kiểm tra q8-trên-WebGPU)
 * phải đi qua hàm này — đọc thẳng `spec.dtype` sẽ bỏ sót phần ghi đè và tải
 * nhầm file .onnx.
 */
export function resolveDtype(spec: ModelSpec, variant: ModelVariant): DType | undefined {
  return variant.dtype ?? spec.dtype
}
