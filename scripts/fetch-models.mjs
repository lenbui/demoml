#!/usr/bin/env node
/**
 * Tải sẵn model từ Hugging Face Hub về `public/models/` để chạy OFFLINE.
 *
 * Dùng khi nào?
 *   Phòng máy 30 chỗ cùng tải một model 67MB = 2GB qua đường truyền chung.
 *   Thay vào đó: giảng viên chạy script này một lần, copy folder public/models
 *   vào máy sinh viên (hoặc đặt trên server LAN), rồi bật chế độ offline bằng
 *   cách tạo file .env.local với nội dung: VITE_LOCAL_MODELS=true
 *
 * Cách dùng:
 *   npm run fetch-models
 *   npm run fetch-models -- Xenova/all-MiniLM-L6-v2 q8
 *   npm run fetch-models -- Xenova/gpt2 tokenizer      (chỉ tải file tokenizer)
 *
 * Model encoder-decoder (Whisper, T5, BART) cần NHIỀU file .onnx và tên file
 * không theo mẫu 'model_*.onnx'. Với chúng, khai báo thẳng bằng `onnxFiles`:
 *   { model: 'Xenova/whisper-tiny',
 *     onnxFiles: ['encoder_model_quantized.onnx', 'decoder_model_merged_quantized.onnx'] }
 * Cách biết cần những file nào: chạy demo một lần ở chế độ online, mở
 * DevTools > Network, lọc ".onnx".
 */
import { mkdir, writeFile, stat } from 'node:fs/promises'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..')
const OUT_DIR = join(ROOT, 'public', 'models')
const SAMPLE_DIR = join(ROOT, 'public', 'samples')

/**
 * GIỮ ĐỒNG BỘ VỚI MODEL_REGISTRY trong src/lib/modelRegistry.ts.
 * Mỗi khi thêm demo mới, thêm một dòng ở đây.
 */
const DEFAULT_MODELS = [
  // Tokenizer Explorer — chỉ cần file tokenizer, đặt tokenizerOnly để bỏ qua .onnx
  { model: 'Xenova/bert-base-uncased', tokenizerOnly: true },
  { model: 'Xenova/bert-base-multilingual-cased', tokenizerOnly: true },
  { model: 'Xenova/gpt2', tokenizerOnly: true },
  { model: 'Xenova/xlm-roberta-base', tokenizerOnly: true },
  { model: 'Xenova/t5-small', tokenizerOnly: true },
  // Sentiment
  { model: 'Xenova/distilbert-base-uncased-finetuned-sst-2-english', dtype: 'q8' },
  // Embedding & Semantic Search
  { model: 'Xenova/multilingual-e5-small', dtype: 'q8' },
  { model: 'Xenova/all-MiniLM-L6-v2', dtype: 'q8' },
  // Fill-mask — CÙNG repo với entry tokenizerOnly ở trên, nhưng lần này cần cả
  // trọng số. File tokenizer đã có sẵn nên chỉ tải thêm phần .onnx.
  { model: 'Xenova/bert-base-uncased', dtype: 'q8' },
  // Zero-shot (NLI)
  { model: 'Xenova/mobilebert-uncased-mnli', dtype: 'q8' },
  // NER
  { model: 'Xenova/bert-base-NER', dtype: 'q8' },
  // Reranker (cross-encoder)
  { model: 'Xenova/ms-marco-MiniLM-L-6-v2', dtype: 'q8' },
  // Thị giác máy tính
  { model: 'Xenova/vit-base-patch16-224', dtype: 'q8' },
  { model: 'Xenova/clip-vit-base-patch32', dtype: 'q8' },
  { model: 'Xenova/yolos-tiny', dtype: 'q8' },
  { model: 'Xenova/detr-resnet-50', dtype: 'q8' },
  // Tách nền — hai variant CỐ Ý dùng hai dtype khác nhau, xem ghi chú ở entry
  // 'background-removal' trong src/lib/modelRegistry.ts. Sửa dtype ở một trong
  // hai chỗ mà quên chỗ kia thì chế độ offline sẽ thiếu đúng file .onnx cần dùng.
  { model: 'Xenova/modnet', dtype: 'fp32' },
  { model: 'briaai/RMBG-1.4', dtype: 'q8' },
  // Âm thanh
  //
  // Whisper là model ENCODER-DECODER: nó cần HAI file onnx thay vì một, và tên
  // file không theo mẫu 'model_*.onnx'. Dùng `onnxFiles` để khai báo thẳng.
  // Danh sách này lấy từ DevTools > Network khi chạy demo ở chế độ online.
  {
    model: 'Xenova/whisper-tiny',
    onnxFiles: ['encoder_model_quantized.onnx', 'decoder_model_merged_quantized.onnx'],
  },
  { model: 'Xenova/ast-finetuned-audioset-10-10-0.4593', dtype: 'q8' },
  { model: 'Xenova/mms-tts-vie', dtype: 'q8' },
  { model: 'Xenova/mms-tts-eng', dtype: 'q8' },
  // Sinh văn bản — decoder-only, cũng dùng file 'decoder_model_merged_*' chứ
  // không phải 'model_*'. Repo gpt2 đã có entry tokenizerOnly ở trên; entry này
  // tải thêm phần trọng số.
  {
    model: 'Xenova/gpt2',
    onnxFiles: ['decoder_model_merged_quantized.onnx'],
  },
  {
    model: 'Xenova/distilgpt2',
    onnxFiles: ['decoder_model_merged_quantized.onnx'],
  },
]

/**
 * Ảnh mẫu cho các demo thị giác.
 *
 * GIỮ ĐỒNG BỘ với SAMPLE_IMAGES trong src/lib/samples.ts.
 *
 * Cần tải về vì cùng một lý do với model: ở chế độ offline thì không có
 * Internet để lấy ảnh, mà bắt cả lớp tự đi tìm ảnh trước khi bấm được nút Chạy
 * thì buổi thực hành mất mười lăm phút đầu.
 */
const SAMPLE_IMAGES = [
  'tiger.jpg',
  'corgi.jpg',
  'butterfly.jpg',
  'cats.jpg',
  'football-match.jpg',
  'city-streets.jpg',
  'beach.png',
  'pikachu.png',
  'portrait-of-woman_small.jpg',
  // Âm thanh — GIỮ ĐỒNG BỘ với SAMPLE_AUDIO trong src/lib/samples.ts.
  // Cố ý không lấy ted_talk.wav (162 MB): bản ted_60_16k.wav dài 60 giây là đủ
  // để minh hoạ cửa sổ 30 giây của Whisper mà chỉ nặng 1.9 MB.
  'jfk.wav',
  'french-audio.wav',
  'japanese-audio.wav',
  'ted_60_16k.wav',
  'cat_meow.wav',
  'dog_barking.wav',
  'courtroom.wav',
]

const SAMPLE_BASE =
  'https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/'

/** dtype trong ModelSpec -> tên file .onnx tương ứng trên Hub. */
const WEIGHTS_FILE = {
  fp32: 'model.onnx',
  fp16: 'model_fp16.onnx',
  q8: 'model_quantized.onnx',
  int8: 'model_int8.onnx',
  uint8: 'model_uint8.onnx',
  q4: 'model_q4.onnx',
  q4f16: 'model_q4f16.onnx',
  bnb4: 'model_bnb4.onnx',
}

/** File luôn cần. */
const REQUIRED_FILES = ['config.json']

/** File có thể không tồn tại tuỳ loại model — thiếu thì bỏ qua, không coi là lỗi. */
const OPTIONAL_FILES = [
  'tokenizer.json',
  'tokenizer_config.json',
  'special_tokens_map.json',
  'vocab.txt',
  'vocab.json',
  'merges.txt',
  'spiece.model',
  'sentencepiece.bpe.model',
  'preprocessor_config.json',
  'generation_config.json',
]

function formatBytes(n) {
  return n < 1024 * 1024 ? `${(n / 1024).toFixed(0)} KB` : `${(n / 1024 / 1024).toFixed(1)} MB`
}

async function exists(path) {
  try {
    await stat(path)
    return true
  } catch {
    return false
  }
}

async function download(model, filePath, { optional = false } = {}) {
  const target = join(OUT_DIR, model, filePath)

  if (await exists(target)) {
    console.log(`  = ${filePath} (đã có, bỏ qua)`)
    return true
  }

  const url = `https://huggingface.co/${model}/resolve/main/${filePath}`
  const response = await fetch(url)

  if (!response.ok) {
    if (optional && response.status === 404) {
      console.log(`  - ${filePath} (không có trên Hub — bỏ qua)`)
      return false
    }
    throw new Error(`HTTP ${response.status} khi tải ${url}`)
  }

  const buffer = Buffer.from(await response.arrayBuffer())
  await mkdir(dirname(target), { recursive: true })
  await writeFile(target, buffer)
  console.log(`  + ${filePath} — ${formatBytes(buffer.byteLength)}`)
  return true
}

async function fetchModel({ model, dtype, tokenizerOnly = false, onnxFiles }) {
  // Ba chế độ: chỉ tokenizer, danh sách file onnx tự khai, hoặc suy ra từ dtype.
  const weights = onnxFiles ?? (tokenizerOnly ? [] : [WEIGHTS_FILE[dtype]])

  if (!tokenizerOnly && !onnxFiles && !weights[0]) {
    throw new Error(
      `dtype không hợp lệ: "${dtype}". Chọn một trong: ${Object.keys(WEIGHTS_FILE).join(', ')}`,
    )
  }

  const mode = tokenizerOnly
    ? 'chỉ tokenizer'
    : onnxFiles
      ? `${onnxFiles.length} file onnx`
      : `dtype=${dtype}`
  console.log(`\n${model}  (${mode})`)

  for (const file of REQUIRED_FILES) {
    await download(model, file)
  }
  for (const file of OPTIONAL_FILES) {
    await download(model, file, { optional: true })
  }

  // Demo mode 'tokenizer' không cần trọng số — bỏ qua file .onnx hàng chục MB.
  if (tokenizerOnly) return

  for (const file of weights) {
    const ok = await download(model, `onnx/${file}`, { optional: true })
    if (!ok) {
      console.log(
        `  ! Không tìm thấy onnx/${file}. Mở https://huggingface.co/${model}/tree/main/onnx\n` +
          `    để xem repo này có những file onnx nào, rồi sửa dtype (hoặc onnxFiles) cho khớp.`,
      )
    }
  }
}

/** Tải ảnh và âm thanh mẫu về public/samples/ để demo chạy được khi offline. */
async function fetchSamples() {
  console.log('\nẢnh và âm thanh mẫu')

  for (const name of SAMPLE_IMAGES) {
    const target = join(SAMPLE_DIR, name)

    if (await exists(target)) {
      console.log(`  = ${name} (đã có, bỏ qua)`)
      continue
    }

    const response = await fetch(`${SAMPLE_BASE}${name}`)
    if (!response.ok) {
      // Thiếu một ảnh mẫu không đáng làm hỏng cả lần chạy — model mới là thứ chính.
      console.log(`  ! ${name} — HTTP ${response.status}, bỏ qua`)
      continue
    }

    const buffer = Buffer.from(await response.arrayBuffer())
    await mkdir(dirname(target), { recursive: true })
    await writeFile(target, buffer)
    console.log(`  + ${name} — ${formatBytes(buffer.byteLength)}`)
  }
}

async function main() {
  const args = process.argv.slice(2)
  const targets =
    args.length >= 1
      ? [
          args[1] === 'tokenizer'
            ? { model: args[0], tokenizerOnly: true }
            : { model: args[0], dtype: args[1] ?? 'q8' },
        ]
      : DEFAULT_MODELS

  console.log(`Tải model về: ${OUT_DIR}`)

  for (const target of targets) {
    await fetchModel(target)
  }

  // Chỉ tải ảnh mẫu khi chạy không tham số (tức là tải trọn bộ cho lớp học).
  // Gọi có tham số nghĩa là đang lấy lẻ một model, không cần đụng tới ảnh.
  if (args.length === 0) {
    await fetchSamples()
  }

  console.log(
    '\nXong. Tạo file .env.local ở thư mục gốc với nội dung:\n' +
      '  VITE_LOCAL_MODELS=true\n' +
      'rồi chạy lại `npm run dev` để đọc model và file mẫu từ public/.\n' +
      '\nGHI CHÚ — model encoder-decoder (T5, BART, MarianMT) cần nhiều file onnx. Khai báo\n' +
      'bằng `onnxFiles` như entry Whisper ở đầu file. Cách biết cần file nào: chạy demo một\n' +
      'lần ở chế độ online, mở DevTools > Network, lọc ".onnx".',
  )
}

main().catch((err) => {
  console.error(`\nLỗi: ${err.message}`)
  process.exit(1)
})
