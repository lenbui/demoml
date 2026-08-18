/**
 * Ảnh mẫu cho các demo thị giác máy tính.
 *
 * Giải quyết đúng bài toán mà env.ts giải cho model: ở chế độ mặc định thì tải
 * từ Hugging Face, ở chế độ offline thì đọc từ thư mục cục bộ. Nhờ vậy buổi
 * thực hành không mạng vẫn có ảnh để bấm thử ngay, không bắt mọi người phải tự
 * tìm ảnh.
 *
 *   npm run fetch-models       # tải cả model lẫn ảnh mẫu về public/
 *   VITE_LOCAL_MODELS=true     # trong .env.local -> đọc từ public/samples/
 *
 * File này không import '@huggingface/transformers' (xem lý do ở lib/config.ts).
 */
import { LOCAL_SAMPLE_PATH, USE_LOCAL_MODELS } from './config'

/** Repo dataset chính thức mà tài liệu Transformers.js dùng làm ảnh mẫu. */
const REMOTE_BASE = 'https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/'

/**
 * Mọi ảnh mẫu được dùng trong dashboard.
 *
 * GIỮ ĐỒNG BỘ với mảng SAMPLE_IMAGES trong scripts/fetch-models.mjs — thêm ảnh
 * ở đây thì thêm cả ở đó, nếu không chế độ offline sẽ thiếu ảnh.
 */
export const SAMPLE_IMAGES = [
  'tiger.jpg',
  'corgi.jpg',
  'butterfly.jpg',
  'cats.jpg',
  'football-match.jpg',
  'city-streets.jpg',
  'beach.png',
  'pikachu.png',
  // Ảnh chân dung có tóc bay — trường hợp mà demo Tách nền cần: rìa tóc là chỗ
  // duy nhất thấy rõ mặt nạ alpha là số thực trong [0,1] chứ không phải 0/1.
  'portrait-of-woman_small.jpg',
] as const

export type SampleImage = (typeof SAMPLE_IMAGES)[number]

/**
 * Âm thanh mẫu cho các demo audio.
 *
 * Cũng GIỮ ĐỒNG BỘ với scripts/fetch-models.mjs.
 *
 * Cố ý KHÔNG lấy ted_talk.wav (162 MB) — bản ted_60_16k.wav dài 60 giây là đủ
 * để minh hoạ cửa sổ 30 giây của Whisper mà chỉ nặng 1.9 MB.
 */
export const SAMPLE_AUDIO = [
  'jfk.wav',
  'french-audio.wav',
  'japanese-audio.wav',
  'ted_60_16k.wav',
  'cat_meow.wav',
  'dog_barking.wav',
  'courtroom.wav',
] as const

export type SampleAudio = (typeof SAMPLE_AUDIO)[number]

/** URL thực tế của một ảnh mẫu, tuỳ chế độ online/offline. */
export function sampleImageUrl(name: SampleImage): string {
  return USE_LOCAL_MODELS ? `${LOCAL_SAMPLE_PATH}${name}` : `${REMOTE_BASE}${name}`
}

/** URL thực tế của một file âm thanh mẫu, tuỳ chế độ online/offline. */
export function sampleAudioUrl(name: SampleAudio): string {
  return USE_LOCAL_MODELS ? `${LOCAL_SAMPLE_PATH}${name}` : `${REMOTE_BASE}${name}`
}

export interface SampleChoice {
  name: SampleImage
  label: string
  /** Vì sao ảnh này được chọn — hiện khi hover, và là lý do dạy học. */
  hint?: string
}

export interface AudioChoice {
  name: SampleAudio
  label: string
  hint?: string
}
