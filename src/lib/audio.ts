/**
 * Giải mã âm thanh cho các demo audio.
 *
 * ── VÌ SAO PHẢI LÀM Ở MAIN THREAD ─────────────────────────────────────────
 * Mọi demo khác đưa thẳng input vào worker. Audio thì không được: giải mã file
 * nén (mp3/wav/webm) cần `AudioContext.decodeAudioData`, mà Web Audio API
 * KHÔNG tồn tại trong Web Worker. Nên luồng bắt buộc là:
 *
 *     file/mic ──▶ main thread: decode + resample ──▶ Float32Array ──▶ worker
 *
 * Đây cũng là lý do các pipeline audio của Transformers.js nhận thẳng
 * Float32Array chứ không nhận URL như pipeline ảnh.
 *
 * ── VÌ SAO PHẢI RESAMPLE VỀ 16 kHz ────────────────────────────────────────
 * Whisper và AST đều được huấn luyện trên âm thanh 16 kHz và KHÔNG tự kiểm tra
 * tần số lấy mẫu. Đưa vào 44.1 kHz thì model vẫn chạy, vẫn trả kết quả, chỉ là
 * nó "nghe" mọi thứ nhanh gấp 2.75 lần — giọng nói thành tiếng chíp chíp và kết
 * quả thành rác. Một lỗi im lặng kinh điển của xử lý âm thanh.
 */

/** Tần số lấy mẫu mà Whisper và AST yêu cầu. */
export const TARGET_SAMPLE_RATE = 16000

export interface PreparedAudio {
  /** Dữ liệu thô đưa vào model: biên độ trong [-1, 1], mono, 16 kHz. */
  samples: Float32Array
  /** URL cho thẻ <audio> nghe thử. */
  src: string
  label: string
  /** Độ dài (giây). */
  duration: number
  /** Tần số lấy mẫu GỐC của file, trước khi resample. */
  originalSampleRate: number
  /** Số kênh gốc — stereo bị trộn thành mono. */
  originalChannels: number
}

/**
 * Giải mã một ArrayBuffer âm thanh thành mono 16 kHz.
 *
 * Việc trộn kênh và đổi tần số được giao cho OfflineAudioContext: khai báo
 * destination 1 kênh @ 16 kHz thì trình duyệt tự downmix và resample bằng bộ lọc
 * chất lượng cao của chính nó. Tự viết tay phần này gần như chắc chắn sẽ tạo
 * aliasing.
 */
export async function decodeAudioBuffer(
  data: ArrayBuffer,
): Promise<{ samples: Float32Array; duration: number; sampleRate: number; channels: number }> {
  // AudioContext chỉ dùng để decode rồi đóng ngay — không phát ra tiếng nào.
  const context = new AudioContext()
  let decoded: AudioBuffer
  try {
    decoded = await context.decodeAudioData(data)
  } finally {
    void context.close()
  }

  const originalSampleRate = decoded.sampleRate
  const originalChannels = decoded.numberOfChannels

  if (originalSampleRate === TARGET_SAMPLE_RATE && originalChannels === 1) {
    return {
      samples: decoded.getChannelData(0).slice(),
      duration: decoded.duration,
      sampleRate: originalSampleRate,
      channels: originalChannels,
    }
  }

  const frames = Math.max(1, Math.ceil(decoded.duration * TARGET_SAMPLE_RATE))
  const offline = new OfflineAudioContext(1, frames, TARGET_SAMPLE_RATE)
  const source = offline.createBufferSource()
  source.buffer = decoded
  source.connect(offline.destination)
  source.start()

  const rendered = await offline.startRendering()

  return {
    samples: rendered.getChannelData(0).slice(),
    duration: decoded.duration,
    sampleRate: originalSampleRate,
    channels: originalChannels,
  }
}

/** Tải một file âm thanh từ URL rồi giải mã. */
export async function prepareAudioUrl(url: string, label: string): Promise<PreparedAudio> {
  const response = await fetch(url)
  if (!response.ok) throw new Error(`Không tải được âm thanh: HTTP ${response.status}`)

  const decoded = await decodeAudioBuffer(await response.arrayBuffer())
  return {
    samples: decoded.samples,
    src: url,
    label,
    duration: decoded.duration,
    originalSampleRate: decoded.sampleRate,
    originalChannels: decoded.channels,
  }
}

/** Giải mã một Blob (file người dùng chọn, hoặc bản ghi từ micro). */
export async function prepareAudioBlob(blob: Blob, label: string): Promise<PreparedAudio> {
  const decoded = await decodeAudioBuffer(await blob.arrayBuffer())
  return {
    samples: decoded.samples,
    // Object URL cho thẻ <audio>. Không revoke ngay: thẻ audio còn đang dùng nó,
    // và một card chỉ giữ vài bản ghi nên rò rỉ ở đây là không đáng kể.
    src: URL.createObjectURL(blob),
    label,
    duration: decoded.duration,
    originalSampleRate: decoded.sampleRate,
    originalChannels: decoded.channels,
  }
}

/**
 * Nén dãy mẫu thành các cặp (min, max) để vẽ waveform.
 *
 * Một đoạn 60 giây có 960.000 mẫu — vẽ hết thì vô nghĩa và treo máy. Lấy min/max
 * theo từng khoảng (chứ không lấy trung bình) để giữ được biên độ đỉnh, nếu
 * không waveform sẽ phẳng lì.
 */
export function waveformPeaks(samples: Float32Array, buckets = 240): Array<[number, number]> {
  const size = Math.max(1, Math.floor(samples.length / buckets))
  const peaks: Array<[number, number]> = []

  for (let i = 0; i < buckets; i++) {
    const start = i * size
    if (start >= samples.length) break

    const end = Math.min(start + size, samples.length)
    let min = samples[start]
    let max = samples[start]
    for (let j = start + 1; j < end; j++) {
      if (samples[j] < min) min = samples[j]
      if (samples[j] > max) max = samples[j]
    }
    peaks.push([min, max])
  }

  return peaks
}

/** Micro có dùng được không (cần HTTPS hoặc localhost). */
export const HAS_MICROPHONE =
  typeof navigator !== 'undefined' && Boolean(navigator.mediaDevices?.getUserMedia)
