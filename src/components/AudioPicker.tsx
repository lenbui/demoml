import { useEffect, useRef, useState } from 'react'

import { USE_LOCAL_MODELS } from '../lib/config'
import {
  HAS_MICROPHONE,
  TARGET_SAMPLE_RATE,
  prepareAudioBlob,
  prepareAudioUrl,
  type PreparedAudio,
} from '../lib/audio'
import { sampleAudioUrl, type AudioChoice } from '../lib/samples'
import { Badge } from './Badge'
import { Info } from './Info'
import { Waveform } from './Waveform'

/**
 * Chọn âm thanh cho các demo audio: mẫu có sẵn, file của bạn, hoặc thu từ micro.
 *
 * Khác ImagePicker ở một điểm quan trọng: ảnh được đưa vào model dưới dạng URL,
 * còn âm thanh BẮT BUỘC phải giải mã trước ở main thread (worker không có Web
 * Audio API). Vì vậy component này luôn trả về Float32Array đã sẵn sàng, không
 * chỉ trả về một đường dẫn — xem lib/audio.ts.
 */
export function AudioPicker({
  samples,
  value,
  onChange,
  disabled,
}: {
  samples: AudioChoice[]
  value: PreparedAudio | null
  onChange: (audio: PreparedAudio | null) => void
  disabled?: boolean
}) {
  const inputRef = useRef<HTMLInputElement>(null)
  const recorderRef = useRef<MediaRecorder | null>(null)
  const chunksRef = useRef<Blob[]>([])

  const [busy, setBusy] = useState(false)
  const [recording, setRecording] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const initialised = useRef(false)
  useEffect(() => {
    if (initialised.current || value) return
    initialised.current = true
    void pickSample(samples[0])
    // Chỉ chạy một lần lúc mount.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // Đang thu mà rời demo thì phải tắt micro, nếu không đèn micro vẫn sáng.
  useEffect(() => {
    return () => {
      recorderRef.current?.stream.getTracks().forEach((track) => track.stop())
    }
  }, [])

  async function load(task: () => Promise<PreparedAudio>) {
    setError(null)
    setBusy(true)
    try {
      onChange(await task())
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Không đọc được âm thanh.')
      onChange(null)
    } finally {
      setBusy(false)
    }
  }

  async function pickSample(choice: AudioChoice) {
    setError(null)
    setBusy(true)
    try {
      onChange(await prepareAudioUrl(sampleAudioUrl(choice.name), choice.label))
    } catch {
      setError(
        USE_LOCAL_MODELS
          ? `Không tìm thấy public/samples/${choice.name}. Chạy \`npm run fetch-models\` để tải mẫu về.`
          : 'Không tải được âm thanh mẫu — cần Internet, hoặc dùng file của bạn.',
      )
      onChange(null)
    } finally {
      setBusy(false)
    }
  }

  async function startRecording() {
    setError(null)
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      const recorder = new MediaRecorder(stream)
      chunksRef.current = []

      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) chunksRef.current.push(event.data)
      }
      recorder.onstop = () => {
        // Tắt micro ngay khi dừng — đèn micro của trình duyệt phải tắt theo.
        stream.getTracks().forEach((track) => track.stop())
        const blob = new Blob(chunksRef.current, { type: recorder.mimeType })
        void load(() => prepareAudioBlob(blob, 'Bản thu của bạn'))
      }

      recorder.start()
      recorderRef.current = recorder
      setRecording(true)
    } catch {
      setError('Không truy cập được micro. Kiểm tra quyền của trình duyệt.')
    }
  }

  function stopRecording() {
    recorderRef.current?.stop()
    recorderRef.current = null
    setRecording(false)
  }

  const resampled = value != null && value.originalSampleRate !== TARGET_SAMPLE_RATE

  return (
    <div className="field">
      <div className="field-head">
        <span className="field-label">Âm thanh</span>
        {value && <Badge mono>{value.duration.toFixed(1)} s</Badge>}
        {value && (
          <Badge mono tone={resampled ? 'warn' : 'default'}>
            {(value.originalSampleRate / 1000).toFixed(1)} kHz
            {resampled ? ` → 16 kHz` : ''}
          </Badge>
        )}
        {value && <Badge mono>{value.samples.length.toLocaleString('vi-VN')} mẫu</Badge>}
        <Info title="Model “nghe” thấy gì">
          <p>
            Không có gì bí ẩn: chỉ là một dãy số thực trong <code>[-1, 1]</code>, đúng{' '}
            <strong>16.000 số cho mỗi giây</strong>. Dạng sóng bên dưới vẽ trực tiếp từ chính dãy số
            được đưa vào model.
          </p>
          <p>
            Whisper và AST đều được huấn luyện ở 16 kHz và <strong>không tự kiểm tra</strong> tần số
            lấy mẫu. Đưa vào file 44.1 kHz mà quên resample thì model vẫn chạy, vẫn trả kết quả —
            chỉ là nó nghe mọi thứ nhanh gấp 2,75 lần và kết quả thành rác. Đây là lỗi im lặng kinh
            điển của xử lý âm thanh, nên demo hiện luôn tần số gốc để bạn thấy bước đổi tần số đã
            xảy ra.
          </p>
          <p className="hint">
            Việc giải mã và đổi tần số phải làm ở main thread vì Web Audio API không tồn tại trong
            Web Worker — xem <code>src/lib/audio.ts</code>.
          </p>
        </Info>
        <div className="chip-row">
          {samples.map((choice) => (
            <button
              key={choice.name}
              type="button"
              className={`chip${value?.label === choice.label ? ' chip--active' : ''}`}
              onClick={() => void pickSample(choice)}
              disabled={disabled || busy || recording}
              title={choice.hint}
            >
              {choice.label}
            </button>
          ))}
          <button
            type="button"
            className="chip"
            onClick={() => inputRef.current?.click()}
            disabled={disabled || busy || recording}
          >
            + File của bạn
          </button>
          {HAS_MICROPHONE && (
            <button
              type="button"
              className={recording ? 'chip chip--active' : 'chip'}
              onClick={() => (recording ? stopRecording() : void startRecording())}
              disabled={disabled || busy}
            >
              {recording ? '■ Dừng thu' : '● Thu từ micro'}
            </button>
          )}
        </div>
      </div>

      <input
        ref={inputRef}
        type="file"
        accept="audio/*"
        hidden
        onChange={(e) => {
          const file = e.target.files?.[0]
          if (file) void load(() => prepareAudioBlob(file, file.name))
          e.target.value = ''
        }}
      />

      {error && <div className="callout callout--warn">{error}</div>}
      {busy && <div className="empty-state">Đang giải mã âm thanh…</div>}
      {recording && (
        <div className="callout callout--warn">
          Đang thu. Nói một câu rồi bấm <strong>Dừng thu</strong>.
        </div>
      )}

      {value && (
        <>
          <Waveform samples={value.samples} />
          {/* controls: nghe lại chính đoạn vừa đưa vào model. */}
          <audio className="audio-player" src={value.src} controls preload="metadata" />
        </>
      )}

      <p className="hint">
        Âm thanh của bạn <strong>không rời khỏi máy</strong>: nó được giải mã ngay trong trình duyệt
        rồi đưa thẳng vào model.
      </p>
    </div>
  )
}
