import { useState } from 'react'

import { AudioPicker } from '../../components/AudioPicker'
import { Badge } from '../../components/Badge'
import { Info } from '../../components/Info'
import { Math as Tex } from '../../components/Math'
import { ModelBar } from '../../components/ModelBar'
import { UnderTheHood } from '../../components/UnderTheHood'
import { useModel } from '../../hooks/useModel'
import type { PreparedAudio } from '../../lib/audio'
import { formatMs } from '../../lib/format'
import type { AudioChoice } from '../../lib/samples'
import type { DebugInfo } from '../../workers/protocol'

/** Cửa sổ ngữ cảnh của Whisper — cố định 30 giây, không đổi được. */
const WINDOW_SECONDS = 30

interface Chunk {
  timestamp: [number, number | null]
  text: string
}

interface AsrOutput {
  text: string
  chunks?: Chunk[]
}

const SAMPLES: AudioChoice[] = [
  { name: 'jfk.wav', label: 'JFK (Anh)', hint: 'Câu nói nổi tiếng, 11 giây, thu rõ.' },
  { name: 'french-audio.wav', label: 'Tiếng Pháp', hint: 'Thử nút Dịch sang tiếng Anh.' },
  { name: 'japanese-audio.wav', label: 'Tiếng Nhật', hint: 'Chữ viết khác hẳn — cùng một model.' },
  {
    name: 'ted_60_16k.wav',
    label: 'TED 60 giây',
    hint: 'Dài hơn cửa sổ 30 giây — bật/tắt chunking để thấy phần bị mất.',
  },
]

const LANGUAGES = [
  { id: '', label: 'Tự nhận' },
  { id: 'english', label: 'Anh' },
  { id: 'french', label: 'Pháp' },
  { id: 'japanese', label: 'Nhật' },
  { id: 'vietnamese', label: 'Việt' },
]

export function AsrDemo() {
  const model = useModel('asr')

  const [audio, setAudio] = useState<PreparedAudio | null>(null)
  const [language, setLanguage] = useState('')
  const [translate, setTranslate] = useState(false)
  const [chunking, setChunking] = useState(true)
  const [timestamps, setTimestamps] = useState(true)
  const [result, setResult] = useState<AsrOutput | null>(null)
  const [ms, setMs] = useState<number | null>(null)
  const [debug, setDebug] = useState<DebugInfo | undefined>(undefined)

  const tooLong = audio != null && audio.duration > WINDOW_SECONDS
  /** Phần âm thanh bị bỏ nếu không bật chunking. */
  const droppedSeconds = tooLong && !chunking ? audio.duration - WINDOW_SECONDS : 0

  async function handleRun() {
    if (!audio) return
    try {
      const res = await model.run<AsrOutput>(audio.samples, {
        pipelineOptions: {
          // language: undefined -> Whisper tự đoán ngôn ngữ từ chính âm thanh.
          language: language || undefined,
          task: translate ? 'translate' : 'transcribe',
          return_timestamps: timestamps,
          // chunk_length_s chỉ có tác dụng với đoạn dài hơn 30 giây.
          ...(chunking ? { chunk_length_s: WINDOW_SECONDS, stride_length_s: 5 } : {}),
        },
      })
      setResult(res.output)
      setMs(res.ms)
      setDebug(res.debug)
    } catch {
      setResult(null)
      setDebug(undefined)
    }
  }

  const canRun = audio != null && !model.isBusy
  /** Tỉ lệ thời gian thực: <1 nghĩa là chạy nhanh hơn nghe. */
  const realtimeFactor = ms != null && audio ? ms / 1000 / audio.duration : null

  return (
    <>
      <ModelBar model={model} />

      <AudioPicker
        samples={SAMPLES}
        value={audio}
        onChange={(next) => {
          setAudio(next)
          setResult(null)
          setDebug(undefined)
        }}
        disabled={model.isBusy}
      />

      <div className="config-grid">
        <div className="field">
          <div className="field-head">
            <span className="field-label">Ngôn ngữ</span>
            <Info title="Một model cho 99 ngôn ngữ">
              <p>
                Whisper không có bản riêng cho từng thứ tiếng. Ngôn ngữ được truyền vào như một{' '}
                <strong>token điều khiển</strong> đứng ngay đầu chuỗi mà decoder sinh ra:
              </p>
              <p className="mono">
                {'<|startoftranscript|> <|fr|> <|transcribe|> <|notimestamps|> …'}
              </p>
              <p>
                Để "Tự nhận" thì model tự đoán token ngôn ngữ đó từ chính âm thanh. Chọn sai ngôn
                ngữ là một cách hay để thấy model hỏng ra sao — nó sẽ cố phiên âm tiếng Pháp bằng
                chính tả tiếng Anh.
              </p>
              <p className="hint">
                Chính vì ngôn ngữ và task chỉ là token, nên cùng một trọng số làm được cả nhận dạng
                lẫn dịch mà không cần đổi model.
              </p>
            </Info>
          </div>
          <div className="chip-row">
            {LANGUAGES.map((item) => (
              <button
                key={item.id}
                type="button"
                className={`chip${language === item.id ? ' chip--active' : ''}`}
                onClick={() => {
                  setLanguage(item.id)
                  setResult(null)
                }}
              >
                {item.label}
              </button>
            ))}
          </div>

          <div className="field-head" style={{ marginTop: 8 }}>
            <span className="field-label">Tác vụ</span>
          </div>
          <div className="chip-row">
            <button
              type="button"
              className={`chip${!translate ? ' chip--active' : ''}`}
              onClick={() => {
                setTranslate(false)
                setResult(null)
              }}
            >
              Chép lại nguyên ngữ
            </button>
            <button
              type="button"
              className={`chip${translate ? ' chip--active' : ''}`}
              onClick={() => {
                setTranslate(true)
                setResult(null)
              }}
            >
              Dịch sang tiếng Anh
            </button>
          </div>
        </div>

        <div className="field">
          <div className="field-head">
            <span className="field-label">Cửa sổ 30 giây</span>
            {tooLong && (
              <Badge mono tone={chunking ? 'ok' : 'danger'}>
                đoạn dài {audio.duration.toFixed(0)} s
              </Badge>
            )}
            <Info title="Whisper chỉ nhìn được 30 giây một lần">
              <p>
                Encoder của Whisper có đầu vào <strong>cố định</strong> đúng 30 giây. Âm thanh ngắn
                hơn thì bị đệm im lặng cho đủ; dài hơn thì bị{' '}
                <strong>cắt cụt và bỏ đi phần thừa</strong> — không có cảnh báo nào.
              </p>
              <p>
                <code>chunk_length_s</code> bật chế độ cắt đoạn dài thành nhiều cửa sổ 30 giây rồi
                ghép kết quả lại. <code>stride_length_s</code> cho các cửa sổ chồng lấn nhau 5 giây
                để một từ nằm đúng chỗ cắt không bị mất.
              </p>
              <p className="hint">
                Thử với mẫu TED 60 giây: tắt chunking thì mất sạch nửa sau, mà văn bản trả về vẫn
                trông hoàn toàn bình thường. Đúng kiểu lỗi khiến người ta tin nhầm vào kết quả.
              </p>
            </Info>
          </div>
          <label className="toggle">
            <input
              type="checkbox"
              checked={chunking}
              onChange={(e) => {
                setChunking(e.target.checked)
                setResult(null)
              }}
            />
            cắt đoạn dài (chunk_length_s)
          </label>
          <label className="toggle">
            <input
              type="checkbox"
              checked={timestamps}
              onChange={(e) => {
                setTimestamps(e.target.checked)
                setResult(null)
              }}
            />
            trả về mốc thời gian
          </label>

          {droppedSeconds > 0 && (
            <div className="callout callout--danger">
              Đang tắt chunking với đoạn dài {audio!.duration.toFixed(0)} s →{' '}
              <strong>{droppedSeconds.toFixed(0)} giây cuối sẽ bị bỏ đi</strong> mà không báo lỗi.
            </div>
          )}
        </div>
      </div>

      <div className="control-row">
        <button type="button" className="primary" onClick={handleRun} disabled={!canRun}>
          {model.status === 'running'
            ? 'Đang nghe…'
            : translate
              ? 'Nghe và dịch'
              : 'Chuyển thành văn bản'}
        </button>
        {realtimeFactor != null && (
          <>
            <Badge mono tone={realtimeFactor < 1 ? 'ok' : 'warn'}>
              {realtimeFactor.toFixed(2)}× thời gian thực
            </Badge>
            <Info title="Đọc chỉ số thời gian thực">
              <Tex tex="\text{RTF} = \frac{\text{thời gian xử lý}}{\text{độ dài âm thanh}}" block />
              <p>
                Nhỏ hơn 1 nghĩa là máy chép nhanh hơn người nói — điều kiện cần để làm phụ đề trực
                tiếp. Lớn hơn 1 thì chỉ dùng được cho file có sẵn.
              </p>
              <p className="hint">
                Con số này phụ thuộc máy và backend. Trên WASM đơn luồng nó tệ hơn nhiều lần so với
                khi có WASM đa luồng — xem badge môi trường ở góc trên bên phải.
              </p>
            </Info>
          </>
        )}
      </div>

      {!result && !model.isBusy && (
        <div className="empty-state">Chọn hoặc thu một đoạn âm thanh rồi bấm để chép lại.</div>
      )}

      {result && (
        <>
          <div className="field">
            <div className="field-head">
              <span className="field-label">
                {translate ? 'Bản dịch tiếng Anh' : 'Văn bản'}
              </span>
              {ms != null && <Badge mono>{formatMs(ms)}</Badge>}
              {result.chunks && <Badge mono>{result.chunks.length} đoạn</Badge>}
              <Info title="Văn bản này được SINH ra, không phải được chọn">
                <p>
                  Đây là model duy nhất trong dashboard <em>sinh</em> ra kết quả. Mọi card khác chọn
                  một nhãn trong danh sách có sẵn; Whisper đẻ ra văn bản từng token một, mỗi token
                  được sinh ra <strong>dựa trên tất cả token đã sinh trước đó</strong>:
                </p>
                <Tex tex="P(y) = \prod_{t} P(y_t \mid y_{<t},\, \text{audio})" block />
                <p>
                  Hệ quả thực tế: một token sai ở đầu có thể kéo lệch cả câu sau đó, và model có thể{' '}
                  <strong>bịa</strong> ra chữ không hề có trong âm thanh — nhất là ở đoạn im lặng
                  hoặc nhiễu. Đó là mặt trái của việc sinh văn bản trôi chảy.
                </p>
              </Info>
            </div>
            <div className={result.text.trim() ? 'transcript' : 'transcript transcript--empty'}>
              {result.text.trim() || '(không nghe ra chữ nào)'}
            </div>
          </div>

          {result.chunks && result.chunks.length > 0 && (
            <div className="field">
              <div className="field-head">
                <span className="field-label">Mốc thời gian</span>
                <Info title="Mốc thời gian cũng chỉ là token">
                  <p>
                    Whisper không đo thời gian bằng thuật toán riêng. Bộ từ vựng của nó có sẵn các
                    token <code>{'<|0.00|>'}</code>, <code>{'<|0.02|>'}</code>… và model học cách
                    sinh chúng xen giữa văn bản.
                  </p>
                  <p className="hint">
                    Vì thế mốc thời gian cũng có thể sai như chữ có thể sai — độ phân giải chỉ 20 ms
                    và nó thường lệch ở ranh giới giữa các đoạn.
                  </p>
                </Info>
              </div>
              <div className="table-scroll">
                <table className="data">
                  <thead>
                    <tr>
                      <th className="num">Bắt đầu</th>
                      <th className="num">Kết thúc</th>
                      <th>Nội dung</th>
                    </tr>
                  </thead>
                  <tbody>
                    {result.chunks.map((chunk, i) => (
                      <tr key={i}>
                        <td className="num">{chunk.timestamp[0]?.toFixed(2) ?? '—'}</td>
                        <td className="num">{chunk.timestamp[1]?.toFixed(2) ?? '—'}</td>
                        <td>{chunk.text}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </>
      )}

      <UnderTheHood debug={debug}>
        {audio && (
          <div className="hood-block">
            <h4>
              Từ sóng âm đến log-mel spectrogram
              <Info title="“Tokenizer” của âm thanh">
                <p>
                  Whisper không đọc thẳng 16.000 số mỗi giây. Trước tiên âm thanh được biến thành một{' '}
                  <strong>ảnh</strong>: cắt thành các khung 25 ms trượt mỗi 10 ms, mỗi khung qua
                  biến đổi Fourier để lấy phổ tần số, rồi nén theo thang mel cho gần với cách tai
                  người nghe.
                </p>
                <p>
                  Kết quả là ma trận <Tex tex="[80, 3000]" /> — 80 dải tần × 3000 khung thời gian
                  (30 giây ÷ 10 ms). Encoder xử lý ma trận đó y như một ảnh.
                </p>
                <p className="hint">
                  Cùng một ý tưởng ở card Phân loại ảnh và card Tokenizer: dữ liệu thô luôn phải qua
                  một bước biến đổi cố định trước khi model nhìn thấy gì, và bước đó không hề trung
                  tính.
                </p>
              </Info>
            </h4>
            <table className="data">
              <tbody>
                <tr>
                  <td>Âm thanh gốc</td>
                  <td className="mono">
                    {audio.duration.toFixed(2)} s · {audio.originalSampleRate} Hz ·{' '}
                    {audio.originalChannels === 1 ? 'mono' : `${audio.originalChannels} kênh`}
                  </td>
                </tr>
                <tr>
                  <td>Đưa vào model</td>
                  <td className="mono">
                    {audio.samples.length.toLocaleString('vi-VN')} mẫu · 16.000 Hz · mono
                  </td>
                </tr>
                <tr>
                  <td>Cửa sổ encoder</td>
                  <td className="mono">
                    {WINDOW_SECONDS} s cố định
                    {audio.duration > WINDOW_SECONDS
                      ? ` → cần ${Math.ceil(audio.duration / WINDOW_SECONDS)} cửa sổ`
                      : ' (đoạn này vừa 1 cửa sổ)'}
                  </td>
                </tr>
                <tr>
                  <td>Log-mel spectrogram</td>
                  <td className="mono">[80, 3000] — 80 dải mel × 3000 khung 10 ms</td>
                </tr>
                <tr>
                  <td>Kiến trúc</td>
                  <td className="mono">encoder-decoder · sinh tự hồi quy</td>
                </tr>
              </tbody>
            </table>
          </div>
        )}
      </UnderTheHood>
    </>
  )
}
