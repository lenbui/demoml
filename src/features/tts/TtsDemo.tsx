import { useEffect, useRef, useState } from 'react'

import { Badge } from '../../components/Badge'
import { Info } from '../../components/Info'
import { Math as Tex } from '../../components/Math'
import { ModelBar } from '../../components/ModelBar'
import { UnderTheHood } from '../../components/UnderTheHood'
import { VariantPicker } from '../../components/VariantPicker'
import { Waveform } from '../../components/Waveform'
import { useModel } from '../../hooks/useModel'
import { formatBytes, formatMs } from '../../lib/format'
import { encodeWav } from '../../lib/wav'
import type { DebugInfo } from '../../workers/protocol'

/** Output của pipeline('text-to-speech'). */
interface TtsOutput {
  audio: Float32Array
  sampling_rate: number
}

interface Spoken {
  samples: Float32Array
  sampleRate: number
  url: string
  bytes: number
  ms: number
}

const EXAMPLES: Record<string, Array<{ label: string; text: string; note?: string }>> = {
  vie: [
    { label: 'Câu thường', text: 'Xin chào, đây là mô hình chạy hoàn toàn trong trình duyệt.' },
    {
      label: 'Có dấu khó',
      text: 'Nguyễn Thị Tuyết Nhung khuỵu xuống giữa khuya.',
      note: 'Nhiều nguyên âm đôi và dấu nặng — chỗ dễ lộ giới hạn của model nhỏ.',
    },
    {
      label: 'Số & ký hiệu',
      text: 'Giá 1.250.000 đồng, giảm 15% cho 3 người đầu.',
      note: 'Model KHÔNG tự đọc số thành chữ. Đây là việc của bước chuẩn hoá văn bản.',
    },
    {
      label: 'Sai ngôn ngữ',
      text: 'The quick brown fox jumps over the lazy dog.',
      note: 'Đưa tiếng Anh vào model tiếng Việt — nó vẫn đọc, theo âm tiếng Việt.',
    },
  ],
  eng: [
    { label: 'Câu thường', text: 'Hello, this model runs entirely inside your browser.' },
    {
      label: 'Số & ký hiệu',
      text: 'It costs $1,250 and ships in 3 days.',
      note: 'Model KHÔNG tự đọc số thành chữ — cần bước chuẩn hoá văn bản trước.',
    },
    {
      label: 'Sai ngôn ngữ',
      text: 'Xin chào, tôi đang học máy học.',
      note: 'Tiếng Việt đưa vào model tiếng Anh — dấu bị bỏ và đọc sai hẳn.',
    },
  ],
}

export function TtsDemo() {
  const model = useModel('tts')
  const variants = model.spec.variants ?? []

  const [variantId, setVariantId] = useState(variants[0]?.id ?? 'vie')
  const [text, setText] = useState(EXAMPLES.vie[0].text)
  const [note, setNote] = useState<string | undefined>(undefined)
  const [spoken, setSpoken] = useState<Spoken | null>(null)
  const [debug, setDebug] = useState<DebugInfo | undefined>(undefined)

  // Object URL của lần trước phải được thu hồi, nếu không mỗi lần bấm lại giữ
  // thêm một bản WAV trong bộ nhớ cho tới khi đóng tab.
  const urlRef = useRef<string | null>(null)
  useEffect(() => {
    return () => {
      if (urlRef.current) URL.revokeObjectURL(urlRef.current)
    }
  }, [])

  const examples = EXAMPLES[variantId] ?? EXAMPLES.vie

  async function handleRun() {
    try {
      const res = await model.run<TtsOutput>(text, { variantId })
      const samples = res.output.audio
      const sampleRate = res.output.sampling_rate

      const blob = encodeWav(samples, sampleRate)
      if (urlRef.current) URL.revokeObjectURL(urlRef.current)
      urlRef.current = URL.createObjectURL(blob)

      setSpoken({ samples, sampleRate, url: urlRef.current, bytes: blob.size, ms: res.ms })
      setDebug(res.debug)
    } catch {
      setSpoken(null)
      setDebug(undefined)
    }
  }

  const canRun = text.trim().length > 0 && !model.isBusy
  const duration = spoken ? spoken.samples.length / spoken.sampleRate : null
  /** <1 nghĩa là tổng hợp nhanh hơn nghe — điều kiện cần để đọc trực tiếp. */
  const realtimeFactor = spoken && duration ? spoken.ms / 1000 / duration : null

  return (
    <>
      <div className="field">
        <div className="field-head">
          <span className="field-label">Model</span>
          <Info title="Một model cho mỗi ngôn ngữ, khác hẳn Whisper">
            <p>
              Whisper ở card bên cạnh dùng <strong>một</strong> bộ trọng số cho 99 ngôn ngữ, chọn
              ngôn ngữ bằng một token điều khiển. MMS-TTS thì ngược lại: mỗi ngôn ngữ là một model
              riêng, huấn luyện riêng.
            </p>
            <p>
              Lý do nằm ở bản chất bài toán. Nhận dạng là <em>nhiều-về-một</em>: mọi thứ tiếng đều
              quy về văn bản, nên chia sẻ biểu diễn giữa các ngôn ngữ rất có lợi. Tổng hợp là{' '}
              <em>một-về-nhiều</em>: đầu ra là âm thanh mang đặc trưng ngữ âm riêng của từng thứ
              tiếng, khó gộp chung.
            </p>
            <p className="hint">
              Đổi lại mỗi model chỉ 38 MB, nên tải vài ngôn ngữ vẫn nhẹ hơn một Whisper.
            </p>
          </Info>
        </div>
        <VariantPicker
          variants={variants}
          value={variantId}
          onChange={(id) => {
            setVariantId(id)
            setText((EXAMPLES[id] ?? EXAMPLES.vie)[0].text)
            setNote(undefined)
            setSpoken(null)
            setDebug(undefined)
          }}
          readyVariants={model.readyVariants}
          loadingVariantId={model.loadingVariantId}
          disabled={model.isBusy}
        />
      </div>

      <ModelBar model={model} variantId={variantId} />

      <div className="field">
        <div className="field-head">
          <label className="field-label" htmlFor="tts-input">
            Văn bản cần đọc
          </label>
          <div className="chip-row">
            {examples.map((example) => (
              <button
                key={example.label}
                type="button"
                className={`chip${text === example.text ? ' chip--active' : ''}`}
                onClick={() => {
                  setText(example.text)
                  setNote(example.note)
                  setSpoken(null)
                }}
              >
                {example.label}
              </button>
            ))}
          </div>
        </div>
        <textarea
          id="tts-input"
          className="input--compact"
          value={text}
          onChange={(e) => {
            setText(e.target.value)
            setNote(undefined)
            setSpoken(null)
          }}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && (e.ctrlKey || e.metaKey) && canRun) void handleRun()
          }}
        />
        {note && <p className="field-hint">{note}</p>}
      </div>

      <div className="control-row">
        <button type="button" className="primary" onClick={handleRun} disabled={!canRun}>
          {model.status === 'running' ? 'Đang tổng hợp…' : 'Đọc thành tiếng'}
        </button>
        <span className="kbd-hint">Ctrl + Enter</span>
        {realtimeFactor != null && (
          <>
            <Badge mono tone={realtimeFactor < 1 ? 'ok' : 'warn'}>
              {realtimeFactor.toFixed(2)}× thời gian thực
            </Badge>
            <Info title="Đọc chỉ số thời gian thực">
              <Tex
                tex="\text{RTF} = \frac{\text{thời gian tổng hợp}}{\text{độ dài âm thanh}}"
                block
              />
              <p>
                Nhỏ hơn 1 nghĩa là máy tạo ra âm thanh nhanh hơn thời gian phát nó — điều kiện cần
                để đọc trực tiếp mà không giật.
              </p>
              <p className="hint">
                Card Whisper cũng có chỉ số này, cho chiều ngược lại. Hai con số cạnh nhau cho thấy
                nhận dạng và tổng hợp có chi phí rất khác nhau.
              </p>
            </Info>
          </>
        )}
      </div>

      {!spoken && !model.isBusy && (
        <div className="empty-state">Nhập câu rồi bấm để model tạo ra sóng âm.</div>
      )}

      {spoken && duration != null && (
        <div className="field">
          <div className="field-head">
            <span className="field-label">Kết quả</span>
            <Badge mono>{duration.toFixed(2)} s</Badge>
            <Badge mono>{(spoken.sampleRate / 1000).toFixed(1)} kHz</Badge>
            <Badge mono>{spoken.samples.length.toLocaleString('vi-VN')} mẫu</Badge>
            <Badge mono>WAV {formatBytes(spoken.bytes)}</Badge>
            <Info title="Model trả về dãy số, không trả về file">
              <p>
                Đầu ra của model là một <code>Float32Array</code> — biên độ trong{' '}
                <code>[-1, 1]</code>, {spoken.sampleRate.toLocaleString('vi-VN')} số mỗi giây. Thẻ{' '}
                <code>&lt;audio&gt;</code> không phát được dãy số đó, nên demo phải tự đóng gói
                thành file.
              </p>
              <p>
                WAV được chọn vì nó gần như không có gì: <strong>44 byte header</strong> rồi tới mẫu
                âm thanh, không nén, không thư viện. Xem <code>encodeWav()</code> trong{' '}
                <code>src/lib/wav.ts</code> — toàn bộ đặc tả nằm trong ngần ấy byte.
              </p>
              <p>
                Mẫu được đổi từ float32 sang PCM 16-bit nguyên, nên file nặng đúng{' '}
                <Tex tex="44 + 2n" /> byte với <Tex tex="n" /> là số mẫu.
              </p>
              <p className="hint">
                Đây chính là dãy số mà card Whisper <em>nhận vào</em>. Hai card là hai chiều ngược
                nhau của cùng một biểu diễn.
              </p>
            </Info>
          </div>
          <Waveform samples={spoken.samples} />
          <audio className="audio-player" src={spoken.url} controls autoPlay />
        </div>
      )}

      <UnderTheHood debug={debug}>
        {spoken && duration != null && (
          <div className="hood-block">
            <h4>
              Từ chữ ra sóng âm
              <Info title="VITS gộp cả hai tầng vào một mạng">
                <p>Hệ tổng hợp tiếng nói cổ điển có hai tầng tách rời:</p>
                <ul className="notes">
                  <li>
                    <strong>acoustic model</strong>: văn bản → mel spectrogram;
                  </li>
                  <li>
                    <strong>vocoder</strong>: mel spectrogram → sóng âm.
                  </li>
                </ul>
                <p>
                  Đó cũng là lý do SpeechT5 phải tải ba file onnx riêng. VITS thì huấn luyện cả hai
                  tầng cùng lúc, đối kháng với một discriminator, nên chỉ còn <strong>một</strong>{' '}
                  mạng đi thẳng từ chữ ra sóng.
                </p>
                <p>
                  Chú ý phần Tokenize phía trên: VITS cắt văn bản thành{' '}
                  <strong>từng ký tự</strong>, không phải subword như BERT hay GPT. Vì đầu ra là âm
                  thanh nên thứ nó cần là chuỗi âm vị, mà ký tự là xấp xỉ gần nhất — với những ngôn
                  ngữ có chính tả đều đặn như tiếng Việt thì xấp xỉ đó khá tốt.
                </p>
                <p>
                  Và cứ hai ký tự lại có một <strong>token trống</strong> xen vào (nó hiện ra thành{' '}
                  <code>ụ</code> vì đó là ký tự ở vị trí 0 của bộ từ vựng). Bộ dự đoán trường độ dùng
                  các ô trống này để đặt khoảng chuyển tiếp giữa các âm — nên số token luôn xấp xỉ
                  gấp đôi số ký tự.
                </p>
                <p className="hint">
                  VITS còn có một bộ dự đoán trường độ mang tính ngẫu nhiên, nên đọc lại cùng một
                  câu có thể ra độ dài hơi khác. Bấm lại vài lần và nhìn số giây.
                </p>
              </Info>
            </h4>
            {/* Round-trip giống card Tokenizer: phần chênh lệch chính là thông
                tin đã mất trước khi model kịp nhìn thấy. */}
            {debug?.tokens && (
              <div className="field" style={{ marginBottom: 12 }}>
                <div className="field-head">
                  <span className="field-label">Model thật sự nhận được gì</span>
                  <Badge tone="danger">mất dấu câu &amp; chữ hoa</Badge>
                  <Info title="Tokenizer bỏ dấu câu — model không thể ngắt nhịp theo dấu phẩy">
                    <p>
                      Ghép các token lại (bỏ ô trống xen giữa) rồi so với câu bạn gõ. Đã kiểm chứng
                      với câu mẫu tiếng Việt:
                    </p>
                    <table className="data">
                      <tbody>
                        <tr>
                          <td>Bạn gõ</td>
                          <td className="mono">Xin chào, đây là mô hình…</td>
                        </tr>
                        <tr>
                          <td>Model nhận</td>
                          <td className="mono">xin chào đây là mô hình…</td>
                        </tr>
                      </tbody>
                    </table>
                    <p>
                      Chữ hoa bị hạ xuống, dấu phẩy và dấu chấm bị <strong>bỏ hẳn</strong>. Nghĩa là
                      model <em>không có cách nào</em> biết chỗ nào nên ngắt nhịp — mọi khoảng nghỉ
                      nghe thấy đều do nó tự suy ra từ chuỗi ký tự.
                    </p>
                    <p className="hint">
                      Đây đúng là bài học của card Tokenizer Explorer, lần này ở đầu ra âm thanh: cái
                      gì tokenizer làm mất thì mất vĩnh viễn.
                    </p>
                  </Info>
                </div>
                <div className="roundtrip roundtrip--diff">
                  {debug.tokens
                    .filter((_, i) => i % 2 === 1)
                    .join('') || <em>(rỗng)</em>}
                </div>
              </div>
            )}

            <table className="data">
              <tbody>
                <tr>
                  <td>Văn bản vào</td>
                  <td className="mono">{text.length} ký tự</td>
                </tr>
                <tr>
                  <td>Sau tokenize</td>
                  <td className="mono">
                    {debug?.tokens?.length ?? '—'} token — ký tự xen với ô trống
                  </td>
                </tr>
                <tr>
                  <td>Sóng âm ra</td>
                  <td className="mono">
                    {spoken.samples.length.toLocaleString('vi-VN')} mẫu ·{' '}
                    {spoken.sampleRate.toLocaleString('vi-VN')} Hz · mono
                  </td>
                </tr>
                <tr>
                  <td>Độ dài</td>
                  <td className="mono">{duration.toFixed(3)} s</td>
                </tr>
                <tr>
                  <td>Thời gian tổng hợp</td>
                  <td className="mono">{formatMs(spoken.ms)}</td>
                </tr>
                <tr>
                  <td>Kiến trúc</td>
                  <td className="mono">VITS — end-to-end, không có vocoder rời</td>
                </tr>
                <tr>
                  <td>File WAV</td>
                  <td className="mono">
                    44 + 2 × {spoken.samples.length.toLocaleString('vi-VN')} ={' '}
                    {formatBytes(spoken.bytes)}
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        )}
      </UnderTheHood>
    </>
  )
}
