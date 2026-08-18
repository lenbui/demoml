import { useMemo, useState } from 'react'

import { AudioPicker } from '../../components/AudioPicker'
import { Badge } from '../../components/Badge'
import { Info } from '../../components/Info'
import { Math as Tex } from '../../components/Math'
import { ModelBar } from '../../components/ModelBar'
import { UnderTheHood } from '../../components/UnderTheHood'
import { useModel } from '../../hooks/useModel'
import type { PreparedAudio } from '../../lib/audio'
import { formatPercent } from '../../lib/format'
import type { AudioChoice } from '../../lib/samples'
import type { DebugInfo } from '../../workers/protocol'

const TOP_K = 8

/** Kích thước spectrogram mà AST nhận — đọc từ preprocessor_config.json. */
const MEL_BINS = 128
const MAX_FRAMES = 1024

interface Prediction {
  label: string
  score: number
}

const SAMPLES: AudioChoice[] = [
  { name: 'cat_meow.wav', label: 'Mèo kêu', hint: 'Âm thanh ngắn, một nguồn duy nhất.' },
  { name: 'dog_barking.wav', label: 'Chó sủa', hint: 'AudioSet có riêng nhãn Bark, Howl, Growling.' },
  { name: 'jfk.wav', label: 'Giọng nói', hint: 'Speech, Narration, Male speech — nhiều nhãn cùng đúng.' },
  { name: 'courtroom.wav', label: 'Phòng xử án', hint: 'Nhiều nguồn âm chồng nhau.' },
]

export function AudioClassificationDemo() {
  const model = useModel('audio-classification')

  const [audio, setAudio] = useState<PreparedAudio | null>(null)
  const [result, setResult] = useState<Prediction[] | null>(null)
  const [debug, setDebug] = useState<DebugInfo | undefined>(undefined)

  /**
   * Tổng điểm top-k.
   *
   * Đã đo: con số này bằng ĐÚNG 1.00 — tức pipeline áp softmax lên cả 527 nhãn.
   * Đáng chú ý vì AudioSet vốn là bài toán ĐA NHÃN (model được huấn luyện bằng
   * binary cross-entropy, mỗi nhãn một sigmoid độc lập). Hiện con số này ra để
   * người học tự thấy sự vênh đó — xem popover ở phần verdict.
   */
  const topKMass = useMemo(
    () => (result ? result.reduce((sum, p) => sum + p.score, 0) : null),
    [result],
  )

  async function handleRun() {
    if (!audio) return
    try {
      const res = await model.run<Prediction[]>(audio.samples, {
        pipelineOptions: { top_k: TOP_K },
      })
      setResult(res.output)
      setDebug(res.debug)
    } catch {
      setResult(null)
      setDebug(undefined)
    }
  }

  const canRun = audio != null && !model.isBusy

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

      <div className="control-row">
        <button type="button" className="primary" onClick={handleRun} disabled={!canRun}>
          {model.status === 'running' ? 'Đang nghe…' : 'Nhận diện âm thanh'}
        </button>
        <Info title="AST — Vision Transformer, nhưng cho âm thanh">
          <p>
            Tên đầy đủ là <strong>Audio Spectrogram Transformer</strong>, và cái tên đó nói đúng
            nghĩa đen những gì nó làm: biến âm thanh thành một <em>ảnh</em> spectrogram rồi chạy
            đúng kiến trúc ViT lên trên.
          </p>
          <ul className="notes">
            <li>âm thanh → log-mel spectrogram, ma trận {MEL_BINS} × {MAX_FRAMES};</li>
            <li>cắt ma trận đó thành các ô 16×16, y hệt card Phân loại ảnh;</li>
            <li>mỗi ô thành một token đi vào Transformer.</li>
          </ul>
          <p>
            AST thậm chí khởi tạo từ trọng số của một ViT đã học ImageNet — model "nhìn ảnh" được
            dùng lại để "nghe". Đây là ví dụ rõ nhất trong cả dashboard cho thấy Transformer là một
            kiến trúc chung, không gắn với riêng loại dữ liệu nào.
          </p>
          <p className="hint">
            Cùng bước biến đổi log-mel với Whisper ở card bên cạnh — chỉ khác phần phía sau: AST
            phân loại, Whisper sinh văn bản.
          </p>
        </Info>
      </div>

      {!result && !model.isBusy && (
        <div className="empty-state">Chọn hoặc thu âm thanh rồi bấm để nhận diện.</div>
      )}

      {result && result.length > 0 && (
        <>
          <div className="verdict">
            <span className="verdict-label">{result[0].label}</span>
            <Badge mono>{formatPercent(result[0].score, 1)}</Badge>
            {topKMass != null && (
              <Badge mono tone={topKMass > 0.97 ? 'warn' : 'default'}>
                tổng top-{TOP_K} = {topKMass.toFixed(2)}
              </Badge>
            )}
            <Info title="Con số “tổng = 1.00” đang nói lên một chỗ vênh thật sự">
              <p>
                AudioSet là bài toán <strong>đa nhãn</strong>: các nhãn trên không loại trừ nhau mà
                xếp thành một cây phân cấp. Tiếng mèo kêu <em>đồng thời</em> là "Meow", "Cat",
                "Domestic animals, pets" và "Animal" — cả bốn cùng đúng 100%.
              </p>
              <p>
                Model cũng được huấn luyện đúng như vậy: binary cross-entropy, mỗi nhãn một sigmoid
                độc lập, nên đáng ra phải chấm bằng
              </p>
              <Tex tex="p_k = \sigma(z_k) = \frac{1}{1 + e^{-z_k}}" block />
              <p>
                Nhưng tổng điểm đo được là <strong>1.00</strong>, tức pipeline đã áp{' '}
                <strong>softmax</strong> lên cả 527 nhãn — biến chúng thành các lựa chọn cạnh tranh:
              </p>
              <Tex tex="p_k = \frac{e^{z_k}}{\sum_{j=1}^{527} e^{z_j}}" block />
              <p>
                Hậu quả nhìn thấy được ngay: bốn nhãn <em>cùng đúng</em> phải chia nhau một phần
                trăm duy nhất, nên "Meow" chỉ được 56% và "Cat" tụt xuống 23% — không phải vì model
                lưỡng lự, mà vì bước hậu xử lý buộc chúng phải giành nhau.
              </p>
              <p className="hint">
                Bài học: bước hậu xử lý là một <strong>lựa chọn nằm ngoài model</strong>, và chọn
                sai thì không có gì báo lỗi — kết quả vẫn là những con số trông rất hợp lý. Muốn
                đúng thì phải lấy logits thô rồi tự sigmoid, như card Phân loại cảm xúc tự tính
                softmax thay vì để pipeline làm.
              </p>
            </Info>
          </div>

          <div className="pred-list">
            {result.map((prediction, i) => (
              <div
                key={`${prediction.label}-${i}`}
                className={`pred-row${i === 0 ? ' pred-row--winner' : ''}`}
              >
                <span className="pred-label" title={prediction.label}>
                  {prediction.label}
                </span>
                <div className="pred-track">
                  <div
                    className="pred-bar"
                    style={{
                      // Vẽ theo giá trị tuyệt đối, KHÔNG chuẩn hoá theo nhãn cao
                      // nhất: chính việc các thanh cùng nhau lấp đầy đúng 100% là
                      // thứ cho thấy softmax đang bắt các nhãn chia nhau phần trăm.
                      width: `${prediction.score * 100}%`,
                      background: i === 0 ? 'var(--positive)' : 'var(--accent)',
                      opacity: i === 0 ? 1 : 0.45,
                    }}
                  />
                </div>
                <span className="pred-value">{formatPercent(prediction.score, 1)}</span>
              </div>
            ))}
          </div>
        </>
      )}

      <UnderTheHood debug={debug}>
        {audio && (
          <div className="hood-block">
            <h4>
              Âm thanh trở thành ảnh như thế nào
              <Info title="Log-mel spectrogram">
                <p>
                  Âm thanh được cắt thành các khung ngắn chồng lấn nhau, mỗi khung qua biến đổi
                  Fourier để lấy phổ tần số. Các dải tần được gộp theo thang <em>mel</em> — thang
                  phi tuyến mô phỏng việc tai người phân biệt tần số thấp tinh hơn tần số cao.
                </p>
                <p>
                  Kết quả là ma trận {MEL_BINS} × {MAX_FRAMES}: trục dọc là tần số, trục ngang là
                  thời gian, giá trị là năng lượng. Đó chính là một bức ảnh xám.
                </p>
              </Info>
            </h4>
            <table className="data">
              <tbody>
                <tr>
                  <td>Âm thanh gốc</td>
                  <td className="mono">
                    {audio.duration.toFixed(2)} s · {audio.originalSampleRate} Hz
                  </td>
                </tr>
                <tr>
                  <td>Đưa vào model</td>
                  <td className="mono">
                    {audio.samples.length.toLocaleString('vi-VN')} mẫu · 16.000 Hz · mono
                  </td>
                </tr>
                <tr>
                  <td>Log-mel spectrogram</td>
                  <td className="mono">
                    [{MEL_BINS}, {MAX_FRAMES}] — dải mel × khung thời gian
                  </td>
                </tr>
                <tr>
                  <td>Cắt thành ô</td>
                  <td className="mono">16 × 16, y hệt ViT ở card Phân loại ảnh</td>
                </tr>
                <tr>
                  <td>Số nhãn</td>
                  <td className="mono">527 (AudioSet) · cây phân cấp, các nhãn cùng đúng được</td>
                </tr>
                <tr>
                  <td>Huấn luyện</td>
                  <td className="mono">đa nhãn · binary cross-entropy · sigmoid từng nhãn</td>
                </tr>
                <tr>
                  <td>Hậu xử lý của pipeline</td>
                  <td className="mono">softmax trên 527 nhãn → tổng = 1.00 (đã đo)</td>
                </tr>
              </tbody>
            </table>
          </div>
        )}
      </UnderTheHood>
    </>
  )
}
