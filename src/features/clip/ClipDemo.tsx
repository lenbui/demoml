import { useMemo, useState } from 'react'

import { Badge } from '../../components/Badge'
import { ImagePicker } from '../../components/ImagePicker'
import { Info } from '../../components/Info'
import { Math as Tex } from '../../components/Math'
import { ModelBar } from '../../components/ModelBar'
import { UnderTheHood } from '../../components/UnderTheHood'
import { useModel } from '../../hooks/useModel'
import { formatPercent } from '../../lib/format'
import type { PreparedImage } from '../../lib/image'
import type { SampleChoice } from '../../lib/samples'
import type { DebugInfo } from '../../workers/protocol'

const DEFAULT_TEMPLATE = 'a photo of a {}'

interface Prediction {
  label: string
  score: number
}

const SAMPLES: SampleChoice[] = [
  { name: 'pikachu.png', label: 'Pikachu', hint: 'ViT không biết Pikachu. CLIP thì có.' },
  { name: 'cats.jpg', label: 'Hai con mèo', hint: 'Thử nhãn đếm số: "two cats" vs "one cat".' },
  { name: 'beach.png', label: 'Bãi biển', hint: 'Nhãn cảnh vật, không phải vật thể.' },
  { name: 'tiger.jpg', label: 'Hổ', hint: 'So trực tiếp với card ViT trên cùng một ảnh.' },
  { name: 'city-streets.jpg', label: 'Phố', hint: 'Thử nhãn trừu tượng: "a busy street".' },
]

/** Bộ nhãn gợi ý, chọn để lộ ra cả điểm mạnh lẫn điểm yếu của CLIP. */
const LABEL_PRESETS: Array<{ label: string; value: string }> = [
  { label: 'Vật thể', value: 'a cat, a dog, a tiger, Pikachu, a car' },
  { label: 'Cảnh vật', value: 'a beach, a city street, a forest, an office' },
  { label: 'Đếm số', value: 'one cat, two cats, three cats, no cats' },
  { label: 'Phong cách', value: 'a photograph, a cartoon, a painting, a 3D render' },
  { label: 'Trừu tượng', value: 'something cute, something scary, something expensive' },
]

function parseLabels(raw: string): string[] {
  return raw
    .split(',')
    .map((s) => s.trim())
    .filter((s) => s.length > 0)
}

export function ClipDemo() {
  const model = useModel('clip')

  const [image, setImage] = useState<PreparedImage | null>(null)
  const [labelsRaw, setLabelsRaw] = useState(LABEL_PRESETS[0].value)
  const [template, setTemplate] = useState(DEFAULT_TEMPLATE)
  const [result, setResult] = useState<Prediction[] | null>(null)
  const [debug, setDebug] = useState<DebugInfo | undefined>(undefined)

  const candidates = useMemo(() => parseLabels(labelsRaw), [labelsRaw])
  const templateOk = template.includes('{}')
  const canRun = image != null && candidates.length >= 2 && templateOk && !model.isBusy

  async function handleRun() {
    if (!image) return
    try {
      // candidate_labels là tham số VỊ TRÍ thứ hai của pipeline, không nằm trong
      // options — nên phải truyền qua `args`. Xem WorkerRequest trong protocol.ts.
      const res = await model.run<Prediction[]>(image.src, {
        args: [candidates],
        pipelineOptions: { hypothesis_template: template },
      })
      setResult(res.output)
      setDebug(res.debug)
    } catch {
      setResult(null)
      setDebug(undefined)
    }
  }

  return (
    <>
      <ModelBar model={model} />

      <ImagePicker
        samples={SAMPLES}
        value={image}
        onChange={(next) => {
          setImage(next)
          setResult(null)
        }}
        disabled={model.isBusy}
      />

      <div className="config-grid">
        <div className="field">
          <div className="field-head">
            <label className="field-label" htmlFor="clip-labels">
              Nhãn của bạn
            </label>
            <Badge mono>{candidates.length}</Badge>
            <Info title="Ảnh và chữ nằm trong CÙNG một không gian vector">
              <p>
                CLIP có hai nhánh: một encoder ảnh và một encoder văn bản. Chúng được huấn luyện
                cùng nhau trên 400 triệu cặp (ảnh, chú thích) với mục tiêu <em>contrastive</em>: kéo
                vector của cặp đúng lại gần nhau, đẩy mọi cặp sai ra xa.
              </p>
              <p>Kết quả là phân loại ảnh thu về đúng phép tính ở card Embedding:</p>
              <Tex tex="\text{score}_k = \cos\big(f_{\text{ảnh}}(I),\; f_{\text{chữ}}(t_k)\big)" block />
              <p>
                Nhãn không còn là chỉ số 0…999 khoá cứng lúc huấn luyện, mà là một{' '}
                <strong>câu tiếng Anh được mã hoá lúc chạy</strong>. Nên bạn gõ nhãn nào cũng được.
              </p>
              <p className="hint">
                Cùng ý tưởng với card Zero-shot: biến bài toán phân loại thành bài toán so khớp
                ngôn ngữ. Khác ở chỗ CLIP so ảnh với chữ, còn NLI so chữ với chữ.
              </p>
            </Info>
            <div className="chip-row">
              {LABEL_PRESETS.map((preset) => (
                <button
                  key={preset.label}
                  type="button"
                  className={`chip${labelsRaw === preset.value ? ' chip--active' : ''}`}
                  onClick={() => {
                    setLabelsRaw(preset.value)
                    setResult(null)
                  }}
                >
                  {preset.label}
                </button>
              ))}
            </div>
          </div>
          <textarea
            id="clip-labels"
            className="input--compact"
            value={labelsRaw}
            onChange={(e) => {
              setLabelsRaw(e.target.value)
              setResult(null)
            }}
          />
        </div>

        <div className="field">
          <div className="field-head">
            <label className="field-label" htmlFor="clip-template">
              Mẫu câu
            </label>
            <Info title="“a photo of a {}” không phải chi tiết thừa">
              <p>
                CLIP học từ chú thích ảnh trên Internet, mà chú thích thường là một câu chứ không
                phải một từ trơ trọi. Bọc nhãn vào một câu giống dữ liệu huấn luyện làm vector chữ
                nằm đúng vùng mà encoder quen — nhóm tác giả CLIP đo được điều này cải thiện độ
                chính xác trên ImageNet vài phần trăm.
              </p>
              <p>Thử đổi giữa các mẫu trên cùng một ảnh và bộ nhãn:</p>
              <table className="data">
                <tbody>
                  <tr>
                    <td className="mono">{'{}'}</td>
                    <td>nhãn trơ, không có ngữ cảnh</td>
                  </tr>
                  <tr>
                    <td className="mono">a photo of a {'{}'}</td>
                    <td>mặc định, hợp với ảnh chụp</td>
                  </tr>
                  <tr>
                    <td className="mono">a blurry photo of a {'{}'}</td>
                    <td>hợp với ảnh mờ, kém nét</td>
                  </tr>
                  <tr>
                    <td className="mono">a cartoon of a {'{}'}</td>
                    <td>hợp với hình vẽ</td>
                  </tr>
                </tbody>
              </table>
              <p className="hint">
                Đây là prompt engineering, chỉ là cho ảnh. Cùng bản chất với ô "Mẫu giả thuyết" ở
                card Zero-shot.
              </p>
            </Info>
          </div>
          <textarea
            id="clip-template"
            className="input--compact"
            value={template}
            onChange={(e) => {
              setTemplate(e.target.value)
              setResult(null)
            }}
          />
          {!templateOk && (
            <p className="field-hint">
              Mẫu phải chứa <code>{'{}'}</code> — đó là chỗ nhãn được chèn vào.
            </p>
          )}
        </div>
      </div>

      <div className="control-row">
        <button type="button" className="primary" onClick={handleRun} disabled={!canRun}>
          {model.status === 'running' ? 'Đang so khớp…' : `So ảnh với ${candidates.length} nhãn`}
        </button>
      </div>

      {!result && !model.isBusy && (
        <div className="empty-state">
          Bấm để mã hoá ảnh và từng nhãn thành vector, rồi đo góc giữa chúng.
        </div>
      )}

      {result && result.length > 0 && (
        <>
          <div className="verdict">
            <span className="verdict-label">{result[0].label}</span>
            <Badge mono>{formatPercent(result[0].score, 2)}</Badge>
            <Info title="Con số này là xác suất TƯƠNG ĐỐI giữa các nhãn bạn gõ">
              <p>
                Các điểm cosine được đưa qua softmax trên đúng những nhãn bạn nhập, nên chúng luôn
                cộng thành 1 — <strong>kể cả khi không nhãn nào đúng</strong>.
              </p>
              <p>
                Thử ảnh Pikachu với bộ nhãn "Cảnh vật": CLIP vẫn phải chọn một cái và vẫn đưa ra một
                con số trông rất tự tin. Bỏ nhãn đúng ra khỏi danh sách là cách nhanh nhất để thấy
                điều đó.
              </p>
              <p className="hint">
                Muốn có ngưỡng "không nhãn nào đúng" thì phải tự thêm một nhãn nền kiểu "a random
                photo" và tự đặt ngưỡng — CLIP không tự làm việc đó.
              </p>
            </Info>
          </div>

          <div className="pred-list">
            {result.map((prediction, i) => (
              <div
                key={prediction.label}
                className={`pred-row${i === 0 ? ' pred-row--winner' : ''}`}
              >
                <span className="pred-label" title={prediction.label}>
                  {prediction.label}
                </span>
                <div className="pred-track">
                  <div
                    className="pred-bar"
                    style={{
                      width: `${prediction.score * 100}%`,
                      background: i === 0 ? 'var(--positive)' : 'var(--accent)',
                      opacity: i === 0 ? 1 : 0.45,
                    }}
                  />
                </div>
                <span className="pred-value">{formatPercent(prediction.score, 2)}</span>
              </div>
            ))}
          </div>
        </>
      )}

      <UnderTheHood debug={debug}>
        {result && (
          <div className="hood-block">
            <h4>
              Các câu thật sự được mã hoá
              <Info title="Chi phí: ảnh 1 lần, chữ N lần">
                <p>
                  Encoder ảnh chạy <strong>một</strong> lần. Encoder văn bản chạy{' '}
                  <strong>{candidates.length}</strong> lần — mỗi nhãn một lần.
                </p>
                <p>
                  Nhưng vector chữ chỉ phụ thuộc vào nhãn, không phụ thuộc ảnh. Nên trong sản phẩm
                  thật, bộ nhãn cố định được mã hoá trước một lần rồi lưu lại; lúc chạy chỉ còn một
                  phép nhân ma trận — y hệt cách vector database làm ở card Embedding.
                </p>
                <p className="hint">
                  Đảo lại cũng đúng: mã hoá trước hàng triệu <em>ảnh</em> rồi tìm bằng câu chữ, đó
                  chính là tìm kiếm ảnh theo ngôn ngữ tự nhiên.
                </p>
              </Info>
            </h4>
            <div className="table-scroll">
              <table className="data">
                <thead>
                  <tr>
                    <th>Câu được mã hoá</th>
                    <th className="num">softmax</th>
                  </tr>
                </thead>
                <tbody>
                  {result.map((prediction, i) => (
                    <tr key={prediction.label} className={i === 0 ? 'row--active' : undefined}>
                      <td className="mono">{template.replace('{}', prediction.label)}</td>
                      <td className="num">{prediction.score.toFixed(4)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <ul className="notes">
              <li>
                Encoder ảnh: 1 forward pass · encoder văn bản: {candidates.length} forward pass.
              </li>
              <li>
                Đây là lý do file model nặng 154 MB: nó gộp cả nhánh ảnh (89 MB) lẫn nhánh văn bản
                (64 MB).
              </li>
            </ul>
          </div>
        )}
      </UnderTheHood>
    </>
  )
}
