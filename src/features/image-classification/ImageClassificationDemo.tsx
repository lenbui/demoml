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

const TOP_K = 8

/** Kích thước đầu vào cố định của ViT-base/16 — đọc từ preprocessor_config.json. */
const INPUT_SIZE = 224
const PATCH = 16

interface Prediction {
  label: string
  score: number
}

const SAMPLES: SampleChoice[] = [
  { name: 'tiger.jpg', label: 'Hổ', hint: 'Một vật thể rõ ràng, đúng kiểu ảnh ImageNet.' },
  { name: 'corgi.jpg', label: 'Chó corgi', hint: 'ImageNet có tới 120 giống chó riêng biệt.' },
  { name: 'butterfly.jpg', label: 'Bướm', hint: 'Nhiều lớp côn trùng rất giống nhau.' },
  {
    name: 'cats.jpg',
    label: 'Hai con mèo',
    hint: 'Nhiều vật thể — nhưng model chỉ được chọn MỘT nhãn.',
  },
  {
    name: 'pikachu.png',
    label: 'Pikachu',
    hint: 'KHÔNG có trong 1000 lớp ImageNet. Model buộc phải đoán bừa.',
  },
]

export function ImageClassificationDemo() {
  const model = useModel('image-classification')

  const [image, setImage] = useState<PreparedImage | null>(null)
  const [result, setResult] = useState<Prediction[] | null>(null)
  const [debug, setDebug] = useState<DebugInfo | undefined>(undefined)

  /** Tổng xác suất top-k — cho thấy 1000 lớp còn lại chiếm bao nhiêu. */
  const topKMass = useMemo(
    () => (result ? result.reduce((sum, p) => sum + p.score, 0) : null),
    [result],
  )

  async function handleRun() {
    if (!image) return
    try {
      const res = await model.run<Prediction[]>(image.src, {
        pipelineOptions: { top_k: TOP_K },
      })
      setResult(res.output)
      setDebug(res.debug)
    } catch {
      setResult(null)
      setDebug(undefined)
    }
  }

  const canRun = image != null && !model.isBusy
  /** Số pixel bị vứt đi khi ép ảnh về 224×224 — con số gây bất ngờ nhất demo. */
  const pixelRatio = image
    ? (image.naturalWidth * image.naturalHeight) / (INPUT_SIZE * INPUT_SIZE)
    : null

  return (
    <>
      <ModelBar model={model} />

      <ImagePicker
        samples={SAMPLES}
        value={image}
        onChange={(next) => {
          setImage(next)
          setResult(null)
          setDebug(undefined)
        }}
        disabled={model.isBusy}
      />

      <div className="control-row">
        <button type="button" className="primary" onClick={handleRun} disabled={!canRun}>
          {model.status === 'running' ? 'Đang phân loại…' : 'Phân loại ảnh'}
        </button>
        {pixelRatio != null && pixelRatio > 2 && (
          <>
            <Badge mono tone="warn">
              vứt {formatPercent(1 - 1 / pixelRatio, 1)} số pixel
            </Badge>
            <Info title="Tiền xử lý ảnh — “tokenizer” của thị giác">
              <p>
                Model có đầu vào <strong>cố định</strong> {INPUT_SIZE}×{INPUT_SIZE}. Mọi ảnh đều bị
                ép về đúng kích thước đó trước khi model nhìn thấy gì:
              </p>
              <ul className="notes">
                <li>thu nhỏ cạnh ngắn rồi cắt giữa (center crop) — mất luôn phần rìa;</li>
                <li>
                  chuẩn hoá mỗi kênh màu về phân phối mà model quen:{' '}
                  <Tex tex="x' = (x - \mu) / \sigma" />;
                </li>
                <li>
                  cắt thành {(INPUT_SIZE / PATCH) ** 2} ô vuông {PATCH}×{PATCH} — mỗi ô là một
                  “token” đi vào Transformer, y hệt subword của văn bản.
                </li>
              </ul>
              <p>
                Ảnh {image?.naturalWidth}×{image?.naturalHeight} của bạn có{' '}
                {((image!.naturalWidth * image!.naturalHeight) / 1000).toFixed(0)}k pixel, model chỉ
                nhận {((INPUT_SIZE * INPUT_SIZE) / 1000).toFixed(0)}k.
              </p>
              <p className="hint">
                Đây là cùng một bài học ở card Tokenizer Explorer: bước tiền xử lý không hề trung
                tính, nó quyết định model được nhìn thấy gì. Vật thể nhỏ nằm ở rìa ảnh thì coi như
                không tồn tại.
              </p>
            </Info>
          </>
        )}
      </div>

      {!result && !model.isBusy && (
        <div className="empty-state">Chọn ảnh rồi bấm Phân loại để xem 1000 lớp ImageNet.</div>
      )}

      {result && result.length > 0 && (
        <>
          <div className="verdict">
            <span className="verdict-label">{result[0].label}</span>
            <Badge mono>{formatPercent(result[0].score, 2)}</Badge>
            {topKMass != null && (
              <Badge mono tone={topKMass < 0.5 ? 'warn' : 'default'}>
                top-{TOP_K} chiếm {formatPercent(topKMass, 1)}
              </Badge>
            )}
            <Info title="1000 lớp — và không có lớp “không biết”">
              <p>
                Softmax chạy trên đúng 1000 lớp của ImageNet-1k. Chúng luôn cộng lại thành 1, nên
                model <strong>bắt buộc</strong> phải chia hết xác suất cho 1000 nhãn đó — kể cả khi
                vật trong ảnh chẳng thuộc nhãn nào.
              </p>
              <p>
                Thử ảnh <strong>Pikachu</strong>: nó không có trong ImageNet, nhưng model vẫn trả về
                một nhãn kèm một con số. Không có cách nào để nó nói “tôi không biết”.
              </p>
              <p>
                Dấu hiệu nhận ra: xác suất cao nhất thấp và trải đều trên nhiều nhãn. Nhưng đó chỉ
                là <em>dấu hiệu</em>, không phải cơ chế — mạng neural vẫn có thể rất tự tin mà sai,
                y như ở card phân loại cảm xúc.
              </p>
              <p className="hint">
                Đây chính là vấn đề mà card CLIP giải quyết: nhãn ở đó do bạn gõ ra lúc chạy, không
                bị khoá cứng từ lúc huấn luyện.
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
                      width: `${(prediction.score / result[0].score) * 100}%`,
                      background: 'var(--accent)',
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
        {image && (
          <div className="hood-block">
            <h4>
              Ảnh của bạn trở thành tensor như thế nào
              <Info title="Từ pixel đến chuỗi token">
                <p>
                  Vision Transformer không có gì đặc biệt so với BERT ngoài bước đầu vào: thay vì
                  cắt câu thành subword, nó cắt ảnh thành các ô vuông.
                </p>
                <Tex tex="n_{\text{patch}} = \left(\frac{224}{16}\right)^2 = 196" block />
                <p>
                  196 ô, cộng một token <code>[CLS]</code> ở đầu, cho ra chuỗi 197 token — đúng dạng
                  mà Transformer xử lý. Vector <code>[CLS]</code> ở lớp cuối là thứ đi vào bộ phân
                  loại 1000 lớp.
                </p>
              </Info>
            </h4>
            <table className="data">
              <tbody>
                <tr>
                  <td>Ảnh gốc</td>
                  <td className="mono">
                    {image.naturalWidth} × {image.naturalHeight}
                  </td>
                </tr>
                <tr>
                  <td>Model nhận vào</td>
                  <td className="mono">
                    {INPUT_SIZE} × {INPUT_SIZE} × 3
                  </td>
                </tr>
                <tr>
                  <td>Số ô (patch)</td>
                  <td className="mono">
                    ({INPUT_SIZE}/{PATCH})² = {(INPUT_SIZE / PATCH) ** 2}
                  </td>
                </tr>
                <tr>
                  <td>Chuỗi vào Transformer</td>
                  <td className="mono">[1, {(INPUT_SIZE / PATCH) ** 2 + 1}, 768] — gồm cả [CLS]</td>
                </tr>
                <tr>
                  <td>Số lớp đầu ra</td>
                  <td className="mono">1000 (ImageNet-1k)</td>
                </tr>
              </tbody>
            </table>
          </div>
        )}
      </UnderTheHood>
    </>
  )
}
