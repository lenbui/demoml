import { useMemo, useState } from 'react'

import { Badge } from '../../components/Badge'
import { BoxOverlay, boxColor, type DetectionBox } from '../../components/BoxOverlay'
import { ImagePicker } from '../../components/ImagePicker'
import { Info } from '../../components/Info'
import { Math as Tex } from '../../components/Math'
import { ModelBar } from '../../components/ModelBar'
import { UnderTheHood } from '../../components/UnderTheHood'
import { VariantPicker } from '../../components/VariantPicker'
import { useModel } from '../../hooks/useModel'
import { formatMs, formatPercent } from '../../lib/format'
import type { PreparedImage } from '../../lib/image'
import type { SampleChoice } from '../../lib/samples'
import type { DebugInfo } from '../../workers/protocol'

/** Một kết quả của pipeline('object-detection') với `percentage: true`. */
interface Detection {
  label: string
  score: number
  box: { xmin: number; ymin: number; xmax: number; ymax: number }
}

interface VariantResult {
  detections: Detection[]
  ms: number
}

const SAMPLES: SampleChoice[] = [
  { name: 'football-match.jpg', label: 'Trận bóng', hint: 'Nhiều người chồng lấn nhau.' },
  { name: 'city-streets.jpg', label: 'Phố', hint: 'Xe cộ đủ mọi kích cỡ, có cái rất nhỏ.' },
  { name: 'cats.jpg', label: 'Hai con mèo', hint: 'Ít vật thể, dễ — dùng để đối chiếu.' },
  { name: 'beach.png', label: 'Bãi biển', hint: 'Vật thể nhỏ, ở xa.' },
]

/**
 * Ngưỡng gửi cho pipeline. Cố ý ĐỂ THẤP rồi lọc lại ở phía UI, để kéo slider
 * không phải chạy lại model — nhờ vậy thấy ngay ngưỡng là một lựa chọn HẬU xử
 * lý, không phải một tham số của model.
 */
const PIPELINE_THRESHOLD = 0.05

export function DetectionDemo() {
  const model = useModel('detection')
  const variants = model.spec.variants ?? []

  const [variantId, setVariantId] = useState(variants[0]?.id ?? 'default')
  const [image, setImage] = useState<PreparedImage | null>(null)
  const [results, setResults] = useState<Record<string, VariantResult>>({})
  const [threshold, setThreshold] = useState(0.5)
  const [debug, setDebug] = useState<DebugInfo | undefined>(undefined)

  const current = results[variantId]

  const visible = useMemo(
    () => (current ? current.detections.filter((d) => d.score >= threshold) : []),
    [current, threshold],
  )

  const boxes: DetectionBox[] = visible.map((d) => ({
    label: d.label,
    score: d.score,
    xmin: d.box.xmin,
    ymin: d.box.ymin,
    xmax: d.box.xmax,
    ymax: d.box.ymax,
    color: boxColor(d.label),
  }))

  /** Đếm theo nhãn — bảng này mới là "kết quả" mà người dùng thật sự muốn. */
  const counts = useMemo(() => {
    const map = new Map<string, number>()
    for (const d of visible) map.set(d.label, (map.get(d.label) ?? 0) + 1)
    return [...map.entries()].sort((a, b) => b[1] - a[1])
  }, [visible])

  async function runVariant(id: string) {
    if (!image) return
    try {
      const res = await model.run<Detection[]>(image.src, {
        variantId: id,
        // percentage: true -> toạ độ về dạng tỉ lệ 0–1, nhờ đó lớp phủ không cần
        // biết ảnh đang hiển thị ở kích thước nào.
        pipelineOptions: { threshold: PIPELINE_THRESHOLD, percentage: true },
      })
      setResults((prev) => ({ ...prev, [id]: { detections: res.output, ms: res.ms } }))
      if (id === variantId) setDebug(res.debug)
    } catch {
      /* lỗi đã hiện ở ModelBar */
    }
  }

  async function runAll() {
    // Tuần tự: hai model tải cùng lúc thì progress bar thành vô nghĩa.
    for (const variant of variants) await runVariant(variant.id)
  }

  const canRun = image != null && !model.isBusy
  const comparedCount = Object.keys(results).length

  return (
    <>
      <div className="field">
        <div className="field-head">
          <span className="field-label">Model</span>
          <Info title="Dung lượng file KHÔNG cho biết model chạy nhanh hay chậm">
            <p>
              Hai model cùng làm một việc, cùng bộ 91 nhãn COCO, chênh nhau 4 lần dung lượng. Bấm
              "Chạy cả 2" rồi đọc bảng bên dưới — kết quả thường ngược với dự đoán.
            </p>
            <p>Đo trên máy viết demo này (WASM, ảnh trận bóng, cả hai đã chạy nóng):</p>
            <table className="data">
              <tbody>
                <tr>
                  <td>YOLOS-tiny (10 MB)</td>
                  <td className="num">7.7 s</td>
                  <td className="num">16 hộp</td>
                </tr>
                <tr>
                  <td>DETR ResNet-50 (43 MB)</td>
                  <td className="num">4.4 s</td>
                  <td className="num">8 hộp</td>
                </tr>
              </tbody>
            </table>
            <p>
              Model <strong>nhỏ hơn 4 lần lại chậm hơn gần 2 lần</strong>. Lý do: YOLOS là một ViT
              thuần, attention phải chạy trên rất nhiều patch token nên khối lượng tính toán lớn dù
              ít tham số. DETR dùng backbone CNN (ResNet-50) vốn rất hiệu quả, phần transformer phía
              sau thì nhỏ.
            </p>
            <p>
              Bài học: dung lượng file đo <strong>số tham số</strong>, còn thời gian chạy phụ thuộc{' '}
              <strong>khối lượng tính toán</strong> (FLOPs). Hai đại lượng đó không tỉ lệ với nhau,
              nên đừng chọn model theo số MB.
            </p>
            <p className="hint">
              Còn 16 hộp so với 8 thì <em>chưa</em> kết luận được model nào tốt hơn — phải tự nhìn
              ảnh xem hộp nào đúng. Hạ ngưỡng xuống 0.05 rồi so hai bên là cách nhanh nhất.
            </p>
          </Info>
        </div>
        <VariantPicker
          variants={variants}
          value={variantId}
          onChange={setVariantId}
          readyVariants={model.readyVariants}
          loadingVariantId={model.loadingVariantId}
          disabled={model.isBusy}
        />
      </div>

      <ModelBar model={model} variantId={variantId} />

      <ImagePicker
        samples={SAMPLES}
        value={image}
        onChange={(next) => {
          setImage(next)
          setResults({})
          setDebug(undefined)
        }}
        disabled={model.isBusy}
      >
        {boxes.length > 0 && <BoxOverlay boxes={boxes} />}
      </ImagePicker>

      <div className="control-row">
        <button
          type="button"
          className="primary"
          onClick={() => runVariant(variantId)}
          disabled={!canRun}
        >
          {model.status === 'running' ? 'Đang phát hiện…' : 'Phát hiện vật thể'}
        </button>
        <button type="button" onClick={runAll} disabled={!canRun}>
          Chạy cả {variants.length} để so sánh
        </button>
      </div>

      {!current && !model.isBusy && (
        <div className="empty-state">Chọn ảnh rồi bấm Phát hiện để vẽ bounding box.</div>
      )}

      {current && (
        <>
          <div className="metrics">
            <Badge mono tone="concept">
              {visible.length} vật thể
            </Badge>
            <Badge mono>{formatMs(current.ms)}</Badge>
            <Badge mono>
              {current.detections.length} hộp thô trước khi lọc
            </Badge>
            {counts.map(([label, count]) => (
              <Badge key={label} mono>
                {label} ×{count}
              </Badge>
            ))}
          </div>

          <div className="field">
            <div className="field-head">
              <label className="field-label" htmlFor="det-threshold">
                Ngưỡng tin cậy
              </label>
              <Badge mono>{threshold.toFixed(2)}</Badge>
              <Info title="Ngưỡng là lựa chọn của bạn, không phải của model">
                <p>
                  Model trả về <strong>{current.detections.length}</strong> hộp cho ảnh này, mỗi hộp
                  kèm một điểm tin cậy. Slider chỉ lọc danh sách đó — kéo qua kéo lại{' '}
                  <em>không</em> chạy lại model, nên bạn thấy ngay đây là bước hậu xử lý thuần tuý.
                </p>
                <p>Cùng một đánh đổi như ở card phân loại cảm xúc, chỉ khác ngữ cảnh:</p>
                <table className="data">
                  <tbody>
                    <tr>
                      <td>Hạ ngưỡng</td>
                      <td>bắt được nhiều vật hơn (recall ↑), nhiều hộp rác hơn (precision ↓)</td>
                    </tr>
                    <tr>
                      <td>Nâng ngưỡng</td>
                      <td>chỉ giữ hộp chắc chắn (precision ↑), bỏ sót nhiều hơn (recall ↓)</td>
                    </tr>
                  </tbody>
                </table>
                <p>
                  <Tex tex="\text{precision} = \frac{TP}{TP+FP}" />
                  {' · '}
                  <Tex tex="\text{recall} = \frac{TP}{TP+FN}" />
                </p>
                <p className="hint">
                  Hãy hạ xuống 0.05 để thấy model thật sự "nghĩ" gì trước khi bị lọc — thường có rất
                  nhiều hộp chồng lên nhau ở mức tin cậy thấp.
                </p>
              </Info>
            </div>
            <input
              id="det-threshold"
              type="range"
              min={PIPELINE_THRESHOLD}
              max={0.95}
              step={0.01}
              value={threshold}
              onChange={(e) => setThreshold(Number(e.target.value))}
            />
          </div>
        </>
      )}

      {comparedCount > 1 && (
        <div className="field">
          <div className="field-head">
            <span className="field-label">So sánh {comparedCount} model</span>
          </div>
          <div className="table-scroll">
            <table className="data">
              <thead>
                <tr>
                  <th>Model</th>
                  <th className="num">MB</th>
                  <th className="num">Vật thể</th>
                  <th className="num">Điểm cao nhất</th>
                  <th className="num">ms</th>
                </tr>
              </thead>
              <tbody>
                {variants
                  .filter((v) => results[v.id])
                  .map((v) => {
                    const r = results[v.id]
                    const kept = r.detections.filter((d) => d.score >= threshold)
                    const best = kept.length ? Math.max(...kept.map((d) => d.score)) : 0
                    return (
                      <tr key={v.id} className={v.id === variantId ? 'row--active' : undefined}>
                        <td>{v.label}</td>
                        <td className="num">{v.approxSizeMB}</td>
                        <td className="num">{kept.length}</td>
                        <td className="num">{kept.length ? formatPercent(best, 1) : '—'}</td>
                        <td className="num">{r.ms}</td>
                      </tr>
                    )
                  })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      <UnderTheHood debug={debug}>
        {current && (
          <div className="hood-block">
            <h4>
              Toàn bộ hộp model trả về
              <Info title="Vì sao toạ độ là tỉ lệ chứ không phải pixel">
                <p>
                  Tuỳ chọn <code>percentage: true</code> cho toạ độ trong khoảng{' '}
                  <Tex tex="[0, 1]" /> thay vì pixel. Cần thiết vì pixel mà model làm việc là pixel
                  của ảnh <em>đã qua tiền xử lý</em> (DETR ép cạnh ngắn về 800), không phải của ảnh
                  gốc bạn đưa vào.
                </p>
                <p>
                  Dùng tỉ lệ thì lớp phủ chỉ cần đặt theo phần trăm và tự đúng ở mọi kích thước hiển
                  thị — kể cả khi ảnh bị thu nhỏ cho vừa card.
                </p>
              </Info>
            </h4>
            <div className="table-scroll">
              <table className="data">
                <thead>
                  <tr>
                    <th>Nhãn</th>
                    <th className="num">Điểm</th>
                    <th className="num">x</th>
                    <th className="num">y</th>
                    <th className="num">rộng</th>
                    <th className="num">cao</th>
                  </tr>
                </thead>
                <tbody>
                  {current.detections
                    .slice()
                    .sort((a, b) => b.score - a.score)
                    .map((d, i) => (
                      <tr
                        key={`${d.label}-${i}`}
                        className={d.score >= threshold ? 'row--active' : undefined}
                      >
                        <td>{d.label}</td>
                        <td className="num">{d.score.toFixed(3)}</td>
                        <td className="num">{d.box.xmin.toFixed(3)}</td>
                        <td className="num">{d.box.ymin.toFixed(3)}</td>
                        <td className="num">{(d.box.xmax - d.box.xmin).toFixed(3)}</td>
                        <td className="num">{(d.box.ymax - d.box.ymin).toFixed(3)}</td>
                      </tr>
                    ))}
                </tbody>
              </table>
            </div>
            <ul className="notes">
              <li>
                Hàng tô đậm là các hộp vượt ngưỡng {threshold.toFixed(2)} hiện tại —{' '}
                {visible.length}/{current.detections.length}.
              </li>
              <li>
                DETR không cần bước NMS (non-maximum suppression) như các model phát hiện đời trước:
                nó dự đoán một số cố định các "query" và được huấn luyện để mỗi query bắt đúng một
                vật thể.
              </li>
            </ul>
          </div>
        )}
      </UnderTheHood>
    </>
  )
}
