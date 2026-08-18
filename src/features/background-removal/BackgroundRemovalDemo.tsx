import { useEffect, useMemo, useState } from 'react'

import { Badge } from '../../components/Badge'
import { ImagePicker } from '../../components/ImagePicker'
import { Info } from '../../components/Info'
import { Math as Tex } from '../../components/Math'
import { ModelBar } from '../../components/ModelBar'
import { UnderTheHood } from '../../components/UnderTheHood'
import { VariantPicker } from '../../components/VariantPicker'
import { useModel } from '../../hooks/useModel'
import { formatMs, formatPercent } from '../../lib/format'
import type { PreparedImage } from '../../lib/image'
import {
  alphaStats,
  compositeCutout,
  maskToDataUrl,
  type AlphaMask,
  type AlphaStats,
  type Background,
} from '../../lib/matte'
import type { SampleChoice } from '../../lib/samples'
import type { DebugInfo } from '../../workers/protocol'

/**
 * Một phần tử của pipeline('image-segmentation') khi model không có bước
 * post_process_* nào — tức là các model matting như MODNet / RMBG.
 *
 * `label` và `score` luôn null: không có nhãn nào để gán, vì model chỉ trả lời
 * "thuộc chủ thể bao nhiêu phần", không trả lời "đây là con gì".
 */
interface SegmentationResult {
  label: string | null
  score: number | null
  mask: AlphaMask
}

const SAMPLES: SampleChoice[] = [
  {
    name: 'portrait-of-woman_small.jpg',
    label: 'Chân dung',
    hint: 'Tóc bay — chỗ duy nhất thấy rõ alpha là số thực, không phải 0/1.',
  },
  { name: 'corgi.jpg', label: 'Chó corgi', hint: 'Lông xù: rìa mềm, nhưng không phải người.' },
  {
    name: 'butterfly.jpg',
    label: 'Bướm',
    hint: 'Không có người trong ảnh — MODNet sẽ gãy, RMBG thì không.',
  },
  {
    name: 'cats.jpg',
    label: 'Hai con mèo',
    hint: 'Hai chủ thể. Mặt nạ không phân biệt được đâu là con nào.',
  },
  {
    name: 'city-streets.jpg',
    label: 'Phố',
    hint: 'Không có "một chủ thể nổi bật" nào — bài toán vô nghĩa với ảnh này.',
  },
]

/**
 * Kích thước đầu vào của từng model, đọc từ preprocessor_config.json.
 *
 * Để ở đây chứ không hardcode một con số chung, vì chênh lệch này chính là bài
 * học: MODNet co ảnh theo CẠNH NGẮN nên giữ nguyên tỉ lệ, còn RMBG ép cứng
 * 1024×1024 nên ảnh bị bóp méo trước khi model kịp nhìn.
 */
const INPUT_SIZE: Record<string, { text: string; note: string }> = {
  modnet: {
    text: 'cạnh ngắn 512 (bội số của 32)',
    note: 'giữ nguyên tỉ lệ khung hình',
  },
  rmbg: {
    text: '1024 × 1024 cố định',
    note: 'ảnh bị bóp méo về hình vuông',
  },
}

type View = 'original' | 'mask' | 'cutout'

const BACKGROUNDS: Array<{ id: string; label: string; background: Background }> = [
  { id: 'transparent', label: 'Trong suốt', background: { kind: 'transparent' } },
  { id: 'white', label: 'Trắng', background: { kind: 'color', color: '#ffffff' } },
  { id: 'green', label: 'Phông xanh', background: { kind: 'color', color: '#00b140' } },
  { id: 'magenta', label: 'Hồng sẫm', background: { kind: 'color', color: '#c2185b' } },
]

interface VariantResult {
  mask: AlphaMask
  stats: AlphaStats
  ms: number
}

export function BackgroundRemovalDemo() {
  const model = useModel('background-removal')
  const variants = model.spec.variants ?? []

  const [variantId, setVariantId] = useState(variants[0]?.id ?? 'default')
  const [image, setImage] = useState<PreparedImage | null>(null)
  const [results, setResults] = useState<Record<string, VariantResult>>({})
  const [debug, setDebug] = useState<DebugInfo | undefined>(undefined)

  const [view, setView] = useState<View>('cutout')
  const [backgroundId, setBackgroundId] = useState('transparent')
  /** null = giữ alpha mềm. Số = nhị phân hoá ở ngưỡng đó. */
  const [threshold, setThreshold] = useState<number | null>(null)
  const [cutout, setCutout] = useState<string | null>(null)
  const [maskUrl, setMaskUrl] = useState<string | null>(null)
  const [compositeError, setCompositeError] = useState<string | null>(null)

  const current = results[variantId]
  const background =
    BACKGROUNDS.find((b) => b.id === backgroundId)?.background ?? BACKGROUNDS[0].background

  /**
   * Vẽ lại ảnh mỗi khi mặt nạ, nền hoặc ngưỡng đổi.
   *
   * Chú ý: KHÔNG chạy lại model. Đổi nền và kéo ngưỡng chỉ đụng tới vòng lặp
   * pixel trong lib/matte.ts — đó là cách demo chứng minh rằng "xoá nền" nằm ở
   * bước hậu xử lý, không nằm trong mạng neural.
   */
  useEffect(() => {
    if (!image || !current) {
      setCutout(null)
      setMaskUrl(null)
      return
    }

    let cancelled = false
    setCompositeError(null)

    void (async () => {
      try {
        const url = await compositeCutout(image.src, current.mask, background, {
          hardThreshold: threshold,
        })
        if (cancelled) return
        setCutout(url)
        setMaskUrl(maskToDataUrl(current.mask, threshold))
      } catch (err) {
        if (cancelled) return
        setCutout(null)
        setMaskUrl(null)
        setCompositeError(
          err instanceof Error ? err.message : 'Không ghép được ảnh từ mặt nạ alpha.',
        )
      }
    })()

    return () => {
      cancelled = true
    }
  }, [image, current, background, threshold])

  async function runVariant(id: string) {
    if (!image) return
    try {
      const res = await model.run<SegmentationResult[]>(image.src, { variantId: id })
      const mask = res.output?.[0]?.mask
      if (!mask) throw new Error('Model không trả về mặt nạ nào.')
      setResults((prev) => ({
        ...prev,
        [id]: { mask, stats: alphaStats(mask), ms: res.ms },
      }))
      setDebug(res.debug)
    } catch {
      setResults((prev) => {
        const next = { ...prev }
        delete next[id]
        return next
      })
      setDebug(undefined)
    }
  }

  function resetResults() {
    setResults({})
    setDebug(undefined)
    setCutout(null)
    setMaskUrl(null)
    setCompositeError(null)
  }

  const canRun = image != null && !model.isBusy

  /** Ảnh đang hiện ở khung xem, tuỳ tab. */
  const shown = view === 'original' ? (image?.src ?? null) : view === 'mask' ? maskUrl : cutout

  /** So sánh hai model khi cả hai đã chạy trên cùng tấm ảnh. */
  const comparison = useMemo(
    () =>
      variants
        .map((variant) => ({ variant, result: results[variant.id] }))
        .filter((row) => row.result != null),
    [variants, results],
  )

  return (
    <>
      <VariantPicker
        variants={variants}
        value={variantId}
        onChange={setVariantId}
        readyVariants={model.readyVariants}
        loadingVariantId={model.loadingVariantId}
        disabled={model.isBusy}
      />
      <ModelBar model={model} variantId={variantId} />

      <ImagePicker
        samples={SAMPLES}
        value={image}
        onChange={(next) => {
          setImage(next)
          resetResults()
        }}
        disabled={model.isBusy}
      />

      <div className="control-row">
        <button type="button" className="primary" onClick={() => void runVariant(variantId)} disabled={!canRun}>
          {model.status === 'running' ? 'Đang tách nền…' : 'Tách nền'}
        </button>
        <Info title="Model trả về cái gì?">
          <p>
            Không phải một tấm ảnh. Model trả về một <strong>ma trận alpha</strong> đúng bằng kích
            thước ảnh: mỗi pixel một số trong <Tex tex="[0, 1]" />, trả lời câu hỏi{' '}
            <em>“pixel này thuộc chủ thể bao nhiêu phần”</em>.
          </p>
          <p>
            Ảnh nền trong suốt là do đoạn code ở <code>src/lib/matte.ts</code> gán ma trận đó vào
            kênh alpha của ảnh gốc — một vòng lặp gán, không có mạng neural nào ở bước này:
          </p>
          <Tex tex="\text{RGBA}_i = (R_i,\; G_i,\; B_i,\; 255 \cdot \alpha_i)" block />
          <p className="hint">
            Đổi nền và kéo ngưỡng bên dưới đều <strong>không</strong> chạy lại model — chỉ chạy lại
            vòng lặp đó. Cùng một bài học với slider ngưỡng ở card Phát hiện vật thể.
          </p>
        </Info>
      </div>

      {compositeError && <div className="callout callout--danger">{compositeError}</div>}

      {!current && !model.isBusy && (
        <div className="empty-state">Chọn ảnh rồi bấm Tách nền để xem mặt nạ alpha model sinh ra.</div>
      )}

      {current && (
        <>
          <div className="field">
            <div className="field-head">
              <span className="field-label">Xem</span>
              <div className="chip-row">
                {(
                  [
                    ['original', 'Ảnh gốc'],
                    ['mask', 'Mặt nạ alpha'],
                    ['cutout', 'Đã tách nền'],
                  ] as Array<[View, string]>
                ).map(([id, label]) => (
                  <button
                    key={id}
                    type="button"
                    className={`chip${view === id ? ' chip--active' : ''}`}
                    onClick={() => setView(id)}
                  >
                    {label}
                  </button>
                ))}
              </div>
            </div>

            {view === 'cutout' && (
              <div className="field-head">
                <span className="field-label">Nền</span>
                <div className="chip-row">
                  {BACKGROUNDS.map((option) => (
                    <button
                      key={option.id}
                      type="button"
                      className={`chip${backgroundId === option.id ? ' chip--active' : ''}`}
                      onClick={() => setBackgroundId(option.id)}
                    >
                      {option.label}
                    </button>
                  ))}
                </div>
              </div>
            )}

            {shown && (
              <div className="image-stage image-stage--alpha">
                <img src={shown} alt="Kết quả tách nền" className="image-preview" />
              </div>
            )}

            {view === 'cutout' && cutout && (
              <div className="control-row">
                <a className="chip" href={cutout} download="tach-nen.png">
                  ⬇ Tải ảnh PNG
                </a>
                <span className="hint">
                  Chỉ PNG giữ được kênh alpha. Xuất ra JPEG là mất sạch phần trong suốt.
                </span>
              </div>
            )}
          </div>

          <div className="field">
            <div className="field-head">
              <label className="field-label" htmlFor="matte-threshold">
                Nhị phân hoá mặt nạ
              </label>
              <Badge mono tone={threshold == null ? 'default' : 'warn'}>
                {threshold == null ? 'alpha mềm (đúng)' : `ngưỡng ${threshold.toFixed(2)}`}
              </Badge>
              <Info title="Vì sao gọi là matting chứ không phải segmentation">
                <p>
                  Phân đoạn nhị phân trả lời <em>có / không</em> cho mỗi pixel. Nhưng một sợi tóc
                  mảnh hơn một pixel: pixel đó vừa là tóc vừa là nền. Ép nó về 0 hoặc 1 là vứt đi
                  thông tin có thật.
                </p>
                <p>
                  Kéo slider để ép mặt nạ về nhị phân rồi nhìn vào rìa tóc. Ảnh lập tức trông như
                  hình dán: đó chính là{' '}
                  <strong>{formatPercent(current.stats.partial / current.stats.total, 2)}</strong>{' '}
                  số pixel có alpha nằm lưng chừng bị làm tròn đi.
                </p>
                <p className="hint">
                  Bấm “alpha mềm” để trả về mặc định. Model không đổi, chỉ có bước hậu xử lý đổi.
                </p>
              </Info>
            </div>
            <div className="control-row">
              <input
                id="matte-threshold"
                type="range"
                min={0.05}
                max={0.95}
                step={0.05}
                value={threshold ?? 0.5}
                onChange={(e) => setThreshold(Number(e.target.value))}
              />
              <button
                type="button"
                className={`chip${threshold == null ? ' chip--active' : ''}`}
                onClick={() => setThreshold(null)}
              >
                alpha mềm
              </button>
            </div>
          </div>

          <div className="verdict">
            <span className="verdict-label">
              Chủ thể chiếm {formatPercent(current.stats.coverage, 1)} diện tích
            </span>
            <Badge mono>{formatMs(current.ms)}</Badge>
            <Badge mono tone={current.stats.partial / current.stats.total > 0.02 ? 'warn' : 'default'}>
              {formatPercent(current.stats.partial / current.stats.total, 2)} pixel bán trong suốt
            </Badge>
          </div>

          {comparison.length > 1 && (
            <div className="field">
              <div className="field-head">
                <span className="field-label">So sánh hai model</span>
                <Info title="Chuyên biệt vs tổng quát">
                  <p>
                    MODNet nhỏ hơn nhưng chỉ biết một thứ: <strong>người</strong>. Nó không có khái
                    niệm “vật thể nổi bật”. Thử ảnh <strong>Bướm</strong>: nó trả về một mặt nạ gần
                    như rỗng — chủ thể chiếm khoảng 0.1% diện tích, tức là <em>không tìm thấy gì</em>.
                  </p>
                  <p>
                    Và nó không báo lỗi. Không có ngoại lệ nào được ném ra, không có điểm tin cậy nào
                    tụt xuống — bạn vẫn nhận về một mảng số đúng kích thước ảnh, chỉ toàn số 0. Muốn
                    biết model vừa thất bại thì phải tự nhìn vào kết quả mà đoán, y như card Phân
                    loại ảnh với tấm ảnh Pikachu.
                  </p>
                  <p>
                    RMBG-1.4 được huấn luyện để tách bất kỳ chủ thể nổi bật nào, nên dùng được cho
                    ảnh chung — đổi lại nó ép mọi ảnh về 1024×1024 và tốn nhiều phép tính hơn hẳn.
                  </p>
                  <p>
                    Nhìn cột <strong>pixel lưng chừng</strong>: hai model cho ra dải chuyển tiếp rất
                    khác nhau ở rìa. Đây không phải chuyện model nào “tốt hơn” — đó là hai mục tiêu
                    huấn luyện khác nhau (matting chân dung vs phân đoạn vật thể nổi bật) hiện thẳng
                    ra trong output. Ghép người vào phông nền mới thì dải mềm là thứ bạn cần; cắt
                    ảnh sản phẩm cho trang bán hàng thì rìa sắc mới đúng.
                  </p>
                  <p className="hint">
                    Cùng một đánh đổi ở card Phát hiện vật thể: dung lượng file đo số tham số, không
                    đo khối lượng tính toán — model nhỏ hơn ở đây lại là model chạy nhanh hơn, nhưng
                    vì nó chạy được trên WebGPU chứ không phải vì nó ít tham số.
                  </p>
                </Info>
              </div>
              <table className="data">
                <thead>
                  <tr>
                    <th>Model</th>
                    <th>Đầu vào</th>
                    <th>Thời gian</th>
                    <th>Diện tích chủ thể</th>
                    <th>Pixel lưng chừng</th>
                  </tr>
                </thead>
                <tbody>
                  {comparison.map(({ variant, result }) => (
                    <tr key={variant.id}>
                      <td>{variant.label}</td>
                      <td className="mono">{INPUT_SIZE[variant.id]?.text ?? '—'}</td>
                      <td className="mono">{formatMs(result!.ms)}</td>
                      <td className="mono">{formatPercent(result!.stats.coverage, 1)}</td>
                      <td className="mono">
                        {formatPercent(result!.stats.partial / result!.stats.total, 2)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </>
      )}

      <UnderTheHood debug={debug}>
        {current && image && (
          <>
            <div className="hood-block">
              <h4>
                Phân bố giá trị alpha
                <Info title="Đọc biểu đồ này thế nào">
                  <p>
                    Trục ngang là giá trị alpha từ 0 (nền) tới 1 (chủ thể), chia thành 16 khoảng.
                    Gần như toàn bộ pixel dồn về hai đầu — model rất tự tin ở phần lớn ảnh.
                  </p>
                  <p>
                    Phần bụng ở giữa nhỏ nhưng là phần quan trọng nhất: đó là rìa chủ thể. Với ảnh
                    chân dung, nó nằm gần hết ở tóc.
                  </p>
                  <p>
                    Model sinh ra dãy số này bằng <strong>sigmoid trên từng pixel</strong>, không
                    phải softmax trên các nhãn:
                  </p>
                  <Tex tex="\alpha_i = \sigma(z_i) = \frac{1}{1 + e^{-z_i}}" block />
                  <p className="hint">
                    Khác biệt cốt lõi với card Phân loại ảnh: ở đó các xác suất cộng lại thành 1 vì
                    chỉ được chọn một nhãn. Ở đây mỗi pixel quyết định độc lập, không có ràng buộc
                    tổng nào cả.
                  </p>
                </Info>
              </h4>
              <div className="pred-list">
                {current.stats.histogram.map((count, i) => {
                  const peak = Math.max(...current.stats.histogram)
                  return (
                    <div key={i} className="pred-row">
                      <span className="pred-label mono">
                        {(i / 16).toFixed(2)}–{((i + 1) / 16).toFixed(2)}
                      </span>
                      <div className="pred-track">
                        <div
                          className="pred-bar"
                          style={{
                            width: `${peak > 0 ? (count / peak) * 100 : 0}%`,
                            background: 'var(--accent)',
                            opacity: i === 0 || i === 15 ? 1 : 0.5,
                          }}
                        />
                      </div>
                      <span className="pred-value mono">
                        {formatPercent(count / current.stats.total, 2)}
                      </span>
                    </div>
                  )
                })}
              </div>
            </div>

            <div className="hood-block">
              <h4>Đường đi của một tấm ảnh</h4>
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
                      {INPUT_SIZE[variantId]?.text ?? '—'}
                      {INPUT_SIZE[variantId] && ` — ${INPUT_SIZE[variantId].note}`}
                    </td>
                  </tr>
                  <tr>
                    <td>Model trả ra</td>
                    <td className="mono">1 kênh × H × W — một số thực cho MỖI pixel</td>
                  </tr>
                  <tr>
                    <td>Mặt nạ sau resize</td>
                    <td className="mono">
                      {current.mask.width} × {current.mask.height} × {current.mask.channels} kênh
                    </td>
                  </tr>
                  <tr>
                    <td>Số giá trị alpha</td>
                    <td className="mono">
                      {current.stats.total.toLocaleString('vi-VN')} — bằng đúng số pixel
                    </td>
                  </tr>
                  <tr>
                    <td>alpha ≈ 0 (nền)</td>
                    <td className="mono">
                      {formatPercent(current.stats.transparent / current.stats.total, 2)}
                    </td>
                  </tr>
                  <tr>
                    <td>alpha ≈ 1 (chủ thể)</td>
                    <td className="mono">
                      {formatPercent(current.stats.opaque / current.stats.total, 2)}
                    </td>
                  </tr>
                  <tr>
                    <td>lưng chừng (rìa chủ thể)</td>
                    <td className="mono">
                      {formatPercent(current.stats.partial / current.stats.total, 2)}
                    </td>
                  </tr>
                </tbody>
              </table>
              <p className="hint">
                Ba dòng cuối dùng dấu <strong>≈</strong> chứ không phải dấu bằng, và đó không phải
                chuyện làm tròn cho đẹp: alpha sinh ra từ sigmoid, mà sigmoid không bao giờ đạt đúng
                0 hay đúng 1 — nó chỉ tiệm cận. Pixel nền thật sự thường ra <code>1/255</code> chứ
                không ra <code>0</code>. Đếm theo dấu bằng tuyệt đối sẽ báo hơn một nửa số pixel là
                “bán trong suốt”, trong khi biểu đồ ngay trên cho thấy chúng nằm gọn ở hai đầu.
              </p>
              <p className="hint">
                Cũng không có dòng “nhãn” nào trong bảng, vì model không sinh ra nhãn nào.{' '}
                <code>label</code> và <code>score</code> mà pipeline trả về đều là <code>null</code>:
                câu hỏi ở đây là “thuộc chủ thể bao nhiêu phần”, không phải “đây là con gì”.
              </p>
            </div>
          </>
        )}
      </UnderTheHood>
    </>
  )
}
