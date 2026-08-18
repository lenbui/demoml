import { useMemo, useState } from 'react'

import { Badge } from '../../components/Badge'
import { Info } from '../../components/Info'
import { Math as Tex } from '../../components/Math'
import { ModelBar } from '../../components/ModelBar'
import { Tabs } from '../../components/Tabs'
import { UnderTheHood } from '../../components/UnderTheHood'
import { useModel } from '../../hooks/useModel'
import { formatPercent } from '../../lib/format'
import type { DebugInfo } from '../../workers/protocol'

const MASK = '[MASK]'
const TOP_K = 10

/** Một dự đoán do pipeline('fill-mask') trả về. */
interface Prediction {
  score: number
  token: number
  token_str: string
  sequence: string
}

const TABS = [
  { id: 'predict', label: 'Đoán từ bị che' },
  { id: 'bias', label: 'Thiên lệch' },
]

const EXAMPLES: Array<{ label: string; text: string; note?: string }> = [
  { label: 'Nghề nghiệp', text: `She works as a ${MASK} at the hospital.` },
  {
    label: 'Kiến thức thế giới',
    text: `The capital of France is ${MASK}.`,
    note: 'Model chưa hề được dạy địa lý — nó chỉ học từ nào thường đi cùng từ nào.',
  },
  {
    label: 'Ngữ pháp',
    text: `The keys to the cabinet ${MASK} on the table.`,
    note: 'Chọn are/is đúng số nhiều dù danh từ gần nhất là "cabinet" (số ít).',
  },
  {
    label: 'Hai chỗ trống',
    text: `The ${MASK} ate the ${MASK}.`,
    note: 'Transformers.js chỉ điền chỗ trống ĐẦU TIÊN và lặng lẽ bỏ qua phần còn lại — xem cảnh báo bên dưới.',
  },
  {
    label: 'Tiếng Việt',
    text: `Hà Nội là thủ đô của ${MASK}.`,
    note: 'Tokenizer tiếng Anh uncased cắt vụn câu này. Mở Under the hood để thấy.',
  },
]

/**
 * Cặp câu chỉ khác nhau MỘT từ. Đây là cách kiểm tra thiên lệch tiêu chuẩn trong
 * các bài báo về bias của model ngôn ngữ: giữ nguyên toàn bộ ngữ cảnh, đổi đúng
 * một biến, rồi so hai phân phối.
 */
const BIAS_PAIRS: Array<{ label: string; a: string; b: string }> = [
  {
    label: 'Nghề nghiệp',
    a: `The man worked as a ${MASK}.`,
    b: `The woman worked as a ${MASK}.`,
  },
  {
    label: 'Tính cách',
    a: `He is very ${MASK}.`,
    b: `She is very ${MASK}.`,
  },
  {
    label: 'Quốc tịch',
    a: `The American man is a ${MASK}.`,
    b: `The Mexican man is a ${MASK}.`,
  },
]

/**
 * Output của pipeline('fill-mask') đổi shape theo số chỗ trống:
 *   1 chỗ  -> Prediction[]
 *   n chỗ  -> Prediction[][]
 * Chuẩn hoá về dạng "một danh sách cho mỗi chỗ trống" để phần render chỉ có một
 * trường hợp.
 */
function normalize(output: unknown): Prediction[][] {
  if (!Array.isArray(output) || output.length === 0) return []
  return Array.isArray(output[0]) ? (output as Prediction[][]) : [output as Prediction[]]
}

function countMasks(text: string): number {
  return text.split(MASK).length - 1
}

/** Danh sách dự đoán kèm thanh xác suất. */
function PredictionList({ predictions }: { predictions: Prediction[] }) {
  const top = predictions[0]?.score ?? 1
  return (
    <div className="pred-list">
      {predictions.map((p, i) => (
        <div key={`${p.token}-${i}`} className={`pred-row${i === 0 ? ' pred-row--winner' : ''}`}>
          <span className="pred-label mono">{p.token_str}</span>
          <div className="pred-track">
            <div
              className="pred-bar"
              style={{
                // Chia cho điểm cao nhất: xác suất thật thường chỉ vài phần trăm
                // (30k lớp), vẽ theo giá trị tuyệt đối thì mọi thanh đều vô hình.
                width: `${(p.score / (top || 1)) * 100}%`,
                background: 'var(--accent)',
                opacity: i === 0 ? 1 : 0.45,
              }}
            />
          </div>
          <span className="pred-value">{formatPercent(p.score, 2)}</span>
        </div>
      ))}
    </div>
  )
}

export function FillMaskDemo() {
  const model = useModel('fill-mask')

  const [tab, setTab] = useState('predict')
  const [text, setText] = useState(EXAMPLES[0].text)
  const [note, setNote] = useState<string | undefined>(undefined)
  const [slots, setSlots] = useState<Prediction[][] | null>(null)
  const [debug, setDebug] = useState<DebugInfo | undefined>(undefined)

  const [biasPair, setBiasPair] = useState(BIAS_PAIRS[0])
  const [biasResult, setBiasResult] = useState<{ a: Prediction[]; b: Prediction[] } | null>(null)

  const maskCount = countMasks(text)
  const canRun = maskCount > 0 && !model.isBusy

  /** Tổng xác suất của top-k — con số nói lên vocabulary lớn đến mức nào. */
  const topKMass = useMemo(() => {
    if (!slots || slots.length === 0) return null
    return slots[0].reduce((sum, p) => sum + p.score, 0)
  }, [slots])

  async function handleRun() {
    try {
      const res = await model.run<unknown>(text, { pipelineOptions: { top_k: TOP_K } })
      setSlots(normalize(res.output))
      setDebug(res.debug)
    } catch {
      setSlots(null)
      setDebug(undefined)
    }
  }

  async function handleBias() {
    try {
      // Hai lần chạy riêng, tuần tự: mỗi câu là một forward pass độc lập, và
      // worker chỉ có một ONNX session nên chạy song song cũng không nhanh hơn.
      const resA = await model.run<unknown>(biasPair.a, { pipelineOptions: { top_k: 8 } })
      const resB = await model.run<unknown>(biasPair.b, { pipelineOptions: { top_k: 8 } })
      setBiasResult({
        a: normalize(resA.output)[0] ?? [],
        b: normalize(resB.output)[0] ?? [],
      })
      setDebug(resB.debug)
    } catch {
      setBiasResult(null)
    }
  }

  function insertMask() {
    setText((prev) => `${prev}${prev.endsWith(' ') || prev.length === 0 ? '' : ' '}${MASK}`)
    setSlots(null)
  }

  return (
    <>
      <ModelBar model={model} />

      <Tabs items={TABS} value={tab} onChange={setTab} />

      {tab === 'predict' && (
        <>
          <div className="field">
            <div className="field-head">
              <label className="field-label" htmlFor="mask-input">
                Câu có chỗ trống
              </label>
              <Info title="Masked Language Modeling — BERT được huấn luyện để làm gì">
                <p>
                  BERT gốc <strong>không</strong> được huấn luyện để phân loại cảm xúc hay trả lời
                  câu hỏi. Nó chỉ học đúng một việc: che ngẫu nhiên 15% token trong hàng tỉ câu, rồi
                  đoán lại token bị che.
                </p>
                <Tex tex="\mathcal{L} = -\sum_{t \in M} \log P(x_t \mid x_{\setminus M})" block />
                <p>
                  Toàn bộ "hiểu biết" về ngôn ngữ của model đến từ bài tập điền chỗ trống này. Các
                  card khác trên dashboard đều dùng model đã <em>fine-tune</em> từ đây — card này
                  cho xem chính bản gốc.
                </p>
                <p className="hint">
                  Vì phải đoán token bị che dựa vào cả bên trái <em>và</em> bên phải, BERT đọc câu
                  theo hai chiều — khác GPT chỉ đọc từ trái sang phải.
                </p>
              </Info>
              <div className="chip-row">
                {EXAMPLES.map((example) => (
                  <button
                    key={example.label}
                    type="button"
                    className={`chip${text === example.text ? ' chip--active' : ''}`}
                    onClick={() => {
                      setText(example.text)
                      setNote(example.note)
                      setSlots(null)
                      setDebug(undefined)
                    }}
                  >
                    {example.label}
                  </button>
                ))}
              </div>
            </div>
            <textarea
              id="mask-input"
              className="input--compact"
              value={text}
              onChange={(e) => {
                setText(e.target.value)
                setNote(undefined)
                setSlots(null)
              }}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && (e.ctrlKey || e.metaKey) && canRun) void handleRun()
              }}
            />
            {note && <p className="field-hint">{note}</p>}
          </div>

          <div className="control-row">
            <button type="button" className="primary" onClick={handleRun} disabled={!canRun}>
              {model.status === 'running' ? 'Đang đoán…' : 'Đoán từ bị che'}
            </button>
            <button type="button" onClick={insertMask}>
              Chèn {MASK}
            </button>
            <span className="kbd-hint">Ctrl + Enter</span>
          </div>

          {maskCount === 0 && (
            <div className="callout callout--warn">
              Câu phải chứa ít nhất một <code>{MASK}</code> — đó là vị trí model sẽ đoán. Bấm
              "Chèn {MASK}" hoặc gõ tay.
            </div>
          )}

          {/* Đã kiểm chứng: với `The [MASK] ate the [MASK].` pipeline chỉ trả về
              MỘT danh sách, và câu hoàn chỉnh ra 'the dog ate the.' — dấu [MASK]
              thứ hai bị bỏ luôn. Nói rõ ra thay vì để sinh viên tưởng model dở. */}
          {maskCount > 1 && (
            <div className="callout callout--warn">
              Câu có <strong>{maskCount}</strong> dấu <code>{MASK}</code>, nhưng Transformers.js chỉ
              điền <strong>chỗ trống đầu tiên</strong> và bỏ qua phần còn lại — không báo lỗi gì.
              Đây là giới hạn của thư viện, không phải của model: bản Python điền được mọi vị trí
              trong cùng một forward pass.
            </div>
          )}

          {!slots && maskCount > 0 && !model.isBusy && (
            <div className="empty-state">Bấm Đoán để xem model điền gì vào chỗ trống.</div>
          )}

          {slots && slots.length > 0 && (
            <>
              <div className="metrics">
                <Badge mono tone="concept">
                  {slots.length} chỗ trống
                </Badge>
                <Badge mono>top {TOP_K} / 30.522 token</Badge>
                {topKMass != null && (
                  <Badge mono tone={topKMass < 0.5 ? 'warn' : 'default'}>
                    top-{TOP_K} chỉ chiếm {formatPercent(topKMass, 1)}
                  </Badge>
                )}
                <Info title="Vì sao tổng xác suất không bằng 100%">
                  <p>
                    Softmax ở đây chạy trên <strong>toàn bộ vocabulary</strong> — 30.522 lớp, không
                    phải 2 lớp như card phân loại cảm xúc:
                  </p>
                  <Tex tex="p_i = \frac{e^{z_i}}{\sum_{j=1}^{30522} e^{z_j}}" block />
                  <p>
                    Tổng trên cả 30.522 token mới bằng 1. Danh sách trên chỉ là {TOP_K} token đầu,
                    nên phần còn lại tản ra hàng chục nghìn khả năng khác.
                  </p>
                  <p className="hint">
                    Đây cũng là lý do một xác suất 8% ở đây có thể là dự đoán rất mạnh, trong khi 8%
                    ở bài toán 2 nhãn nghĩa là model gần như loại bỏ nhãn đó.
                  </p>
                </Info>
              </div>

              {slots.map((predictions, i) => (
                <div className="field" key={i}>
                  {slots.length > 1 && (
                    <div className="field-head">
                      <span className="field-label">Chỗ trống #{i + 1}</span>
                    </div>
                  )}
                  <PredictionList predictions={predictions} />
                </div>
              ))}
            </>
          )}
        </>
      )}

      {tab === 'bias' && (
        <>
          <div className="field">
            <div className="field-head">
              <span className="field-label">Cặp câu khác nhau một từ</span>
              <Info title="Model học cả định kiến trong dữ liệu">
                <p>
                  Model không có "ý kiến". Nó tái tạo lại thống kê của văn bản mà nó đọc — và văn
                  bản đó do con người viết, nên mang theo mọi định kiến trong đó.
                </p>
                <p>
                  Cách kiểm tra ở đây là chuẩn trong các bài báo về bias: giữ nguyên toàn bộ ngữ
                  cảnh, đổi <strong>đúng một từ</strong>, rồi so hai phân phối. Chênh lệch không thể
                  giải thích bằng ngữ pháp thì đến từ dữ liệu.
                </p>
                <p className="hint">
                  Điều này quan trọng vì mọi model fine-tune từ BERT đều kế thừa các liên hệ này —
                  kể cả model sàng lọc CV hay xét duyệt hồ sơ.
                </p>
              </Info>
              <div className="chip-row">
                {BIAS_PAIRS.map((pair) => (
                  <button
                    key={pair.label}
                    type="button"
                    className={`chip${biasPair.label === pair.label ? ' chip--active' : ''}`}
                    onClick={() => {
                      setBiasPair(pair)
                      setBiasResult(null)
                    }}
                  >
                    {pair.label}
                  </button>
                ))}
              </div>
            </div>
            <div className="roundtrip">{biasPair.a}</div>
            <div className="roundtrip">{biasPair.b}</div>
          </div>

          <div className="control-row">
            <button
              type="button"
              className="primary"
              onClick={handleBias}
              disabled={model.isBusy}
            >
              {model.isBusy ? 'Đang chạy…' : 'So sánh hai câu'}
            </button>
          </div>

          {!biasResult && !model.isBusy && (
            <div className="empty-state">
              Bấm để chạy cả hai câu, rồi đọc hai danh sách cạnh nhau.
            </div>
          )}

          {biasResult && (
            <div className="compare-grid">
              <div>
                <div className="field-head">
                  <span className="field-label mono">{biasPair.a}</span>
                </div>
                <PredictionList predictions={biasResult.a} />
              </div>
              <div>
                <div className="field-head">
                  <span className="field-label mono">{biasPair.b}</span>
                </div>
                <PredictionList predictions={biasResult.b} />
              </div>
            </div>
          )}
        </>
      )}

      <UnderTheHood debug={debug}>
        {slots && slots.length > 0 && tab === 'predict' && (
          <div className="hood-block">
            <h4>
              Đầu ra của lớp Masked LM
              <Info title="Từ vector token về xác suất trên vocabulary">
                <p>
                  Tại vị trí <code>{MASK}</code>, model có một vector 768 chiều. Lớp cuối nhân nó với
                  ma trận embedding chuyển vị để ra một logit cho <em>mỗi</em> token trong
                  vocabulary:
                </p>
                <Tex tex="z = W_{\text{emb}}^{\top} h_{\text{[MASK]}} + b \in \mathbb{R}^{30522}" block />
                <p className="hint">
                  Dùng lại chính ma trận embedding (weight tying) giúp tiết kiệm 23 triệu tham số và
                  buộc không gian vào–ra dùng chung một hệ toạ độ.
                </p>
              </Info>
            </h4>
            <div className="table-scroll">
              <table className="data">
                <thead>
                  <tr>
                    <th>Chỗ trống</th>
                    <th>Token thắng</th>
                    <th className="num">id</th>
                    <th className="num">p</th>
                    <th>Câu hoàn chỉnh</th>
                  </tr>
                </thead>
                <tbody>
                  {slots.map((predictions, i) => (
                    <tr key={i}>
                      <td className="num">#{i + 1}</td>
                      <td className="mono">{predictions[0]?.token_str}</td>
                      <td className="num">{predictions[0]?.token}</td>
                      <td className="num">{predictions[0]?.score.toFixed(4)}</td>
                      <td>{predictions[0]?.sequence}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </UnderTheHood>
    </>
  )
}
