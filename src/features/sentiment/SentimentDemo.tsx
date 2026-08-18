import { useMemo, useState, type ReactNode } from 'react'

import { Badge } from '../../components/Badge'
import { Info } from '../../components/Info'
import { Math as Tex } from '../../components/Math'
import { ModelBar } from '../../components/ModelBar'
import { UnderTheHood } from '../../components/UnderTheHood'
import { useModel } from '../../hooks/useModel'
import { argmax, entropyBits, softmax } from '../../lib/math'
import { formatPercent } from '../../lib/format'
import type { DebugInfo } from '../../workers/protocol'

/** Output của runner 'sequence-classification' trong pipeline.worker.ts. */
interface SentimentOutput {
  labels: string[]
  logits: number[]
}

interface Example {
  label: string
  text: string
  /** Một dòng ngắn hiện ngay dưới input. */
  hint?: string
  /** Phần giải thích dài, nằm sau nút "?". */
  detail?: ReactNode
}

const EXAMPLES: Example[] = [
  {
    label: 'Tích cực',
    text: 'This course completely changed how I think about machine learning. Best class I have taken.',
  },
  {
    label: 'Trái chiều',
    text: 'The lecture slides were well prepared, but the lab sessions were a complete waste of time.',
    hint: 'Câu có cả ý tốt và ý xấu — nhưng model vẫn rất “chắc chắn”.',
    detail: (
      <>
        <p>
          Model chọn NEGATIVE với 99.97% và entropy chỉ 0.004 bit, tức gần như tuyệt đối chắc
          chắn.
        </p>
        <p>
          Bài học: bộ phân loại 2 nhãn <strong>không có cách nào</strong> diễn đạt “trái chiều”.
          Softmax luôn buộc tổng bằng 1 trên đúng hai lựa chọn, nên độ tự tin cao không hề phản
          ánh việc câu đó có rõ ràng hay không.
        </p>
      </>
    ),
  },
  {
    label: 'Phủ định kép',
    text: 'I cannot say that I did not enjoy this course.',
    hint: 'Nghĩa thật là tích cực. Model trả lời sai — và sai một cách rất tự tin.',
    detail: (
      <>
        <p>NEGATIVE 99.87%, entropy 0.014 bit. Sai, nhưng tự tin hơn cả câu tích cực rõ ràng.</p>
        <p>Hai vấn đề lộ ra cùng lúc:</p>
        <ul className="notes">
          <li>
            model học <strong>tương quan bề mặt</strong> — thấy “cannot”, “did not” là nghiêng về
            negative, chứ không suy luận logic phủ định kép;
          </li>
          <li>
            mạng neural <strong>overconfident</strong>, nên xác suất cao không dùng được để phát
            hiện câu trả lời sai.
          </li>
        </ul>
      </>
    ),
  },
  {
    label: 'Tiếng Việt',
    text: 'Khóa học này thật tuyệt vời, tôi học được rất nhiều điều bổ ích.',
    hint: 'Model chỉ học tiếng Anh. Mở Under the hood để thấy câu bị cắt vụn.',
    detail: (
      <>
        <p>
          Kết quả là NEGATIVE cho một câu rõ ràng tích cực. Nguyên nhân nằm ở tokenizer, không
          phải ở model: đây là WordPiece tiếng Anh <code>uncased</code>, nên “khóa” bị bỏ dấu rồi
          cắt thành <code>k</code> + <code>##ho</code> + <code>##a</code>.
        </p>
        <p>
          Model nhận vào một chuỗi mảnh vụn vô nghĩa. Luôn kiểm tra tokenizer và dữ liệu huấn
          luyện trước khi dùng một model cho ngôn ngữ khác.
        </p>
        <p className="hint">Card Tokenizer Explorer ở trên cho so sánh trực tiếp với model đa ngữ.</p>
      </>
    ),
  },
]

export function SentimentDemo() {
  const model = useModel('sentiment')

  const [text, setText] = useState(EXAMPLES[0].text)
  const [example, setExample] = useState<Example | null>(null)
  const [result, setResult] = useState<SentimentOutput | null>(null)
  const [debug, setDebug] = useState<DebugInfo | undefined>(undefined)
  /** Ngưỡng quyết định cho nhãn POSITIVE. */
  const [threshold, setThreshold] = useState(0.5)

  // Softmax được tính Ở ĐÂY, không phải trong model: model chỉ trả logits.
  const analysis = useMemo(() => {
    if (!result) return null
    const probabilities = softmax(result.logits)
    const predictedIndex = argmax(probabilities)
    const positiveIndex = result.labels.findIndex((l) => /pos/i.test(l))
    return {
      probabilities,
      predictedIndex,
      positiveIndex,
      entropy: entropyBits(probabilities),
      maxLogit: Math.max(...result.logits),
    }
  }, [result])

  async function handleRun() {
    try {
      const res = await model.run<SentimentOutput>(text)
      setResult(res.output)
      setDebug(res.debug)
    } catch {
      // Lỗi đã được useModel đưa vào model.error và ModelBar hiển thị.
      setResult(null)
      setDebug(undefined)
    }
  }

  const canRun = text.trim().length > 0 && !model.isBusy

  const thresholdDecision =
    analysis && analysis.positiveIndex >= 0 && result
      ? analysis.probabilities[analysis.positiveIndex] >= threshold
        ? result.labels[analysis.positiveIndex]
        : result.labels[1 - analysis.positiveIndex]
      : null

  return (
    <>
      <div className="field">
        <div className="field-head">
          <label className="field-label" htmlFor="sentiment-input">
            Câu cần phân loại
          </label>
          <div className="chip-row">
            {EXAMPLES.map((item) => (
              <button
                key={item.label}
                type="button"
                className={`chip${example?.label === item.label ? ' chip--active' : ''}`}
                onClick={() => {
                  setText(item.text)
                  setExample(item)
                  setResult(null)
                  setDebug(undefined)
                }}
              >
                {item.label}
              </button>
            ))}
          </div>
        </div>
        <textarea
          id="sentiment-input"
          className="input--compact"
          value={text}
          onChange={(e) => {
            setText(e.target.value)
            setExample(null)
          }}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && (e.ctrlKey || e.metaKey) && canRun) void handleRun()
          }}
          placeholder="Nhập một câu tiếng Anh…"
        />
        {example?.hint && (
          <p className="field-hint">
            {example.hint}
            {example.detail && (
              <Info title={example.label}>{example.detail}</Info>
            )}
          </p>
        )}
      </div>

      <ModelBar model={model} />

      <div className="control-row">
        <button type="button" className="primary" onClick={handleRun} disabled={!canRun}>
          {model.status === 'running' ? 'Đang phân loại…' : 'Phân loại'}
        </button>
        <span className="kbd-hint">Ctrl + Enter</span>
      </div>

      {!result && !model.isBusy && (
        <div className="empty-state">Bấm Phân loại để chạy model ngay trong trình duyệt.</div>
      )}

      {result && analysis && (
        <>
          <div className="verdict">
            <span
              className="verdict-label"
              style={{
                color: /pos/i.test(result.labels[analysis.predictedIndex])
                  ? 'var(--positive)'
                  : 'var(--negative)',
              }}
            >
              {result.labels[analysis.predictedIndex]}
            </span>
            <Badge mono>{formatPercent(analysis.probabilities[analysis.predictedIndex], 2)}</Badge>
            <Badge mono tone={analysis.entropy > 0.5 ? 'warn' : 'default'}>
              entropy {analysis.entropy.toFixed(3)} bit
            </Badge>
            <Info title="Entropy — model đang do dự bao nhiêu">
              <Tex tex="H = -\sum_i p_i \log_2 p_i" block />
              <table className="data">
                <tbody>
                  <tr>
                    <td className="mono">0 bit</td>
                    <td>chắc chắn tuyệt đối</td>
                  </tr>
                  <tr>
                    <td className="mono">1 bit</td>
                    <td>hoàn toàn do dự (với 2 nhãn: 50/50)</td>
                  </tr>
                </tbody>
              </table>
              <p className="hint">
                Tổng quát: entropy tối đa là <Tex tex="\log_2 n" /> với <Tex tex="n" /> nhãn.
              </p>
            </Info>
          </div>

          <div className="pred-list">
            {result.labels.map((label, i) => {
              const p = analysis.probabilities[i]
              const isWinner = i === analysis.predictedIndex
              return (
                <div key={label} className={`pred-row${isWinner ? ' pred-row--winner' : ''}`}>
                  <span className="pred-label">{label}</span>
                  <div className="pred-track">
                    <div
                      className="pred-bar"
                      style={{
                        width: `${p * 100}%`,
                        background: /pos/i.test(label) ? 'var(--positive)' : 'var(--negative)',
                        opacity: isWinner ? 1 : 0.4,
                      }}
                    />
                  </div>
                  <span className="pred-value">{formatPercent(p, 2)}</span>
                </div>
              )
            })}
          </div>

          {analysis.positiveIndex >= 0 && (
            <div className="field">
              <div className="field-head">
                <label className="field-label" htmlFor="sentiment-threshold">
                  Ngưỡng quyết định
                </label>
                <Badge mono>{threshold.toFixed(2)}</Badge>
                <Badge tone={/pos/i.test(thresholdDecision ?? '') ? 'ok' : 'danger'}>
                  {thresholdDecision}
                </Badge>
                <Info title="Ngưỡng và đánh đổi precision–recall">
                  <p>
                    Mặc định ta lấy nhãn có xác suất cao nhất (argmax), tương đương ngưỡng 0.50.
                    Nhưng ngưỡng là một <strong>lựa chọn</strong>, không phải thuộc tính của model.
                  </p>
                  <table className="data">
                    <tbody>
                      <tr>
                        <td>Hạ ngưỡng</td>
                        <td>bắt được nhiều câu tích cực hơn (recall ↑), báo sai nhiều hơn (precision ↓)</td>
                      </tr>
                      <tr>
                        <td>Nâng ngưỡng</td>
                        <td>chỉ báo khi rất chắc (precision ↑), bỏ sót nhiều hơn (recall ↓)</td>
                      </tr>
                    </tbody>
                  </table>
                  <p>
                    <Tex tex="\text{precision} = \frac{TP}{TP+FP}" />
                    {' · '}
                    <Tex tex="\text{recall} = \frac{TP}{TP+FN}" />
                  </p>
                  <p className="hint">
                    Đây là lý do một mô hình phân loại không thể tóm lại bằng một con số “độ chính
                    xác” duy nhất.
                  </p>
                </Info>
              </div>
              <input
                id="sentiment-threshold"
                type="range"
                min={0.05}
                max={0.95}
                step={0.01}
                value={threshold}
                onChange={(e) => setThreshold(Number(e.target.value))}
              />
            </div>
          )}
        </>
      )}

      <UnderTheHood debug={debug}>
        {result && analysis && (
          <div className="hood-block">
            <h4>
              Logits → Softmax → Xác suất
              <Info title="Vì sao phải trừ max(z)">
                <p>
                  <Tex tex="e^{1000}" /> tràn số thành <code>Infinity</code> và kết quả ra{' '}
                  <code>NaN</code>. Trừ đi một hằng số bất kỳ không làm đổi kết quả:
                </p>
                <Tex
                  tex="\frac{e^{z_i - C}}{\sum_j e^{z_j - C}} = \frac{e^{z_i}/e^{C}}{\sum_j e^{z_j}/e^{C}} = \frac{e^{z_i}}{\sum_j e^{z_j}}"
                  block
                />
                <p>
                  Chọn <Tex tex="C = \max(z)" /> thì mọi số hạng <Tex tex="e^{z_i - C} \le 1" />.
                  Mẹo “numerical stability” này có trong mọi thư viện ML.
                </p>
              </Info>
              <Info title="Softmax không làm model chính xác hơn">
                <p>
                  Softmax chỉ <em>chuẩn hoá</em> logits thành một phân phối. Nó không thêm thông
                  tin nào.
                </p>
                <p>
                  Xác suất 99% <strong>không</strong> có nghĩa model đúng 99% số lần — thử ví dụ
                  “Phủ định kép” để thấy model sai ở mức 99.87%. Việc làm cho xác suất khớp với tần
                  suất đúng thực tế gọi là <em>calibration</em>, và cần một bước hiệu chỉnh riêng.
                </p>
              </Info>
            </h4>

            <Tex tex="p_i = \frac{e^{z_i - \max(z)}}{\sum_j e^{z_j - \max(z)}}" block />

            <div className="table-scroll">
              <table className="data">
                <thead>
                  <tr>
                    <th>Nhãn</th>
                    <th className="num">z</th>
                    <th className="num">z − max</th>
                    <th className="num">e^(z−max)</th>
                    <th className="num">p</th>
                  </tr>
                </thead>
                <tbody>
                  {result.labels.map((label, i) => {
                    const z = result.logits[i]
                    const shifted = z - analysis.maxLogit
                    return (
                      <tr key={label}>
                        <td>{label}</td>
                        <td className="num">{z.toFixed(4)}</td>
                        <td className="num">{shifted.toFixed(4)}</td>
                        <td className="num">{globalThis.Math.exp(shifted).toFixed(4)}</td>
                        <td className="num">{analysis.probabilities[i].toFixed(4)}</td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </UnderTheHood>
    </>
  )
}
