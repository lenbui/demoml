import { useMemo, useState } from 'react'

import { Badge } from '../../components/Badge'
import { Info } from '../../components/Info'
import { Math as Tex } from '../../components/Math'
import { ModelBar } from '../../components/ModelBar'
import { UnderTheHood } from '../../components/UnderTheHood'
import { useModel } from '../../hooks/useModel'
import { formatPercent } from '../../lib/format'
import { softmax } from '../../lib/math'
import type { DebugInfo, PairScoreOutput } from '../../workers/protocol'

const DEFAULT_TEMPLATE = 'This example is {}.'

const EXAMPLES: Array<{ label: string; text: string; labels: string; note?: string }> = [
  {
    label: 'Chủ đề',
    text: 'The new graphics card delivers twice the frame rate at half the power draw.',
    labels: 'technology, cooking, politics, sports',
  },
  {
    label: 'Ý định',
    text: 'My order arrived broken and nobody answers the support line.',
    labels: 'complaint, praise, question, refund request',
    note: 'Bộ nhãn này chưa từng tồn tại lúc huấn luyện model. Không cần fine-tune gì cả.',
  },
  {
    label: 'Nhãn khó',
    text: 'The lecture slides were fine but the lab was a waste of time.',
    labels: 'positive, negative, mixed',
    note: 'Thêm nhãn "mixed" — thứ mà bộ phân loại 2 nhãn ở card cảm xúc không thể diễn đạt.',
  },
  {
    label: 'Tiếng Việt',
    text: 'Khóa học này thật tuyệt vời, tôi học được rất nhiều điều bổ ích.',
    labels: 'positive, negative',
    note: 'Model NLI này chỉ học tiếng Anh — kết quả gần như vô nghĩa, nhưng vẫn trả về số.',
  },
]

/** Nhận diện index của nhãn NLI theo tên, không hardcode theo vị trí. */
function findNliIndices(labels: string[]) {
  const find = (needle: string) => labels.findIndex((l) => l.toLowerCase().includes(needle))
  return {
    entail: find('entail'),
    neutral: find('neutral'),
    contradict: find('contradic'),
  }
}

function parseLabels(raw: string): string[] {
  return raw
    .split(',')
    .map((s) => s.trim())
    .filter((s) => s.length > 0)
}

export function ZeroShotDemo() {
  const model = useModel('zero-shot')

  const [text, setText] = useState(EXAMPLES[0].text)
  const [labelsRaw, setLabelsRaw] = useState(EXAMPLES[0].labels)
  const [template, setTemplate] = useState(DEFAULT_TEMPLATE)
  const [multiLabel, setMultiLabel] = useState(false)
  const [note, setNote] = useState<string | undefined>(undefined)
  const [result, setResult] = useState<{
    candidates: string[]
    hypotheses: string[]
    nliLabels: string[]
    logits: number[][]
  } | null>(null)
  const [debug, setDebug] = useState<DebugInfo | undefined>(undefined)

  const candidates = useMemo(() => parseLabels(labelsRaw), [labelsRaw])
  const templateOk = template.includes('{}')
  const canRun = text.trim().length > 0 && candidates.length >= 2 && templateOk && !model.isBusy

  /**
   * Từ logits NLI thô ra điểm cho từng nhãn người dùng nhập.
   *
   * Hai chế độ, đúng như thư viện Transformers làm:
   *  • single-label : lấy logit ENTAILMENT của mọi nhãn rồi softmax NGANG các
   *                   nhãn -> tổng bằng 1, các nhãn cạnh tranh nhau.
   *  • multi-label  : với TỪNG nhãn, softmax trên 2 giá trị
   *                   [entailment, contradiction] -> mỗi nhãn độc lập, tổng có
   *                   thể lớn hơn 1.
   */
  const scored = useMemo(() => {
    if (!result) return null
    const { entail, contradict } = findNliIndices(result.nliLabels)
    if (entail < 0) return null

    // Phân phối NLI đầy đủ của từng cặp — phần "mở hộp đen" của demo này.
    const nliProbabilities = result.logits.map((row) => softmax(row))

    let scores: number[]
    if (multiLabel && contradict >= 0) {
      scores = result.logits.map((row) => {
        const [pEntail] = softmax([row[entail], row[contradict]])
        return pEntail
      })
    } else {
      scores = softmax(result.logits.map((row) => row[entail]))
    }

    const order = candidates
      .map((label, i) => ({ label, score: scores[i], index: i }))
      .sort((a, b) => b.score - a.score)

    return { scores, order, nliProbabilities, entail, contradict }
  }, [result, multiLabel, candidates])

  async function handleRun() {
    const hypotheses = candidates.map((label) => template.replace('{}', label))
    try {
      // Mỗi nhãn thành MỘT cặp (câu gốc, giả thuyết). Toàn bộ chạy trong 1 batch.
      const res = await model.run<PairScoreOutput>({
        a: candidates.map(() => text),
        b: hypotheses,
      })
      setResult({
        candidates: [...candidates],
        hypotheses,
        nliLabels: res.output.labels,
        logits: res.output.logits,
      })
      setDebug(res.debug)
    } catch {
      setResult(null)
      setDebug(undefined)
    }
  }

  return (
    <>
      <ModelBar model={model} />

      <div className="field">
        <div className="field-head">
          <label className="field-label" htmlFor="zs-input">
            Câu cần phân loại
          </label>
          <Info title="Zero-shot: phân loại mà không huấn luyện">
            <p>
              Model này <strong>chưa từng thấy nhãn của bạn</strong>. Nó chỉ được huấn luyện trên một
              task duy nhất: Natural Language Inference — cho hai câu, quyết định câu thứ hai là{' '}
              <em>suy ra được</em> (entailment), <em>trung tính</em>, hay <em>trái ngược</em>
              (contradiction) với câu thứ nhất.
            </p>
            <p>Mẹo biến NLI thành bộ phân loại bất kỳ:</p>
            <ul className="notes">
              <li>câu của bạn làm <strong>premise</strong>;</li>
              <li>mỗi nhãn được nhét vào một mẫu câu để thành <strong>hypothesis</strong>;</li>
              <li>nhãn nào có xác suất entailment cao nhất thì thắng.</li>
            </ul>
            <p className="hint">
              Không có gì huyền bí ở đây — bạn thấy được toàn bộ các cặp câu thật sự được đưa vào
              model ở bảng dưới cùng.
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
                  setLabelsRaw(example.labels)
                  setNote(example.note)
                  setResult(null)
                  setDebug(undefined)
                }}
              >
                {example.label}
              </button>
            ))}
          </div>
        </div>
        <textarea
          id="zs-input"
          className="input--compact"
          value={text}
          onChange={(e) => {
            setText(e.target.value)
            setNote(undefined)
            setResult(null)
          }}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && (e.ctrlKey || e.metaKey) && canRun) void handleRun()
          }}
        />
        {note && <p className="field-hint">{note}</p>}
      </div>

      <div className="config-grid">
        <div className="field">
          <div className="field-head">
            <label className="field-label" htmlFor="zs-labels">
              Nhãn của bạn
            </label>
            <Badge mono>{candidates.length}</Badge>
          </div>
          <textarea
            id="zs-labels"
            className="input--compact"
            value={labelsRaw}
            onChange={(e) => {
              setLabelsRaw(e.target.value)
              setResult(null)
            }}
            placeholder="technology, cooking, politics"
          />
          <p className="hint">Cách nhau bằng dấu phẩy. Gõ nhãn nào cũng được, kể cả nhãn bạn vừa nghĩ ra.</p>
        </div>

        <div className="field">
          <div className="field-head">
            <label className="field-label" htmlFor="zs-template">
              Mẫu giả thuyết
            </label>
            <Info title="Mẫu câu không phải chi tiết vô hại">
              <p>
                Nhãn <strong>không</strong> là một ký hiệu trung tính. Nó bị nhét vào một câu tiếng
                Anh và model phải hiểu câu đó — nên cách bạn viết mẫu ảnh hưởng trực tiếp tới kết
                quả.
              </p>
              <p>Thử đổi qua lại trên cùng một câu và bộ nhãn:</p>
              <table className="data">
                <tbody>
                  <tr>
                    <td className="mono">This example is {'{}'}.</td>
                    <td>mặc định của thư viện, chung chung</td>
                  </tr>
                  <tr>
                    <td className="mono">This text is about {'{}'}.</td>
                    <td>hướng về chủ đề</td>
                  </tr>
                  <tr>
                    <td className="mono">The customer is {'{}'}.</td>
                    <td>hướng về cảm xúc/ý định</td>
                  </tr>
                </tbody>
              </table>
              <p className="hint">
                Đây là dạng prompt engineering sớm nhất: chất lượng phụ thuộc vào việc diễn đạt bài
                toán bằng ngôn ngữ mà model đã quen.
              </p>
            </Info>
          </div>
          <textarea
            id="zs-template"
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
          <label className="toggle">
            <input
              type="checkbox"
              checked={multiLabel}
              onChange={(e) => setMultiLabel(e.target.checked)}
            />
            multi-label (các nhãn không cạnh tranh)
          </label>
        </div>
      </div>

      <div className="control-row">
        <button type="button" className="primary" onClick={handleRun} disabled={!canRun}>
          {model.status === 'running' ? 'Đang chạy…' : `Phân loại · ${candidates.length} cặp câu`}
        </button>
        <span className="kbd-hint">Ctrl + Enter</span>
      </div>

      {!result && !model.isBusy && (
        <div className="empty-state">
          Bấm Phân loại — mỗi nhãn sẽ thành một cặp câu đưa vào model NLI.
        </div>
      )}

      {result && scored && (
        <>
          <div className="metrics">
            <Badge mono tone="concept">
              {result.candidates.length} nhãn = {result.candidates.length} forward pass
            </Badge>
            <Badge mono>{multiLabel ? 'multi-label' : 'single-label'}</Badge>
            <Badge mono>NLI: {result.nliLabels.join(' · ')}</Badge>
            <Info title="Hai cách quy đổi ra điểm cuối">
              <p>
                <strong>single-label</strong> — lấy logit entailment của mọi nhãn rồi softmax ngang
                các nhãn:
              </p>
              <Tex tex="p_k = \frac{e^{z^{\text{entail}}_k}}{\sum_j e^{z^{\text{entail}}_j}}" block />
              <p>Tổng bằng 1, các nhãn cạnh tranh nhau — dùng khi đúng một nhãn đúng.</p>
              <p>
                <strong>multi-label</strong> — với từng nhãn, softmax trên đúng 2 giá trị:
              </p>
              <Tex tex="p_k = \frac{e^{z^{\text{entail}}_k}}{e^{z^{\text{entail}}_k} + e^{z^{\text{contra}}_k}}" block />
              <p>
                Mỗi nhãn được xét độc lập nên tổng có thể lớn hơn 1 — dùng khi một câu có thể mang
                nhiều nhãn.
              </p>
              <p className="hint">
                Cùng một lần chạy model, chỉ khác cách đọc logits. Bật/tắt ô multi-label để thấy con
                số đổi mà không phải chạy lại.
              </p>
            </Info>
          </div>

          <div className="pred-list">
            {scored.order.map((entry, rank) => (
              <div
                key={entry.label}
                className={`pred-row${rank === 0 ? ' pred-row--winner' : ''}`}
              >
                <span className="pred-label">{entry.label}</span>
                <div className="pred-track">
                  <div
                    className="pred-bar"
                    style={{
                      width: `${entry.score * 100}%`,
                      background: rank === 0 ? 'var(--positive)' : 'var(--accent)',
                      opacity: rank === 0 ? 1 : 0.45,
                    }}
                  />
                </div>
                <span className="pred-value">{formatPercent(entry.score, 2)}</span>
              </div>
            ))}
          </div>
        </>
      )}

      <UnderTheHood debug={debug}>
        {result && scored && (
          <div className="hood-block">
            <h4>
              Các cặp câu thật sự được đưa vào model
              <Info title="Đây là toàn bộ 'phép màu' của zero-shot">
                <p>
                  Mỗi hàng là một lần chạy model trên chuỗi ghép{' '}
                  <code>[CLS] câu-của-bạn [SEP] giả-thuyết [SEP]</code>.
                </p>
                <p>
                  Ba cột entail/neutral/contra là phân phối NLI gốc của model. Cột cuối là điểm sau
                  khi quy đổi. Nhìn bảng này là hiểu vì sao đổi mẫu câu lại làm đổi kết quả.
                </p>
                <p className="hint">
                  Chú ý chi phí: N nhãn = N cặp câu = N forward pass. Zero-shot rẻ về công sức nhưng
                  đắt về tính toán so với một bộ phân loại đã fine-tune (1 forward pass cho mọi
                  nhãn).
                </p>
              </Info>
            </h4>
            <div className="table-scroll">
              <table className="data">
                <thead>
                  <tr>
                    <th>Giả thuyết (hypothesis)</th>
                    {result.nliLabels.map((label) => (
                      <th className="num" key={label}>
                        {label.toLowerCase().slice(0, 7)}
                      </th>
                    ))}
                    <th className="num">điểm</th>
                  </tr>
                </thead>
                <tbody>
                  {result.candidates.map((label, i) => (
                    <tr key={label} className={scored.order[0]?.index === i ? 'row--active' : undefined}>
                      <td className="mono">{result.hypotheses[i]}</td>
                      {scored.nliProbabilities[i].map((p, j) => (
                        <td className="num" key={j}>
                          {p.toFixed(4)}
                        </td>
                      ))}
                      <td className="num">{scored.scores[i].toFixed(4)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <ul className="notes">
              <li>
                Thứ tự nhãn NLI của model này là <code>{result.nliLabels.join(', ')}</code> — đọc từ{' '}
                <code>config.id2label</code>, không hardcode: bart-large-mnli xếp ngược lại, hardcode
                index sẽ cho kết quả sai mà không báo lỗi.
              </li>
            </ul>
          </div>
        )}
      </UnderTheHood>
    </>
  )
}
