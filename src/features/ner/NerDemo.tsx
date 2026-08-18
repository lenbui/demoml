import { useMemo, useState } from 'react'

import { Badge } from '../../components/Badge'
import { Info } from '../../components/Info'
import { Math as Tex } from '../../components/Math'
import { ModelBar } from '../../components/ModelBar'
import { UnderTheHood } from '../../components/UnderTheHood'
import { useModel } from '../../hooks/useModel'
import { formatPercent } from '../../lib/format'
import { softmax } from '../../lib/math'
import {
  ENTITY_LABELS,
  entityColor,
  labelTokens,
  mergeEntities,
  type LabeledToken,
} from '../../lib/ner'
import type { DebugInfo, TokenLabelOutput } from '../../workers/protocol'

const EXAMPLES: Array<{ label: string; text: string; note?: string }> = [
  {
    label: 'Tin tức',
    text: 'Apple announced that Tim Cook will visit Berlin and Paris next March.',
  },
  {
    label: 'Tên bị cắt vụn',
    text: 'Nguyen Thanh Trung works for Vinaconex in Ho Chi Minh City.',
    note: 'Tên tiếng Việt bị WordPiece cắt thành nhiều mảnh — chính chỗ bước gộp phải hoạt động.',
  },
  {
    label: 'Nhập nhằng',
    text: 'Washington told Jordan that Amazon is bigger than Amazon.',
    note: 'Cùng một chữ, hai loại entity khác nhau. Model phải dựa vào ngữ cảnh, không tra từ điển.',
  },
  {
    label: 'Không có entity',
    text: 'The weather was cold and nobody wanted to go outside yesterday.',
    note: 'Mọi token nhận nhãn O. Kiểm tra model có bịa ra entity không.',
  },
  {
    label: 'Tiếng Việt',
    text: 'Trường Đại học Khoa học Tự nhiên nằm ở Thành phố Hồ Chí Minh.',
    note: 'Model chỉ học tiếng Anh (CoNLL-2003). Kết quả sẽ lộn xộn — và đó là bài học.',
  },
]

export function NerDemo() {
  const model = useModel('ner')

  const [text, setText] = useState(EXAMPLES[0].text)
  const [note, setNote] = useState<string | undefined>(undefined)
  const [result, setResult] = useState<TokenLabelOutput | null>(null)
  const [debug, setDebug] = useState<DebugInfo | undefined>(undefined)
  const [merged, setMerged] = useState(true)

  const analysis = useMemo(() => {
    if (!result) return null
    const labeled = labelTokens(result.tokens, result.ids, result.logits, result.labels)
    const entities = mergeEntities(labeled)
    const nonO = labeled.filter((t) => !t.special && t.type).length
    return { labeled, entities, nonO }
  }, [result])

  async function handleRun() {
    try {
      const res = await model.run<TokenLabelOutput>(text)
      setResult(res.output)
      setDebug(res.debug)
    } catch {
      setResult(null)
      setDebug(undefined)
    }
  }

  const canRun = text.trim().length > 0 && !model.isBusy

  /** Token nào thuộc entity nào — để tô màu hàng token theo loại entity. */
  const tokenEntityType = useMemo(() => {
    const map = new Map<number, string>()
    if (!analysis) return map
    for (const entity of analysis.entities) {
      for (const index of entity.tokenIndices) map.set(index, entity.type)
    }
    return map
  }, [analysis])

  function renderToken(item: LabeledToken) {
    const type = merged ? tokenEntityType.get(item.index) : item.type
    const color = type ? entityColor(type) : null

    return (
      <span
        key={`${item.token}-${item.index}`}
        className={item.special ? 'token token--special' : 'token'}
        style={
          color
            ? {
                color,
                borderColor: `color-mix(in srgb, ${color} 45%, transparent)`,
                background: `color-mix(in srgb, ${color} 12%, var(--bg-sunken))`,
              }
            : undefined
        }
        title={`vị trí ${item.index} · ${item.label} · ${formatPercent(item.score, 1)}`}
      >
        <span className="token-text">{item.token}</span>
        <span className="token-id">{item.special ? '—' : item.label}</span>
      </span>
    )
  }

  return (
    <>
      <ModelBar model={model} />

      <div className="field">
        <div className="field-head">
          <label className="field-label" htmlFor="ner-input">
            Văn bản
          </label>
          <Info title="Token classification — một nhãn cho MỖI token">
            <p>
              Khác mọi card phân loại khác trên dashboard: ở đây model không cho một dự đoán cho cả
              câu, mà một dự đoán cho <strong>từng token</strong>.
            </p>
            <p>
              Vì thế shape của logits là <Tex tex="[\text{batch}, \text{seq}, 9]" /> — có thêm chiều
              độ dài câu. Softmax được tính trên chiều cuối, tức trên từng token một.
            </p>
            <p className="hint">
              9 nhãn là sơ đồ BIO của CoNLL-2003: <code>O</code> cộng với B-/I- cho bốn loại PER,
              ORG, LOC, MISC.
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
          id="ner-input"
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

      <div className="control-row">
        <button type="button" className="primary" onClick={handleRun} disabled={!canRun}>
          {model.status === 'running' ? 'Đang nhận diện…' : 'Nhận diện thực thể'}
        </button>
        <span className="kbd-hint">Ctrl + Enter</span>
      </div>

      {!result && !model.isBusy && (
        <div className="empty-state">Bấm để gán nhãn cho từng token trong câu trên.</div>
      )}

      {result && analysis && (
        <>
          <div className="metrics">
            <Badge mono tone="concept">
              {analysis.entities.length} entity
            </Badge>
            <Badge mono>{analysis.nonO} token không phải O</Badge>
            <Badge mono>{result.tokens.length} token</Badge>
            <Badge mono>{result.labels.length} nhãn BIO</Badge>
            {analysis.nonO > 0 && analysis.nonO !== analysis.entities.length && (
              <Info title="Vì sao số token khác số entity">
                <p>
                  <strong>{analysis.nonO}</strong> token được gán nhãn khác <code>O</code>, nhưng chỉ
                  có <strong>{analysis.entities.length}</strong> entity. Chênh lệch đó chính là công
                  việc của bước gộp.
                </p>
                <p>
                  Một entity như "Ho Chi Minh City" chiếm 4 token; "Nguyen" có thể bị WordPiece cắt
                  thành 2–3 mảnh nữa. Không gộp thì kết quả là danh sách mảnh vụn.
                </p>
              </Info>
            )}
          </div>

          <div className="field">
            <div className="field-head">
              <div className="legend">
                {Object.entries(ENTITY_LABELS).map(([type, label]) => (
                  <span key={type} className="legend-item">
                    <i
                      className="swatch"
                      style={{ background: entityColor(type), borderColor: entityColor(type) }}
                    />
                    {label} ({type})
                  </span>
                ))}
              </div>
              <label className="toggle">
                <input
                  type="checkbox"
                  checked={merged}
                  onChange={(e) => setMerged(e.target.checked)}
                />
                đã gộp
              </label>
            </div>
            <div className="token-row">{analysis.labeled.map(renderToken)}</div>
            <p className="hint">
              {merged
                ? 'Đang tô theo entity SAU khi gộp — subword bị nhãn O ở giữa một entity vẫn được nối vào.'
                : 'Đang tô theo nhãn THÔ của từng token. Bật "đã gộp" để so hai cách.'}
            </p>
          </div>

          <div className="field">
            <div className="field-head">
              <span className="field-label">Entity đã gộp</span>
              <Info title="Bước gộp BIO là heuristic, không phải phép biến đổi hiển nhiên">
                <p>Quy tắc dùng trong demo này, theo đúng thứ tự kiểm tra:</p>
                <table className="data">
                  <tbody>
                    <tr>
                      <td className="mono">O</td>
                      <td>đóng entity đang mở</td>
                    </tr>
                    <tr>
                      <td className="mono">B-X</td>
                      <td>đóng entity cũ, mở entity mới loại X</td>
                    </tr>
                    <tr>
                      <td className="mono">I-X</td>
                      <td>nối tiếp nếu đang mở X, ngược lại mở entity mới loại X</td>
                    </tr>
                    <tr>
                      <td className="mono">##…</td>
                      <td>nối vô điều kiện vào entity đang mở</td>
                    </tr>
                  </tbody>
                </table>
                <p>
                  Hai dòng cuối quan trọng nhất. Bộ nhãn CoNLL-2003 dùng quy ước <strong>IOB1</strong>
                  , ở đó <code>B-</code> chỉ xuất hiện khi hai entity cùng loại nằm sát nhau — nên
                  phần lớn entity thật sự bắt đầu bằng <code>I-</code>. Ai áp dụng đúng lý thuyết
                  IOB2 ("phải có B- mới mở entity") sẽ mất gần hết kết quả.
                </p>
                <p>
                  Còn subword được nối bất kể nhãn riêng của nó, vì cắt entity ở giữa một từ thì ra
                  chữ vô nghĩa. Ranh giới từ được ưu tiên hơn nhãn từng mảnh.
                </p>
                <p className="hint">
                  Trong thư viện Python việc này nằm sau tham số <code>aggregation_strategy</code>.
                  Ở đây nó là <code>mergeEntities()</code> trong <code>src/lib/ner.ts</code>, đọc
                  được — và đổi quy tắc thì kết quả đổi theo.
                </p>
              </Info>
            </div>
            {analysis.entities.length === 0 ? (
              <div className="empty-state">
                Không có entity nào — mọi token đều nhận nhãn <code>O</code>.
              </div>
            ) : (
              <div className="table-scroll">
                <table className="data">
                  <thead>
                    <tr>
                      <th>Văn bản</th>
                      <th>Loại</th>
                      <th className="num">Token</th>
                      <th className="num">Điểm</th>
                    </tr>
                  </thead>
                  <tbody>
                    {analysis.entities.map((entity, i) => (
                      <tr key={`${entity.text}-${i}`}>
                        <td>{entity.text}</td>
                        <td>
                          <span
                            className="topic-dot"
                            style={{ background: entityColor(entity.type) }}
                          />{' '}
                          {ENTITY_LABELS[entity.type] ?? entity.type}
                        </td>
                        <td className="num">{entity.tokenIndices.length}</td>
                        <td className="num">{formatPercent(entity.score, 1)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </>
      )}

      <UnderTheHood debug={debug}>
        {result && analysis && (
          <div className="hood-block">
            <h4>
              Nhãn và xác suất của từng token
              <Info title="Softmax theo chiều nào">
                <p>
                  logits có shape <Tex tex="[1, \text{seq}, 9]" />. Softmax phải chạy trên chiều{' '}
                  <strong>cuối</strong> — 9 nhãn của một token cạnh tranh nhau:
                </p>
                <Tex tex="p_{t,i} = \frac{e^{z_{t,i}}}{\sum_{j=1}^{9} e^{z_{t,j}}}" block />
                <p>
                  Chạy sai chiều (trên chiều độ dài câu) vẫn ra số hợp lệ, vẫn tổng bằng 1, và không
                  có gì báo lỗi — chỉ là hoàn toàn vô nghĩa. Đây là lỗi rất dễ mắc khi tự viết phần
                  hậu xử lý.
                </p>
              </Info>
            </h4>
            <div className="table-scroll">
              <table className="data">
                <thead>
                  <tr>
                    <th className="num">#</th>
                    <th>Token</th>
                    <th className="num">id</th>
                    <th>Nhãn</th>
                    <th className="num">p</th>
                    <th className="num">p(O)</th>
                  </tr>
                </thead>
                <tbody>
                  {analysis.labeled.map((item) => {
                    const probabilities = softmax(result.logits[item.index] ?? [])
                    const oIndex = result.labels.indexOf('O')
                    return (
                      <tr
                        key={item.index}
                        className={!item.special && item.type ? 'row--active' : undefined}
                      >
                        <td className="num">{item.index}</td>
                        <td className="mono">{item.token}</td>
                        <td className="num">{item.id}</td>
                        <td className="mono">{item.label}</td>
                        <td className="num">{item.score.toFixed(4)}</td>
                        <td className="num">
                          {oIndex >= 0 ? probabilities[oIndex].toFixed(4) : '—'}
                        </td>
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
