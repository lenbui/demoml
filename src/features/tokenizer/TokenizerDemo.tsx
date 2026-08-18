import { useMemo, useState } from 'react'

import { Badge } from '../../components/Badge'
import { Info } from '../../components/Info'
import { Math as Tex } from '../../components/Math'
import { ModelBar } from '../../components/ModelBar'
import { TokenChips, TokenLegend } from '../../components/TokenChips'
import { UnderTheHood } from '../../components/UnderTheHood'
import { VariantPicker } from '../../components/VariantPicker'
import { useModel } from '../../hooks/useModel'
import type { DebugInfo, TokenizeOutput } from '../../workers/protocol'

const EXAMPLES: Array<{ label: string; text: string }> = [
  { label: 'Tiếng Việt', text: 'Khóa học máy học rất thú vị và bổ ích.' },
  { label: 'Tiếng Anh', text: 'The quick brown fox jumps over the lazy dog.' },
  { label: 'Từ chuyên ngành', text: 'Tokenization uncontrollably outperforms hyperparameterization.' },
  { label: 'Code', text: 'const logits = await model.forward(input_ids);' },
  { label: 'Số & emoji', text: 'Giá: 1.250.000₫ 🎉 giảm 15% cho 3 người đầu.' },
]

interface Measured extends TokenizeOutput {
  ms: number
}

function countWords(text: string): number {
  const trimmed = text.trim()
  return trimmed ? trimmed.split(/\s+/).length : 0
}

export function TokenizerDemo() {
  const model = useModel('tokenizer')
  const variants = model.spec.variants ?? []

  const [text, setText] = useState(EXAMPLES[0].text)
  const [variantId, setVariantId] = useState(variants[0]?.id ?? 'default')
  const [results, setResults] = useState<Record<string, Measured>>({})
  const [debug, setDebug] = useState<DebugInfo | undefined>(undefined)
  const [showIds, setShowIds] = useState(false)

  const current = results[variantId]

  const stats = useMemo(() => {
    if (!current) return null
    const words = countWords(text)
    return {
      tokens: current.tokens.length,
      chars: text.length,
      words,
      charsPerToken: current.tokens.length ? text.length / current.tokens.length : 0,
      fertility: words ? current.tokens.length / words : 0,
      roundTripOk: current.decoded.trim() === text.trim(),
    }
  }, [current, text])

  /** Kết quả cũ không còn ứng với text mới, nên xoá hết khi sửa input. */
  function updateText(next: string) {
    setText(next)
    setResults({})
    setDebug(undefined)
  }

  async function runVariant(id: string) {
    try {
      const res = await model.run<TokenizeOutput>(text, { variantId: id })
      setResults((prev) => ({ ...prev, [id]: { ...res.output, ms: res.ms } }))
      if (id === variantId) setDebug(res.debug)
      return true
    } catch {
      return false
    }
  }

  async function runAll() {
    // Tuần tự, không song song: 5 tokenizer tải cùng lúc thì progress bar thành
    // vô nghĩa và dễ nghẽn mạng phòng máy.
    for (const variant of variants) {
      await runVariant(variant.id)
    }
  }

  const canRun = text.trim().length > 0 && !model.isBusy
  const comparedCount = Object.keys(results).length

  return (
    <>
      <div className="field">
        <div className="field-head">
          <label className="field-label" htmlFor="tok-input">
            Văn bản
          </label>
          <div className="chip-row">
            {EXAMPLES.map((example) => (
              <button
                key={example.label}
                type="button"
                className="chip"
                onClick={() => updateText(example.text)}
              >
                {example.label}
              </button>
            ))}
          </div>
        </div>
        <textarea
          id="tok-input"
          className="input--compact"
          value={text}
          onChange={(e) => updateText(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && (e.ctrlKey || e.metaKey) && canRun) void runVariant(variantId)
          }}
        />
      </div>

      <div className="field">
        <div className="field-head">
          <span className="field-label">Tokenizer</span>
          <Info title="Vì sao có nhiều tokenizer?">
            <p>
              Tokenizer là <strong>một phần của model</strong>, không phải bước tiền xử lý trung
              tính. Mỗi họ model được huấn luyện cùng một tokenizer riêng và không thể thay thế
              lẫn nhau.
            </p>
            <p>
              Cho cùng một câu chạy qua cả 5 để thấy: số token khác nhau, cách cắt từ khác nhau,
              và với tiếng Việt thì chênh lệch rất lớn.
            </p>
            <p className="hint">
              Chỉ tokenizer được tải, không có trọng số model — nên mỗi lựa chọn chỉ 1–9 MB.
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

      <div className="control-row">
        <button type="button" className="primary" onClick={() => runVariant(variantId)} disabled={!canRun}>
          {model.status === 'running' ? 'Đang chạy…' : 'Tokenize'}
        </button>
        <button type="button" onClick={runAll} disabled={!canRun}>
          Chạy cả {variants.length} để so sánh
        </button>
      </div>

      {!current && !model.isBusy && (
        <div className="empty-state">Bấm Tokenize để xem câu trên được cắt thành gì.</div>
      )}

      {current && stats && (
        <>
          <div className="metrics">
            <Badge mono tone="concept">
              {stats.tokens} token
            </Badge>
            <Badge mono>{stats.chars} ký tự</Badge>
            <Badge mono>{stats.charsPerToken.toFixed(2)} ký tự/token</Badge>
            <Badge mono tone={stats.fertility > 2.5 ? 'warn' : 'default'}>
              fertility {stats.fertility.toFixed(2)}
            </Badge>
            <Badge mono>{current.algorithm}</Badge>
            {current.vocabSize && <Badge mono>vocab {current.vocabSize.toLocaleString('vi-VN')}</Badge>}
            <Info title="Fertility — số token mỗi từ">
              <Tex tex="\text{fertility} = \frac{\#\text{token}}{\#\text{từ}}" block />
              <p>
                Càng gần 1 thì tokenizer càng “vừa vặn” với ngôn ngữ đó. Fertility cao nghĩa là
                mỗi từ bị cắt thành nhiều mảnh, kéo theo ba hệ quả:
              </p>
              <ul className="notes">
                <li>chuỗi dài hơn → chi phí attention tăng theo <Tex tex="O(n^2)" /></li>
                <li>tốn nhiều token hơn cho cùng một nội dung → đắt hơn khi gọi API</li>
                <li>ngữ nghĩa bị chia vụn → model khó học quan hệ giữa các từ</li>
              </ul>
              <p className="hint">
                Đây là lý do model đa ngữ thường tốt hơn hẳn model tiếng Anh khi xử lý tiếng Việt,
                kể cả trước khi xét đến dữ liệu huấn luyện.
              </p>
            </Info>
          </div>

          <div className="field">
            <div className="field-head">
              <TokenLegend algorithm={current.algorithm} />
              <label className="toggle">
                <input
                  type="checkbox"
                  checked={showIds}
                  onChange={(e) => setShowIds(e.target.checked)}
                />
                hiện id
              </label>
            </div>
            <TokenChips
              tokens={current.tokens}
              ids={current.ids}
              algorithm={current.algorithm}
              specialTokens={current.specialTokens}
              unkToken={current.unkToken}
              showIds={showIds}
            />
          </div>

          <div className="field">
            <div className="field-head">
              <span className="field-label">Decode ngược</span>
              <Badge tone={stats.roundTripOk ? 'ok' : 'danger'}>
                {stats.roundTripOk ? 'khớp bản gốc' : 'khác bản gốc'}
              </Badge>
              <Info title="Round-trip: tokenizer làm mất gì">
                <p>
                  Lấy dãy id vừa sinh ra rồi decode ngược lại thành chữ. Nếu kết quả khác bản gốc
                  thì phần chênh lệch chính là thông tin đã <strong>mất vĩnh viễn</strong> — model
                  không bao giờ thấy được nó.
                </p>
                <p>
                  Tokenizer <code>uncased</code> của BERT tiếng Anh bỏ cả chữ hoa và dấu tiếng
                  Việt: “Khóa” trở thành “khoa”. Ngược lại BPE mức byte của GPT-2 không mất gì vì
                  nó làm việc trên byte thô.
                </p>
              </Info>
            </div>
            <div className={stats.roundTripOk ? 'roundtrip' : 'roundtrip roundtrip--diff'}>
              {current.decoded || <em>(rỗng)</em>}
            </div>
          </div>
        </>
      )}

      {comparedCount > 1 && (
        <div className="field">
          <div className="field-head">
            <span className="field-label">So sánh {comparedCount} tokenizer</span>
          </div>
          <div className="table-scroll">
            <table className="data">
              <thead>
                <tr>
                  <th>Tokenizer</th>
                  <th>Thuật toán</th>
                  <th className="num">Token</th>
                  <th className="num">Fertility</th>
                  <th>Round-trip</th>
                  <th className="num">ms</th>
                </tr>
              </thead>
              <tbody>
                {variants
                  .filter((v) => results[v.id])
                  .map((v) => {
                    const r = results[v.id]
                    const words = countWords(text)
                    const fertility = words ? r.tokens.length / words : 0
                    const ok = r.decoded.trim() === text.trim()
                    return (
                      <tr key={v.id} className={v.id === variantId ? 'row--active' : undefined}>
                        <td>{v.label}</td>
                        <td className="mono">{r.algorithm}</td>
                        <td className="num">{r.tokens.length}</td>
                        <td className="num">{fertility.toFixed(2)}</td>
                        <td>
                          <Badge tone={ok ? 'ok' : 'danger'}>{ok ? 'khớp' : 'khác'}</Badge>
                        </td>
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
            <h4>Tokenizer đang dùng</h4>
            <table className="data">
              <tbody>
                <tr>
                  <td>Class</td>
                  <td className="mono">{current.tokenizerClass}</td>
                </tr>
                <tr>
                  <td>Thuật toán</td>
                  <td className="mono">{current.algorithm}</td>
                </tr>
                <tr>
                  <td>Kích thước vocabulary</td>
                  <td className="mono">{current.vocabSize?.toLocaleString('vi-VN') ?? '—'}</td>
                </tr>
                <tr>
                  <td>Token đặc biệt</td>
                  <td className="mono">{current.specialTokens.join(' ') || '(không có)'}</td>
                </tr>
                <tr>
                  <td>Token UNK</td>
                  <td className="mono">{current.unkToken ?? '(không có)'}</td>
                </tr>
              </tbody>
            </table>
          </div>
        )}
      </UnderTheHood>
    </>
  )
}
