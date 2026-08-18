import { useMemo, useState } from 'react'

import { Badge } from '../../components/Badge'
import { Info } from '../../components/Info'
import { Math as Tex } from '../../components/Math'
import { ModelBar } from '../../components/ModelBar'
import { Tabs } from '../../components/Tabs'
import { UnderTheHood } from '../../components/UnderTheHood'
import { VariantPicker } from '../../components/VariantPicker'
import { useModel } from '../../hooks/useModel'
import { formatMs, formatPercent } from '../../lib/format'
import {
  SAMPLING_PRESETS,
  applySampling,
  type SamplingParams,
} from '../../lib/sampling'
import type { DebugInfo, GenerateOutput, NextTokenOutput } from '../../workers/protocol'

const TABS = [
  { id: 'next', label: 'Token tiếp theo' },
  { id: 'generate', label: 'Sinh văn bản' },
]

const PROMPTS: Array<{ label: string; text: string; note?: string }> = [
  { label: 'Câu dở dang', text: 'The capital of France is' },
  {
    label: 'Kể chuyện',
    text: 'Once upon a time, in a small village by the sea, there lived',
  },
  {
    label: 'Hỏi đáp',
    text: 'Question: What is machine learning?\nAnswer:',
    note: 'GPT-2 là base model — nó nối tiếp văn bản chứ không thật sự trả lời.',
  },
  {
    label: 'Lặp vòng',
    text: 'I like cats. I like cats. I like cats.',
    note: 'Để nhiệt độ 0 rồi sinh — model sẽ kẹt trong vòng lặp.',
  },
  {
    label: 'Tiếng Việt',
    text: 'Hà Nội là thủ đô của',
    note: 'GPT-2 gần như chỉ học tiếng Anh. Kết quả sẽ rất tệ.',
  },
]

const MAX_TOKENS = [20, 40, 80]

export function LlmDemo() {
  const model = useModel('llm')
  const variants = model.spec.variants ?? []

  const [variantId, setVariantId] = useState(variants[0]?.id ?? 'default')
  const [tab, setTab] = useState('next')
  const [prompt, setPrompt] = useState(PROMPTS[0].text)
  const [note, setNote] = useState<string | undefined>(undefined)

  const [params, setParams] = useState<SamplingParams>({ temperature: 0.7, topK: 50, topP: 0.9 })
  const [maxTokens, setMaxTokens] = useState(40)

  const [distribution, setDistribution] = useState<NextTokenOutput | null>(null)
  const [streamed, setStreamed] = useState('')
  const [genStats, setGenStats] = useState<{ ms: number; tokens: number } | null>(null)
  const [debug, setDebug] = useState<DebugInfo | undefined>(undefined)

  /**
   * Toàn bộ phép sampling chạy Ở ĐÂY, trên logits đã có sẵn — kéo slider KHÔNG
   * gọi lại model. Đó chính là điều demo muốn chứng minh.
   */
  const sampling = useMemo(
    () => (distribution ? applySampling(distribution.candidates, params) : null),
    [distribution, params],
  )

  async function loadDistribution() {
    try {
      const res = await model.run<NextTokenOutput>(prompt, {
        variantId,
        pipelineOptions: { mode: 'distribution' },
      })
      setDistribution(res.output)
      setDebug(res.debug)
    } catch {
      setDistribution(null)
    }
  }

  async function generate() {
    setStreamed('')
    setGenStats(null)
    try {
      const res = await model.run<GenerateOutput>(prompt, {
        variantId,
        // onToken không đi qua postMessage — xem RunOptions trong useModel.
        onToken: (text) => setStreamed((prev) => prev + text),
        pipelineOptions: {
          max_new_tokens: maxTokens,
          // temperature 0 không hợp lệ với do_sample; dùng greedy đúng nghĩa.
          do_sample: params.temperature > 0,
          ...(params.temperature > 0
            ? {
                temperature: params.temperature,
                top_k: params.topK > 0 ? params.topK : 0,
                top_p: params.topP,
              }
            : {}),
        },
      })
      setGenStats({ ms: res.ms, tokens: res.output.newTokens })
      setDebug(res.debug)
      // Chốt lại bằng kết quả cuối: streamer có thể gộp mẩu khác với text cuối.
      setStreamed(res.output.text)
    } catch {
      setGenStats(null)
    }
  }

  const busy = model.isBusy
  const canRun = prompt.trim().length > 0 && !busy
  const tokensPerSecond =
    genStats && genStats.ms > 0 ? genStats.tokens / (genStats.ms / 1000) : null

  function updateParams(next: Partial<SamplingParams>) {
    setParams((prev) => ({ ...prev, ...next }))
  }

  return (
    <>
      <div className="field">
        <div className="field-head">
          <span className="field-label">Model</span>
          <Info title="Vì sao GPT-2 chứ không phải một model chat đời mới">
            <p>
              GPT-2 là <strong>base model</strong>: nó chỉ được huấn luyện để đoán token kế tiếp
              trên văn bản Internet, chưa qua bước instruction tuning nào.
            </p>
            <p>
              Nên nó <em>nối tiếp</em> văn bản chứ không trả lời bạn. Thử prompt "Hỏi đáp": nó sẽ
              viết tiếp thêm câu hỏi khác thay vì trả lời. Đúng điều đó giải thích vì sao phải có
              bước instruction tuning — thứ mà một model chat đã che mất.
            </p>
            <p className="hint">
              DistilGPT-2 là bản chưng cất: 6 lớp thay vì 12, học cách bắt chước đầu ra của GPT-2.
              Chạy cả hai trên cùng prompt để thấy mất mát chất lượng đổi lấy 43 MB.
            </p>
          </Info>
        </div>
        <VariantPicker
          variants={variants}
          value={variantId}
          onChange={(id) => {
            setVariantId(id)
            setDistribution(null)
            setStreamed('')
            setGenStats(null)
          }}
          readyVariants={model.readyVariants}
          loadingVariantId={model.loadingVariantId}
          disabled={busy}
        />
      </div>

      <ModelBar model={model} variantId={variantId} />

      <div className="field">
        <div className="field-head">
          <label className="field-label" htmlFor="llm-prompt">
            Prompt
          </label>
          <div className="chip-row">
            {PROMPTS.map((item) => (
              <button
                key={item.label}
                type="button"
                className={`chip${prompt === item.text ? ' chip--active' : ''}`}
                onClick={() => {
                  setPrompt(item.text)
                  setNote(item.note)
                  setDistribution(null)
                  setStreamed('')
                  setGenStats(null)
                }}
              >
                {item.label}
              </button>
            ))}
          </div>
        </div>
        <textarea
          id="llm-prompt"
          className="input--compact"
          value={prompt}
          onChange={(e) => {
            setPrompt(e.target.value)
            setNote(undefined)
            setDistribution(null)
            setStreamed('')
          }}
        />
        {note && <p className="field-hint">{note}</p>}
      </div>

      {/* Bộ tham số dùng chung cho cả hai tab — cùng một phép toán, một bên xem
          tĩnh từng bước, một bên chạy thật nhiều bước liên tiếp. */}
      <div className="field">
        <div className="field-head">
          <span className="field-label">Tham số lấy mẫu</span>
          <Badge mono>T {params.temperature.toFixed(2)}</Badge>
          <Badge mono>top-k {params.topK || '—'}</Badge>
          <Badge mono>top-p {params.topP.toFixed(2)}</Badge>
          <Info title="Ba tham số, ba việc khác nhau">
            <p>
              Model chỉ trả về logits. Cả ba tham số dưới đây là <strong>hậu xử lý</strong> chạy bên
              ngoài model — đó là lý do kéo slider ở tab bên cạnh không cần chạy lại model.
            </p>
            <p>
              <strong>Nhiệt độ</strong> chia logits trước khi softmax:
            </p>
            <Tex tex="p_i = \frac{e^{z_i / T}}{\sum_j e^{z_j / T}}" block />
            <table className="data">
              <tbody>
                <tr>
                  <td className="mono">T → 0</td>
                  <td>phân phối nhọn hoắt → luôn chọn token cao nhất (tất định)</td>
                </tr>
                <tr>
                  <td className="mono">T = 1</td>
                  <td>đúng phân phối model đã học</td>
                </tr>
                <tr>
                  <td className="mono">T &gt; 1</td>
                  <td>san phẳng → token hiếm có cơ hội, dễ lảm nhảm hơn</td>
                </tr>
              </tbody>
            </table>
            <p>
              <strong>top-k</strong> giữ đúng k token cao nhất — một con số cứng, không quan tâm
              phân phối nhọn hay phẳng.
            </p>
            <p>
              <strong>top-p</strong> (nucleus) giữ các token cao nhất cho tới khi xác suất dồn đủ p.
              Thích ứng được: khi model rất chắc chắn thì chỉ giữ 1–2 token, khi model lưỡng lự thì
              giữ hàng chục.
            </p>
            <p className="hint">
              Đó là lý do top-p thường được ưa hơn top-k. Dùng cả hai thì token phải sống sót qua cả
              hai bước cắt.
            </p>
          </Info>
        </div>

        <div className="chip-row">
          {SAMPLING_PRESETS.map((preset) => (
            <button
              key={preset.label}
              type="button"
              className={`chip${
                params.temperature === preset.params.temperature &&
                params.topK === preset.params.topK &&
                params.topP === preset.params.topP
                  ? ' chip--active'
                  : ''
              }`}
              onClick={() => setParams(preset.params)}
              title={preset.note}
            >
              {preset.label}
            </button>
          ))}
        </div>

        <div className="config-grid">
          <div className="field">
            <label className="field-label" htmlFor="llm-temp">
              Nhiệt độ
            </label>
            <input
              id="llm-temp"
              type="range"
              min={0}
              max={2}
              step={0.05}
              value={params.temperature}
              onChange={(e) => updateParams({ temperature: Number(e.target.value) })}
            />
            <label className="field-label" htmlFor="llm-topk">
              top-k (0 = tắt)
            </label>
            <input
              id="llm-topk"
              type="range"
              min={0}
              max={60}
              step={1}
              value={params.topK}
              onChange={(e) => updateParams({ topK: Number(e.target.value) })}
            />
          </div>
          <div className="field">
            <label className="field-label" htmlFor="llm-topp">
              top-p
            </label>
            <input
              id="llm-topp"
              type="range"
              min={0.05}
              max={1}
              step={0.01}
              value={params.topP}
              onChange={(e) => updateParams({ topP: Number(e.target.value) })}
            />
          </div>
        </div>
      </div>

      <Tabs items={TABS} value={tab} onChange={setTab} />

      {tab === 'next' && (
        <>
          <div className="control-row">
            <button type="button" className="primary" onClick={loadDistribution} disabled={!canRun}>
              {model.status === 'running' ? 'Đang tính…' : 'Xem phân phối token kế tiếp'}
            </button>
            {distribution && <Badge tone="ok">kéo slider không chạy lại model</Badge>}
          </div>

          {!distribution && !busy && (
            <div className="empty-state">
              Chạy một forward pass để lấy logits, rồi kéo slider xem chúng bị biến đổi thế nào.
            </div>
          )}

          {distribution && sampling && (
            <>
              <div className="metrics">
                <Badge mono tone="concept">
                  {sampling.keptCount} / {distribution.candidates.length} token sống sót
                </Badge>
                <Badge mono>
                  vocab {distribution.vocabSize.toLocaleString('vi-VN')}
                </Badge>
                <Badge mono tone={sampling.entropy > 3 ? 'warn' : 'default'}>
                  entropy {sampling.entropy.toFixed(2)} bit
                </Badge>
                <Badge mono tone={sampling.coverage < 0.9 ? 'warn' : 'default'}>
                  phủ {formatPercent(sampling.coverage, 1)}
                </Badge>
                <Info title="Đọc mấy con số này">
                  <p>
                    <strong>Sống sót</strong> — bao nhiêu token còn được phép bốc sau khi cắt. Bằng 1
                    nghĩa là bước này đã tất định.
                  </p>
                  <p>
                    <strong>Entropy</strong> — còn bao nhiêu lựa chọn thật sự, tính trên phân phối
                    sau khi cắt. 0 bit = chỉ một khả năng; 3 bit ≈ 8 khả năng ngang nhau.
                  </p>
                  <p>
                    <strong>Phủ</strong> — bảng này chỉ có{' '}
                    {distribution.candidates.length} ứng viên trong{' '}
                    {distribution.vocabSize.toLocaleString('vi-VN')} token của vocabulary. Con số cho
                    biết chúng chiếm bao nhiêu phần trăm xác suất. Ở nhiệt độ thấp nó gần 100%; kéo
                    nhiệt độ lên cao sẽ thấy nó tụt — phần đuôi khổng lồ bắt đầu có trọng lượng.
                  </p>
                </Info>
              </div>

              <div className="table-scroll">
                <table className="data">
                  <thead>
                    <tr>
                      <th>Token</th>
                      <th className="num">logit</th>
                      <th className="num">p (sau T)</th>
                      <th className="num">dồn</th>
                      <th>Phân phối cuối</th>
                      <th className="num">p cuối</th>
                    </tr>
                  </thead>
                  <tbody>
                    {sampling.tokens.slice(0, 25).map((item) => (
                      <tr key={item.id} className={item.kept ? 'row--active' : undefined}>
                        <td className="mono">
                          {item.token === ' ' ? '␣' : item.token.replace(/\n/g, '\\n')}
                        </td>
                        <td className="num">{item.logit.toFixed(3)}</td>
                        <td className="num">{item.probability.toFixed(4)}</td>
                        <td className="num">{item.cumulative.toFixed(3)}</td>
                        <td>
                          <div className="pred-track">
                            <div
                              className="pred-bar"
                              style={{
                                width: `${item.finalProbability * 100}%`,
                                background: item.kept ? 'var(--accent)' : 'var(--border-strong)',
                              }}
                            />
                          </div>
                        </td>
                        <td className="num">
                          {item.kept ? (
                            item.finalProbability.toFixed(4)
                          ) : (
                            <Badge tone="danger">{item.cutBy}</Badge>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <p className="hint">
                Hàng tô đậm là token còn được phép bốc. Kéo nhiệt độ về 0 để thấy mọi xác suất dồn
                hết vào token đầu; kéo lên 2 để thấy chúng san phẳng ra.
              </p>
            </>
          )}
        </>
      )}

      {tab === 'generate' && (
        <>
          <div className="control-row">
            <button type="button" className="primary" onClick={generate} disabled={!canRun}>
              {model.status === 'running' ? 'Đang sinh…' : 'Sinh văn bản'}
            </button>
            <div className="chip-row">
              {MAX_TOKENS.map((n) => (
                <button
                  key={n}
                  type="button"
                  className={`chip${maxTokens === n ? ' chip--active' : ''}`}
                  onClick={() => setMaxTokens(n)}
                >
                  {n} token
                </button>
              ))}
            </div>
            {tokensPerSecond != null && (
              <Badge mono>{tokensPerSecond.toFixed(1)} token/s</Badge>
            )}
            {genStats && <Badge mono>{formatMs(genStats.ms)}</Badge>}
          </div>

          {!streamed && !busy && (
            <div className="empty-state">
              Bấm Sinh — chữ sẽ hiện dần từng token một, đúng như model đẻ ra chúng.
            </div>
          )}

          {(streamed || model.status === 'running') && (
            <div className="field">
              <div className="field-head">
                <span className="field-label">Kết quả</span>
                <Info title="Vì sao chữ hiện dần chứ không hiện một lúc">
                  <p>
                    Không phải hiệu ứng giao diện. Model thật sự sinh{' '}
                    <strong>từng token một</strong>, mỗi token cần một forward pass riêng và phụ
                    thuộc mọi token trước đó:
                  </p>
                  <Tex tex="P(y) = \prod_{t} P(y_t \mid y_{<t})" block />
                  <p>
                    Vì thế chi phí tăng tuyến tính theo độ dài đầu ra, và không có cách nào sinh 80
                    token nhanh bằng sinh 1 token. Mọi giao diện chat bạn từng dùng đều đang stream
                    đúng như thế này.
                  </p>
                  <p className="hint">
                    Kỹ thuật: worker gửi từng mẩu qua message <code>kind: 'token'</code>, hook{' '}
                    <code>useModel</code> tra theo <code>requestId</code> rồi gọi{' '}
                    <code>onToken</code>. Xem <code>src/workers/protocol.ts</code>.
                  </p>
                </Info>
              </div>
              <div className="transcript">
                <span className="llm-prompt-echo">{prompt}</span>
                {streamed}
                {model.status === 'running' && <span className="caret" />}
              </div>
            </div>
          )}

          {genStats && params.temperature === 0 && (
            <div className="callout callout--warn">
              Nhiệt độ 0 = greedy: chạy lại bao nhiêu lần cũng ra <strong>đúng một kết quả</strong>.
              Thử prompt "Lặp vòng" để thấy mặt trái — model kẹt trong vòng lặp và không thoát ra
              được, vì token nào cao nhất thì mãi mãi cao nhất.
            </div>
          )}
        </>
      )}

      <UnderTheHood debug={debug} />
    </>
  )
}
