import { useMemo, useState } from 'react'

import { Badge } from '../../components/Badge'
import { Info } from '../../components/Info'
import { Math as Tex } from '../../components/Math'
import { ModelBar } from '../../components/ModelBar'
import { ScatterPlot, type ScatterPoint } from '../../components/ScatterPlot'
import { Tabs } from '../../components/Tabs'
import { UnderTheHood } from '../../components/UnderTheHood'
import { VariantPicker } from '../../components/VariantPicker'
import { useModel } from '../../hooks/useModel'
import { buildBm25Index, bm25Search } from '../../lib/bm25'
import { dot, POOLING_LABELS, type Pooling } from '../../lib/embedding'
import { pca2d } from '../../lib/pca'
import type { DebugInfo, EmbedOutput } from '../../workers/protocol'
import { CORPORA, getCorpus, TOPICS } from './corpus'

const TABS = [
  { id: 'search', label: 'Tìm kiếm' },
  { id: 'pair', label: 'So sánh 2 câu' },
  { id: 'map', label: 'Bản đồ 2D' },
]

const TOP_K = 5

/**
 * Bề rộng thanh so sánh, tính TƯƠNG ĐỐI trong các kết quả đang hiện.
 *
 * Vì sao không dùng thẳng giá trị cosine làm bề rộng? Model E5 dồn mọi điểm vào
 * dải rất hẹp (đo được: 0.81–0.87 trên corpus này), nên thanh vẽ theo giá trị
 * tuyệt đối sẽ dài gần bằng nhau và không thấy được thứ hạng.
 *
 * Sàn 12% để kết quả cuối cùng vẫn có thanh nhìn thấy được. Con số cosine thô
 * luôn hiện bên cạnh, nên không có gì bị bóp méo — chỉ là thanh mang nghĩa
 * "so với các kết quả khác đang xem", và popover "?" nói rõ điều đó.
 */
function relativeWidth(score: number, min: number, max: number): string {
  const t = (score - min) / (max - min || 1)
  return `${(12 + t * 88).toFixed(1)}%`
}

/** Cặp câu mặc định: cùng ý, không trùng một từ nào. */
const DEFAULT_PAIR = {
  a: 'Mô hình học thuộc lòng dữ liệu huấn luyện.',
  b: 'Máy ghi nhớ từng ví dụ thay vì rút ra quy luật.',
}

/**
 * Các câu B để so với cùng một câu A. Bấm lần lượt cả bốn rồi đọc bảng bên dưới —
 * đó là cách duy nhất để con số cosine có nghĩa.
 *
 * Số đo trên model đa ngữ E5: 0.8725 / 0.8441 / 0.8124 theo thứ tự đồng nghĩa,
 * bản dịch, vô nghĩa. Thứ tự đúng nhưng chênh nhau chỉ 0.06.
 */
const PAIR_B_PRESETS = [
  { label: 'Đồng nghĩa', text: 'Máy ghi nhớ từng ví dụ thay vì rút ra quy luật.' },
  { label: 'Bản dịch Anh', text: 'The model memorises its training data.' },
  { label: 'Cùng lĩnh vực', text: 'Tốc độ học quá lớn làm quá trình tối ưu phát tán.' },
  { label: 'Vô nghĩa', text: 'Hôm nay trời mưa và tôi ăn phở.' },
]

export function EmbeddingDemo() {
  const model = useModel('embeddings')
  const variants = model.spec.variants ?? []

  const [variantId, setVariantId] = useState(variants[0]?.id ?? 'default')
  const [corpusId, setCorpusId] = useState(CORPORA[0].id)
  const [pooling, setPooling] = useState<Pooling>('mean')
  const [tab, setTab] = useState('search')

  const corpus = getCorpus(corpusId)

  /** Vector của corpus, kèm khoá cho biết chúng ứng với cấu hình nào. */
  const [embedded, setEmbedded] = useState<{ key: string; vectors: number[][] } | null>(null)
  const [query, setQuery] = useState(CORPORA[0].sampleQueries[0])
  const [queryVector, setQueryVector] = useState<number[] | null>(null)
  const [debug, setDebug] = useState<DebugInfo | undefined>(undefined)

  const [pairA, setPairA] = useState(DEFAULT_PAIR.a)
  const [pairB, setPairB] = useState(DEFAULT_PAIR.b)
  /**
   * Lịch sử vài lần so sánh gần nhất.
   *
   * Cần thiết vì một con số cosine đứng một mình không đọc được: model E5 cho
   * 0.8725 với cặp đồng nghĩa và 0.8124 với cặp hoàn toàn vô nghĩa — chênh 0.06.
   * Đặt cạnh nhau thì thứ tự mới hiện ra, và đó đúng là cách dùng cosine.
   */
  const [pairHistory, setPairHistory] = useState<Array<{ b: string; score: number }>>([])

  // Đổi corpus, model hay pooling là mọi vector cũ hết giá trị.
  const cacheKey = `${corpusId}:${variantId}:${pooling}`
  const corpusReady = embedded?.key === cacheKey

  // ── BM25 chạy hoàn toàn phía UI, không cần model, nên có kết quả tức thì ──
  const bm25 = useMemo(() => {
    const index = buildBm25Index(corpus.passages.map((p) => p.text))
    return bm25Search(index, query).filter((hit) => hit.score > 0)
  }, [corpus, query])

  const semantic = useMemo(() => {
    if (!corpusReady || !queryVector) return null
    return corpus.passages
      .map((_, i) => ({ docIndex: i, score: dot(queryVector, embedded.vectors[i]) }))
      .sort((a, b) => b.score - a.score)
  }, [corpusReady, queryVector, embedded, corpus])

  const projection = useMemo(() => {
    if (!corpusReady) return null
    // Gộp truy vấn vào cùng phép PCA để dấu ✕ nằm đúng hệ toạ độ với các đoạn văn.
    const input = queryVector ? [...embedded.vectors, queryVector] : embedded.vectors
    const { points, explainedRatio } = pca2d(input)
    return {
      corpusPoints: points.slice(0, corpus.passages.length),
      queryPoint: queryVector ? points[points.length - 1] : null,
      explainedRatio,
    }
  }, [corpusReady, embedded, queryVector, corpus])

  /**
   * Thêm tiền tố mà model yêu cầu (E5 dùng 'query: ' / 'passage: ').
   *
   * Cố ý làm ở đây thay vì trong worker: đây là bước xử lý DỮ LIỆU, và để sinh
   * viên thấy được rằng cùng một câu phải được gắn nhãn khác nhau tuỳ vai trò
   * của nó — truy vấn hay tài liệu.
   */
  const prefixes = variants.find((v) => v.id === variantId)?.prefixes
  const asQuery = (text: string) => `${prefixes?.query ?? ''}${text}`
  const asPassage = (text: string) => `${prefixes?.passage ?? ''}${text}`

  async function ensureCorpusVectors(): Promise<number[][]> {
    if (embedded?.key === cacheKey) return embedded.vectors

    const res = await model.run<EmbedOutput>(
      corpus.passages.map((p) => asPassage(p.text)),
      { variantId, pipelineOptions: { pooling } },
    )
    setEmbedded({ key: cacheKey, vectors: res.output.vectors })
    setDebug(res.debug)
    return res.output.vectors
  }

  async function handleSearch() {
    try {
      await ensureCorpusVectors()
      const res = await model.run<EmbedOutput>([asQuery(query)], {
        variantId,
        pipelineOptions: { pooling },
      })
      setQueryVector(res.output.vectors[0])
    } catch {
      setQueryVector(null)
    }
  }

  async function handlePair() {
    try {
      // So sánh hai câu là bài toán ĐỐI XỨNG, nên cả hai đều dùng tiền tố query.
      const res = await model.run<EmbedOutput>([asQuery(pairA), asQuery(pairB)], {
        variantId,
        pipelineOptions: { pooling },
      })
      const [a, b] = res.output.vectors
      const score = dot(a, b)
      setPairHistory((prev) => [{ b: pairB, score }, ...prev.filter((p) => p.b !== pairB)].slice(0, 5))
      setDebug(res.debug)
    } catch {
      /* lỗi đã được ModelBar hiển thị */
    }
  }

  /** Đổi cấu hình thì kết quả cũ không còn đúng — xoá đi thay vì để người dùng nhầm. */
  function resetResults() {
    setQueryVector(null)
    setPairHistory([])
  }

  const busy = model.isBusy
  const bm25Empty = bm25.length === 0
  const topSemanticIndices = new Set((semantic ?? []).slice(0, 3).map((h) => h.docIndex))

  return (
    <>
      <div className="config-grid">
        <div className="field">
          <div className="field-head">
            <span className="field-label">Model</span>
            <Info title="Vì sao model đa ngữ nặng gấp 5 lần?">
              <p>
                Hai model đều 384 chiều và cùng kiến trúc MiniLM. Chênh lệch 22 MB so với 113 MB
                nằm gần hết ở <strong>ma trận embedding</strong>: 30k token (chỉ tiếng Anh) so với
                250k token (50 ngôn ngữ).
              </p>
              <p>
                Hãy thử tìm bằng corpus tiếng Việt với model “Chỉ tiếng Anh” — thứ hạng sẽ lộn xộn
                hẳn. Đó là cùng một bài học ở card Tokenizer Explorer, nhưng lần này thấy được hậu
                quả trên kết quả cuối.
              </p>
            </Info>
          </div>
          <VariantPicker
            variants={variants}
            value={variantId}
            onChange={(id) => {
              setVariantId(id)
              resetResults()
            }}
            readyVariants={model.readyVariants}
            loadingVariantId={model.loadingVariantId}
            disabled={busy}
          />
        </div>

        <div className="field">
          <div className="field-head">
            <span className="field-label">Corpus</span>
            <Badge mono>{corpus.passages.length} đoạn</Badge>
          </div>
          <div className="chip-row">
            {CORPORA.map((item) => (
              <button
                key={item.id}
                type="button"
                className={`chip${corpusId === item.id ? ' chip--active' : ''}`}
                onClick={() => {
                  setCorpusId(item.id)
                  setQuery(item.sampleQueries[0])
                  resetResults()
                }}
              >
                {item.label}
              </button>
            ))}
          </div>

          <div className="field-head" style={{ marginTop: 8 }}>
            <span className="field-label">Pooling</span>
            <Info title="Pooling — “embedding của câu” không có định nghĩa duy nhất">
              <p>
                Model trả về một vector cho <em>mỗi token</em>, tức ma trận{' '}
                <Tex tex="[\text{seq}, 384]" />. Muốn có một vector cho cả câu thì phải gộp lại, và
                cách gộp là một lựa chọn:
              </p>
              <Tex tex="v_{\text{mean}} = \frac{1}{|M|}\sum_{t \in M} h_t" block />
              <p>
                với <Tex tex="M" /> là tập token thật (loại bỏ <code>[PAD]</code>). CLS pooling thì
                lấy đúng <Tex tex="h_0" />, vector của token <code>[CLS]</code>.
              </p>
              <p>
                Hai model ở đây thuộc họ sentence-transformers, <strong>được huấn luyện với mean
                pooling</strong>. Đổi sang CLS sẽ thấy thứ hạng xấu đi rõ — bằng chứng rằng pooling
                không phải chi tiết kỹ thuật vô hại.
              </p>
              <p className="hint">
                Bỏ mask khi mean pooling là lỗi im lặng phổ biến: câu ngắn bị pha loãng bởi các
                token đệm.
              </p>
            </Info>
          </div>
          <div className="chip-row">
            {(['mean', 'cls'] as Pooling[]).map((mode) => (
              <button
                key={mode}
                type="button"
                className={`chip${pooling === mode ? ' chip--active' : ''}`}
                onClick={() => {
                  setPooling(mode)
                  resetResults()
                }}
              >
                {POOLING_LABELS[mode]}
              </button>
            ))}
          </div>
        </div>
      </div>

      <ModelBar model={model} variantId={variantId} />

      <Tabs items={TABS} value={tab} onChange={setTab} />

      {tab === 'search' && (
        <>
          <div className="field">
            <div className="field-head">
              <label className="field-label" htmlFor="emb-query">
                Truy vấn
              </label>
              <div className="chip-row">
                {corpus.sampleQueries.map((sample, i) => (
                  <button
                    key={i}
                    type="button"
                    className="chip"
                    onClick={() => {
                      setQuery(sample)
                      setQueryVector(null)
                    }}
                  >
                    {i + 1}
                  </button>
                ))}
              </div>
            </div>
            <textarea
              id="emb-query"
              className="input--compact"
              value={query}
              onChange={(e) => {
                setQuery(e.target.value)
                setQueryVector(null)
              }}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && (e.ctrlKey || e.metaKey) && !busy) void handleSearch()
              }}
            />
          </div>

          <div className="control-row">
            <button
              type="button"
              className="primary"
              onClick={handleSearch}
              disabled={busy || query.trim().length === 0}
            >
              {busy ? 'Đang encode…' : corpusReady ? 'Tìm' : `Encode ${corpus.passages.length} đoạn + tìm`}
            </button>
            {corpusReady && <Badge tone="ok">corpus đã encode</Badge>}
          </div>

          <div className="compare-grid">
            <div>
              <div className="field-head">
                <span className="field-label">Ngữ nghĩa (embedding)</span>
                <Info title="Cosine similarity">
                  <Tex tex="\cos(a,b) = \frac{a \cdot b}{\|a\|\,\|b\|}" block />
                  <p>
                    Vector đã được L2-normalize trong worker, nên <Tex tex="\|a\|=\|b\|=1" /> và
                    công thức thu về đúng một phép nhân vô hướng. Đó là lý do mọi vector database
                    đều lưu vector đã normalize: tìm kiếm chỉ còn là một phép nhân ma trận.
                  </p>
                  <p>
                    Cosine đo <strong>góc</strong>, không đo khoảng cách — nên độ dài vector (và do
                    đó độ dài câu) không ảnh hưởng.
                  </p>
                  <p>
                    <strong>Đừng đọc ngưỡng tuyệt đối.</strong> Mỗi model có thang riêng, đo được
                    trên đúng corpus này:
                  </p>
                  <table className="data">
                    <tbody>
                      <tr>
                        <td>Đa ngữ (E5)</td>
                        <td className="mono">0.81 – 0.87</td>
                        <td>dồn rất hẹp</td>
                      </tr>
                      <tr>
                        <td>Chỉ tiếng Anh</td>
                        <td className="mono">0.18 – 0.45</td>
                        <td>giãn rộng</td>
                      </tr>
                    </tbody>
                  </table>
                  <p className="hint">
                    Vì thế thanh ngang vẽ theo mức <em>tương đối trong 5 kết quả đang xem</em>, còn
                    con số bên cạnh là cosine thô. Chỉ so sánh các con số do cùng một model sinh ra.
                  </p>
                </Info>
              </div>
              {!semantic ? (
                <div className="empty-state">Chưa encode. Bấm nút trên.</div>
              ) : (
                (() => {
                  const shown = semantic.slice(0, TOP_K)
                  const scores = shown.map((h) => h.score)
                  const min = Math.min(...scores)
                  const max = Math.max(...scores)
                  return (
                    <ol className="rank">
                      {shown.map((hit) => {
                        const passage = corpus.passages[hit.docIndex]
                        const foundByBm25 = bm25.some((b) => b.docIndex === hit.docIndex)
                        return (
                          <li key={passage.id} className="rank-item">
                            <div className="rank-head">
                              <span className="rank-score">{hit.score.toFixed(3)}</span>
                              <span
                                className="topic-dot"
                                style={{ background: TOPICS[passage.topic].color }}
                                title={TOPICS[passage.topic].label}
                              />
                              {!foundByBm25 && <Badge tone="warn">BM25 bỏ sót</Badge>}
                            </div>
                            <div className="rank-bar-track">
                              <div
                                className="rank-bar"
                                style={{ width: relativeWidth(hit.score, min, max) }}
                              />
                            </div>
                            <p className="rank-text">{passage.text}</p>
                          </li>
                        )
                      })}
                    </ol>
                  )
                })()
              )}
            </div>

            <div>
              <div className="field-head">
                <span className="field-label">Từ khoá (BM25)</span>
                <Info title="BM25 chỉ đếm từ trùng nhau">
                  <Tex
                    tex="\text{score} = \sum_{t \in Q} \text{IDF}(t)\cdot\frac{f(t,D)\,(k_1+1)}{f(t,D) + k_1\left(1-b+b\frac{|D|}{\text{avgdl}}\right)}"
                    block
                  />
                  <p>
                    <Tex tex="f(t,D)" /> là số lần từ <Tex tex="t" /> xuất hiện trong tài liệu,{' '}
                    <Tex tex="\text{IDF}" /> hạ trọng số những từ có ở khắp nơi, và mẫu số chuẩn hoá
                    theo độ dài tài liệu.
                  </p>
                  <p>
                    Điểm mấu chốt: nếu truy vấn <strong>không dùng đúng từ</strong> của tài liệu thì{' '}
                    <Tex tex="f(t,D)=0" /> với mọi <Tex tex="t" />, và điểm bằng 0. BM25 không có
                    cách nào biết “học vẹt” và “ghi nhớ” là cùng một ý.
                  </p>
                  <p className="hint">
                    Đổi lại BM25 không cần model, chạy tức thì, và giải thích được vì sao nó trả về
                    một tài liệu. Thực tế người ta thường kết hợp cả hai (hybrid search).
                  </p>
                </Info>
              </div>
              {bm25Empty ? (
                <div className="callout callout--warn">
                  Không đoạn nào chứa từ nào của truy vấn, nên BM25 trả về{' '}
                  <strong>rỗng</strong>. Cột bên trái vẫn tìm đúng — đó chính là lý do embedding tồn
                  tại.
                </div>
              ) : (
                (() => {
                  const shown = bm25.slice(0, TOP_K)
                  const scores = shown.map((h) => h.score)
                  const min = Math.min(...scores)
                  const max = Math.max(...scores)
                  return (
                    <ol className="rank">
                      {shown.map((hit) => {
                        const passage = corpus.passages[hit.docIndex]
                        return (
                          <li key={passage.id} className="rank-item">
                            <div className="rank-head">
                              <span className="rank-score">{hit.score.toFixed(3)}</span>
                              <span
                                className="topic-dot"
                                style={{ background: TOPICS[passage.topic].color }}
                                title={TOPICS[passage.topic].label}
                              />
                              <span className="matched">{hit.matchedTerms.join(' · ')}</span>
                            </div>
                            <div className="rank-bar-track">
                              <div
                                className="rank-bar rank-bar--keyword"
                                style={{ width: relativeWidth(hit.score, min, max) }}
                              />
                            </div>
                            <p className="rank-text">{passage.text}</p>
                          </li>
                        )
                      })}
                    </ol>
                  )
                })()
              )}
            </div>
          </div>
        </>
      )}

      {tab === 'pair' && (
        <>
          <div className="pair-grid">
            <div className="field">
              <span className="field-label">Câu A</span>
              <textarea
                className="input--compact"
                value={pairA}
                onChange={(e) => {
                  setPairA(e.target.value)
                  // Đổi câu A là mọi con số cũ hết nghĩa (chúng so với câu A cũ).
                  setPairHistory([])
                }}
              />
            </div>
            <div className="field">
              <span className="field-label">Câu B</span>
              <textarea
                className="input--compact"
                value={pairB}
                onChange={(e) => setPairB(e.target.value)}
              />
              <div className="chip-row">
                {PAIR_B_PRESETS.map((preset) => (
                  <button
                    key={preset.label}
                    type="button"
                    className="chip"
                    onClick={() => setPairB(preset.text)}
                  >
                    {preset.label}
                  </button>
                ))}
              </div>
            </div>
          </div>

          <div className="control-row">
            <button type="button" className="primary" onClick={handlePair} disabled={busy}>
              {busy ? 'Đang tính…' : 'Tính cosine'}
            </button>
            <Info title="Đọc con số này thế nào">
              <p>
                Về lý thuyết cosine nằm trong <Tex tex="[-1, 1]" />, nhưng embedding của model
                ngôn ngữ hầu như không bao giờ âm — nên dải thực tế hẹp hơn nhiều và{' '}
                <strong>khác nhau theo từng model</strong>.
              </p>
              <p>
                Cách dùng đúng: đổi một câu rồi xem con số <em>tăng hay giảm</em>, thay vì hỏi
                “0.83 là cao hay thấp”. Muốn có ngưỡng dùng được trong sản phẩm thì phải tự đo trên
                dữ liệu của mình.
              </p>
              <p>Đo được trên model đa ngữ E5, cùng một câu A:</p>
              <table className="data">
                <tbody>
                  <tr>
                    <td>Cùng lĩnh vực, khác nghĩa</td>
                    <td className="num">0.8779</td>
                  </tr>
                  <tr>
                    <td>Đồng nghĩa</td>
                    <td className="num">0.8725</td>
                  </tr>
                  <tr>
                    <td>Bản dịch tiếng Anh</td>
                    <td className="num">0.8441</td>
                  </tr>
                  <tr>
                    <td>Hoàn toàn vô nghĩa</td>
                    <td className="num">0.8124</td>
                  </tr>
                </tbody>
              </table>
              <p>
                Hai điều đáng chú ý. Thứ nhất, toàn bộ dải chỉ rộng <strong>0.065</strong> — nên một
                con số đứng một mình không đọc được, phải xếp cạnh nhau.
              </p>
              <p>
                Thứ hai, câu <em>cùng lĩnh vực nhưng khác nghĩa</em> lại xếp trên câu{' '}
                <em>đồng nghĩa</em>. Embedding kiểu này đo độ gần về <strong>chủ đề</strong> mạnh
                hơn nghĩa chính xác. Đây là hạn chế quan trọng khi dùng chúng cho RAG: hệ thống sẽ
                lấy về tài liệu “đúng chủ đề” chứ chưa chắc “trả lời được câu hỏi”.
              </p>
              <p className="hint">
                Thử lại đúng bốn câu này với model <strong>Chỉ tiếng Anh</strong>: câu vô nghĩa được
                0.4356 còn câu đồng nghĩa chỉ 0.3764 — nó xếp <em>ngược</em>. Model không biết tiếng
                Việt thì cosine của nó vô giá trị, dù vẫn trả về một con số trông rất bình thường.
              </p>
            </Info>
          </div>

          {pairHistory.length === 0 ? (
            <div className="empty-state">
              Bấm lần lượt cả 4 câu B mẫu ở trên, rồi so các con số với nhau.
            </div>
          ) : (
            (() => {
              const scores = pairHistory.map((h) => h.score)
              const min = Math.min(...scores)
              const max = Math.max(...scores)
              const sorted = [...pairHistory].sort((a, b) => b.score - a.score)
              return (
                <ol className="rank">
                  {sorted.map((entry) => (
                    <li key={entry.b} className="rank-item">
                      <div className="rank-head">
                        <span className="rank-score">{entry.score.toFixed(4)}</span>
                        {entry.b === pairB && <Badge tone="concept">đang nhập</Badge>}
                      </div>
                      <div className="rank-bar-track">
                        <div
                          className="rank-bar"
                          style={{
                            width:
                              pairHistory.length > 1 ? relativeWidth(entry.score, min, max) : '100%',
                          }}
                        />
                      </div>
                      <p className="rank-text">{entry.b}</p>
                    </li>
                  ))}
                </ol>
              )
            })()
          )}
        </>
      )}

      {tab === 'map' && (
        <>
          <div className="control-row">
            <button
              type="button"
              className="primary"
              onClick={() => void ensureCorpusVectors()}
              disabled={busy || corpusReady}
            >
              {corpusReady ? 'Đã encode' : `Encode ${corpus.passages.length} đoạn`}
            </button>
            {projection && (
              <>
                <Badge mono tone={projection.explainedRatio < 0.3 ? 'warn' : 'default'}>
                  giữ {(projection.explainedRatio * 100).toFixed(0)}% phương sai
                </Badge>
                <Info title="Bản đồ 2D này méo đến mức nào?">
                  <p>
                    PCA tìm hai hướng mà dữ liệu biến thiên mạnh nhất rồi chiếu 384 chiều xuống
                    chúng. Con số bên cạnh là phần phương sai mà hai trục đó giữ lại.
                  </p>
                  <p>
                    Đo trên corpus này: <strong>18%</strong> với model đa ngữ, <strong>22%</strong>{' '}
                    với model tiếng Anh. Nghĩa là gần 80% thông tin đã bị bỏ đi, ở cả hai model —
                    đây không phải khuyết điểm của một model cụ thể mà là hệ quả của việc nén 384
                    chiều xuống 2.
                  </p>
                  <p>
                    Vì vậy <strong>đừng đọc khoảng cách trên hình theo nghĩa chữ</strong>. Hai điểm
                    trông sát nhau có thể xa nhau trong không gian thật, và ngược lại. Bạn sẽ thấy
                    ngay: một số kết quả tốt nhất (vòng viền) nằm gần dấu ✕, nhưng có kết quả nằm
                    tít phía khác — chính là bằng chứng phép chiếu bị méo.
                  </p>
                  <p className="hint">
                    Thứ hạng tìm kiếm luôn tính bằng cosine trên vector 384 chiều đầy đủ, không bao
                    giờ tính trên toạ độ 2D. Bản đồ chỉ để có trực giác về “không gian vector”.
                  </p>
                </Info>
              </>
            )}
          </div>

          {projection ? (
            <>
              <ScatterPlot
                points={projection.corpusPoints.map(([x, y], i): ScatterPoint => {
                  const passage = corpus.passages[i]
                  return {
                    x,
                    y,
                    color: TOPICS[passage.topic].color,
                    label: `[${TOPICS[passage.topic].label}] ${passage.text}`,
                    highlighted: topSemanticIndices.has(i),
                  }
                })}
                queryPoint={
                  projection.queryPoint
                    ? {
                        x: projection.queryPoint[0],
                        y: projection.queryPoint[1],
                        label: `Truy vấn: ${query}`,
                      }
                    : undefined
                }
              />
              <div className="legend">
                {Object.entries(TOPICS).map(([id, topic]) => (
                  <span key={id} className="legend-item">
                    <i className="swatch" style={{ background: topic.color, borderColor: topic.color }} />
                    {topic.label}
                  </span>
                ))}
                {projection.queryPoint && (
                  <span className="legend-item">
                    <span className="mono">✕</span> truy vấn
                  </span>
                )}
              </div>
              <p className="hint">
                Hover một điểm để xem nội dung đoạn. Vòng viền = 3 kết quả tốt nhất của truy vấn
                hiện tại (chạy tab Tìm kiếm trước). Màu chỉ là chủ đề tôi gán sẵn khi viết corpus —
                model không biết gì về các nhãn đó.
              </p>
            </>
          ) : (
            <div className="empty-state">Encode corpus để vẽ bản đồ.</div>
          )}
        </>
      )}

      <UnderTheHood debug={debug}>
        {corpusReady && (
          <div className="hood-block">
            <h4>
              Từ ma trận token về một vector câu
              <Info title="Tại sao pooling phải biết attention_mask">
                <p>
                  Encode nhiều câu một lượt thì câu ngắn được đệm <code>[PAD]</code> cho bằng câu
                  dài nhất. Lấy trung bình cả phần đệm sẽ pha loãng câu ngắn bằng vector vô nghĩa —
                  càng ngắn càng sai lệch.
                </p>
                <p>
                  Xem <code>poolHiddenStates()</code> trong <code>src/lib/embedding.ts</code>: nó bỏ
                  qua mọi vị trí có mask bằng 0 và chia cho số token thật.
                </p>
              </Info>
            </h4>
            <table className="data">
              <tbody>
                <tr>
                  <td>Số câu đã encode</td>
                  <td className="mono">{embedded.vectors.length}</td>
                </tr>
                <tr>
                  <td>Số chiều mỗi vector</td>
                  <td className="mono">{embedded.vectors[0]?.length}</td>
                </tr>
                <tr>
                  <td>Pooling</td>
                  <td className="mono">{pooling}</td>
                </tr>
                <tr>
                  <td>Độ dài vector sau normalize</td>
                  <td className="mono">
                    {globalThis.Math.sqrt(dot(embedded.vectors[0], embedded.vectors[0])).toFixed(6)}
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        )}
      </UnderTheHood>
    </>
  )
}
