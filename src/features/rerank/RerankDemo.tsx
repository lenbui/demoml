import { useMemo, useState } from 'react'

import { Badge } from '../../components/Badge'
import { Info } from '../../components/Info'
import { Math as Tex } from '../../components/Math'
import { ModelBar } from '../../components/ModelBar'
import { UnderTheHood } from '../../components/UnderTheHood'
import { useModel } from '../../hooks/useModel'
import { buildBm25Index, bm25Search } from '../../lib/bm25'
import { formatMs } from '../../lib/format'
import { sigmoid } from '../../lib/math'
import { CORPORA, getCorpus, TOPICS } from '../embeddings/corpus'
import type { DebugInfo, PairScoreOutput } from '../../workers/protocol'

type Stage = 'bm25' | 'all'

interface Ranked {
  docIndex: number
  score: number
  /** Thứ hạng ở tầng retrieve (0-based), để tính số bậc đã dịch chuyển. */
  retrieveRank: number
}

const CANDIDATE_OPTIONS = [3, 5, 10, 20]

export function RerankDemo() {
  const model = useModel('rerank')

  // Model ms-marco chỉ học tiếng Anh, nên mặc định corpus tiếng Anh.
  const [corpusId, setCorpusId] = useState('en')
  const [query, setQuery] = useState(getCorpus('en').sampleQueries[0])
  const [stage, setStage] = useState<Stage>('bm25')
  const [topN, setTopN] = useState(10)
  const [result, setResult] = useState<{
    key: string
    ranked: Ranked[]
    rawScores: number[]
    ms: number
  } | null>(null)
  const [debug, setDebug] = useState<DebugInfo | undefined>(undefined)

  const corpus = getCorpus(corpusId)

  /** Tầng retrieve. BM25 không cần model nên có kết quả tức thì. */
  const retrieved = useMemo(() => {
    if (stage === 'all') {
      return corpus.passages.map((_, i) => ({ docIndex: i, score: 0, matchedTerms: [] }))
    }
    const index = buildBm25Index(corpus.passages.map((p) => p.text))
    return bm25Search(index, query)
      .filter((hit) => hit.score > 0)
      .slice(0, topN)
  }, [corpus, query, stage, topN])

  const cacheKey = `${corpusId}:${query}:${stage}:${stage === 'bm25' ? topN : 'all'}`
  const fresh = result?.key === cacheKey

  async function handleRerank() {
    if (retrieved.length === 0) return
    const passages = retrieved.map((hit) => corpus.passages[hit.docIndex].text)
    try {
      // Mỗi tài liệu ứng viên thành MỘT cặp (truy vấn, tài liệu).
      const res = await model.run<PairScoreOutput>({
        a: passages.map(() => query),
        b: passages,
      })
      // num_labels = 1 -> mỗi cặp chỉ có một logit, là điểm liên quan.
      const rawScores = res.output.logits.map((row) => row[0])
      const ranked: Ranked[] = retrieved
        .map((hit, i) => ({ docIndex: hit.docIndex, score: rawScores[i], retrieveRank: i }))
        .sort((a, b) => b.score - a.score)

      setResult({ key: cacheKey, ranked, rawScores, ms: res.ms })
      setDebug(res.debug)
    } catch {
      setResult(null)
      setDebug(undefined)
    }
  }

  const msPerPair = result && result.ranked.length ? result.ms / result.ranked.length : null
  /** Ngoại suy sang quy mô thật — con số này là toàn bộ lý do có tầng retrieve. */
  const millionDocEstimate = msPerPair ? (msPerPair * 1_000_000) / 1000 : null

  const canRun = query.trim().length > 0 && retrieved.length > 0 && !model.isBusy

  return (
    <>
      <div className="config-grid">
        <div className="field">
          <div className="field-head">
            <span className="field-label">Corpus</span>
            <Badge mono>{corpus.passages.length} đoạn</Badge>
            {corpusId !== 'en' && <Badge tone="warn">model chỉ biết tiếng Anh</Badge>}
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
                  setResult(null)
                }}
              >
                {item.label}
              </button>
            ))}
          </div>
        </div>

        <div className="field">
          <div className="field-head">
            <span className="field-label">Tầng retrieve</span>
            <Info title="Vì sao phải có hai tầng">
              <p>
                Cross-encoder phải chạy model <strong>một lần cho mỗi tài liệu</strong>. Với 1 triệu
                tài liệu thì mỗi truy vấn cần 1 triệu forward pass — không khả thi.
              </p>
              <p>Nên hệ thống thật luôn xếp thành hai tầng:</p>
              <table className="data">
                <tbody>
                  <tr>
                    <td>retrieve</td>
                    <td>nhanh, rẻ, hơi thô (BM25 hoặc bi-encoder) → lấy vài chục ứng viên</td>
                  </tr>
                  <tr>
                    <td>rerank</td>
                    <td>chậm, đắt, chính xác → xếp lại đúng mấy chục ứng viên đó</td>
                  </tr>
                </tbody>
              </table>
              <p className="hint">
                Chọn "toàn bộ corpus" để xem thứ hạng lý tưởng nếu tiền bạc là vô hạn — rồi so với
                thứ hạng khi chỉ được xem top-N của BM25.
              </p>
            </Info>
          </div>
          <div className="chip-row">
            <button
              type="button"
              className={`chip${stage === 'bm25' ? ' chip--active' : ''}`}
              onClick={() => {
                setStage('bm25')
                setResult(null)
              }}
            >
              BM25 top-N
            </button>
            <button
              type="button"
              className={`chip${stage === 'all' ? ' chip--active' : ''}`}
              onClick={() => {
                setStage('all')
                setResult(null)
              }}
            >
              Toàn bộ corpus ({corpus.passages.length})
            </button>
          </div>

          {stage === 'bm25' && (
            <>
              <div className="field-head" style={{ marginTop: 8 }}>
                <span className="field-label">N ứng viên</span>
                <Badge mono>{retrieved.length} thực tế</Badge>
              </div>
              <div className="chip-row">
                {CANDIDATE_OPTIONS.map((n) => (
                  <button
                    key={n}
                    type="button"
                    className={`chip${topN === n ? ' chip--active' : ''}`}
                    onClick={() => {
                      setTopN(n)
                      setResult(null)
                    }}
                  >
                    {n}
                  </button>
                ))}
              </div>
            </>
          )}
        </div>
      </div>

      <ModelBar model={model} />

      <div className="field">
        <div className="field-head">
          <label className="field-label" htmlFor="rr-query">
            Truy vấn
          </label>
          <Info title="Bi-encoder vs cross-encoder">
            <p>
              Card Embedding mã hoá truy vấn và tài liệu <strong>độc lập</strong> thành hai vector
              rồi đo góc:
            </p>
            <Tex tex="\text{score} = \cos\big(f(q),\, f(d)\big)" block />
            <p>
              Vì <Tex tex="f(d)" /> không phụ thuộc truy vấn, mọi tài liệu được mã hoá{' '}
              <em>một lần</em> rồi lưu vào vector database. Tìm kiếm chỉ còn là một phép nhân ma
              trận — nên chạy được trên hàng triệu tài liệu.
            </p>
            <p>Cross-encoder thì đọc cả cặp cùng lúc:</p>
            <Tex tex="\text{score} = g\big([\text{CLS}]\ q\ [\text{SEP}]\ d\ [\text{SEP}]\big)" block />
            <p>
              Mọi token của truy vấn attend được tới mọi token của tài liệu, nên nó bắt được các
              quan hệ mà hai vector rời rạc không thể diễn đạt — chính xác hơn rõ rệt.
            </p>
            <p>
              Cái giá: <strong>không cache được gì cả</strong>. Điểm số phụ thuộc cả cặp, nên đổi
              truy vấn là phải chạy lại toàn bộ N cặp.
            </p>
          </Info>
          <div className="chip-row">
            {corpus.sampleQueries.map((sample, i) => (
              <button
                key={i}
                type="button"
                className="chip"
                onClick={() => {
                  setQuery(sample)
                  setResult(null)
                }}
              >
                {i + 1}
              </button>
            ))}
          </div>
        </div>
        <textarea
          id="rr-query"
          className="input--compact"
          value={query}
          onChange={(e) => {
            setQuery(e.target.value)
            setResult(null)
          }}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && (e.ctrlKey || e.metaKey) && canRun) void handleRerank()
          }}
        />
      </div>

      <div className="control-row">
        <button type="button" className="primary" onClick={handleRerank} disabled={!canRun}>
          {model.status === 'running'
            ? 'Đang chấm điểm…'
            : `Rerank ${retrieved.length} ứng viên`}
        </button>
        <span className="kbd-hint">Ctrl + Enter</span>
      </div>

      {retrieved.length === 0 && (
        <div className="callout callout--warn">
          BM25 không tìm được đoạn nào chứa từ của truy vấn, nên tầng retrieve trả về{' '}
          <strong>rỗng</strong> — và reranker không có gì để xếp lại.
          <br />
          Đây là <strong>trần recall</strong>: một reranker giỏi đến đâu cũng không thể tìm ra tài
          liệu mà tầng retrieve không đưa cho nó. Đổi sang "toàn bộ corpus" để thấy đoạn đúng vẫn tồn
          tại, chỉ là BM25 không lấy được.
        </div>
      )}

      {retrieved.length > 0 && !fresh && !model.isBusy && (
        <div className="empty-state">
          {retrieved.length} ứng viên đã sẵn sàng. Bấm Rerank để chấm điểm từng cặp (truy vấn, tài
          liệu).
        </div>
      )}

      {fresh && result && (
        <>
          <div className="metrics">
            <Badge mono tone="concept">
              {result.ranked.length} cặp
            </Badge>
            <Badge mono>{formatMs(result.ms)}</Badge>
            {msPerPair != null && <Badge mono>{msPerPair.toFixed(1)} ms/cặp</Badge>}
            {millionDocEstimate != null && (
              <Badge mono tone="warn">
                ≈ {(millionDocEstimate / 60).toFixed(0)} phút cho 1 triệu tài liệu
              </Badge>
            )}
            <Info title="Con số này là lý do tồn tại của vector database">
              <p>
                Chi phí cross-encoder <strong>tuyến tính theo số tài liệu</strong>, và nhân với mỗi
                truy vấn:
              </p>
              <Tex tex="T_{\text{rerank}} = N \times t_{\text{forward}}" block />
              <p>
                Đo được trên máy này: {msPerPair?.toFixed(1)} ms mỗi cặp. Với 1 triệu tài liệu là{' '}
                {millionDocEstimate ? (millionDocEstimate / 60).toFixed(0) : '—'} phút — cho{' '}
                <em>một</em> truy vấn.
              </p>
              <p>
                Bi-encoder thì mã hoá tài liệu trước một lần, lúc tìm chỉ còn một phép nhân ma trận:
              </p>
              <Tex tex="T_{\text{bi}} = t_{\text{forward}} + N \times d \text{ phép nhân}" block />
              <p className="hint">
                Đó là lý do kiến trúc hai tầng thắng cả hai phương án thuần: dùng bi-encoder để hạ N
                từ một triệu xuống vài chục, rồi mới trả tiền cho cross-encoder.
              </p>
            </Info>
          </div>

          <div className="compare-grid">
            <div>
              <div className="field-head">
                <span className="field-label">
                  Trước · {stage === 'bm25' ? 'BM25' : 'thứ tự corpus'}
                </span>
              </div>
              <ol className="rank">
                {retrieved.map((hit) => {
                  const passage = corpus.passages[hit.docIndex]
                  return (
                    <li key={passage.id} className="rank-item">
                      <div className="rank-head">
                        <span className="rank-score">
                          {stage === 'bm25' ? hit.score.toFixed(3) : '—'}
                        </span>
                        <span
                          className="topic-dot"
                          style={{ background: TOPICS[passage.topic].color }}
                          title={TOPICS[passage.topic].label}
                        />
                      </div>
                      <p className="rank-text">{passage.text}</p>
                    </li>
                  )
                })}
              </ol>
            </div>

            <div>
              <div className="field-head">
                <span className="field-label">Sau · cross-encoder</span>
                <Info title="Đọc điểm của reranker">
                  <p>
                    Model này có <code>num_labels = 1</code>: không có nhãn nào để softmax với nhau,
                    nên đầu ra là <strong>một số thực bất kỳ</strong>, không phải xác suất. Số âm là
                    bình thường.
                  </p>
                  <p>Muốn ép về khoảng (0, 1) thì dùng sigmoid:</p>
                  <Tex tex="\sigma(z) = \frac{1}{1 + e^{-z}}" block />
                  <p>
                    Nhưng <Tex tex="\sigma(z)" /> ở đây <strong>không</strong> là "khả năng liên
                    quan". Model được huấn luyện để xếp <em>thứ tự</em> đúng, không phải để hiệu
                    chỉnh xác suất. Chỉ nên so các điểm với nhau trong cùng một truy vấn.
                  </p>
                  <p className="hint">
                    Đo trên corpus tiếng Anh này: điểm rơi vào khoảng −11.4 tới −6.5, tức MS MARCO
                    coi mọi đoạn là "không mấy liên quan" (nó được huấn luyện trên đoạn web, không
                    phải văn bản giáo trình). Nhưng <strong>thứ tự vẫn đúng</strong> — đó chính là
                    thứ model này được huấn luyện để làm, và là lý do đừng đọc giá trị tuyệt đối.
                  </p>
                </Info>
              </div>
              <ol className="rank">
                {result.ranked.map((hit, newRank) => {
                  const passage = corpus.passages[hit.docIndex]
                  const delta = hit.retrieveRank - newRank
                  return (
                    <li key={passage.id} className="rank-item">
                      <div className="rank-head">
                        <span className="rank-score">{hit.score.toFixed(3)}</span>
                        <span
                          className="topic-dot"
                          style={{ background: TOPICS[passage.topic].color }}
                          title={TOPICS[passage.topic].label}
                        />
                        <Badge mono>σ {sigmoid(hit.score).toFixed(3)}</Badge>
                        {delta !== 0 && (
                          <Badge tone={delta > 0 ? 'ok' : 'danger'}>
                            {delta > 0 ? `↑${delta}` : `↓${-delta}`}
                          </Badge>
                        )}
                      </div>
                      <p className="rank-text">{passage.text}</p>
                    </li>
                  )
                })}
              </ol>
            </div>
          </div>
        </>
      )}

      <UnderTheHood debug={debug}>
        {fresh && result && (
          <div className="hood-block">
            <h4>
              Dải điểm của cross-encoder
              <Info title="So với dải cosine của bi-encoder">
                <p>
                  Card Embedding đo được cosine dồn trong 0.81–0.87 với model E5 — chênh nhau 0.06
                  giữa câu đồng nghĩa và câu vô nghĩa, gần như không đọc được.
                </p>
                <p>
                  Cross-encoder không bị giới hạn trong <Tex tex="[-1, 1]" /> và được huấn luyện
                  trực tiếp cho việc xếp hạng, nên nó tách các mức liên quan ra xa hơn hẳn: đo trên
                  corpus này dải rộng khoảng <strong>4.9</strong>, so với <strong>0.06</strong> của
                  cosine. Đó là phần "chính xác hơn" thể hiện thành số.
                </p>
                <p className="hint">
                  Thử truy vấn số 1 với BM25 top-6: đoạn trả lời đúng bị BM25 xếp <strong>cuối
                  bảng</strong>, cross-encoder kéo thẳng lên hạng 1 (badge ↑5). Đó là toàn bộ công
                  việc của một reranker.
                </p>
              </Info>
            </h4>
            <table className="data">
              <tbody>
                <tr>
                  <td>Điểm cao nhất</td>
                  <td className="mono">{Math.max(...result.rawScores).toFixed(4)}</td>
                </tr>
                <tr>
                  <td>Điểm thấp nhất</td>
                  <td className="mono">{Math.min(...result.rawScores).toFixed(4)}</td>
                </tr>
                <tr>
                  <td>Độ giãn của dải</td>
                  <td className="mono">
                    {(Math.max(...result.rawScores) - Math.min(...result.rawScores)).toFixed(4)}
                  </td>
                </tr>
                <tr>
                  <td>Số cặp đã chạy</td>
                  <td className="mono">{result.ranked.length}</td>
                </tr>
              </tbody>
            </table>
            <ul className="notes">
              <li>
                Số tài liệu đổi thứ hạng sau khi rerank:{' '}
                <strong>{result.ranked.filter((h, i) => h.retrieveRank !== i).length}</strong> /{' '}
                {result.ranked.length}. Càng nhiều nghĩa là tầng retrieve càng thô.
              </li>
            </ul>
          </div>
        )}
      </UnderTheHood>
    </>
  )
}
