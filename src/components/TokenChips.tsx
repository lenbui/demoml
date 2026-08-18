import { Info } from './Info'

export type TokenKind = 'special' | 'unk' | 'glued' | 'word'

/**
 * Phân loại một token để tô màu.
 *
 * Điểm dễ hiểu sai: ký hiệu đánh dấu KHÁC NHAU tuỳ thuật toán, và ý nghĩa còn
 * NGƯỢC nhau —
 *   WordPiece (BERT)        : '##' ở đầu = token này DÁN LIỀN token trước
 *   SentencePiece (T5, XLM) : '▁'  ở đầu = token này MỞ ĐẦU từ mới (có space)
 *   BPE mức byte (GPT-2)    : 'Ġ'  ở đầu = token này MỞ ĐẦU từ mới (có space)
 * Nghĩa là với SentencePiece/BPE, token KHÔNG có marker mới là token dán liền.
 */
export function classifyToken(
  token: string,
  index: number,
  algorithm: string,
  specialTokens: string[],
  unkToken?: string,
): TokenKind {
  if (unkToken && token === unkToken) return 'unk'
  if (specialTokens.includes(token)) return 'special'

  if (token.startsWith('##')) return 'glued'
  if (token.startsWith('▁') || token.startsWith('Ġ')) return 'word'

  // Với SentencePiece/BPE, thiếu marker = không có khoảng trắng phía trước.
  const markerMeansWordStart = /Unigram|BPE/i.test(algorithm)
  if (markerMeansWordStart && index > 0) return 'glued'

  return 'word'
}

const KIND_CLASS: Record<TokenKind, string> = {
  special: 'token token--special',
  unk: 'token token--unk',
  glued: 'token token--glued',
  word: 'token',
}

export function TokenChips({
  tokens,
  ids,
  algorithm = '',
  specialTokens = [],
  unkToken,
  showIds = false,
}: {
  tokens: string[]
  ids?: number[]
  algorithm?: string
  specialTokens?: string[]
  unkToken?: string
  showIds?: boolean
}) {
  return (
    <div className="token-row">
      {tokens.map((token, i) => {
        const kind = classifyToken(token, i, algorithm, specialTokens, unkToken)
        return (
          <span
            key={`${token}-${i}`}
            className={KIND_CLASS[kind]}
            title={ids ? `vị trí ${i} · id ${ids[i]}` : `vị trí ${i}`}
          >
            <span className="token-text">{token}</span>
            {showIds && ids && <span className="token-id">{ids[i]}</span>}
          </span>
        )
      })}
    </div>
  )
}

/** Chú giải màu — gọn, phần giải thích dài nằm trong popover "?". */
export function TokenLegend({ algorithm }: { algorithm?: string }) {
  return (
    <div className="legend">
      <span className="legend-item">
        <i className="swatch swatch--word" /> từ mới
      </span>
      <span className="legend-item">
        <i className="swatch swatch--glued" /> dán liền
      </span>
      <span className="legend-item">
        <i className="swatch swatch--special" /> đặc biệt
      </span>
      <span className="legend-item">
        <i className="swatch swatch--unk" /> UNK
      </span>
      <Info title="Ký hiệu đánh dấu token">
        <p>
          Mỗi thuật toán dùng một ký hiệu khác nhau, và ý nghĩa còn <strong>ngược nhau</strong>:
        </p>
        <table className="data">
          <tbody>
            <tr>
              <td>
                <code>##</code>
              </td>
              <td>WordPiece (BERT)</td>
              <td>dán liền token trước</td>
            </tr>
            <tr>
              <td>
                <code>▁</code>
              </td>
              <td>SentencePiece (T5, XLM-R)</td>
              <td>mở đầu từ mới</td>
            </tr>
            <tr>
              <td>
                <code>Ġ</code>
              </td>
              <td>BPE mức byte (GPT-2)</td>
              <td>mở đầu từ mới</td>
            </tr>
          </tbody>
        </table>
        <p>
          Với SentencePiece và BPE, token <em>thiếu</em> marker mới là token dán liền — nên dấu
          cách được mã hoá thành một phần của token, không phải bị bỏ đi.
        </p>
        {algorithm && (
          <p className="hint">
            Tokenizer đang chọn dùng thuật toán <code>{algorithm}</code>.
          </p>
        )}
      </Info>
    </div>
  )
}
