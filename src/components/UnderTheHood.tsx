import { useState, type ReactNode } from 'react'

import type { DebugInfo } from '../workers/protocol'
import { Info } from './Info'
import { Panel } from './Panel'
import { TokenChips, TokenLegend } from './TokenChips'

/**
 * Panel "mở hộp đen" — phần có giá trị dạy học cao nhất của dashboard.
 *
 * Hiện những thứ mà `pipeline()` bình thường che đi: chuỗi token, input_ids,
 * shape tensor, và output thô trước hậu xử lý.
 *
 * `children` để mỗi demo chèn thêm phần giải thích riêng (ví dụ card sentiment
 * chèn bảng tính softmax từng bước).
 */
export function UnderTheHood({
  debug,
  children,
  defaultOpen = false,
}: {
  debug?: DebugInfo
  children?: ReactNode
  defaultOpen?: boolean
}) {
  const [showIds, setShowIds] = useState(true)

  if (!debug && !children) return null

  const hasTokens = Boolean(debug?.tokens?.length)

  return (
    <Panel title="Under the hood" defaultOpen={defaultOpen}>
      {hasTokens && debug?.tokens && (
        <div className="hood-block">
          <h4>
            Tokenize
            <span className="hood-count">{debug.tokens.length} token</span>
            <Info title="Model không đọc chữ">
              <p>
                Model chỉ nhận số. Tokenizer cắt câu thành các mảnh có trong vocabulary rồi đổi mỗi
                mảnh thành một số nguyên — đó là <code>input_ids</code>.
              </p>
              <p>
                Từ không có trong vocabulary bị cắt nhỏ tiếp thành subword. Nhờ vậy tokenizer không
                bao giờ gặp từ “lạ hoàn toàn”, nhưng đổi lại chuỗi dài ra và ngữ nghĩa bị chia vụn.
              </p>
            </Info>
          </h4>
          <div className="field-head" style={{ marginBottom: 8 }}>
            <TokenLegend />
            <label className="toggle">
              <input
                type="checkbox"
                checked={showIds}
                onChange={(e) => setShowIds(e.target.checked)}
              />
              hiện id
            </label>
          </div>
          <TokenChips tokens={debug.tokens} ids={debug.inputIds} showIds={showIds} />
        </div>
      )}

      {debug?.tensors && debug.tensors.length > 0 && (
        <div className="hood-block">
          <h4>
            Tensor vào / ra
            <Info title="Đọc shape của tensor">
              <table className="data">
                <tbody>
                  <tr>
                    <td className="mono">[1, 19]</td>
                    <td>1 câu trong batch, 19 token</td>
                  </tr>
                  <tr>
                    <td className="mono">[1, 2]</td>
                    <td>1 câu, 2 nhãn — mỗi nhãn một logit</td>
                  </tr>
                  <tr>
                    <td className="mono">[1, 19, 768]</td>
                    <td>1 câu, 19 token, mỗi token một vector 768 chiều</td>
                  </tr>
                </tbody>
              </table>
              <p className="hint">
                Chiều đầu luôn là batch. <code>int64</code> cho id vì vocabulary có thể rất lớn;{' '}
                <code>float32</code> cho mọi giá trị tính toán.
              </p>
            </Info>
          </h4>
          <div className="table-scroll">
            <table className="data">
              <thead>
                <tr>
                  <th>Tên</th>
                  <th>Shape</th>
                  <th>dtype</th>
                </tr>
              </thead>
              <tbody>
                {debug.tensors.map((t) => (
                  <tr key={t.name}>
                    <td className="mono">{t.name}</td>
                    <td className="mono">[{t.dims.join(', ')}]</td>
                    <td className="mono">{t.dtype}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {children}

      {debug?.notes && debug.notes.length > 0 && (
        <ul className="notes">
          {debug.notes.map((note, i) => (
            <li key={i}>{note}</li>
          ))}
        </ul>
      )}

      {debug?.raw !== undefined && (
        <details className="raw-output">
          <summary>Output thô từ model</summary>
          <pre className="code">{JSON.stringify(debug.raw, null, 2)}</pre>
        </details>
      )}
    </Panel>
  )
}
