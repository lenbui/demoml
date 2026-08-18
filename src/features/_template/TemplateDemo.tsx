import { useState } from 'react'

import { Info } from '../../components/Info'
import { Math as Tex } from '../../components/Math'
import { ModelBar } from '../../components/ModelBar'
import { UnderTheHood } from '../../components/UnderTheHood'
import { useModel } from '../../hooks/useModel'
import type { DebugInfo } from '../../workers/protocol'

/**
 * ══════════════════════════════════════════════════════════════════════════
 * KHUNG MẪU CHO DEMO MỚI — copy cả folder `_template` rồi đổi tên.
 * ══════════════════════════════════════════════════════════════════════════
 *
 * Bốn bước:
 *   1. Thêm entry vào MODEL_REGISTRY (src/lib/modelRegistry.ts) với id trùng
 *      với id trong index.ts của demo.
 *   2. Đổi 'TODO-demo-id' bên dưới thành id đó.
 *   3. Sửa phần input + phần hiển thị kết quả cho đúng task của bạn.
 *   4. Đăng ký demo trong src/features/index.ts.
 *
 * Kiểu của `output` phụ thuộc task. Tra bảng ở
 * https://huggingface.co/docs/transformers.js/api/pipelines
 * hoặc bấm "Chạy" một lần rồi đọc mục "Output thô từ model" trong Under the hood.
 *
 * ── QUY ƯỚC GIAO DIỆN (được chấm điểm) ─────────────────────────────────────
 *  • KHÔNG in đoạn giải thích dài ra card. Đưa vào <Info title="…">…</Info> —
 *    nút "?" nhỏ, bấm mới mở. Card chỉ chứa thứ người dùng cần nhìn.
 *  • Công thức viết bằng <Tex tex="p_i = \frac{e^{z_i}}{\sum_j e^{z_j}}" />,
 *    thêm `block` cho công thức đứng riêng dòng. Không viết công thức bằng ký
 *    tự thường.
 *  • Kết quả phải trực quan (thanh xác suất, bounding box, bảng…), không chỉ
 *    JSON.
 *  • KHÔNG cần thêm spellCheck={false} vào input: đã tắt một lần cho cả trang ở
 *    <body> trong index.html (spellcheck di truyền xuống mọi thẻ con).
 */

/** TODO: thay bằng kiểu output thật của task bạn chọn. */
type Output = unknown

export function TemplateDemo() {
  const model = useModel('TODO-demo-id')

  const [input, setInput] = useState('Nhập dữ liệu thử ở đây')
  const [output, setOutput] = useState<Output>(null)
  const [debug, setDebug] = useState<DebugInfo | undefined>(undefined)

  async function handleRun() {
    try {
      const res = await model.run<Output>(input)
      setOutput(res.output)
      setDebug(res.debug)
    } catch {
      setOutput(null)
      setDebug(undefined)
    }
  }

  return (
    <>
      <ModelBar model={model} />

      <div className="field">
        <div className="field-head">
          <label className="field-label" htmlFor="template-input">
            Input
          </label>
          {/* Mẫu dùng Info + Math — xoá nếu demo không cần. */}
          <Info title="TODO: khái niệm demo này dạy">
            <p>Giải thích ngắn gọn khái niệm, kèm công thức nếu có:</p>
            <Tex tex="\text{TODO}" block />
          </Info>
        </div>
        <textarea
          id="template-input"
          className="input--compact"
          value={input}
          onChange={(e) => setInput(e.target.value)}
        />
      </div>

      <div className="control-row">
        <button
          type="button"
          className="primary"
          onClick={handleRun}
          disabled={model.isBusy || input.trim().length === 0}
        >
          Chạy
        </button>
      </div>

      {/* TODO: thay <pre> bằng phần hiển thị trực quan (thanh xác suất, bounding
          box, danh sách kết quả…). Chấm điểm dựa vào việc kết quả có được trình
          bày dễ hiểu hay không, không chỉ ở việc in ra JSON. */}
      {output != null && <pre className="code">{JSON.stringify(output, null, 2)}</pre>}

      {/* TODO: thêm phần giải thích riêng của demo (công thức, bảng trung gian)
          làm children của UnderTheHood — xem SentimentDemo.tsx làm mẫu. */}
      <UnderTheHood debug={debug} />
    </>
  )
}
