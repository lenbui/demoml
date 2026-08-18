import type { useModel } from '../hooks/useModel'
import { resolveVariant } from '../lib/modelRegistry'
import { HAS_WEBGPU } from '../lib/device'
import { formatMs } from '../lib/format'
import { Badge } from './Badge'
import { Info } from './Info'
import { ProgressBar } from './ProgressBar'

type Model = ReturnType<typeof useModel>

/**
 * Thanh trạng thái model: nút tải, progress bar, và các số đo (backend, thời
 * gian load, thời gian inference).
 *
 * Các con số ở đây chính là dữ liệu sinh viên đưa vào phần benchmark của báo cáo.
 * Phần giải thích dài nằm sau nút "?" để card không bị ngập chữ.
 */
export function ModelBar({ model, variantId }: { model: Model; variantId?: string }) {
  const { spec, status, overall, currentFile, device, deviceNote, loadMs, lastMs, error } = model
  const variant = resolveVariant(spec, variantId)

  // Mode 'tokenizer' không tạo ONNX session nào, nên nói "chạy trên webgpu/wasm"
  // là sai lệch — ẩn hẳn phần backend đi thay vì báo một con số vô nghĩa.
  const hasBackend = spec.mode !== 'tokenizer'

  const blockedByWebGPU = spec.requiresWebGPU && !HAS_WEBGPU

  if (blockedByWebGPU) {
    return (
      <div className="callout callout--warn">
        Demo này cần <strong>WebGPU</strong> nhưng trình duyệt/máy hiện tại không hỗ trợ.
      </div>
    )
  }

  if (status === 'error') {
    return (
      <div className="stack">
        <div className="callout callout--danger">
          <strong>Lỗi:</strong> {error}
        </div>
        <button type="button" onClick={() => model.load(variantId)}>
          Thử lại
        </button>
      </div>
    )
  }

  if (status === 'idle') {
    return (
      <div className="control-row">
        <button type="button" onClick={() => model.load(variantId)}>
          Tải model · {variant.approxSizeMB} MB
        </button>
        <Info title="Model tải về máy bạn">
          <p>
            Trọng số được tải từ Hugging Face Hub xuống trình duyệt và chạy tại đây. Không có
            backend nào nhận dữ liệu của bạn.
          </p>
          <p>
            Trình duyệt cache lại vào Cache Storage, nên lần mở sau gần như tức thì. Đổi model
            khác rồi quay lại cũng không phải tải lại.
          </p>
        </Info>
      </div>
    )
  }

  if (status === 'loading' && overall < 1) {
    return (
      <ProgressBar
        value={overall}
        left={currentFile ?? 'Đang khởi tạo…'}
        right={`${Math.round(overall * 100)}%`}
      />
    )
  }

  return (
    <div className="stack">
      <div className="metrics">
        <Badge tone="ok">sẵn sàng</Badge>
        {hasBackend && device && (
          <Badge tone={device === 'webgpu' ? 'ok' : 'default'} mono>
            {device}
          </Badge>
        )}
        {hasBackend && spec.dtype && <Badge mono>{spec.dtype}</Badge>}
        {!hasBackend && <Badge mono>chỉ tokenizer</Badge>}
        {loadMs != null && <Badge mono>load {formatMs(loadMs)}</Badge>}
        {lastMs != null && <Badge mono>run {formatMs(lastMs)}</Badge>}
        <Info title="Các số đo này nghĩa là gì">
          <table className="data">
            <tbody>
              {hasBackend ? (
                <>
                  <tr>
                    <td>
                      <code>wasm</code> / <code>webgpu</code>
                    </td>
                    <td>backend thực sự chạy: CPU qua WebAssembly, hay GPU qua WebGPU</td>
                  </tr>
                  <tr>
                    <td>
                      <code>{spec.dtype ?? 'fp32'}</code>
                    </td>
                    <td>độ chính xác trọng số — q8 nhẹ hơn fp32 khoảng 4 lần</td>
                  </tr>
                </>
              ) : (
                <tr>
                  <td>
                    <code>chỉ tokenizer</code>
                  </td>
                  <td>
                    không tải trọng số và không tạo ONNX session, nên không có backend nào để
                    báo — tokenize chạy bằng JavaScript thuần
                  </td>
                </tr>
              )}
              <tr>
                <td>
                  <code>load</code>
                </td>
                <td>
                  thời gian tải {hasBackend ? '+ khởi tạo session ONNX' : 'file tokenizer'}
                </td>
              </tr>
              <tr>
                <td>
                  <code>run</code>
                </td>
                <td>thời gian của lần chạy gần nhất</td>
              </tr>
            </tbody>
          </table>
          {hasBackend && (
            <p className="hint">
              Muốn lấy số liệu cho báo cáo: đổi <code>dtype</code> hoặc <code>device</code> trong{' '}
              <code>src/lib/modelRegistry.ts</code>, reload, chạy vài lần rồi lấy trung bình.
            </p>
          )}
        </Info>

        {/* Cảnh báo đổi backend là thông tin đúng và quan trọng, nhưng in cả đoạn
            ra card thì chiếm ba dòng trên mọi demo. Nén thành badge + "?". */}
        {hasBackend && deviceNote && (
          <>
            <Badge tone="warn">backend đã bị hạ</Badge>
            <Info title="Vì sao scaffold tự đổi backend">
              <p>{deviceNote}</p>
              <p className="hint">
                Đây là lỗi <strong>âm thầm</strong>: model vẫn chạy và vẫn trả về số, chỉ là số rác.
                Xem <code>resolveDevice()</code> trong <code>src/workers/pipeline.worker.ts</code>.
              </p>
            </Info>
          </>
        )}
      </div>
    </div>
  )
}
