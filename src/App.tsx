import { Badge } from './components/Badge'
import { DemoCard } from './components/DemoCard'
import { DemoGallery } from './components/DemoGallery'
import { Info } from './components/Info'
import { DEMOS, findDemo } from './features'
import { useHashRoute } from './hooks/useHashRoute'
import { USE_LOCAL_MODELS } from './lib/config'
import { CPU_THREADS, HAS_SHARED_ARRAY_BUFFER, HAS_WEBGPU } from './lib/device'

export default function App() {
  const { route, navigate } = useHashRoute()
  const demo = route ? findDemo(route) : undefined

  return (
    <>
      <nav className="topbar">
        <a
          className="topbar-brand"
          href="#"
          onClick={(e) => {
            e.preventDefault()
            navigate(null)
          }}
        >
          ML · Transformers.js
        </a>

        {/* Chỉ hiện danh sách nhảy nhanh khi đang ở trong một demo. Ở gallery thì
            bản thân lưới card đã là danh sách rồi. */}
        {demo && (
          <div className="topbar-links">
            {DEMOS.map((item) => (
              <a
                key={item.id}
                href={`#${item.id}`}
                className={item.id === demo.id ? 'is-active' : undefined}
              >
                {item.title}
              </a>
            ))}
          </div>
        )}

        <div className="topbar-env">
          <Badge tone={HAS_WEBGPU ? 'ok' : 'warn'}>{HAS_WEBGPU ? 'WebGPU' : 'WASM'}</Badge>
          <Info title="Môi trường chạy trên máy này">
            <table className="data">
              <tbody>
                <tr>
                  <td>WebGPU</td>
                  <td>{HAS_WEBGPU ? 'có' : 'không — mọi model chạy trên CPU'}</td>
                </tr>
                <tr>
                  <td>WASM đa luồng</td>
                  <td>{HAS_SHARED_ARRAY_BUFFER ? 'có' : 'không — chậm hơn 2–4 lần'}</td>
                </tr>
                <tr>
                  <td>Luồng CPU</td>
                  <td>{CPU_THREADS || 'không rõ'}</td>
                </tr>
                <tr>
                  <td>Nguồn model</td>
                  <td>{USE_LOCAL_MODELS ? 'public/models (offline)' : 'huggingface.co'}</td>
                </tr>
              </tbody>
            </table>
            <p className="hint">
              WASM đa luồng cần header COOP/COEP — đã cấu hình trong <code>vite.config.ts</code>.
            </p>
          </Info>
        </div>
      </nav>

      <div className="app">
        {demo ? (
          <>
            <div className="detail-head">
              <button
                type="button"
                className="back"
                onClick={() => navigate(null)}
                aria-label="Quay lại danh sách demo"
              >
                ← Tất cả demo
              </button>
            </div>
            {/* key: đổi demo là unmount hoàn toàn cái cũ, nên worker của nó bị
                terminate và trọng số được giải phóng khỏi RAM. */}
            <DemoCard key={demo.id} demo={demo} />
          </>
        ) : (
          <>
            <header className="hero">
              <h1>Machine Learning trong trình duyệt</h1>
              <p>
                Chọn một demo. Mọi model chạy tại máy bạn, không có backend inference.
                <Info title="Chạy trong trình duyệt nghĩa là gì">
                  <p>
                    Trọng số model ở định dạng ONNX được tải từ Hugging Face Hub xuống trình duyệt,
                    rồi thực thi bằng WebAssembly (CPU) hoặc WebGPU (GPU).
                  </p>
                  <p>
                    Hệ quả: dữ liệu bạn nhập <strong>không rời khỏi máy</strong>, không cần server,
                    và sau lần tải đầu thì dùng được cả khi mất mạng.
                  </p>
                  <p className="hint">
                    Con số MB trên mỗi card là dung lượng phải tải lần đầu. Trình duyệt cache lại
                    nên lần sau gần như tức thì.
                  </p>
                </Info>
              </p>
            </header>

            <DemoGallery demos={DEMOS} />

            <footer className="app-footer">
              Thêm demo mới: khai báo model trong <code>src/lib/modelRegistry.ts</code>, copy{' '}
              <code>src/features/_template/</code>, đăng ký ở <code>src/features/index.ts</code>.
              Chi tiết trong README.
            </footer>
          </>
        )}
      </div>
    </>
  )
}
