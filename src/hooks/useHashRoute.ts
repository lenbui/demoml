import { useCallback, useEffect, useState } from 'react'

/**
 * Điều hướng bằng hash URL — không cần thư viện router.
 *
 *   #             → gallery (danh sách card nhỏ)
 *   #tokenizer    → chỉ demo tokenizer chiếm toàn trang
 *
 * Dùng hash thay vì history API vì:
 *  • deep-link được: gửi link #embeddings cho sinh viên là mở đúng demo đó;
 *  • F5 không mất chỗ;
 *  • nút back/forward của browser hoạt động sẵn;
 *  • không cần server rewrite, nên deploy tĩnh lên GitHub Pages / HF Spaces vẫn chạy.
 *
 * Hệ quả quan trọng về bộ nhớ: mỗi lúc chỉ một demo được mount, nên chỉ một
 * worker tồn tại. Rời demo là worker bị terminate (xem useModel) và vài trăm MB
 * trọng số được giải phóng.
 */
function readHash(): string | null {
  const raw = window.location.hash.replace(/^#/, '')
  return raw.length > 0 ? decodeURIComponent(raw) : null
}

export function useHashRoute() {
  const [route, setRoute] = useState<string | null>(() => readHash())

  useEffect(() => {
    const sync = () => setRoute(readHash())
    window.addEventListener('hashchange', sync)
    // popstate: bấm back/forward giữa hai state không đổi hash cũng cần đồng bộ.
    window.addEventListener('popstate', sync)
    return () => {
      window.removeEventListener('hashchange', sync)
      window.removeEventListener('popstate', sync)
    }
  }, [])

  // Đổi demo thì cuộn lên đầu — nếu không, mở demo mới ở giữa trang rất mất phương hướng.
  useEffect(() => {
    window.scrollTo({ top: 0 })
  }, [route])

  const navigate = useCallback((id: string | null) => {
    window.location.hash = id ?? ''
  }, [])

  return { route, navigate }
}
