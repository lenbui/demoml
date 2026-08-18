/**
 * Render công thức toán bằng KaTeX.
 *
 * Cũng lazy import như highlight.ts: KaTeX ~280 KB, chỉ cần khi có công thức
 * thực sự xuất hiện trên màn hình (phần lớn nằm trong popover "?").
 *
 * CSS của KaTeX được import ở src/main.tsx vì nó kèm font, cần có sẵn trước khi
 * công thức đầu tiên render, nếu không chữ sẽ nhảy layout.
 */

/**
 * Chỉ khai báo phần API thực sự dùng. KaTeX export cả `default` lẫn named nên
 * kiểu của module không ổn định giữa các bundler — tự định nghĩa gọn hơn.
 */
interface KatexApi {
  renderToString(tex: string, options?: Record<string, unknown>): string
}

let katexPromise: Promise<KatexApi> | null = null

function getKatex(): Promise<KatexApi> {
  if (!katexPromise) {
    katexPromise = import('katex').then((m) => (m.default ?? m) as unknown as KatexApi)
  }
  return katexPromise
}

export async function renderMath(tex: string, displayMode: boolean): Promise<string> {
  const katex = await getKatex()
  return katex.renderToString(tex, {
    displayMode,
    throwOnError: false,
    // Hiện công thức gốc màu đỏ nếu sai cú pháp, thay vì làm sập cả card.
    errorColor: '#e5636c',
    strict: false,
  })
}
