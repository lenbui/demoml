import { useEffect, useState } from 'react'

import { renderMath } from '../lib/math-render'

/**
 * Công thức toán render bằng KaTeX.
 *
 *   <Math tex="p_i = \frac{e^{z_i}}{\sum_j e^{z_j}}" />        (inline)
 *   <Math tex="…" block />                                     (display)
 *
 * Trong lúc KaTeX đang tải, hiện chuỗi TeX gốc dạng monospace — vẫn đọc được,
 * và không để lại khoảng trống nhảy layout.
 */
export function Math({ tex, block = false }: { tex: string; block?: boolean }) {
  const [html, setHtml] = useState<string | null>(null)

  useEffect(() => {
    let active = true
    renderMath(tex, block).then((result) => {
      if (active) setHtml(result)
    })
    return () => {
      active = false
    }
  }, [tex, block])

  if (!html) {
    return <code className={block ? 'math-fallback math-fallback--block' : 'math-fallback'}>{tex}</code>
  }

  // dangerouslySetInnerHTML là cách dùng đúng của KaTeX: nó sinh ra MathML +
  // HTML đã escape sẵn, và `tex` ở đây luôn là hằng số trong code, không phải
  // input của người dùng.
  return (
    <span
      className={block ? 'math math--block' : 'math'}
      dangerouslySetInnerHTML={{ __html: html }}
    />
  )
}
