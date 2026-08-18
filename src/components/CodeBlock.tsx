import { useEffect, useState } from 'react'

import { highlightCode } from '../lib/highlight'

/**
 * Khối code có syntax highlighting (Shiki) và nút copy.
 *
 * Mỗi demo đính kèm đúng đoạn code sinh ra kết quả đang hiện trên màn hình, để
 * sinh viên copy về Colab/CodeSandbox chạy lại.
 */
export function CodeBlock({ code }: { code: string }) {
  const [html, setHtml] = useState<string | null>(null)
  const [copied, setCopied] = useState(false)

  useEffect(() => {
    let active = true
    highlightCode(code).then((result) => {
      if (active) setHtml(result)
    })
    return () => {
      active = false
    }
  }, [code])

  async function copy() {
    try {
      await navigator.clipboard.writeText(code)
      setCopied(true)
      window.setTimeout(() => setCopied(false), 1500)
    } catch {
      setCopied(false)
    }
  }

  return (
    <div className="code-wrap">
      {html ? (
        // Shiki trả về HTML đã escape, và `code` luôn là hằng số trong source.
        <div className="code-shiki" dangerouslySetInnerHTML={{ __html: html }} />
      ) : (
        <pre className="code">{code}</pre>
      )}
      <button type="button" className="code-copy" onClick={copy}>
        {copied ? 'Đã copy' : 'Copy'}
      </button>
    </div>
  )
}
