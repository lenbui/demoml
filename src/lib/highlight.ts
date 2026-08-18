import type { HighlighterCore } from 'shiki/core'

/**
 * Syntax highlighting cho các snippet code.
 *
 * Ba lựa chọn thiết kế:
 *  1. Lazy import — Shiki + grammar nặng vài trăm KB, chỉ tải khi có card mở
 *     panel "Xem code", không tính vào lần tải trang đầu.
 *  2. JavaScript regex engine thay vì Oniguruma — không cần nạp thêm WASM.
 *  3. Dual theme với `defaultColor: false` — Shiki sinh ra CSS variable
 *     --shiki-light / --shiki-dark cho từng token, styles.css chọn theo
 *     prefers-color-scheme. Nhờ vậy code đổi màu theo theme mà không cần
 *     highlight lại.
 */

export const CODE_THEMES = { light: 'github-light-default', dark: 'github-dark-default' } as const

let highlighterPromise: Promise<HighlighterCore> | null = null

function getHighlighter(): Promise<HighlighterCore> {
  if (!highlighterPromise) {
    highlighterPromise = (async () => {
      const [{ createHighlighterCore }, { createJavaScriptRegexEngine }] = await Promise.all([
        import('shiki/core'),
        import('shiki/engine/javascript'),
      ])

      return createHighlighterCore({
        themes: [
          import('shiki/themes/github-light-default.mjs'),
          import('shiki/themes/github-dark-default.mjs'),
        ],
        langs: [import('shiki/langs/javascript.mjs')],
        engine: createJavaScriptRegexEngine(),
      })
    })()
  }
  return highlighterPromise
}

export async function highlightCode(code: string): Promise<string> {
  const highlighter = await getHighlighter()
  return highlighter.codeToHtml(code, {
    lang: 'javascript',
    themes: CODE_THEMES,
    defaultColor: false,
  })
}
