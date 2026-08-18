import { useEffect, useId, useLayoutEffect, useRef, useState, type ReactNode } from 'react'
import { createPortal } from 'react-dom'

const PANEL_WIDTH = 360
const MARGIN = 12

/**
 * Nút "?" mở popover giải thích.
 *
 * Đây là cách giữ giao diện gọn: mọi đoạn giải thích lý thuyết dài đều nằm sau
 * một nút "?" thay vì in thẳng ra card. Người đã hiểu thì không bị nhiễu, người
 * cần thì bấm một cái là có — kèm cả công thức KaTeX và code nếu cần.
 *
 *   <Info title="Softmax">Nội dung, có thể chứa <Math tex="…" /></Info>
 *   <Info trigger="logits" title="Logits">…</Info>   // gắn vào một từ trong câu
 *
 * Popover render qua portal vào body vì .card có overflow:hidden — nếu render
 * tại chỗ thì panel bị cắt mất.
 */
export function Info({
  title,
  children,
  trigger,
}: {
  title: string
  children: ReactNode
  /** Nếu có, dùng chuỗi này làm phần tử bấm thay cho hình tròn "?". */
  trigger?: string
}) {
  const [open, setOpen] = useState(false)
  const [style, setStyle] = useState<React.CSSProperties | null>(null)
  const triggerRef = useRef<HTMLButtonElement>(null)
  const panelRef = useRef<HTMLDivElement>(null)
  const panelId = useId()

  useLayoutEffect(() => {
    if (!open || !triggerRef.current) return

    const rect = triggerRef.current.getBoundingClientRect()
    // Trên màn hình hẹp, thu bề rộng lại cho vừa thay vì để tràn ra ngoài.
    const width = Math.min(PANEL_WIDTH, window.innerWidth - MARGIN * 2)
    const left = Math.max(
      MARGIN,
      Math.min(rect.left + rect.width / 2 - width / 2, window.innerWidth - width - MARGIN),
    )

    // Nút nằm ở nửa dưới màn hình thì mở popover lên trên, để không bị tràn đáy.
    const openUpward = rect.bottom > window.innerHeight * 0.6
    setStyle({
      left,
      width,
      ...(openUpward
        ? { bottom: window.innerHeight - rect.top + 8 }
        : { top: rect.bottom + 8 }),
      maxHeight: openUpward ? rect.top - MARGIN - 8 : window.innerHeight - rect.bottom - MARGIN - 8,
    })
  }, [open])

  useEffect(() => {
    if (!open) return

    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setOpen(false)
    }
    const onPointerDown = (e: PointerEvent) => {
      const target = e.target as Node
      if (!panelRef.current?.contains(target) && !triggerRef.current?.contains(target)) {
        setOpen(false)
      }
    }
    // Popover neo theo toạ độ cố định, nên đóng lại khi trang cuộn/đổi kích thước.
    const onReflow = () => setOpen(false)

    document.addEventListener('keydown', onKeyDown)
    document.addEventListener('pointerdown', onPointerDown)
    window.addEventListener('scroll', onReflow, true)
    window.addEventListener('resize', onReflow)
    return () => {
      document.removeEventListener('keydown', onKeyDown)
      document.removeEventListener('pointerdown', onPointerDown)
      window.removeEventListener('scroll', onReflow, true)
      window.removeEventListener('resize', onReflow)
    }
  }, [open])

  return (
    <>
      <button
        ref={triggerRef}
        type="button"
        className={trigger ? 'info-trigger info-trigger--text' : 'info-trigger'}
        aria-label={trigger ? undefined : `Giải thích: ${title}`}
        aria-expanded={open}
        aria-controls={open ? panelId : undefined}
        onClick={() => setOpen((v) => !v)}
      >
        {trigger ?? '?'}
      </button>

      {open &&
        style &&
        createPortal(
          <div
            ref={panelRef}
            id={panelId}
            className="info-panel"
            role="dialog"
            aria-label={title}
            style={style}
          >
            <div className="info-panel-head">
              <strong>{title}</strong>
              <button type="button" onClick={() => setOpen(false)} aria-label="Đóng">
                ✕
              </button>
            </div>
            <div className="info-panel-body">{children}</div>
          </div>,
          document.body,
        )}
    </>
  )
}
