import type { ReactNode } from 'react'

/** Khối gập/mở dùng cho phần "Under the hood" và "Xem code" ở chân mỗi card. */
export function Panel({
  title,
  children,
  defaultOpen = false,
}: {
  title: ReactNode
  children: ReactNode
  defaultOpen?: boolean
}) {
  return (
    <details className="panel" open={defaultOpen}>
      <summary>{title}</summary>
      <div className="panel-content">{children}</div>
    </details>
  )
}
