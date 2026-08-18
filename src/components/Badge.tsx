import type { ReactNode } from 'react'

export type BadgeTone = 'default' | 'concept' | 'ok' | 'warn' | 'danger'

const TONE_CLASS: Record<BadgeTone, string> = {
  default: '',
  concept: 'badge--concept',
  ok: 'badge--ok',
  warn: 'badge--warn',
  danger: 'badge--danger',
}

export function Badge({
  children,
  tone = 'default',
  mono = false,
  title,
}: {
  children: ReactNode
  tone?: BadgeTone
  mono?: boolean
  title?: string
}) {
  const classes = ['badge', TONE_CLASS[tone], mono ? 'badge--mono' : ''].filter(Boolean).join(' ')
  return (
    <span className={classes} title={title}>
      {children}
    </span>
  )
}
