export function ProgressBar({
  value,
  left,
  right,
}: {
  /** 0–1 */
  value: number
  left?: string
  right?: string
}) {
  const percent = Math.round(Math.min(Math.max(value, 0), 1) * 100)
  return (
    <div className="progress">
      <div
        className="progress-track"
        role="progressbar"
        aria-valuenow={percent}
        aria-valuemin={0}
        aria-valuemax={100}
      >
        <div className="progress-fill" style={{ width: `${percent}%` }} />
      </div>
      {(left || right) && (
        <div className="progress-meta">
          <span>{left ?? ''}</span>
          <span>{right ?? `${percent}%`}</span>
        </div>
      )}
    </div>
  )
}
