import { useMemo } from 'react'

import { waveformPeaks } from '../lib/audio'

const HEIGHT = 72

/**
 * Waveform vẽ thẳng từ chính Float32Array được đưa vào model.
 *
 * Mục đích dạy học, không phải trang trí: nó cho thấy "âm thanh" mà model nhận
 * chỉ là một dãy số trong [-1, 1], 16.000 số mỗi giây. Cùng ý với hàng token ở
 * card Tokenizer — phơi ra đúng thứ model thực sự đọc.
 */
export function Waveform({ samples }: { samples: Float32Array }) {
  const peaks = useMemo(() => waveformPeaks(samples), [samples])

  if (peaks.length === 0) return null

  const step = 100 / peaks.length

  return (
    <div className="waveform">
      <svg
        viewBox={`0 0 100 ${HEIGHT}`}
        preserveAspectRatio="none"
        role="img"
        aria-label="Dạng sóng của đoạn âm thanh"
      >
        {/* Đường 0 — biên độ âm nằm dưới, dương nằm trên. */}
        <line x1={0} y1={HEIGHT / 2} x2={100} y2={HEIGHT / 2} className="waveform-axis" />
        {peaks.map(([min, max], i) => {
          const top = (HEIGHT / 2) * (1 - max)
          const bottom = (HEIGHT / 2) * (1 - min)
          return (
            <rect
              key={i}
              x={i * step}
              y={top}
              width={step * 0.8}
              // Biên độ rất nhỏ vẫn phải thấy được một vạch, nếu không đoạn im
              // lặng trông như dữ liệu bị mất.
              height={Math.max(bottom - top, 0.6)}
              className="waveform-bar"
            />
          )
        })}
      </svg>
    </div>
  )
}
