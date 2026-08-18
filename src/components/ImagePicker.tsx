import { useEffect, useRef, useState, type ReactNode } from 'react'

import { prepareImageFile, readImageSize, type PreparedImage } from '../lib/image'
import { sampleImageUrl, type SampleChoice } from '../lib/samples'
import { USE_LOCAL_MODELS } from '../lib/config'
import { Badge } from './Badge'

/**
 * Chọn ảnh cho các demo thị giác: bấm một ảnh mẫu, hoặc tải ảnh của mình lên.
 *
 * Ảnh của người dùng KHÔNG rời khỏi máy — nó được đọc thành data URL rồi đưa
 * thẳng vào worker. Đây là điểm bán hàng lớn nhất của cả dashboard, nên nói rõ
 * ngay dưới ô chọn ảnh.
 *
 * `children` để mỗi demo chèn lớp phủ riêng lên trên ảnh (demo phát hiện vật thể
 * vẽ bounding box ở đây).
 */
export function ImagePicker({
  samples,
  value,
  onChange,
  disabled,
  children,
}: {
  samples: SampleChoice[]
  value: PreparedImage | null
  onChange: (image: PreparedImage | null) => void
  disabled?: boolean
  children?: ReactNode
}) {
  const inputRef = useRef<HTMLInputElement>(null)
  const [error, setError] = useState<string | null>(null)
  const [loadingSample, setLoadingSample] = useState<string | null>(null)

  // Chọn sẵn ảnh mẫu đầu tiên để card không mở ra với một ô trống.
  const initialised = useRef(false)
  useEffect(() => {
    if (initialised.current || value) return
    initialised.current = true
    void pickSample(samples[0])
    // Chỉ chạy một lần lúc mount.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  async function pickSample(choice: SampleChoice) {
    const src = sampleImageUrl(choice.name)
    setError(null)
    setLoadingSample(choice.name)
    try {
      // Ảnh mẫu được đưa thẳng vào model bằng URL, không qua canvas — kích thước
      // đọc riêng để hiện được "ảnh gốc bao nhiêu pixel".
      const { width, height } = await readImageSize(src)
      onChange({
        src,
        label: choice.label,
        naturalWidth: width,
        naturalHeight: height,
        width,
        height,
        bytes: 0,
      })
    } catch {
      setError(
        USE_LOCAL_MODELS
          ? `Không tìm thấy public/samples/${choice.name}. Chạy \`npm run fetch-models\` để tải ảnh mẫu về.`
          : 'Không tải được ảnh mẫu — cần Internet, hoặc dùng nút tải ảnh lên.',
      )
      onChange(null)
    } finally {
      setLoadingSample(null)
    }
  }

  async function handleFile(file: File | undefined) {
    if (!file) return
    setError(null)
    try {
      onChange(await prepareImageFile(file))
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Không đọc được ảnh.')
      onChange(null)
    }
  }

  const resized = value != null && value.width !== value.naturalWidth

  return (
    <div className="field">
      <div className="field-head">
        <span className="field-label">Ảnh</span>
        {value && (
          <Badge mono>
            {value.naturalWidth}×{value.naturalHeight}
          </Badge>
        )}
        {resized && (
          <Badge mono tone="warn">
            đã thu nhỏ về {value.width}×{value.height}
          </Badge>
        )}
        <div className="chip-row">
          {samples.map((choice) => (
            <button
              key={choice.name}
              type="button"
              className={`chip${value?.label === choice.label ? ' chip--active' : ''}`}
              onClick={() => void pickSample(choice)}
              disabled={disabled || loadingSample !== null}
              title={choice.hint}
            >
              {choice.label}
            </button>
          ))}
          <button
            type="button"
            className="chip"
            onClick={() => inputRef.current?.click()}
            disabled={disabled}
          >
            + Ảnh của bạn
          </button>
        </div>
      </div>

      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        hidden
        onChange={(e) => {
          void handleFile(e.target.files?.[0])
          // Reset để chọn lại đúng file vừa chọn vẫn kích hoạt onChange.
          e.target.value = ''
        }}
      />

      {error && <div className="callout callout--warn">{error}</div>}

      {value && (
        <div className="image-stage">
          <img src={value.src} alt={value.label} className="image-preview" />
          {children}
        </div>
      )}

      <p className="hint">
        Ảnh bạn tải lên <strong>không rời khỏi máy</strong>: nó được đọc thành data URL rồi đưa
        thẳng vào model đang chạy trong trình duyệt. Không có request nào mang ảnh đi đâu cả.
      </p>
    </div>
  )
}
