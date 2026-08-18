import type { ModelVariant } from '../lib/modelRegistry'

/**
 * Bộ chọn model cho các demo khai báo `variants` — dùng để SO SÁNH nhiều model
 * cùng task. Dấu chấm xanh = variant đã tải xong, đổi qua lại không phải tải lại.
 */
export function VariantPicker({
  variants,
  value,
  onChange,
  readyVariants,
  loadingVariantId,
  disabled,
}: {
  variants: ModelVariant[]
  value: string
  onChange: (variantId: string) => void
  readyVariants: string[]
  loadingVariantId?: string | null
  disabled?: boolean
}) {
  return (
    <div className="variant-picker" role="radiogroup" aria-label="Chọn model">
      {variants.map((variant) => {
        const isReady = readyVariants.includes(variant.id)
        const isLoading = loadingVariantId === variant.id
        return (
          <button
            key={variant.id}
            type="button"
            role="radio"
            aria-checked={value === variant.id}
            className={`variant${value === variant.id ? ' variant--active' : ''}`}
            onClick={() => onChange(variant.id)}
            disabled={disabled}
            title={variant.model}
          >
            <span className="variant-head">
              <span className="variant-label">{variant.label}</span>
              <span
                className={
                  isLoading ? 'dot dot--loading' : isReady ? 'dot dot--ready' : 'dot'
                }
                aria-hidden="true"
              />
            </span>
            {variant.note && <span className="variant-note">{variant.note}</span>}
            <span className="variant-size">{variant.approxSizeMB} MB</span>
          </button>
        )
      })}
    </div>
  )
}
