import type { DemoDefinition } from '../features/types'
import { GROUP_LABELS, type DemoGroup } from '../features/types'
import { MODEL_REGISTRY, type ModelSpec } from '../lib/modelRegistry'
import { Badge } from './Badge'

const GROUP_ORDER: DemoGroup[] = ['text', 'vector', 'vision', 'audio', 'llm']

/** Dung lượng phải tải: một số, hoặc khoảng nếu demo có nhiều variant. */
function sizeLabel(spec?: ModelSpec): string | null {
  if (!spec) return null

  if (spec.variants?.length) {
    const sizes = spec.variants.map((v) => v.approxSizeMB)
    const min = Math.min(...sizes)
    const max = Math.max(...sizes)
    return min === max ? `${min} MB` : `${min}–${max} MB`
  }

  return spec.approxSizeMB ? `${spec.approxSizeMB} MB` : null
}

/**
 * Trang chủ: lưới card nhỏ, mỗi card là một demo.
 *
 * Vì sao không hiện tất cả demo trên một trang như trước? Mỗi demo là một card
 * cao vài trăm pixel, nên 10 demo thành một trang cuộn rất dài và không ai tìm
 * được gì. Quan trọng hơn: hiện tất cả cùng lúc là mount tất cả, mỗi demo một
 * worker — dễ ngốn hàng GB RAM nếu người dùng bấm tải vài model.
 */
export function DemoGallery({ demos }: { demos: DemoDefinition[] }) {
  const groups = GROUP_ORDER.map((group) => ({
    group,
    demos: demos.filter((d) => d.group === group),
  })).filter((g) => g.demos.length > 0)

  return (
    <>
      {groups.map(({ group, demos: groupDemos }) => (
        <section key={group}>
          <h3 className="section-title">{GROUP_LABELS[group]}</h3>
          <div className="gallery">
            {groupDemos.map((demo) => {
              const size = sizeLabel(MODEL_REGISTRY[demo.id])
              return (
                <a key={demo.id} className="gallery-card" href={`#${demo.id}`}>
                  <span className="gallery-title">{demo.title}</span>
                  <span className="gallery-tagline">{demo.tagline ?? demo.subtitle}</span>
                  <span className="gallery-foot">
                    <span className="gallery-concepts">
                      {demo.concepts.slice(0, 2).map((concept) => (
                        <Badge key={concept} tone="concept">
                          {concept}
                        </Badge>
                      ))}
                    </span>
                    {size && <span className="gallery-size">{size}</span>}
                  </span>
                </a>
              )
            })}
          </div>
        </section>
      ))}
    </>
  )
}
