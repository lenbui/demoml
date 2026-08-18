import { MODEL_REGISTRY } from '../lib/modelRegistry'
import type { DemoDefinition } from '../features/types'
import { Badge } from './Badge'
import { CodeBlock } from './CodeBlock'
import { Info } from './Info'
import { Panel } from './Panel'

const VISIBLE_CONCEPTS = 3

/**
 * Vỏ chung của mọi card demo: tiêu đề, các khái niệm ML, thân demo, panel
 * "Xem code", và dòng thông tin model.
 *
 * Phần thân do chính demo render (demo.Component). Demo tự đặt <ModelBar> và
 * <UnderTheHood> bên trong — xem src/features/tokenizer/TokenizerDemo.tsx.
 */
export function DemoCard({ demo }: { demo: DemoDefinition }) {
  const spec = MODEL_REGISTRY[demo.id]
  const { Component } = demo

  const visible = demo.concepts.slice(0, VISIBLE_CONCEPTS)
  const hidden = demo.concepts.slice(VISIBLE_CONCEPTS)

  return (
    <article className="card" id={demo.id}>
      <header className="card-header">
        <h2>{demo.title}</h2>
        <p className="subtitle">{demo.subtitle}</p>

        <div className="concept-row">
          {visible.map((concept) => (
            <Badge key={concept} tone="concept">
              {concept}
            </Badge>
          ))}
          {hidden.length > 0 && (
            <Info title="Khái niệm trong card này" trigger={`+${hidden.length}`}>
              <ul className="notes">
                {demo.concepts.map((concept) => (
                  <li key={concept}>{concept}</li>
                ))}
              </ul>
            </Info>
          )}
        </div>
      </header>

      <div className="card-body">
        <Component />
      </div>

      <footer className="card-footer">
        <Panel title="Xem code">
          <CodeBlock code={demo.snippet} />
        </Panel>

        {spec && (
          <div className="card-meta">
            {spec.variants ? (
              <span>
                {spec.variants.length} model ·{' '}
                {spec.variants.map((v, i) => (
                  <span key={v.id}>
                    {i > 0 && ' · '}
                    <a href={`https://huggingface.co/${v.model}`} target="_blank" rel="noreferrer">
                      {v.label}
                    </a>
                  </span>
                ))}
              </span>
            ) : (
              <span>
                <a href={`https://huggingface.co/${spec.model}`} target="_blank" rel="noreferrer">
                  {spec.model}
                </a>
                {' · '}
                {spec.task}
                {spec.dtype ? ` · ${spec.dtype}` : ''}
                {spec.approxSizeMB ? ` · ${spec.approxSizeMB} MB` : ''}
              </span>
            )}
          </div>
        )}
      </footer>
    </article>
  )
}
