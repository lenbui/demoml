import { asrDemo } from './asr'
import { audioClassificationDemo } from './audio-classification'
import { backgroundRemovalDemo } from './background-removal'
import { clipDemo } from './clip'
import { detectionDemo } from './detection'
import { embeddingsDemo } from './embeddings'
import { fillMaskDemo } from './fill-mask'
import { imageClassificationDemo } from './image-classification'
import { llmDemo } from './llm'
import { nerDemo } from './ner'
import { rerankDemo } from './rerank'
import { sentimentDemo } from './sentiment'
import { tokenizerDemo } from './tokenizer'
import { ttsDemo } from './tts'
import type { DemoDefinition } from './types'
import { zeroShotDemo } from './zero-shot'

/**
 * ĐĂNG KÝ DEMO.
 *
 * Đây là file duy nhất mà mọi nhóm đều phải sửa (thêm 2 dòng: 1 import + 1 phần
 * tử trong mảng), nên khi merge git chỉ xung đột ở đây và rất dễ giải quyết.
 */
export const DEMOS: DemoDefinition[] = [
  tokenizerDemo,
  fillMaskDemo,
  sentimentDemo,
  zeroShotDemo,
  nerDemo,
  embeddingsDemo,
  rerankDemo,
  imageClassificationDemo,
  clipDemo,
  detectionDemo,
  backgroundRemovalDemo,
  asrDemo,
  audioClassificationDemo,
  ttsDemo,
  llmDemo,

  // ── Sinh viên thêm demo ở đây ──────────────────────────────────────────
].sort((a, b) => (a.order ?? 999) - (b.order ?? 999))

export function findDemo(id: string): DemoDefinition | undefined {
  return DEMOS.find((demo) => demo.id === id)
}
