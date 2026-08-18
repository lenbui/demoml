import type { DemoDefinition } from '../types'
import { TemplateDemo } from './TemplateDemo'

const SNIPPET = `import { pipeline } from '@huggingface/transformers'

// TODO: thay task và model cho đúng demo của bạn
const pipe = await pipeline('text-classification', 'Xenova/...', {
  dtype: 'q8',
})

const output = await pipe('dữ liệu thử')
console.log(output)`

/**
 * TODO: đổi toàn bộ các trường bên dưới.
 * Demo này KHÔNG được đăng ký trong src/features/index.ts nên không hiện trên
 * dashboard — nó chỉ là khung để copy.
 */
export const templateDemo: DemoDefinition = {
  id: 'TODO-demo-id',
  title: 'TODO: tên demo',
  subtitle: 'TODO: một câu mô tả demo làm gì.',
  // Bắt buộc: demo phải minh hoạ được khái niệm ML cụ thể, không chỉ "chạy được".
  concepts: ['TODO: khái niệm 1', 'TODO: khái niệm 2'],
  group: 'text',
  Component: TemplateDemo,
  snippet: SNIPPET,
  order: 999,
}
