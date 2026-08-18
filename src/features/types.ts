import type { ComponentType } from 'react'

/**
 * HỢP ĐỒNG GIỮA SCAFFOLD VÀ MỖI DEMO.
 *
 * Sinh viên tạo folder `src/features/<ten-demo>/`, export một object thoả
 * interface này, rồi đăng ký nó trong `src/features/index.ts`.
 * Nhờ vậy các nhóm làm song song mà không sửa chung file nào ngoài 2 dòng
 * đăng ký -> merge git gần như không xung đột.
 */
export interface DemoDefinition {
  /** Trùng với key trong MODEL_REGISTRY. */
  id: string
  title: string
  /** Một câu mô tả demo làm gì, hiện dưới tiêu đề khi mở demo. */
  subtitle: string
  /**
   * Bản rất ngắn (dưới ~70 ký tự) hiện trên card nhỏ ở gallery.
   * Không có thì gallery dùng `subtitle`, nhưng card sẽ cao lộn xộn.
   */
  tagline?: string
  /**
   * Các khái niệm Machine Learning mà demo này minh hoạ.
   * Đây là tiêu chí chấm điểm: demo phải dạy được điều gì đó, không chỉ "chạy được".
   */
  concepts: string[]
  group: DemoGroup
  /** Component render phần thân của card. */
  Component: ComponentType
  /**
   * Đoạn code (~10–20 dòng) tái hiện đúng kết quả người dùng đang xem, hiện ở
   * mục "Xem code". Viết bằng API công khai của Transformers.js.
   */
  snippet: string
  /** Thứ tự hiện trên dashboard (nhỏ hơn hiện trước). */
  order?: number
}

export type DemoGroup = 'text' | 'vector' | 'vision' | 'audio' | 'llm'

export const GROUP_LABELS: Record<DemoGroup, string> = {
  // Tránh dùng đúng chữ "Văn bản": nó trùng với nhãn ô input bên trong card.
  text: 'Xử lý ngôn ngữ',
  vector: 'Biểu diễn vector',
  vision: 'Thị giác máy tính',
  audio: 'Âm thanh',
  llm: 'Sinh văn bản (LLM)',
}
