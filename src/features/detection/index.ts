import type { DemoDefinition } from '../types'
import { DetectionDemo } from './DetectionDemo'

const SNIPPET = `import { pipeline } from '@huggingface/transformers'

const detect = await pipeline('object-detection', 'Xenova/detr-resnet-50', {
  dtype: 'q8',
})

// percentage: true -> toạ độ 0..1 thay vì pixel.
// Quan trọng: pixel của model là pixel của ảnh ĐÃ tiền xử lý (DETR ép cạnh
// ngắn về 800), không phải của ảnh gốc bạn đưa vào.
const output = await detect(imageUrl, { threshold: 0.5, percentage: true })
// [ { label: 'person', score: 0.9971,
//     box: { xmin: 0.412, ymin: 0.233, xmax: 0.549, ymax: 0.887 } }, ... ]

// ── Ngưỡng là bước HẬU xử lý, không phải tham số của model ────────────
// Hạ threshold xuống 0.05 rồi tự lọc ở phía UI: kéo slider không cần chạy
// lại model. Đó là cách demo này làm, và cũng là cách thấy rõ nhất rằng
// ngưỡng là một LỰA CHỌN:
//   hạ  -> recall ↑, precision ↓   (bắt được nhiều, kèm nhiều hộp rác)
//   nâng -> precision ↑, recall ↓  (chỉ hộp chắc chắn, bỏ sót nhiều)
const raw = await detect(imageUrl, { threshold: 0.05, percentage: true })
const kept = raw.filter((d) => d.score >= myThreshold)

// ── DETR không cần NMS ────────────────────────────────────────────────
// Các model phát hiện đời trước sinh ra hàng nghìn hộp chồng nhau rồi phải
// dùng non-maximum suppression để dọn. DETR dự đoán một số cố định "query"
// và được huấn luyện bằng Hungarian matching để mỗi query bắt đúng MỘT vật
// thể — nên đầu ra đã sạch sẵn.

// ── Dung lượng file KHÔNG cho biết model chạy nhanh hay chậm ──────────
// Đo thật trên WASM, cùng một ảnh, cả hai đã chạy nóng:
//   'Xenova/yolos-tiny'      10 MB -> 7.7 s, 16 hộp ≥ 0.5
//   'Xenova/detr-resnet-50'  43 MB -> 4.4 s,  8 hộp ≥ 0.5
//
// Model nhỏ hơn 4 lần lại CHẬM hơn gần 2 lần. YOLOS là ViT thuần nên
// attention chạy trên rất nhiều patch token -> FLOPs cao dù ít tham số.
// DETR dùng backbone CNN vốn rất hiệu quả.
//
// Dung lượng đo SỐ THAM SỐ, thời gian chạy phụ thuộc KHỐI LƯỢNG TÍNH TOÁN.
// Hai đại lượng đó không tỉ lệ với nhau — đừng chọn model theo số MB.`

export const detectionDemo: DemoDefinition = {
  id: 'detection',
  title: 'Phát hiện vật thể',
  subtitle:
    'Tìm và khoanh vùng nhiều vật thể trong một ảnh, so hai model chênh nhau 4 lần dung lượng — và thấy rằng số MB không hề cho biết model nào chạy nhanh hơn.',
  tagline: 'Bounding box, và cái giá của ngưỡng',
  concepts: [
    'Object detection',
    'Bounding box',
    'Ngưỡng & precision–recall',
    'Tham số ≠ khối lượng tính toán',
    'NMS và vì sao DETR không cần',
    'Toạ độ tỉ lệ vs pixel',
  ],
  group: 'vision',
  Component: DetectionDemo,
  snippet: SNIPPET,
  order: 35,
}
