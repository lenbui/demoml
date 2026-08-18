import type { DemoDefinition } from '../types'
import { BackgroundRemovalDemo } from './BackgroundRemovalDemo'

const SNIPPET = `import { pipeline } from '@huggingface/transformers'

// CHÚ Ý task: 'image-segmentation', KHÔNG phải 'background-removal'.
// Transformers.js CÓ task 'background-removal' và nó ngắn hơn:
//
//     const remove = await pipeline('background-removal', 'Xenova/modnet')
//     const [image] = await remove(imageUrl)   // -> RawImage RGBA đã trong suốt
//
// Nhưng nó nhân mặt nạ vào ảnh ở bên trong thư viện rồi chỉ trả về kết quả
// cuối. Demo này cần chính cái mặt nạ đó, nên gọi task phân đoạn để lấy thô.

const segment = await pipeline(
  'image-segmentation',
  'Xenova/modnet',
  { dtype: 'fp32' },   // fp32 chứ không q8: sai số lượng tử hoá hiện ra
)                      // thành rìa răng cưa quanh sợi tóc.

const [{ label, score, mask }] = await segment(imageUrl)
// label === null, score === null  <- model không sinh ra nhãn nào
// mask: RawImage { data: Uint8ClampedArray, width, height, channels: 1 }

// ── Model KHÔNG "xoá nền" ─────────────────────────────────────────────
// Nó sinh ra một MA TRẬN ALPHA: mỗi pixel một số trong [0, 1] trả lời
// "pixel này thuộc chủ thể bao nhiêu phần". Alpha = sigmoid(z) tính RIÊNG
// cho từng pixel — không phải softmax trên các nhãn, nên không có ràng
// buộc "cộng lại bằng 1" như ở card Phân loại ảnh.

// ── Bước "xoá nền" thật sự chỉ là gán kênh alpha ──────────────────────
const canvas = document.createElement('canvas')
canvas.width = mask.width
canvas.height = mask.height
const ctx = canvas.getContext('2d')
ctx.drawImage(await createImageBitmap(await (await fetch(imageUrl)).blob()), 0, 0)

const frame = ctx.getImageData(0, 0, mask.width, mask.height)
for (let i = 0; i < mask.width * mask.height; i++) {
  frame.data[i * 4 + 3] = mask.data[i]   // RGB giữ nguyên, chỉ thay A
}
ctx.putImageData(frame, 0, 0)
canvas.toDataURL('image/png')            // PNG mới giữ được alpha; JPEG thì không

// Đây là toàn bộ phép "xoá nền". Không có mạng neural nào chạy ở bước này —
// nên đổi màu nền hay nhị phân hoá mặt nạ đều không cần chạy lại model.
// Code đầy đủ, có chú thích: src/lib/matte.ts`

export const backgroundRemovalDemo: DemoDefinition = {
  id: 'background-removal',
  title: 'Tách nền ảnh (matting)',
  subtitle:
    'Model không sinh ra tấm ảnh nào — nó sinh ra một ma trận alpha, mỗi pixel một số trong [0,1]. Ảnh nền trong suốt là do bạn nhân ma trận đó vào ảnh gốc.',
  tagline: 'Output là ảnh, nhưng model chỉ trả về alpha',
  concepts: [
    'Image matting',
    'Dense prediction (mỗi pixel một dự đoán)',
    'Sigmoid từng pixel vs softmax trên nhãn',
    'Kênh alpha & bán trong suốt',
    'Hậu xử lý nằm ngoài model',
    'Model chuyên biệt vs tổng quát',
  ],
  group: 'vision',
  Component: BackgroundRemovalDemo,
  snippet: SNIPPET,
  // Ngay sau Phát hiện vật thể (order 35): đây là bước tiếp theo của cùng một
  // mạch — hộp bao quanh vật thể là câu trả lời thô nhất về "vật ở đâu", còn
  // mặt nạ alpha là câu trả lời mịn nhất, tới từng pixel.
  order: 37,
}
