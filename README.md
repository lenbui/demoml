# ML Dashboard — Transformers.js

15 demo Machine Learning chạy **hoàn toàn trong trình duyệt**. Không có backend: model ONNX được
tải từ Hugging Face về máy bạn rồi chạy bằng WebAssembly hoặc WebGPU. Ảnh và âm thanh bạn đưa vào
không rời khỏi máy.

Mỗi demo có panel **Under the hood** phơi ra những bước mà `pipeline()` thường che đi — token,
`input_ids`, shape tensor, output thô trước hậu xử lý — và panel **Xem code** với đoạn code tái
hiện đúng kết quả đang xem.

## Chạy thử

Cần Node ≥ 20 và một browser Chromium/Firefox bản mới.

```bash
npm install
```

```bash
npm run dev
```

Mở http://localhost:5173. Trang chủ là lưới card; bấm một card để mở demo đó.

Bắt đầu bằng **Tokenizer Explorer** — chỉ ~1 MB, chạy ngay lập tức. Các demo khác nặng hơn nên
model chỉ được tải khi bạn bấm nút, không tải sẵn lúc mở trang.

## Danh sách demo

Số MB là dung lượng phải tải về lần đầu; browser cache lại nên lần sau gần như tức thì.

| Demo | Nội dung | Tải về |
|---|---|---|
| **Tokenizer Explorer** | Chữ biến thành số như thế nào — 5 thuật toán | 1–17 MB |
| **Điền chỗ trống (Masked LM)** | Bài tập mà BERT thực sự được học | 111 MB |
| **Phân loại cảm xúc** | Từ logits đến xác suất, từng bước một | 67 MB |
| **Zero-shot Classification** | Nhãn tuỳ ý, không cần fine-tune | 27 MB |
| **Nhận diện thực thể (NER)** | Một nhãn cho mỗi token, rồi gộp lại | 109 MB |
| **Embedding & Semantic Search** | Tìm theo ý nghĩa, so trực tiếp với BM25 | 22–113 MB |
| **Reranker (Cross-encoder)** | Xếp lại kết quả, và cái giá phải trả | 23 MB |
| **Phân loại ảnh (ViT)** | Ảnh cũng bị cắt thành token | 88 MB |
| **CLIP — nhãn ảnh tuỳ ý** | Ảnh và chữ trong cùng một không gian | 154 MB |
| **Phát hiện vật thể** | Bounding box, và cái giá của ngưỡng | 10–43 MB |
| **Tách nền ảnh (matting)** | Output là ảnh, nhưng model chỉ trả về alpha | 25–42 MB |
| **Nhận dạng giọng nói (Whisper)** | Model sinh duy nhất của dashboard | 43 MB |
| **Nhận diện âm thanh (AST)** | Nghe bằng cách nhìn spectrogram | 91 MB |
| **Tổng hợp giọng nói (TTS)** | Chữ thành sóng âm, có tiếng Việt | 38 MB |
| **Sinh văn bản & Sampling** | Model không chọn token, nó chỉ trả logits | 85–128 MB |

Tất cả đều chạy được trên WASM (CPU), **không bắt buộc có GPU**. Máy nào có WebGPU thì nhanh hơn.

## Chạy offline (phòng máy)

30 máy cùng tải một model 67 MB là 2 GB qua đường truyền chung — buổi thực hành sẽ đứng. Chuẩn bị
trước ở một máy:

```bash
npm run fetch-models
```

Model và file mẫu được tải về `public/models/` và `public/samples/`. Tạo file `.env.local`:

```
VITE_LOCAL_MODELS=true
```

Từ giờ app đọc model từ thư mục đó và không gọi ra huggingface.co nữa. Copy cả `public/models/`
sang máy sinh viên, hoặc deploy bản build lên một server trong LAN.

Đây là offline **thật**: mọi request đều cùng origin, kể cả file `.wasm` của onnxruntime — Vite tự
bundle nó vào assets, không cần cấu hình gì thêm.

## Thêm một demo mới

Bốn bước, không phải đụng vào worker:

1. **Khai báo model** trong [`src/lib/modelRegistry.ts`](src/lib/modelRegistry.ts) — task, repo id
   trên Hugging Face, `dtype`, dung lượng.
2. **Copy** `src/features/_template/` thành `src/features/<ten-demo>/`, đổi `'TODO-demo-id'` thành
   id vừa khai báo.
3. **Sửa phần input và hiển thị kết quả.** Chưa biết output có dạng gì? Bấm Chạy một lần rồi đọc
   mục *Output thô từ model* trong panel Under the hood.
4. **Đăng ký** trong [`src/features/index.ts`](src/features/index.ts) — thêm 1 dòng import và 1
   phần tử vào mảng.

Đây là file duy nhất mọi nhóm cùng sửa, nên merge git gần như không xung đột.

## Build & deploy

```bash
npm run build
```

Output tĩnh nằm trong `dist/` (~24 MB, phần lớn là file `.wasm` của onnxruntime).

### GitHub Pages

[`.github/workflows/deploy.yml`](.github/workflows/deploy.yml) tự build và deploy mỗi lần push lên
`main`. Phải bật **một lần bằng tay**:

> Settings › Pages › Build and deployment › Source = **GitHub Actions**

Chọn đúng "GitHub Actions", không phải "Deploy from a branch" — chọn nhầm thì job deploy báo 404.

Workflow tự tính `base` từ tên repo nên không phải sửa `vite.config.ts`.

**Lưu ý:** GitHub Pages không cho set header COOP/COEP, nên bản deploy mất WASM đa luồng và chậm
hơn 2–4 lần so với chạy local. Trang tự hiện tình trạng này ở thanh trên cùng. WebGPU không bị ảnh
hưởng. Muốn giữ đa luồng thì dùng HF Spaces / Netlify / Vercel.

Model **không** nằm trong bản deploy — `public/models/` đã được `.gitignore` bỏ qua, và app ở chế
độ mặc định tải model thẳng từ huggingface.co.

## Lỗi hay gặp

**Trang trắng sau khi deploy lên GitHub Pages** — `base` sai nên mọi file trong `/assets/` trả 404.
Workflow tự tính đúng; nếu build tay thì nhớ `npx vite build --base=/ten-repo/`.

**Windows PowerShell: `&&` báo lỗi cú pháp** — PowerShell 5.1 không có toán tử này. Chạy từng lệnh
riêng, hoặc viết `lệnh A; if ($?) { lệnh B }`.

**Windows Git Bash: base ra `/Program Files/Git/...`** — MSYS dịch đường dẫn. Thêm
`MSYS_NO_PATHCONV=1` vào đầu lệnh.

**Model tải mãi không xong** — file vài chục tới trăm MB, lần đầu chậm là bình thường. Progress bar
hiện tiến độ thật. Tải xong một lần là browser cache lại.

**Kết quả sai một cách khó hiểu** — có thể là tổ hợp `dtype: q8` chạy trên WebGPU, vốn cho kết quả
sai mà không báo lỗi. Dashboard tự phát hiện và hạ về WASM kèm callout vàng giải thích, nên bạn sẽ
thấy cảnh báo chứ không bị lừa im lặng.

## Giới hạn

- **Inference-only.** Không train hay fine-tune được — việc đó vẫn phải làm bằng PyTorch/Python.
- **Model phải có bản ONNX.** Tìm trong [`Xenova`](https://huggingface.co/Xenova) và
  [`onnx-community`](https://huggingface.co/onnx-community), hoặc tự convert bằng Optimum.
- **Không xem được attention weights.** Transformers.js không expose attention map tiện lợi.
- **Kết quả lệch nhẹ so với PyTorch** do lượng tử hoá và khác biệt kernel.

## Cấu trúc thư mục

```
src/
  lib/         modelRegistry.ts (khai báo mọi model), và các hàm ML viết tay
  workers/     pipeline.worker.ts — load model, chạy inference (ít khi phải sửa)
  hooks/       useModel.ts — vòng đời model; useHashRoute.ts — điều hướng
  components/  Thành phần UI dùng chung
  features/    Mỗi demo một thư mục; index.ts là nơi đăng ký
scripts/
  fetch-models.mjs   Tải model về máy để chạy offline
```
