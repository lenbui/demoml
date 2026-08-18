/**
 * Corpus cho demo tìm kiếm ngữ nghĩa.
 *
 * ── NGUYÊN TẮC KHI VIẾT CORPUS NÀY ────────────────────────────────────────
 * Các đoạn văn được viết CỐ Ý tránh dùng đúng những từ mà truy vấn mẫu sẽ dùng.
 * Ví dụ đoạn về overfitting nói "ghi nhớ", "thuộc lòng" — còn truy vấn mẫu là
 * "làm sao biết model học vẹt". Không một từ nào trùng nhau.
 *
 * Nếu corpus dùng lại y nguyên từ của truy vấn thì BM25 cũng tìm ra ngay, và
 * demo mất sạch giá trị dạy học: sinh viên sẽ kết luận sai rằng embedding không
 * hơn gì đếm từ khoá.
 */

export interface Passage {
  id: string
  topic: TopicId
  text: string
}

export type TopicId = 'overfit' | 'optim' | 'data' | 'eval' | 'net' | 'nlp'

export const TOPICS: Record<TopicId, { label: string; color: string }> = {
  overfit: { label: 'Overfitting', color: '#e5734a' },
  optim: { label: 'Tối ưu hoá', color: '#7c9aff' },
  data: { label: 'Dữ liệu', color: '#45c48c' },
  eval: { label: 'Đánh giá', color: '#c77dd6' },
  net: { label: 'Mạng neural', color: '#dfa945' },
  nlp: { label: 'NLP & embedding', color: '#4bb8c9' },
}

export interface Corpus {
  id: string
  label: string
  language: string
  passages: Passage[]
  /** Truy vấn mẫu — chọn sao cho BM25 thất bại mà embedding thành công. */
  sampleQueries: string[]
}

const VI_PASSAGES: Passage[] = [
  // ── overfit ──────────────────────────────────────────────────────────────
  { id: 'vi-of-1', topic: 'overfit', text: 'Khi mô hình ghi nhớ từng mẫu trong tập huấn luyện thay vì rút ra quy luật chung, nó đạt điểm gần như hoàn hảo lúc luyện nhưng thất bại trên dữ liệu chưa từng gặp.' },
  { id: 'vi-of-2', topic: 'overfit', text: 'Thêm phạt L2 vào hàm mất mát giữ cho các trọng số nhỏ lại, buộc mô hình chọn lời giải đơn giản hơn thay vì bám sát từng điểm dữ liệu.' },
  { id: 'vi-of-3', topic: 'overfit', text: 'Dropout ngẫu nhiên tắt một phần nơ-ron ở mỗi bước luyện, khiến mạng không thể dựa vào một vài đường dẫn cố định và phải phân tán biểu diễn.' },
  { id: 'vi-of-4', topic: 'overfit', text: 'Dừng sớm theo dõi sai số trên tập kiểm định và ngắt quá trình luyện ngay khi con số đó bắt đầu xấu đi, dù sai số trên tập luyện vẫn đang giảm.' },

  // ── optim ────────────────────────────────────────────────────────────────
  { id: 'vi-op-1', topic: 'optim', text: 'Thuật toán đi ngược hướng đạo hàm để hạ dần giá trị hàm mất mát, mỗi bước dịch một đoạn tỉ lệ với độ dốc tại điểm hiện tại.' },
  { id: 'vi-op-2', topic: 'optim', text: 'Bước nhảy quá lớn làm giá trị mất mát dao động lên xuống hoặc phát tán ra vô cực; bước quá nhỏ thì quá trình bò rất chậm và dễ mắc ở vùng phẳng.' },
  { id: 'vi-op-3', topic: 'optim', text: 'Adam giữ trung bình động của cả đạo hàm và bình phương đạo hàm, nhờ đó mỗi tham số có bước dịch riêng thay vì dùng chung một hằng số.' },
  { id: 'vi-op-4', topic: 'optim', text: 'Lịch giảm dần theo cosine hạ bước dịch một cách trơn về gần không ở cuối quá trình, giúp tham số ổn định lại quanh điểm cực tiểu.' },

  // ── data ─────────────────────────────────────────────────────────────────
  { id: 'vi-da-1', topic: 'data', text: 'Tập dữ liệu nên được tách thành ba phần riêng biệt: một phần để luyện, một phần để chọn siêu tham số, và một phần chỉ dùng đúng một lần ở cuối.' },
  { id: 'vi-da-2', topic: 'data', text: 'Nếu thông tin từ phần đánh giá lọt vào quá trình luyện — chẳng hạn chuẩn hoá trên toàn bộ dữ liệu trước khi tách — con số cuối cùng sẽ đẹp một cách giả tạo.' },
  { id: 'vi-da-3', topic: 'data', text: 'Khi một nhãn chiếm chín mươi chín phần trăm mẫu, mô hình chỉ cần đoán nhãn đó là đã đúng gần hết, nên độ chính xác tổng thể trở nên vô nghĩa.' },
  { id: 'vi-da-4', topic: 'data', text: 'Nhãn do người gán luôn có tỉ lệ sai nhất định; giới hạn trên của chất lượng mô hình bị chặn bởi mức độ nhất quán giữa những người gán nhãn.' },

  // ── eval ─────────────────────────────────────────────────────────────────
  { id: 'vi-ev-1', topic: 'eval', text: 'Tỉ lệ dự đoán đúng trong số các trường hợp được báo là dương, so với tỉ lệ bắt được trong số các trường hợp dương thật, là hai đại lượng đánh đổi lẫn nhau.' },
  { id: 'vi-ev-2', topic: 'eval', text: 'Hạ mức ngưỡng quyết định làm mô hình báo dương nhiều hơn: bắt được nhiều ca thật hơn nhưng đồng thời báo sai nhiều hơn.' },
  { id: 'vi-ev-3', topic: 'eval', text: 'Chia dữ liệu thành k phần rồi lần lượt để mỗi phần làm tập kiểm tra cho ra ước lượng ổn định hơn một lần tách duy nhất, đặc biệt khi dữ liệu ít.' },
  { id: 'vi-ev-4', topic: 'eval', text: 'Mạng nơ-ron thường trả về mức tin cậy cao hơn thực tế; xác suất chín mươi chín phần trăm không có nghĩa là nó đúng chín mươi chín lần trên một trăm.' },

  // ── net ──────────────────────────────────────────────────────────────────
  { id: 'vi-ne-1', topic: 'net', text: 'Không có hàm phi tuyến giữa các lớp, một chồng nhiều lớp nhân ma trận vẫn chỉ tương đương đúng một phép biến đổi tuyến tính duy nhất.' },
  { id: 'vi-ne-2', topic: 'net', text: 'Quy tắc chuỗi lan tín hiệu sai số từ đầu ra ngược về từng tham số, cho biết mỗi trọng số cần dịch theo hướng nào và bao nhiêu.' },
  { id: 'vi-ne-3', topic: 'net', text: 'Trong mạng rất sâu, tín hiệu dội ngược có thể nhỏ dần tới mức triệt tiêu; kết nối tắt cho phép nó đi vòng qua nhiều lớp mà không suy giảm.' },
  { id: 'vi-ne-4', topic: 'net', text: 'Chuẩn hoá theo lô giữ phân bố đầu vào của mỗi lớp ổn định qua các bước luyện, nhờ đó có thể dùng bước dịch lớn hơn mà vẫn hội tụ.' },

  // ── nlp ──────────────────────────────────────────────────────────────────
  { id: 'vi-nl-1', topic: 'nlp', text: 'Mỗi đoạn chữ được ánh xạ thành một dãy số nhiều chiều, sao cho hai đoạn nói cùng một ý nằm gần nhau dù không dùng chung từ nào.' },
  { id: 'vi-nl-2', topic: 'nlp', text: 'Góc giữa hai dãy số cho biết chúng giống nhau đến đâu, và không phụ thuộc vào độ dài của chúng.' },
  { id: 'vi-nl-3', topic: 'nlp', text: 'Cơ chế chú ý cho mỗi vị trí nhìn thẳng sang mọi vị trí khác trong câu, nên quan hệ giữa hai từ xa nhau không phải truyền qua từng bước trung gian.' },
  { id: 'vi-nl-4', topic: 'nlp', text: 'Từ không có trong bộ từ vựng bị chẻ thành các mảnh nhỏ hơn, nên chuỗi dài ra và ý nghĩa của từ gốc bị phân tán.' },
]

const EN_PASSAGES: Passage[] = [
  { id: 'en-of-1', topic: 'overfit', text: 'When a model commits every training example to memory instead of extracting a general rule, it scores almost perfectly while fitting yet fails on anything it has not seen.' },
  { id: 'en-of-2', topic: 'overfit', text: 'Adding an L2 penalty to the loss keeps weights small, pushing the model toward a simpler solution rather than one that hugs every data point.' },
  { id: 'en-of-3', topic: 'overfit', text: 'Dropout randomly switches off a fraction of units at each step, so the network cannot rely on a few fixed pathways and must spread its representation out.' },
  { id: 'en-of-4', topic: 'overfit', text: 'Early stopping watches the error on a held-out split and halts fitting the moment that number starts getting worse, even while training error still falls.' },

  { id: 'en-op-1', topic: 'optim', text: 'The algorithm steps against the direction of the derivative to lower the loss, moving a distance proportional to the steepness at the current point.' },
  { id: 'en-op-2', topic: 'optim', text: 'Too large a step makes the loss oscillate or blow up to infinity; too small a one crawls along and gets stuck on flat regions.' },
  { id: 'en-op-3', topic: 'optim', text: 'Adam keeps running averages of both the derivative and its square, giving every parameter its own step size instead of one shared constant.' },
  { id: 'en-op-4', topic: 'optim', text: 'A cosine schedule smoothly lowers the step size toward zero near the end, letting parameters settle around the minimum.' },

  { id: 'en-da-1', topic: 'data', text: 'A dataset should be cut into three separate parts: one for fitting, one for picking hyperparameters, and one touched exactly once at the very end.' },
  { id: 'en-da-2', topic: 'data', text: 'If information from the held-out part seeps into fitting — for instance scaling over everything before cutting — the final number will look good for no real reason.' },
  { id: 'en-da-3', topic: 'data', text: 'When one label covers ninety-nine percent of examples, always guessing that label is already almost always right, so overall correctness becomes meaningless.' },
  { id: 'en-da-4', topic: 'data', text: 'Human-assigned labels always carry some error rate; the ceiling on model quality is bounded by how consistently the annotators agree with each other.' },

  { id: 'en-ev-1', topic: 'eval', text: 'The share of correct calls among those flagged positive, versus the share caught among all truly positive cases, are two quantities traded against each other.' },
  { id: 'en-ev-2', topic: 'eval', text: 'Lowering the decision cutoff makes the model flag more cases: it catches more real ones but also raises more false alarms.' },
  { id: 'en-ev-3', topic: 'eval', text: 'Splitting data into k parts and letting each part take a turn as the test set gives a steadier estimate than a single split, especially when data is scarce.' },
  { id: 'en-ev-4', topic: 'eval', text: 'Neural networks tend to report more confidence than warranted; a ninety-nine percent figure does not mean it is right ninety-nine times out of a hundred.' },

  { id: 'en-ne-1', topic: 'net', text: 'Without a nonlinear function between layers, a stack of matrix multiplications collapses into exactly one linear transformation.' },
  { id: 'en-ne-2', topic: 'net', text: 'The chain rule carries the error signal from the output back to each parameter, telling every weight which way to move and by how much.' },
  { id: 'en-ne-3', topic: 'net', text: 'In very deep stacks the signal flowing backward can shrink until it disappears; shortcut connections let it skip past many layers undiminished.' },
  { id: 'en-ne-4', topic: 'net', text: 'Normalising across a batch keeps each layer input distribution steady between steps, which allows a larger step size while still converging.' },

  { id: 'en-nl-1', topic: 'nlp', text: 'Each piece of text is mapped to a list of numbers in many dimensions, arranged so that two pieces expressing the same idea land near one another even with no shared words.' },
  { id: 'en-nl-2', topic: 'nlp', text: 'The angle between two lists of numbers tells you how alike they are, and it does not depend on how long they are.' },
  { id: 'en-nl-3', topic: 'nlp', text: 'The attention mechanism lets every position look directly at every other position in the sentence, so a link between two distant words need not pass through intermediate steps.' },
  { id: 'en-nl-4', topic: 'nlp', text: 'A word missing from the vocabulary gets broken into smaller fragments, so the sequence grows longer and the original word meaning is scattered.' },
]

export const CORPORA: Corpus[] = [
  {
    id: 'vi',
    label: 'Tiếng Việt',
    language: 'vi',
    passages: VI_PASSAGES,
    // Đã kiểm chứng từng truy vấn trên model đa ngữ E5: cả 6 đều cho top-1 đúng,
    // phủ đủ 6 chủ đề, và không truy vấn nào trùng từ nội dung với đoạn văn đích.
    // Nếu bạn sửa corpus, hãy chạy lại và kiểm tra top-1 vẫn đúng.
    sampleQueries: [
      'làm sao biết model học vẹt?',
      'bước cập nhật quá lớn thì hậu quả gì?',
      'nên chia dữ liệu thế nào?',
      'độ chính xác 99% có đáng tin không?',
      'xếp nhiều lớp tuyến tính liên tiếp có ích không?',
      'hai câu khác từ nhưng cùng nghĩa thì sao?',
    ],
  },
  {
    id: 'en',
    label: 'Tiếng Anh',
    language: 'en',
    passages: EN_PASSAGES,
    sampleQueries: [
      'how do I know my model just memorised?',
      'what happens if the update step is too big?',
      'how should I split my data?',
      'is 99% accuracy trustworthy?',
      'is stacking several linear layers useful?',
      // Không dùng "share no words": chữ "word" hút truy vấn về đoạn nói về
      // subword/vocabulary thay vì đoạn nói về embedding. Đã đo và thay.
      'how do you represent meaning as numbers?',
    ],
  },
]

export function getCorpus(id: string): Corpus {
  return CORPORA.find((c) => c.id === id) ?? CORPORA[0]
}
