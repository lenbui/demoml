import type { DemoDefinition } from '../types'
import { TokenizerDemo } from './TokenizerDemo'

const SNIPPET = `import { AutoTokenizer } from '@huggingface/transformers'

// Chỉ tải tokenizer — KHÔNG tải trọng số model, nên chỉ ~1 MB
const tokenizer = await AutoTokenizer.from_pretrained('Xenova/bert-base-uncased')

const text = 'Khóa học máy học rất thú vị.'

// 1. Text -> Tensor int64 shape [1, seq_len]
const encoded = tokenizer(text)
const ids = Array.from(encoded.input_ids.data, Number)
// [ 101, 1047, 6806, 2050, 21929, ... , 102 ]

// 2. Xem từng id ứng với token nào
const tokens = tokenizer.model.convert_ids_to_tokens(ids)
// [ '[CLS]', 'k', '##ho', '##a', 'hoc', ... , '[SEP]' ]

// 3. Decode ngược để biết tokenizer làm mất gì
tokenizer.decode(ids, { skip_special_tokens: true })
// 'khoa hoc may hoc rat thu vi.'   <- mất hết dấu và chữ hoa

// Thuật toán tokenize nằm ở đây
tokenizer.model.constructor.name   // 'WordPieceTokenizer'

// ── Đổi sang tokenizer khác để so sánh ────────────────────────────────
// 'Xenova/bert-base-multilingual-cased'  WordPiece đa ngữ, giữ dấu
// 'Xenova/gpt2'                          BPE mức byte, không mất gì
// 'Xenova/xlm-roberta-base'              SentencePiece Unigram
// 'Xenova/t5-small'                      SentencePiece`

export const tokenizerDemo: DemoDefinition = {
  id: 'tokenizer',
  title: 'Tokenizer Explorer',
  subtitle:
    'Cùng một câu chạy qua 5 thuật toán tokenize khác nhau — xem chữ biến thành số như thế nào, và mất gì trên đường đi.',
  tagline: 'Chữ biến thành số như thế nào — 5 thuật toán',
  concepts: [
    'Subword',
    'Vocabulary & OOV',
    'Fertility',
    'WordPiece · BPE · SentencePiece',
    'Token đặc biệt',
    'Mất mát thông tin',
  ],
  group: 'text',
  Component: TokenizerDemo,
  snippet: SNIPPET,
  order: 5,
}
