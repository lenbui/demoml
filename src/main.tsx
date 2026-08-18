import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'

import App from './App'

// CSS của KaTeX phải có sẵn trước công thức đầu tiên, nếu không chữ sẽ nhảy
// layout khi font toán được nạp. Vite bundle luôn cả font vào assets.
import 'katex/dist/katex.min.css'
import './styles.css'

const container = document.getElementById('root')
if (!container) throw new Error('Không tìm thấy #root trong index.html')

createRoot(container).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
