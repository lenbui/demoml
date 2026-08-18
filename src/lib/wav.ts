/**
 * Đóng gói Float32Array thành file WAV.
 *
 * Model TTS trả về âm thanh ở dạng thô nhất có thể: một dãy số thực trong
 * [-1, 1]. Thẻ <audio> không phát được dãy số đó — nó cần một file có định dạng.
 * WAV là lựa chọn hiển nhiên vì nó gần như không có gì: 44 byte header rồi tới
 * mẫu âm thanh, không nén, không thư viện.
 *
 * Viết tay để thấy "file âm thanh" thực chất là gì. Toàn bộ phần đặc tả nằm
 * trong 44 byte dưới đây.
 */

/** Ghi chuỗi ASCII vào DataView — các nhãn RIFF/WAVE/fmt/data đều 4 ký tự. */
function writeAscii(view: DataView, offset: number, text: string): void {
  for (let i = 0; i < text.length; i++) {
    view.setUint8(offset + i, text.charCodeAt(i))
  }
}

/**
 * @param samples    biên độ trong [-1, 1], mono
 * @param sampleRate số mẫu mỗi giây, lấy từ chính output của model
 *
 * Mẫu được đổi từ float32 sang PCM 16-bit số nguyên — định dạng mà mọi trình
 * duyệt đều phát được. Đó cũng là lý do file WAV nặng gấp đôi số mẫu.
 */
export function encodeWav(samples: Float32Array, sampleRate: number): Blob {
  const bytesPerSample = 2
  const channels = 1
  const dataBytes = samples.length * bytesPerSample

  const buffer = new ArrayBuffer(44 + dataBytes)
  const view = new DataView(buffer)

  // ── Khối RIFF ────────────────────────────────────────────────────────────
  writeAscii(view, 0, 'RIFF')
  // Kích thước toàn file trừ 8 byte đầu. `true` = little-endian, WAV luôn vậy.
  view.setUint32(4, 36 + dataBytes, true)
  writeAscii(view, 8, 'WAVE')

  // ── Khối fmt: mô tả âm thanh được mã hoá thế nào ─────────────────────────
  writeAscii(view, 12, 'fmt ') // dấu cách ở cuối là bắt buộc — nhãn luôn 4 byte
  view.setUint32(16, 16, true) // độ dài khối fmt: 16 byte cho PCM
  view.setUint16(20, 1, true) // 1 = PCM không nén
  view.setUint16(22, channels, true)
  view.setUint32(24, sampleRate, true)
  view.setUint32(28, sampleRate * channels * bytesPerSample, true) // byte mỗi giây
  view.setUint16(32, channels * bytesPerSample, true) // byte mỗi khung
  view.setUint16(34, bytesPerSample * 8, true) // bit mỗi mẫu

  // ── Khối data: chính là dãy số của model ─────────────────────────────────
  writeAscii(view, 36, 'data')
  view.setUint32(40, dataBytes, true)

  let offset = 44
  for (let i = 0; i < samples.length; i++) {
    // Cắt biên trước khi đổi thang: model thỉnh thoảng cho ra giá trị vượt
    // [-1, 1], và tràn số nguyên 16-bit sẽ thành tiếng rẹt rất chói.
    const clamped = Math.max(-1, Math.min(1, samples[i]))
    // Thang âm và dương lệch nhau một đơn vị: int16 chạy từ -32768 tới 32767.
    view.setInt16(offset, clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff, true)
    offset += bytesPerSample
  }

  return new Blob([buffer], { type: 'audio/wav' })
}
