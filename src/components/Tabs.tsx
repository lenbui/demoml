/**
 * Tab trong một demo — dùng khi demo có nhiều chế độ xem trên cùng bộ dữ liệu
 * (card Embedding có 3: tìm kiếm, so sánh 2 câu, bản đồ 2D).
 *
 * Mục đích là chống cuộn dài: 3 phần đó nếu xếp dọc thì card cao gấp ba.
 */
export interface TabItem {
  id: string
  label: string
}

export function Tabs({
  items,
  value,
  onChange,
}: {
  items: TabItem[]
  value: string
  onChange: (id: string) => void
}) {
  return (
    <div className="tabs" role="tablist">
      {items.map((item) => (
        <button
          key={item.id}
          type="button"
          role="tab"
          aria-selected={value === item.id}
          className={`tab${value === item.id ? ' tab--active' : ''}`}
          onClick={() => onChange(item.id)}
        >
          {item.label}
        </button>
      ))}
    </div>
  )
}
