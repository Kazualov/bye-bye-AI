const ALL_MODELS = [
  { id: 'telea', label: 'OpenCV Telea' },
  { id: 'lama', label: 'LaMa' },
  { id: 'vit', label: 'ViTInpainter' },
]

export default function ModelSelector({ selected, onChange }) {
  const toggle = (id) => {
    onChange(
      selected.includes(id)
        ? selected.filter((x) => x !== id)
        : [...selected, id]
    )
  }

  return (
    <div className="model-selector">
      {ALL_MODELS.map((m) => {
        const isActive = selected.includes(m.id)
        return (
          <button
            key={m.id}
            type="button"
            className={`model-pill ${isActive ? 'active' : ''}`}
            onClick={() => toggle(m.id)}
          >
            {m.label}
          </button>
        )
      })}
    </div>
  )
}