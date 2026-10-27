const ALL_MODELS = [
  { id: 'telea', label: 'OpenCV Telea' },
  { id: 'lama', label: 'LaMa' },
  { id: 'vit', label: 'ViTInpainter' },
  //{ id: 'flux', label: 'FLUX.1-Fill' },
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
      {ALL_MODELS.map((m) => (
        <label key={m.id} className="model-option">
          <input
            type="checkbox"
            checked={selected.includes(m.id)}
            onChange={() => toggle(m.id)}
          />
          {m.label}
        </label>
      ))}
    </div>
  )
}