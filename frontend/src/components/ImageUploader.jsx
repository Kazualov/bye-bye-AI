import { useRef } from 'react'

export default function ImageUploader({ onImage }) {
  const inputRef = useRef(null)

  const handleChange = (e) => {
    const f = e.target.files?.[0]
    if (!f) return
    const url = URL.createObjectURL(f)
    const el = new Image()
    el.onload = () => onImage(el, f)
    el.src = url
  }

  return (
    <div className="uploader">
      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        onChange={handleChange}
        style={{ display: 'none' }}
      />
      <button type="button" onClick={() => inputRef.current?.click()}>
        Choose image
      </button>
    </div>
  )
}