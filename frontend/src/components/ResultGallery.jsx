import { useRef } from 'react'

export default function ResultGallery({ job }) {
  const scrollRef = useRef(null)

  const scrollBy = (dir) => {
    const el = scrollRef.current
    if (!el) return
    const card = el.querySelector('.result-card')
    const w = card?.offsetWidth || el.clientWidth * 0.8
    el.scrollBy({ left: dir * (w + 16), behavior: 'smooth' })
  }

  return (
    <section className="gallery-section">
      <h2>3. Results</h2>
      <div className="carousel-wrap">
        <button className="carousel-btn" onClick={() => scrollBy(-1)} aria-label="Previous">‹</button>
        <div className="carousel" ref={scrollRef}>
          {job.results.map((r) => (
            <div className="result-card" key={r.model}>
              <div className="result-header">
                <span className="model-name">{r.model}</span>
                <span className={'status' + (r.error ? ' failed' : '')}>
                  {r.status}
                  {r.runtime_ms ? ` · ${r.runtime_ms} ms` : ''}
                </span>
              </div>
              {r.output_url ? (
                <img src={r.output_url} alt={r.model} />
              ) : (
                <div className="placeholder">
                  {r.status === 'pending' && 'Queued…'}
                  {r.status === 'running' && 'Processing…'}
                  {r.status === 'failed' && (r.error || 'Failed')}
                </div>
              )}
            </div>
          ))}
        </div>
        <button className="carousel-btn" onClick={() => scrollBy(1)} aria-label="Next">›</button>
      </div>
    </section>
  )
}