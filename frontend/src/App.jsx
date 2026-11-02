import { useState } from 'react'
import ImageUploader from './components/ImageUploader'
import MaskCanvas from './components/MaskCanvas'
import ModelSelector from './components/ModelSelector'
import ResultGallery from './components/ResultGallery'
import { createJob } from './api/client'
import { useJobSocket } from './hooks/useJobSocket'
import './App.css'

async function buildMaskBlob(imgEl, paths) {
  const w = imgEl.naturalWidth
  const h = imgEl.naturalHeight
  const m = document.createElement('canvas')
  m.width = w
  m.height = h
  const mx = m.getContext('2d')
  mx.fillStyle = 'black'
  mx.fillRect(0, 0, w, h)
  mx.fillStyle = 'white'
  for (const path of paths) {
    if (path.length < 3) continue
    mx.beginPath()
    mx.moveTo(path[0].x, path[0].y)
    for (let i = 1; i < path.length; i++) mx.lineTo(path[i].x, path[i].y)
    mx.closePath()
    mx.fill()
  }
  return new Promise((res) => m.toBlob(res, 'image/png'))
}

export default function App() {
  const [image, setImage] = useState(null) // { el, blob }
  const [paths, setPaths] = useState([])
  const [models, setModels] = useState(['telea', 'lama', 'vit'])
  const [jobId, setJobId] = useState(null)
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState(null)
  const { job } = useJobSocket(jobId)

  const handleImage = (el, blob) => {
    setImage({ el, blob })
    setPaths([])
    setJobId(null)
    setError(null)
  }

  const handleRun = async () => {
    if (!image || paths.length === 0) return
    setSubmitting(true)
    setError(null)
    try {
      const maskBlob = await buildMaskBlob(image.el, paths)
      const res = await createJob({ imageBlob: image.blob, maskBlob, models })
      setJobId(res.job_id)
    } catch (e) {
      setError(e.message)
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div className="app">
      <header className="hero">
        <h1>Object Removal <span className="accent">AI</span></h1>
        <p className="subtitle">
          Upload an image, draw a lasso around any object, and let our AI models remove it seamlessly.
        </p>
      </header>

      <main className="main-content">
        <section className="step">
          <div className="step-header">
            <span className="step-number">1</span>
            <h2>Upload & Mask</h2>
          </div>
          <div className="step-content">
            <ImageUploader onImage={handleImage} />
            {image ? (
              <div className="canvas-container">
                <MaskCanvas image={image.el} paths={paths} setPaths={setPaths} />
              </div>
            ) : (
              <div className="empty-state">
                <p>No image loaded yet. Upload a photo to start.</p>
              </div>
            )}
          </div>
        </section>

        <section className="step">
          <div className="step-header">
            <span className="step-number">2</span>
            <h2>Choose Models</h2>
          </div>
          <div className="step-content">
            <ModelSelector selected={models} onChange={setModels} />

            <div className="action-area">
              <button
                className="run-btn"
                onClick={handleRun}
                disabled={!image || paths.length === 0 || submitting || !models.length}
              >
                {submitting ? 'Starting...' : 'Run Inpainting'}
              </button>
              {error && <p className="error">{error}</p>}
              git branch
            </div>
          </div>
        </section>

        {job && (
          <section className="step results-step">
            <div className="step-header">
              <span className="step-number">3</span>
              <h2>Results</h2>
            </div>
            <ResultGallery job={job} />
          </section>
        )}
      </main>
    </div>
  )
}