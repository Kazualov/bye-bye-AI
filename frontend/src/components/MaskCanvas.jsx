import { useRef, useEffect, useState } from 'react'

export default function MaskCanvas({ image, paths, setPaths }) {
  const baseRef = useRef(null)
  const overlayRef = useRef(null)
  const [currentPath, setCurrentPath] = useState(null)
  const drawingRef = useRef(false)
  const offsetRef = useRef(0)

  useEffect(() => {
    if (!image || !baseRef.current || !overlayRef.current) return
    const w = image.naturalWidth
    const h = image.naturalHeight
    baseRef.current.width = w
    baseRef.current.height = h
    overlayRef.current.width = w
    overlayRef.current.height = h
    const ctx = baseRef.current.getContext('2d')
    ctx.clearRect(0, 0, w, h)
    ctx.drawImage(image, 0, 0)
  }, [image])

  useEffect(() => {
    let raf
    const loop = () => {
      offsetRef.current = (offsetRef.current + 0.5) % 12
      drawOverlay()
      raf = requestAnimationFrame(loop)
    }
    raf = requestAnimationFrame(loop)
    return () => cancelAnimationFrame(raf)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [paths, currentPath])

  function drawOverlay() {
    const o = overlayRef.current
    if (!o) return
    const ctx = o.getContext('2d')
    ctx.clearRect(0, 0, o.width, o.height)

    const all = currentPath ? [...paths, currentPath] : paths

    for (const path of all) {
      if (path.length < 2) continue
      const isClosed = path !== currentPath

      ctx.beginPath()
      ctx.moveTo(path[0].x, path[0].y)
      for (let i = 1; i < path.length; i++) ctx.lineTo(path[i].x, path[i].y)
      if (isClosed) ctx.closePath()

      if (isClosed) {
        ctx.fillStyle = 'rgba(0, 160, 255, 0.22)'
        ctx.fill()
      }

      ctx.lineWidth = 2
      ctx.setLineDash([6, 6])

      ctx.strokeStyle = 'rgba(0, 0, 0, 0.9)'
      ctx.lineDashOffset = -offsetRef.current
      ctx.stroke()

      ctx.strokeStyle = 'rgba(255, 255, 255, 0.9)'
      ctx.lineDashOffset = -offsetRef.current + 6
      ctx.stroke()
    }
  }

  function getPos(e) {
    const o = overlayRef.current
    const r = o.getBoundingClientRect()
    return {
      x: (e.clientX - r.left) * o.width / r.width,
      y: (e.clientY - r.top) * o.height / r.height,
    }
  }

  function onPointerDown(e) {
    if (!image) return
    drawingRef.current = true
    overlayRef.current.setPointerCapture(e.pointerId)
    setCurrentPath([getPos(e)])
  }

  function onPointerMove(e) {
    if (!drawingRef.current) return
    const p = getPos(e)
    setCurrentPath((prev) => (prev ? [...prev, p] : [p]))
  }

  function onPointerUp() {
    if (!drawingRef.current) return
    drawingRef.current = false
    if (currentPath && currentPath.length >= 3) {
      setPaths((prev) => [...prev, currentPath])
    }
    setCurrentPath(null)
  }

  const undo = () => setPaths((p) => p.slice(0, -1))
  const clear = () => { setPaths([]); setCurrentPath(null) }

  return (
    <>
      <div className="canvas-stack">
        <canvas ref={baseRef} className="base-canvas" />
        <canvas
          ref={overlayRef}
          className="overlay-canvas"
          onPointerDown={onPointerDown}
          onPointerMove={onPointerMove}
          onPointerUp={onPointerUp}
          onPointerCancel={onPointerUp}
        />
      </div>
      <div className="canvas-controls">
        <button type="button" onClick={undo} disabled={!paths.length}>Undo</button>
        <button type="button" onClick={clear} disabled={!paths.length}>Clear</button>
        <span className="hint">
          {paths.length} region{paths.length === 1 ? '' : 's'} selected
        </span>
      </div>
    </>
  )
}