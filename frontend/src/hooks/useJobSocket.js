import { useEffect, useState } from 'react'
import { getJob, jobSocketUrl } from '../api/client'

export function useJobSocket(jobId) {
  const [job, setJob] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    if (!jobId) {
      setJob(null)
      return
    }

    let cancelled = false
    let ws = null
    let pollTimer = null
    let wsConnected = false
    let fallbackTimer = null

    const applyJob = (j) => {
      if (cancelled) return
      setJob(j)
      const done =
        j.status === 'succeeded' || j.status === 'partial' || j.status === 'failed'
      if (done) {
        if (pollTimer) { clearInterval(pollTimer); pollTimer = null }
        if (ws) { try { ws.close() } catch {} ws = null }
      }
    }

    const startPolling = () => {
      if (pollTimer || cancelled) return
      const tick = async () => {
        try {
          const j = await getJob(jobId)
          applyJob(j)
        } catch (e) {
          if (!cancelled) setError(e.message)
        }
      }
      tick()
      pollTimer = setInterval(tick, 1500)
    }

    getJob(jobId).then(applyJob).catch((e) => !cancelled && setError(e.message))

    try {
      ws = new WebSocket(jobSocketUrl(jobId))
      ws.onopen = () => { wsConnected = true }
      ws.onmessage = (ev) => {
        try { applyJob(JSON.parse(ev.data)) } catch { /* ignore */ }
      }
      ws.onerror = () => { if (!wsConnected) startPolling() }
      ws.onclose = () => {
        if (cancelled) return
        getJob(jobId).then(applyJob).catch(() => {})
      }
    } catch {
      startPolling()
    }

    fallbackTimer = setTimeout(() => {
      if (!wsConnected && !cancelled) startPolling()
    }, 2500)

    return () => {
      cancelled = true
      if (pollTimer) clearInterval(pollTimer)
      if (fallbackTimer) clearTimeout(fallbackTimer)
      if (ws) { try { ws.close() } catch {} }
    }
  }, [jobId])

  return { job, error }
}