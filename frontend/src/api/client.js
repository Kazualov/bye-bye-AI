export async function createJob({ imageBlob, maskBlob, models }) {
  const fd = new FormData()
  fd.append('image', imageBlob, 'input.png')
  fd.append('mask', maskBlob, 'mask.png')
  fd.append('models', models.join(','))

  const res = await fetch('/api/v1/jobs', { method: 'POST', body: fd })
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.message || `HTTP ${res.status}`)
  }
  return res.json()
}

export async function getJob(jobId) {
  const res = await fetch(`/api/v1/jobs/${jobId}`)
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  return res.json()
}

export function jobSocketUrl(jobId) {
  const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
  return `${proto}//${window.location.host}/api/v1/ws/jobs/${jobId}`
}