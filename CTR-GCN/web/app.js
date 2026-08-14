const state = {
  stream: null,
  timer: null,
  running: false,
  sending: false,
  backendOk: false,
  lastResult: null,
};

const els = {
  backendStatus: document.getElementById('backendStatus'),
  cameraStatus: document.getElementById('cameraStatus'),
  predictionLabel: document.getElementById('predictionLabel'),
  predictionConfidence: document.getElementById('predictionConfidence'),
  rawLabel: document.getElementById('rawLabel'),
  qualityLabel: document.getElementById('qualityLabel'),
  bufferFill: document.getElementById('bufferFill'),
  latencyMs: document.getElementById('latencyMs'),
  serverFps: document.getElementById('serverFps'),
  top3Box: document.getElementById('top3Box'),
  cameraVideo: document.getElementById('cameraVideo'),
  captureCanvas: document.getElementById('captureCanvas'),
  mirrorFrame: document.getElementById('mirrorFrame'),
  startBtn: document.getElementById('startBtn'),
  stopBtn: document.getElementById('stopBtn'),
};

function setBackendStatus(text, ok) {
  els.backendStatus.textContent = text;
  els.backendStatus.style.color = ok ? 'var(--good)' : 'var(--warn)';
}

function setCameraStatus(text, ok) {
  els.cameraStatus.textContent = text;
  els.cameraStatus.style.color = ok ? 'var(--good)' : 'var(--warn)';
}

async function checkBackend() {
  try {
    const res = await fetch('http://127.0.0.1:8000/health');
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    state.backendOk = true;
    setBackendStatus('Connected', true);
  } catch (err) {
    state.backendOk = false;
    setBackendStatus('Disconnected', false);
  }
}

async function startCamera() {
  await checkBackend();
  if (!state.backendOk) {
    alert('Backend not reachable. Please start docker compose first.');
    return;
  }

  try {
    state.stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: {
        facingMode: 'user',
        width: { ideal: 1280 },
        height: { ideal: 720 },
      },
    });
    els.cameraVideo.srcObject = state.stream;
    await els.cameraVideo.play();
    state.running = true;
    setCameraStatus('Running', true);
    els.startBtn.disabled = true;
    els.stopBtn.disabled = false;
    loop();
  } catch (err) {
    console.error(err);
    alert(`Failed to access camera: ${err.message}`);
    setCameraStatus('Camera error', false);
  }
}

function stopCamera() {
  state.running = false;
  if (state.timer) {
    clearTimeout(state.timer);
    state.timer = null;
  }
  if (state.stream) {
    state.stream.getTracks().forEach((track) => track.stop());
    state.stream = null;
  }
  els.cameraVideo.srcObject = null;
  els.mirrorFrame.style.opacity = 0;
  els.startBtn.disabled = false;
  els.stopBtn.disabled = true;
  setCameraStatus('Idle', false);
}

function drawToCanvas() {
  const canvas = els.captureCanvas;
  const video = els.cameraVideo;
  const ctx = canvas.getContext('2d', { alpha: false });
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  ctx.save();
  ctx.scale(-1, 1);
  ctx.drawImage(video, -canvas.width, 0, canvas.width, canvas.height);
  ctx.restore();
}

function canvasToBlob(canvas) {
  return new Promise((resolve) => {
    canvas.toBlob((blob) => resolve(blob), 'image/jpeg', 0.85);
  });
}

async function sendFrame() {
  if (!state.running || state.sending || !state.backendOk) return;
  const video = els.cameraVideo;
  if (!video.videoWidth || !video.videoHeight) return;

  state.sending = true;
  try {
    drawToCanvas();
    const blob = await canvasToBlob(els.captureCanvas);
    if (!blob) return;

    const form = new FormData();
    form.append('frame', blob, 'frame.jpg');

    const res = await fetch('http://127.0.0.1:8000/infer-frame', {
      method: 'POST',
      body: form,
    });

    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    state.lastResult = data;
    renderResult(data);
  } catch (err) {
    console.error(err);
    setBackendStatus('Backend error', false);
  } finally {
    state.sending = false;
  }
}

function renderResult(data) {
  const payload = data || {};
  const label = payload.predicted_label ?? '---';
  const confidence = Number(payload.confidence ?? 0);
  const rawLabel = payload.raw_label ?? '---';
  const quality = payload.quality_label ?? 'n/a';
  const bufferFill = payload.buffer_fill ?? 0;
  const latencyMs = Number(payload.processing_ms ?? 0);
  const serverFps = Number(payload.processing_fps ?? 0);
  const top3 = payload.top3 ?? [];

  els.predictionLabel.textContent = label;
  els.predictionConfidence.textContent = `${(confidence * 100).toFixed(1)}%`;
  els.rawLabel.textContent = rawLabel;
  els.qualityLabel.textContent = quality;
  els.bufferFill.textContent = String(bufferFill);
  els.latencyMs.textContent = `${latencyMs.toFixed(0)} ms`;
  els.serverFps.textContent = serverFps.toFixed(1);
  els.top3Box.textContent = JSON.stringify(top3, null, 2);
}

function loop() {
  if (!state.running) return;
  sendFrame();
  state.timer = setTimeout(loop, 150);
}

els.startBtn.addEventListener('click', startCamera);
els.stopBtn.addEventListener('click', stopCamera);
window.addEventListener('beforeunload', stopCamera);
checkBackend();
