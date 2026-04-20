const imageInput = document.getElementById("imageInput");
const preview = document.getElementById("preview");
const resultDiv = document.getElementById("result");
const loader = document.getElementById("loader");
const progressBar = document.getElementById("progressBar");
const progressBox = document.getElementById("progressBox");
const dropArea = document.getElementById("drop-area");
const analyzeBtn = document.getElementById("analyzeBtn");
const cameraBtn = document.getElementById("cameraBtn");
const closeCameraBtn = document.getElementById("closeCameraBtn");
const cameraBox = document.getElementById("cameraBox");
const cameraFeed = document.getElementById("cameraFeed");
const captureBtn = document.getElementById("captureBtn");

let currentPreviewUrl = null;
let cameraStream = null;
let selectedImageFile = null;

function setSelectedImage(file) {
  selectedImageFile = file;
}

function stopCamera() {
  if (cameraStream) {
    cameraStream.getTracks().forEach((track) => track.stop());
    cameraStream = null;
  }

  cameraFeed.srcObject = null;
  cameraBox.classList.add("hidden");
  closeCameraBtn.classList.add("hidden");
}

async function startCamera() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    alert("Camera access is not supported in this browser.");
    return;
  }

  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "environment" },
    });

    cameraStream = stream;
    cameraFeed.srcObject = stream;
    cameraBox.classList.remove("hidden");
    closeCameraBtn.classList.remove("hidden");
  } catch (error) {
    alert("Unable to access camera. Please allow camera permission.");
  }
}

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/\"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function renderResult(data) {
  const confidence = Number(data.confidence || 0);
  const scores = Array.isArray(data.all_scores) ? data.all_scores : [];

  resultDiv.classList.remove("hidden");
  resultDiv.innerHTML = `
    <div class="result-item">
      <span class="result-label">Predicted class</span>
      <strong>${escapeHtml(data.predicted_class || "Unknown")}</strong>
    </div>
    <div class="result-item">
      <span class="result-label">Confidence</span>
      <strong>${(confidence * 100).toFixed(2)}%</strong>
    </div>
    <div class="result-item">
      <span class="result-label">Intra-class Severity</span>
      <strong>${escapeHtml(data.intra_class_severity || "Unknown")}</strong>
    </div>
    <div class="result-item">
      <span class="result-label">Severity Score</span>
      <strong>${typeof data.intra_severity_score === "number" ? data.intra_severity_score.toFixed(2) : "Unknown"}</strong>
    </div>
    <div class="result-item">
      <span class="result-label">Rationale</span>
      <strong>${escapeHtml(data.intra_severity_rationale || "Unknown")}</strong>
    </div>
    <div class="result-item">
      <span class="result-label">Device used</span>
      <strong>${escapeHtml(data.device_used || "Unknown")}</strong>
    </div>
    <div class="scores-section">
      <span class="result-label">All scores</span>
      <div class="scores-list">
        ${scores
          .map(
            (score) => `
          <div class="score-row">
            <span>${escapeHtml(score.label)}</span>
            <span>${(Number(score.confidence || 0) * 100).toFixed(2)}%</span>
          </div>
        `,
          )
          .join("")}
      </div>
    </div>
  `;

  progressBox.classList.remove("hidden");
  progressBar.style.width = `${(confidence * 100).toFixed(2)}%`;
}

// Drag & Drop
dropArea.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropArea.classList.add("drag-active");
});

dropArea.addEventListener("dragleave", () => {
  dropArea.classList.remove("drag-active");
});

dropArea.addEventListener("drop", (e) => {
  e.preventDefault();
  dropArea.classList.remove("drag-active");
  const file = e.dataTransfer.files[0];
  imageInput.files = e.dataTransfer.files;
  setSelectedImage(file);
  showPreview(file);
});

dropArea.addEventListener("click", (e) => {
  if (e.target !== imageInput) {
    imageInput.click();
  }
});

dropArea.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") {
    e.preventDefault();
    imageInput.click();
  }
});

// File select
imageInput.addEventListener("change", () => {
  const file = imageInput.files[0];
  setSelectedImage(file);
  showPreview(file);
});

cameraBtn.addEventListener("click", startCamera);
closeCameraBtn.addEventListener("click", stopCamera);

captureBtn.addEventListener("click", () => {
  if (!cameraStream || !cameraFeed.videoWidth || !cameraFeed.videoHeight) {
    return;
  }

  const canvas = document.createElement("canvas");
  canvas.width = cameraFeed.videoWidth;
  canvas.height = cameraFeed.videoHeight;

  const ctx = canvas.getContext("2d");
  if (!ctx) {
    return;
  }

  ctx.drawImage(cameraFeed, 0, 0, canvas.width, canvas.height);

  canvas.toBlob(
    (blob) => {
      if (!blob) {
        return;
      }

      const cameraFile = new File([blob], `camera-capture-${Date.now()}.jpg`, {
        type: "image/jpeg",
      });

      setSelectedImage(cameraFile);
      showPreview(cameraFile);
      stopCamera();
    },
    "image/jpeg",
    0.92,
  );
});

function showPreview(file) {
  if (file) {
    if (currentPreviewUrl) {
      URL.revokeObjectURL(currentPreviewUrl);
    }

    currentPreviewUrl = URL.createObjectURL(file);
    preview.src = currentPreviewUrl;
    preview.hidden = false;
  }
}

async function uploadImage() {
  const file = selectedImageFile || imageInput.files[0];

  if (!file) {
    alert("Upload an image first!");
    return;
  }

  const formData = new FormData();
  formData.append("file", file);

  loader.classList.remove("hidden");
  analyzeBtn.disabled = true;
  analyzeBtn.textContent = "Analyzing...";
  resultDiv.innerHTML = "";
  resultDiv.classList.add("hidden");
  progressBox.classList.add("hidden");

  try {
    const response = await fetch(
      "https://raturihimanshu077-neural-nexus.hf.space/predict",
      {
        method: "POST",
        body: formData,
      },
    );

    if (!response.ok) {
      const errorBody = await response.json().catch(() => ({}));
      throw new Error(errorBody.detail || "Prediction request failed");
    }

    const data = await response.json();

    loader.classList.add("hidden");
    analyzeBtn.disabled = false;
    analyzeBtn.textContent = "Analyze Image";
    renderResult(data);
  } catch (error) {
    loader.classList.add("hidden");
    analyzeBtn.disabled = false;
    analyzeBtn.textContent = "Analyze Image";
    resultDiv.classList.remove("hidden");
    resultDiv.innerHTML = `<div class="result-error">${escapeHtml(error.message || "Server error")}</div>`;
  }
}

window.addEventListener("beforeunload", stopCamera);
