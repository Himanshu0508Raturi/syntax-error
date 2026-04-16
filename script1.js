const imageInput = document.getElementById("imageInput");
const preview = document.getElementById("preview");
const resultDiv = document.getElementById("result");
const loader = document.getElementById("loader");
const progressBar = document.getElementById("progressBar");
const progressBox = document.getElementById("progressBox");
const dropArea = document.getElementById("drop-area");

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
      <strong>${typeof data.intra_severity_score === 'number' ? data.intra_severity_score.toFixed(2) : 'Unknown'}</strong>
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
  dropArea.style.background = "rgba(255,255,255,0.2)";
});

dropArea.addEventListener("dragleave", () => {
  dropArea.style.background = "transparent";
});

dropArea.addEventListener("drop", (e) => {
  e.preventDefault();
  const file = e.dataTransfer.files[0];
  imageInput.files = e.dataTransfer.files;
  showPreview(file);
});

// File select
imageInput.addEventListener("change", () => {
  const file = imageInput.files[0];
  showPreview(file);
});

function showPreview(file) {
  if (file) {
    preview.src = URL.createObjectURL(file);
    preview.hidden = false;
  }
}

async function uploadImage() {
  const file = imageInput.files[0];

  if (!file) {
    alert("Upload an image first!");
    return;
  }

  const formData = new FormData();
  formData.append("file", file);

  loader.classList.remove("hidden");
  resultDiv.innerHTML = "";
  resultDiv.classList.add("hidden");
  progressBox.classList.add("hidden");

  try {
    const response = await fetch("http://127.0.0.1:8000/predict", {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      const errorBody = await response.json().catch(() => ({}));
      throw new Error(errorBody.detail || "Prediction request failed");
    }

    const data = await response.json();

    loader.classList.add("hidden");
    renderResult(data);
  } catch (error) {
    loader.classList.add("hidden");
    resultDiv.classList.remove("hidden");
    resultDiv.innerHTML = `<div class="result-error">${escapeHtml(error.message || "Server error")}</div>`;
  }
}
