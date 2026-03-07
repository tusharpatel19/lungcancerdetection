const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("fileInput");
const patientName = document.getElementById("patientName");
const preview = document.getElementById("previewImage");
const previewWrap = document.querySelector(".preview");
const predictBtn = document.getElementById("predictBtn");
const resetBtn = document.getElementById("resetBtn");
const saveBtn = document.getElementById("saveBtn");
const downloadBtn = document.getElementById("downloadBtn");
const statusEl = document.getElementById("status");
const resultLabel = document.getElementById("resultLabel");
const resultConfidence = document.getElementById("resultConfidence");
const resultProbability = document.getElementById("resultProbability");
const resultNotes = document.getElementById("resultNotes");
const riskLevelEl = document.getElementById("riskLevel");
const chartBars = document.getElementById("chartBars");
const historyBody = document.getElementById("historyBody");

let currentFile = null;
let latestPrediction = null;

const API_BASE = (window.LUNG_GUARD_API_BASE || "").replace(/\/$/, "");
const apiUrl = (path) => `${API_BASE}${path}`;

const setStatus = (msg, isError = false) => {
  statusEl.textContent = msg;
  statusEl.style.color = isError ? "#ff6b6b" : "#9aa7c7";
};

const setRiskPill = (risk) => {
  riskLevelEl.textContent = `Risk: ${risk || "--"}`;
  riskLevelEl.classList.remove("risk-low", "risk-moderate", "risk-high", "risk-critical");
  const key = (risk || "").toLowerCase();
  if (key) {
    riskLevelEl.classList.add(`risk-${key}`);
  }
};

const parseJsonResponse = async (response, fallbackMessage) => {
  const contentType = (response.headers.get("content-type") || "").toLowerCase();
  if (contentType.includes("application/json")) {
    return response.json();
  }

  const raw = await response.text();
  if (raw.trim().startsWith("<")) {
    throw new Error(
      "Backend API unavailable. Flask backend URL set karo: window.LUNG_GUARD_API_BASE = 'https://your-backend-url'."
    );
  }

  throw new Error(fallbackMessage);
};

const resetUI = () => {
  currentFile = null;
  latestPrediction = null;
  fileInput.value = "";
  preview.src = "";
  previewWrap.classList.remove("active");
  resultLabel.textContent = "Awaiting upload";
  resultConfidence.textContent = "--";
  resultProbability.textContent = "--";
  resultNotes.textContent = "Upload an image to begin.";
  chartBars.innerHTML = "";
  setRiskPill("--");
  setStatus("");
};

const renderChart = (scores) => {
  chartBars.innerHTML = "";
  if (!scores || Object.keys(scores).length === 0) return;

  const entries = Object.entries(scores).sort((a, b) => b[1] - a[1]);
  entries.forEach(([label, value]) => {
    const row = document.createElement("div");
    row.className = "chart-bar";

    const header = document.createElement("label");
    const pct = Math.round(value * 100);
    header.textContent = label;

    const span = document.createElement("span");
    span.textContent = `${pct}%`;
    header.appendChild(span);

    const track = document.createElement("div");
    track.className = "bar-track";

    const fill = document.createElement("div");
    fill.className = "bar-fill";
    fill.style.width = `${pct}%`;

    track.appendChild(fill);
    row.appendChild(header);
    row.appendChild(track);
    chartBars.appendChild(row);
  });
};

const handleFile = (file) => {
  if (!file) return;
  currentFile = file;
  const reader = new FileReader();
  reader.onload = (event) => {
    preview.src = event.target.result;
    previewWrap.classList.add("active");
  };
  reader.readAsDataURL(file);
};

const renderHistory = (items) => {
  historyBody.innerHTML = "";
  if (!items || items.length === 0) {
    historyBody.innerHTML = '<tr><td colspan="5">No saved predictions yet.</td></tr>';
    return;
  }

  items.forEach((item) => {
    const tr = document.createElement("tr");
    const dt = new Date(item.created_at);
    tr.innerHTML = `
      <td>${item.patient_name}</td>
      <td>${item.prediction_result}</td>
      <td>${Math.round(item.confidence * 100)}%</td>
      <td>${item.risk_level}</td>
      <td>${Number.isNaN(dt.getTime()) ? item.created_at : dt.toLocaleString()}</td>
    `;
    historyBody.appendChild(tr);
  });
};

const loadHistory = async () => {
  try {
    const response = await fetch(apiUrl("/history"));
    const data = await parseJsonResponse(response, "Failed to load history");
    if (!response.ok) throw new Error(data.error || "Failed to load history");
    renderHistory(data.items || []);
  } catch (error) {
    setStatus(error.message, true);
  }
};

fileInput.addEventListener("change", (event) => {
  const file = event.target.files[0];
  handleFile(file);
});

dropzone.addEventListener("dragover", (event) => {
  event.preventDefault();
  dropzone.classList.add("dragging");
});

dropzone.addEventListener("dragleave", () => {
  dropzone.classList.remove("dragging");
});

dropzone.addEventListener("drop", (event) => {
  event.preventDefault();
  dropzone.classList.remove("dragging");
  const file = event.dataTransfer.files[0];
  if (file) {
    handleFile(file);
  }
});

predictBtn.addEventListener("click", async () => {
  if (!currentFile) {
    setStatus("Please upload a CT scan image first.", true);
    return;
  }

  setStatus("Running prediction...");
  predictBtn.disabled = true;

  const formData = new FormData();
  formData.append("image", currentFile);

  try {
    const response = await fetch(apiUrl("/predict"), {
      method: "POST",
      body: formData,
    });
    const data = await parseJsonResponse(response, "Prediction failed");
    if (!response.ok) throw new Error(data.error || "Prediction failed");

    const isCancer = (data.label || "").toLowerCase().includes("cancer");
    resultLabel.textContent = isCancer ? "Cancer Detected" : "Normal";
    resultConfidence.textContent = `${Math.round((data.confidence || 0) * 100)}%`;
    resultProbability.textContent = `${Math.round((data.confidence || 0) * 100)}%`;
    resultNotes.textContent = `Model label: ${data.label || "N/A"}`;
    setRiskPill(data.risk_level || "--");
    renderChart(data.scores || {});

    latestPrediction = {
      patient_name: patientName.value.trim(),
      prediction_result: data.label || "N/A",
      confidence: Number(data.confidence || 0),
      risk_level: data.risk_level || "N/A",
      image_path: data.image_path || "",
    };

    setStatus("Prediction complete.");
  } catch (error) {
    setStatus(error.message, true);
  } finally {
    predictBtn.disabled = false;
  }
});

saveBtn.addEventListener("click", async () => {
  if (!latestPrediction) {
    setStatus("Run a prediction first.", true);
    return;
  }
  if (!patientName.value.trim()) {
    setStatus("Enter patient name before saving.", true);
    return;
  }

  setStatus("Saving prediction...");
  try {
    const response = await fetch(apiUrl("/save_prediction"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        ...latestPrediction,
        patient_name: patientName.value.trim(),
      }),
    });
    const data = await parseJsonResponse(response, "Failed to save prediction");
    if (!response.ok) throw new Error(data.error || "Failed to save prediction");
    setStatus("Prediction saved to database.");
    await loadHistory();
  } catch (error) {
    setStatus(error.message, true);
  }
});

downloadBtn.addEventListener("click", async () => {
  if (!latestPrediction) {
    setStatus("Run a prediction first.", true);
    return;
  }
  if (!patientName.value.trim()) {
    setStatus("Enter patient name before downloading report.", true);
    return;
  }

  setStatus("Generating PDF report...");
  try {
    const response = await fetch(apiUrl("/download_report"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        ...latestPrediction,
        patient_name: patientName.value.trim(),
      }),
    });
    if (!response.ok) {
      const data = await parseJsonResponse(response, "Failed to generate report");
      throw new Error(data.error || "Failed to generate report");
    }

    const blob = await response.blob();
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `lungguard_report_${Date.now()}.pdf`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    window.URL.revokeObjectURL(url);
    setStatus("PDF report downloaded.");
  } catch (error) {
    setStatus(error.message, true);
  }
});

resetBtn.addEventListener("click", resetUI);

document.querySelectorAll("[data-scroll]").forEach((button) => {
  button.addEventListener("click", () => {
    const target = document.querySelector(button.dataset.scroll);
    if (target) {
      target.scrollIntoView({ behavior: "smooth" });
    }
  });
});

resetUI();
loadHistory();
