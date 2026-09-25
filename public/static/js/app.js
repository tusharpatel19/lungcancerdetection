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
const databaseBody = document.getElementById("databaseBody");
const databaseSummary = document.getElementById("databaseSummary");
const confusionMatrixImage = document.getElementById("confusionMatrixImage");
const matrixEmpty = document.getElementById("matrixEmpty");
const accuracyPlotImage = document.getElementById("accuracyPlotImage");
const accuracyEmpty = document.getElementById("accuracyEmpty");

let currentFile = null;
let latestPrediction = null;
let currentFileLooksLikeCt = false;

const computedOrigin = window.location.origin && window.location.origin !== "null" ? window.location.origin : "";
const API_BASE = (window.LUNG_GUARD_API_BASE || computedOrigin || "http://127.0.0.1:5000").replace(/\/$/, "");
const apiUrl = (path) => `${API_BASE}${path}`;

const escapeHtml = (value) =>
  String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");

const formatDate = (value) => {
  const dt = new Date(value);
  return Number.isNaN(dt.getTime()) ? value : dt.toLocaleString();
};

const formatConfidence = (value) => `${Math.round(Number(value || 0) * 100)}%`;

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
      "Backend API unavailable. Set the Flask backend URL: window.LUNG_GUARD_API_BASE = 'https://your-backend-url'."
    );
  }

  throw new Error(fallbackMessage);
};

const resetUI = () => {
  currentFile = null;
  latestPrediction = null;
  currentFileLooksLikeCt = false;
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

const resetResult = () => {
  latestPrediction = null;
  resultLabel.textContent = "Awaiting upload";
  resultConfidence.textContent = "--";
  resultProbability.textContent = "--";
  resultNotes.textContent = "Upload an image to begin.";
  chartBars.innerHTML = "";
  setRiskPill("--");
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

const readImageFromFile = (file) =>
  new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = (event) => {
      const image = new Image();
      image.onload = () => resolve({ image, dataUrl: event.target.result });
      image.onerror = () => reject(new Error("Could not read the image. Please upload a valid PNG or JPG file."));
      image.src = event.target.result;
    };
    reader.onerror = () => reject(new Error("Could not read the image. Please upload a valid PNG or JPG file."));
    reader.readAsDataURL(file);
  });

const imageLooksLikeCtScan = (image) => {
  const size = 96;
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const context = canvas.getContext("2d");
  context.drawImage(image, 0, 0, size, size);
  const { data } = context.getImageData(0, 0, size, size);

  let totalSaturation = 0;
  let totalDelta = 0;
  let darkPixels = 0;
  let blueDominantPixels = 0;
  const pixels = size * size;

  for (let index = 0; index < data.length; index += 4) {
    const red = data[index];
    const green = data[index + 1];
    const blue = data[index + 2];
    const max = Math.max(red, green, blue);
    const min = Math.min(red, green, blue);
    const delta = max - min;
    const brightness = (red + green + blue) / 3;

    totalDelta += delta;
    totalSaturation += max === 0 ? 0 : delta / max;
    if (brightness < 35) darkPixels += 1;
    if (blue > red + 30 && blue > green + 20) blueDominantPixels += 1;
  }

  const meanSaturation = totalSaturation / pixels;
  const meanDelta = totalDelta / pixels;
  const darkRatio = darkPixels / pixels;
  const blueRatio = blueDominantPixels / pixels;

  return meanSaturation <= 0.12 && meanDelta <= 12 && darkRatio >= 0.05 && blueRatio < 0.08;
};

const handleFile = async (file) => {
  if (!file) return;
  currentFile = file;
  currentFileLooksLikeCt = false;
  resetResult();

  try {
    const { image, dataUrl } = await readImageFromFile(file);
    preview.src = dataUrl;
    previewWrap.classList.add("active");
    currentFileLooksLikeCt = imageLooksLikeCtScan(image);
    if (!currentFileLooksLikeCt) {
      setStatus("This does not look like a CT scan. Please upload a grayscale lung CT image.", true);
      return;
    }
    setStatus("");
  } catch (error) {
    currentFile = null;
    preview.src = "";
    previewWrap.classList.remove("active");
    setStatus(error.message, true);
  }
};

const renderHistory = (items) => {
  historyBody.innerHTML = "";
  if (!items || items.length === 0) {
    historyBody.innerHTML = '<tr><td colspan="5">No saved predictions yet.</td></tr>';
    return;
  }

  items.forEach((item) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${escapeHtml(item.patient_name)}</td>
      <td>${escapeHtml(item.prediction_result)}</td>
      <td>${formatConfidence(item.confidence)}</td>
      <td>${escapeHtml(item.risk_level)}</td>
      <td>${escapeHtml(formatDate(item.created_at))}</td>
    `;
    historyBody.appendChild(tr);
  });
};

const renderDatabase = (payload) => {
  const rows = payload?.tables?.predictions?.rows || [];
  databaseBody.innerHTML = "";
  databaseSummary.textContent = `${rows.length} prediction record${rows.length === 1 ? "" : "s"} from ${payload?.database || "predictions.db"}.`;

  if (rows.length === 0) {
    databaseBody.innerHTML = '<tr><td colspan="7">No rows found in predictions table.</td></tr>';
    return;
  }

  rows.forEach((item) => {
    const tr = document.createElement("tr");
    const imagePath = item.image_path || "";
    const imageLink = imagePath
      ? `<a href="${escapeHtml(imagePath)}" target="_blank" rel="noopener">View image</a>`
      : "--";

    tr.innerHTML = `
      <td>${escapeHtml(item.id)}</td>
      <td>${escapeHtml(item.patient_name)}</td>
      <td class="db-result">${escapeHtml(item.prediction_result)}</td>
      <td>${formatConfidence(item.confidence)}</td>
      <td>${escapeHtml(item.risk_level)}</td>
      <td>${imageLink}</td>
      <td>${escapeHtml(formatDate(item.created_at))}</td>
    `;
    databaseBody.appendChild(tr);
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

const loadDatabase = async () => {
  if (!databaseBody || !databaseSummary) return;

  try {
    const response = await fetch(apiUrl("/database"));
    const data = await parseJsonResponse(response, "Failed to load database");
    if (!response.ok) throw new Error(data.error || "Failed to load database");
    renderDatabase(data);
  } catch (error) {
    databaseSummary.textContent = error.message;
    databaseBody.innerHTML = '<tr><td colspan="7">Database records could not be loaded.</td></tr>';
    setStatus(error.message, true);
  }
};

if (confusionMatrixImage && matrixEmpty) {
  const matrixWrap = matrixEmpty.parentElement;
  confusionMatrixImage.addEventListener("error", () => {
    matrixWrap.classList.add("missing");
  });
  confusionMatrixImage.addEventListener("load", () => {
    matrixWrap.classList.remove("missing");
  });
  if (confusionMatrixImage.complete && confusionMatrixImage.naturalWidth === 0) {
    matrixWrap.classList.add("missing");
  }
}

if (accuracyPlotImage && accuracyEmpty) {
  const graphWrap = accuracyEmpty.parentElement;
  accuracyPlotImage.addEventListener("error", () => {
    graphWrap.classList.add("missing");
  });
  accuracyPlotImage.addEventListener("load", () => {
    graphWrap.classList.remove("missing");
  });
  if (accuracyPlotImage.complete && accuracyPlotImage.naturalWidth === 0) {
    graphWrap.classList.add("missing");
  }
}

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
  if (!currentFileLooksLikeCt) {
    setStatus("This does not look like a CT scan. Please upload a grayscale lung CT image.", true);
    return;
  }

  resetResult();
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

    const isCancer = Boolean(data.is_cancer);
    resultLabel.textContent = isCancer ? "Cancer Detected" : data.display_label || data.label || "Normal";
    resultConfidence.textContent = `${Math.round((data.confidence || 0) * 100)}%`;
    resultProbability.textContent = `${Math.round((data.confidence || 0) * 100)}%`;
    resultNotes.textContent = `Model label: ${data.display_label || data.label || "N/A"}`;
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
    await loadDatabase();
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
loadDatabase();
