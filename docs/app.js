const fileInput = document.getElementById("fileInput");
const preview = document.getElementById("previewImage");
const previewWrap = document.querySelector(".preview");
const predictBtn = document.getElementById("predictBtn");
const resetBtn = document.getElementById("resetBtn");
const statusEl = document.getElementById("status");
const resultLabel = document.getElementById("resultLabel");
const resultConfidence = document.getElementById("resultConfidence");
const resultProbability = document.getElementById("resultProbability");
const resultNotes = document.getElementById("resultNotes");
const chartBars = document.getElementById("chartBars");

let currentFile = null;

const setStatus = (msg, isError = false) => {
  statusEl.textContent = msg;
  statusEl.style.color = isError ? "#ff6b6b" : "#9aa7c7";
};

const resetUI = () => {
  currentFile = null;
  fileInput.value = "";
  preview.src = "";
  previewWrap.classList.remove("active");
  resultLabel.textContent = "Awaiting upload";
  resultConfidence.textContent = "--";
  resultProbability.textContent = "--";
  resultNotes.textContent = "Upload an image to begin.";
  chartBars.innerHTML = "";
  setStatus("");
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

const renderChart = (scores) => {
  chartBars.innerHTML = "";
  if (!scores || Object.keys(scores).length === 0) {
    return;
  }

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

fileInput.addEventListener("change", (event) => {
  const file = event.target.files[0];
  handleFile(file);
});

predictBtn.addEventListener("click", () => {
  if (!currentFile) {
    setStatus("Please upload a CT image first.", true);
    return;
  }

  resultLabel.textContent = "Demo Mode";
  resultConfidence.textContent = "--";
  resultProbability.textContent = "N/A";
  resultNotes.textContent = "Static GitHub Pages demo. Run Flask locally for predictions.";
  renderChart({
    Normal: 0.0,
    "Lung Cancer": 0.0,
    Benign: 0.0,
    Malignant: 0.0,
  });
  setStatus("Demo only. Start the backend locally for real predictions.", true);
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
