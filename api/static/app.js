const state = {
  selectedFile: null,
  selectedStyle: "vangogh",
  selectedNeuralStyle: "starry_night",
  selectedClassifierModel: null,
  selectedColorizerModel: null,
  previewSrc: "",
};

const elements = {
  healthPill: document.getElementById("health-pill"),
  modelState: document.getElementById("model-state"),
  classifierPath: document.getElementById("classifier-path"),
  colorizerPath: document.getElementById("colorizer-path"),
  totals: {
    predictions: document.getElementById("total-predictions"),
    activity: document.getElementById("total-activity"),
    benchmarks: document.getElementById("total-benchmarks"),
    users: document.getElementById("total-users"),
  },
  dropZone: document.getElementById("drop-zone"),
  fileInput: document.getElementById("file-input"),
  dropMessage: document.getElementById("drop-message"),
  preview: document.getElementById("preview"),
  originalImage: document.getElementById("original-image"),
  resultImage: document.getElementById("result-image"),
  resultMeta: document.getElementById("result-meta"),
  resultSection: document.getElementById("result-section"),
  scoreList: document.getElementById("score-list"),
  predictionList: document.getElementById("prediction-list"),
  activityList: document.getElementById("activity-list"),
  benchmarkList: document.getElementById("benchmark-list"),
  benchmarkChart: document.getElementById("benchmark-chart"),
  benchmarkOutput: document.getElementById("benchmark-output"),
  benchmarkDataset: document.getElementById("benchmark-dataset"),
  benchmarkEpochs: document.getElementById("benchmark-epochs"),
};

function setActiveOperation(name) {
  const ops = ["classify", "colorize", "stylize", "neural-style"];
  ops.forEach(op => {
    document.getElementById(`${op}-button`).classList.toggle("active", op === name);
  });
  document.getElementById("classifier-model-row").style.display = name === "classify" ? "" : "none";
  document.getElementById("colorizer-model-row").style.display  = name === "colorize" ? "" : "none";
  document.getElementById("algo-style-row").style.display        = name === "stylize"   ? "" : "none";
  document.getElementById("neural-style-row").style.display      = name === "neural-style" ? "" : "none";
  document.getElementById("nst-hint").style.display              = name === "neural-style" ? "" : "none";
}

function buildModelChips(containerId, models, onSelect) {
  const container = document.getElementById(containerId);
  container.innerHTML = "";
  if (!models.length) {
    container.innerHTML = '<span class="small-note">Моделей не знайдено</span>';
    return;
  }
  models.forEach((m, i) => {
    const chip = document.createElement("div");
    chip.className = "style-chip" + (i === 0 ? " active" : "");
    chip.textContent = m.name;
    chip.title = m.path;
    chip.addEventListener("click", () => {
      container.querySelectorAll(".style-chip").forEach(c => c.classList.remove("active"));
      chip.classList.add("active");
      onSelect(m.path);
    });
    container.appendChild(chip);
    if (i === 0) onSelect(m.path);
  });
}

async function loadClassifierModels() {
  try {
    const models = await apiRequest("/api/classify/models");
    buildModelChips("classifier-chips", models, path => { state.selectedClassifierModel = path; });
    addImageNetChip("classifier-chips");
  } catch (_) {}
}

function addImageNetChip(containerId) {
  const container = document.getElementById(containerId);
  const chip = document.createElement("div");
  chip.className = "style-chip";
  chip.textContent = "\uD83C\uDF10 ImageNet";
  chip.dataset.imagenet = "1";
  chip.addEventListener("click", () => {
    container.querySelectorAll(".style-chip").forEach(c => c.classList.remove("active"));
    chip.classList.add("active");
    state.selectedClassifierModel = null;
  });
  container.appendChild(chip);
}

async function loadColorizerModels() {
  try {
    const models = await apiRequest("/api/colorize/models");
    buildModelChips("colorizer-chips", models, path => { state.selectedColorizerModel = path; });
  } catch (_) {}
}

function init() {
  bindUpload();
  bindStyleChips();
  bindNeuralStyleChips();
  loadClassifierModels();
  loadColorizerModels();
  setActiveOperation("classify");
  document.getElementById("classify-button").addEventListener("click", () => {
    setActiveOperation("classify");
    const imagenetActive = document.querySelector("#classifier-chips .style-chip[data-imagenet].active");
    if (imagenetActive) {
      runClassifyImagenet();
    } else {
      runClassify();
    }
  });
  document.getElementById("colorize-button").addEventListener("click", () => {
    setActiveOperation("colorize");
    runColorize();
  });
  document.getElementById("stylize-button").addEventListener("click", () => {
    setActiveOperation("stylize");
    runStylize();
  });
  document.getElementById("neural-style-button").addEventListener("click", () => {
    setActiveOperation("neural-style");
    runNeuralStyle();
  });
  document.getElementById("refresh-button").addEventListener("click", loadOverview);
  document.getElementById("benchmark-button").addEventListener("click", runBenchmark);
  loadOverview();
  setInterval(loadOverview, 15000);
}

function bindUpload() {
  const { dropZone, fileInput } = elements;
  dropZone.addEventListener("click", () => fileInput.click());
  dropZone.addEventListener("dragover", event => {
    event.preventDefault();
    dropZone.classList.add("drag-over");
  });
  dropZone.addEventListener("dragleave", () => dropZone.classList.remove("drag-over"));
  dropZone.addEventListener("drop", event => {
    event.preventDefault();
    dropZone.classList.remove("drag-over");
    handleFile(event.dataTransfer.files[0]);
  });
  fileInput.addEventListener("change", event => handleFile(event.target.files[0]));
}

function bindStyleChips() {
  document.querySelectorAll("#algo-style-row .style-chip").forEach(chip => {
    chip.addEventListener("click", () => {
      document.querySelectorAll("#algo-style-row .style-chip").forEach(item => item.classList.remove("active"));
      chip.classList.add("active");
      state.selectedStyle = chip.dataset.style;
    });
  });
}

function bindNeuralStyleChips() {
  document.querySelectorAll("#neural-style-row .style-chip").forEach(chip => {
    chip.addEventListener("click", () => {
      document.querySelectorAll("#neural-style-row .style-chip").forEach(item => item.classList.remove("active"));
      chip.classList.add("active");
      state.selectedNeuralStyle = chip.dataset.nstyle;
    });
  });
}

function handleFile(file) {
  if (!file || !file.type.startsWith("image/")) {
    return;
  }

  state.selectedFile = file;
  const reader = new FileReader();
  reader.onload = event => {
    state.previewSrc = event.target.result;
    elements.preview.src = state.previewSrc;
    elements.preview.hidden = false;
    elements.dropMessage.textContent = file.name;
    elements.resultSection.hidden = true;
  };
  reader.readAsDataURL(file);
}

async function toBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = event => resolve(event.target.result.split(",")[1]);
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

async function apiRequest(path, payload) {
  const response = await fetch(path, {
    method: payload ? "POST" : "GET",
    headers: payload ? { "Content-Type": "application/json" } : undefined,
    body: payload ? JSON.stringify(payload) : undefined,
  });

  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.detail || data.message || `HTTP ${response.status}`);
  }
  return data;
}

function ensureFileSelected() {
  if (!state.selectedFile) {
    throw new Error("Спочатку завантажте зображення.");
  }
}

function renderResult({ title, subtitle, imageSrc, scores = [] }) {
  elements.resultSection.hidden = false;
  elements.originalImage.src = state.previewSrc;
  elements.resultImage.src = imageSrc;
  elements.resultMeta.innerHTML = `<strong>${title}</strong><div class="list-meta">${subtitle}</div>`;
  if (scores.length) {
    elements.scoreList.innerHTML = scores
      .map(item => `<div class="list-item"><strong>${item.name}</strong><div class="list-meta">${item.value}</div></div>`)
      .join("");
  } else {
    elements.scoreList.innerHTML = '<div class="small-note">Додаткові score для цього запиту не відображаються.</div>';
  }
}

async function runClassify() {
  try {
    ensureFileSelected();
    const image = await toBase64(state.selectedFile);
    const payload = { image };
    if (state.selectedClassifierModel) payload.model_path = state.selectedClassifierModel;
    const data = await apiRequest("/api/classify", payload);
    const scores = Object.entries(data.all_scores)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 5)
      .map(([name, value]) => ({ name, value: `${(value * 100).toFixed(1)}%` }));
    renderResult({
      title: `Клас: ${data.label}`,
      subtitle: `Впевненість ${(data.confidence * 100).toFixed(1)}%`,
      imageSrc: state.previewSrc,
      scores,
    });
    await loadOverview();
  } catch (error) {
    renderError(error);
  }
}

async function runClassifyImagenet() {
  try {
    ensureFileSelected();
    const image = await toBase64(state.selectedFile);
    const data = await apiRequest("/api/classify/imagenet", { image });
    const scores = Object.entries(data.all_scores)
      .sort((a, b) => b[1] - a[1])
      .map(([name, value]) => ({ name, value: `${(value * 100).toFixed(2)}%` }));
    renderResult({
      title: `ImageNet: ${data.label}`,
      subtitle: `Впевненість ${(data.confidence * 100).toFixed(1)}% · 1000 класів`,
      imageSrc: state.previewSrc,
      scores,
    });
    await loadOverview();
  } catch (error) {
    renderError(error);
  }
}

async function runColorize() {
  try {
    ensureFileSelected();
    const image = await toBase64(state.selectedFile);
    const payload = { image };
    if (state.selectedColorizerModel) payload.model_path = state.selectedColorizerModel;
    const data = await apiRequest("/api/colorize", payload);
    renderResult({
      title: "Колоризація завершена",
      subtitle: "Результат повернуто з endpoint /api/colorize",
      imageSrc: `data:image/jpeg;base64,${data.result_image}`,
    });
    await loadOverview();
  } catch (error) {
    renderError(error);
  }
}

async function runNeuralStyle() {
  try {
    ensureFileSelected();
    const image = await toBase64(state.selectedFile);
    const styleNames = {
      starry_night: "Starry Night",
      great_wave: "Great Wave",
      the_scream: "The Scream",
      composition_viii: "Composition VIII",
    };
    const label = styleNames[state.selectedNeuralStyle] || state.selectedNeuralStyle;
    elements.resultMeta.innerHTML = `<strong>Стилізація: ${label}</strong><div class="list-meta">Обробляється… (~5–15 с)</div>`;
    elements.resultSection.hidden = false;
    elements.resultImage.src = "";
    elements.scoreList.innerHTML = "";

    const data = await apiRequest("/api/neural-stylize", {
      image,
      style: state.selectedNeuralStyle,
      iterations: 80,
      img_size: 256,
    });
    renderResult({
      title: `Стилізація: ${label}`,
      subtitle: "Neural Style Transfer на основі VGG19 (Gatys et al., 2015)",
      imageSrc: `data:image/jpeg;base64,${data.result_image}`,
    });
  } catch (error) {
    renderError(error);
  }
}

async function runStylize() {
  try {
    ensureFileSelected();
    const image = await toBase64(state.selectedFile);
    const data = await apiRequest("/api/stylize", {
      image,
      style: state.selectedStyle,
    });
    renderResult({
      title: `Фільтр: ${state.selectedStyle}`,
      subtitle: "Алгоритмічний фільтр (OpenCV)",
      imageSrc: `data:image/jpeg;base64,${data.result_image}`,
    });
  } catch (error) {
    renderError(error);
  }
}

function renderError(error) {
  elements.resultSection.hidden = false;
  elements.resultMeta.innerHTML = `<strong class="error-note">Помилка</strong><div class="list-meta error-note">${error.message}</div>`;
  elements.scoreList.innerHTML = '<div class="small-note">Перевірте стан моделей і коректність конфігурації API.</div>';
}

async function runBenchmark() {
  const dataset = elements.benchmarkDataset.value;
  const epochs = Number(elements.benchmarkEpochs.value || 100);
  elements.benchmarkOutput.textContent = "Запуск benchmark...";
  try {
    const result = await apiRequest("/api/benchmark/run", { dataset, epochs });
    elements.benchmarkOutput.textContent = `Фоновий запуск стартував: ${result.dataset}, epochs=${epochs}`;
    setTimeout(loadOverview, 1000);
  } catch (error) {
    elements.benchmarkOutput.textContent = error.message;
  }
}

async function loadOverview() {
  try {
    const data = await apiRequest("/api/dashboard/overview");
    renderHealth(data.health);
    renderTotals(data.totals);
    renderPredictions(data.recent_predictions);
    renderActivity(data.recent_activity);
    renderBenchmarks(data.recent_benchmarks);
  } catch (error) {
    elements.healthPill.textContent = "API недоступний";
    elements.healthPill.classList.remove("online");
    elements.modelState.textContent = error.message;
  }
}

function renderHealth(health) {
  const ready = Boolean(health.models_loaded);
  elements.healthPill.textContent = ready ? "API online" : "API online, models pending";
  elements.healthPill.classList.toggle("online", ready);
  elements.modelState.textContent = ready ? "Обидві моделі доступні для inference." : "Моделі ще не натреновані або шляхи не вказані.";
  elements.classifierPath.textContent = health.classifier_path;
  elements.colorizerPath.textContent = health.colorizer_path;
}

function renderTotals(totals) {
  Object.entries(totals).forEach(([key, value]) => {
    if (elements.totals[key]) {
      elements.totals[key].textContent = value;
    }
  });
}

function renderPredictions(items) {
  if (!items.length) {
    elements.predictionList.innerHTML = '<div class="small-note">Prediction history з’явиться після перших викликів classify/colorize.</div>';
    return;
  }
  elements.predictionList.innerHTML = items
    .map(item => {
      const confidence = item.confidence == null ? "без confidence" : `${(item.confidence * 100).toFixed(1)}%`;
      return `
        <div class="list-item">
          <strong>${item.model_type}</strong>
          <div>${item.label || "processed image"}</div>
          <div class="list-meta">${confidence} · ${formatDate(item.created_at)}</div>
        </div>
      `;
    })
    .join("");
}

function renderActivity(items) {
  if (!items.length) {
    elements.activityList.innerHTML = '<div class="small-note">Активність Telegram-бота з’явиться після перших команд.</div>';
    return;
  }
  elements.activityList.innerHTML = items
    .map(item => `
      <div class="list-item">
        <strong>${item.username ? `@${item.username}` : item.telegram_id || "user"}</strong>
        <div class="mono">${escapeHtml(item.command)}</div>
        <div class="list-meta">${formatDate(item.timestamp)}</div>
      </div>
    `)
    .join("");
}

function renderBenchmarks(items) {
  if (!items.length) {
    clearBenchmarkChart("Немає benchmark run для побудови графіка.");
    elements.benchmarkList.innerHTML = '<div class="small-note">Ще немає збережених benchmark run.</div>';
    return;
  }
  renderBenchmarkChart(items[0].results || []);
  elements.benchmarkList.innerHTML = items
    .map(item => {
      const rows = item.results
        .slice(0, 4)
        .map(result => {
          const width = Math.max(6, Math.round(result.accuracy * 100));
          return `
            <div class="list-item">
              <strong>${result.model}</strong>
              <div class="list-meta">${result.framework} · accuracy ${(result.accuracy * 100).toFixed(1)}% · F1 ${(result.f1_score * 100).toFixed(1)}%</div>
              <div class="benchmark-bar"><span style="width:${width}%"></span></div>
            </div>
          `;
        })
        .join("");
      return `
        <div class="list-item">
          <strong>${item.dataset}</strong>
          <div class="list-meta">${formatDate(item.created_at)}</div>
          <div class="benchmark-list">${rows}</div>
        </div>
      `;
    })
    .join("");
}

function clearBenchmarkChart(message) {
  const canvas = elements.benchmarkChart;
  if (!canvas) {
    return;
  }
  const context = canvas.getContext("2d");
  const width = canvas.width = canvas.clientWidth * window.devicePixelRatio;
  const height = canvas.height = canvas.clientHeight * window.devicePixelRatio;
  context.scale(window.devicePixelRatio, window.devicePixelRatio);
  context.clearRect(0, 0, width, height);
  context.fillStyle = "#6f655b";
  context.font = "14px Trebuchet MS";
  context.fillText(message, 12, 24);
}

function renderBenchmarkChart(results) {
  const canvas = elements.benchmarkChart;
  if (!canvas || !results.length) {
    clearBenchmarkChart("Немає даних.");
    return;
  }

  const logicalWidth = canvas.clientWidth || 640;
  const logicalHeight = 220;
  const ratio = window.devicePixelRatio || 1;
  canvas.width = logicalWidth * ratio;
  canvas.height = logicalHeight * ratio;

  const ctx = canvas.getContext("2d");
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.scale(ratio, ratio);
  ctx.clearRect(0, 0, logicalWidth, logicalHeight);

  const margin = { top: 24, right: 18, bottom: 54, left: 40 };
  const chartWidth = logicalWidth - margin.left - margin.right;
  const chartHeight = logicalHeight - margin.top - margin.bottom;
  const maxValue = 1;
  const groupWidth = chartWidth / results.length;
  const barWidth = Math.min(22, groupWidth / 3);

  ctx.strokeStyle = "rgba(31, 29, 27, 0.16)";
  ctx.lineWidth = 1;
  for (let step = 0; step <= 5; step += 1) {
    const y = margin.top + (chartHeight / 5) * step;
    ctx.beginPath();
    ctx.moveTo(margin.left, y);
    ctx.lineTo(logicalWidth - margin.right, y);
    ctx.stroke();
  }

  ctx.fillStyle = "#6f655b";
  ctx.font = "12px Trebuchet MS";
  for (let step = 0; step <= 5; step += 1) {
    const value = (1 - step / 5).toFixed(1);
    const y = margin.top + (chartHeight / 5) * step;
    ctx.fillText(value, 8, y + 4);
  }

  results.forEach((item, index) => {
    const xCenter = margin.left + groupWidth * index + groupWidth / 2;
    const accuracyHeight = Math.max(2, item.accuracy / maxValue * chartHeight);
    const f1Height = Math.max(2, item.f1_score / maxValue * chartHeight);
    const accX = xCenter - barWidth - 4;
    const f1X = xCenter + 4;

    ctx.fillStyle = "#0f7c78";
    ctx.fillRect(accX, margin.top + chartHeight - accuracyHeight, barWidth, accuracyHeight);

    ctx.fillStyle = "#bb4d00";
    ctx.fillRect(f1X, margin.top + chartHeight - f1Height, barWidth, f1Height);

    ctx.fillStyle = "#1f1d1b";
    ctx.font = "11px Trebuchet MS";
    const label = item.model.length > 12 ? `${item.model.slice(0, 12)}…` : item.model;
    ctx.save();
    ctx.translate(xCenter - 6, logicalHeight - 12);
    ctx.rotate(-0.45);
    ctx.fillText(label, 0, 0);
    ctx.restore();
  });

  ctx.fillStyle = "#0f7c78";
  ctx.fillRect(logicalWidth - 170, 12, 12, 12);
  ctx.fillStyle = "#1f1d1b";
  ctx.fillText("Accuracy", logicalWidth - 152, 22);
  ctx.fillStyle = "#bb4d00";
  ctx.fillRect(logicalWidth - 88, 12, 12, 12);
  ctx.fillStyle = "#1f1d1b";
  ctx.fillText("F1", logicalWidth - 68, 22);
}

function formatDate(value) {
  return new Date(value).toLocaleString("uk-UA");
}

function escapeHtml(value) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

init();
