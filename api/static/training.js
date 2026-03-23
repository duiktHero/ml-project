const trainingElements = {
  presetGrid: document.getElementById("preset-grid"),
  jobList: document.getElementById("job-list"),
  refreshButton: document.getElementById("training-refresh-button"),
  apiState: document.getElementById("training-api-state"),
  apiPill: document.getElementById("training-api-pill"),
  pythonNote: document.getElementById("training-python-note"),
};

async function trainingRequest(path, payload) {
  const response = await fetch(path, {
    method: payload ? "POST" : "GET",
    headers: payload ? { "Content-Type": "application/json" } : undefined,
    body: payload ? JSON.stringify(payload) : undefined,
  });
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.detail || `HTTP ${response.status}`);
  }
  return data;
}

async function loadTrainingPage() {
  await Promise.all([loadPresets(), loadJobs()]);
  trainingElements.apiState.textContent = "Training routes доступні.";
  trainingElements.apiPill.textContent = "online";
  trainingElements.apiPill.classList.add("online");
}

async function loadPresets() {
  try {
    const data = await trainingRequest("/api/training/presets");
    renderPresets(data.presets);
  } catch (error) {
    trainingElements.presetGrid.innerHTML = `<div class="small-note error-note">${error.message}</div>`;
    throw error;
  }
}

async function loadJobs() {
  try {
    const data = await trainingRequest("/api/training/jobs");
    renderJobs(data.jobs);
    if (data.jobs.length) {
      const job = data.jobs[0];
      trainingElements.pythonNote.textContent = `${job.runner_label} -> ${job.command.join(" ")}`;
    }
  } catch (error) {
    trainingElements.jobList.innerHTML = `<div class="small-note error-note">${error.message}</div>`;
    throw error;
  }
}

function renderPresets(presets) {
  if (!presets.length) {
    trainingElements.presetGrid.innerHTML = '<div class="small-note">Немає доступних presets.</div>';
    return;
  }

  trainingElements.presetGrid.innerHTML = presets.map(preset => `
    <div class="preset-card">
      <div>
        <div class="metric-label">${preset.id}</div>
        <h3>${preset.title}</h3>
        <p>${preset.description}</p>
        <div class="list-meta">runner: ${preset.runner_label}</div>
      </div>
      <button class="primary" data-preset-id="${preset.id}">Запустити</button>
    </div>
  `).join("");

  trainingElements.presetGrid.querySelectorAll("button[data-preset-id]").forEach(button => {
    button.addEventListener("click", async () => {
      button.disabled = true;
      try {
        await trainingRequest("/api/training/start", { preset: button.dataset.presetId });
        await loadJobs();
      } catch (error) {
        alert(error.message);
      } finally {
        button.disabled = false;
      }
    });
  });
}

function renderJobs(jobs) {
  if (!jobs.length) {
    trainingElements.jobList.innerHTML = '<div class="small-note">Ще немає жодного training job.</div>';
    return;
  }

  trainingElements.jobList.innerHTML = jobs.map(job => {
    const statusClass = job.status === "completed" ? "job-completed" : job.status === "failed" ? "job-failed" : "job-running";
    const args = job.command.slice(2).join(" ");
    const progress = renderProgress(job.progress);
    const logTail = renderLogTail(job.log_tail);
    return `
      <div class="list-item ${statusClass}">
        <strong>${job.title}</strong>
        <div class="list-meta">${job.status} · ${job.runner_label} · PID ${job.pid} · start ${formatJobDate(job.started_at)}</div>
        <div class="mono">scripts/run_local.py ${escapeHtml(args)}</div>
        <div class="list-meta">log: ${job.log_path}</div>
        ${progress}
        ${logTail}
        ${job.finished_at ? `<div class="list-meta">finish: ${formatJobDate(job.finished_at)} · return code: ${job.return_code}</div>` : ""}
      </div>
    `;
  }).join("");
}

function renderProgress(progress) {
  if (!progress) {
    return "";
  }

  const metricEntries = Object.entries(progress.metrics || {});
  const metrics = metricEntries.length
    ? `<div class="list-meta">metrics: ${metricEntries.map(([key, value]) => `${escapeHtml(key)}=${value}`).join(" · ")}</div>`
    : "";
  const runtime = progress.runtime ? `<div class="list-meta">runtime: ${escapeHtml(progress.runtime)}</div>` : "";

  return `
    <div class="job-progress">
      <div class="list-meta">stage: ${escapeHtml(progress.stage || "-")} · epoch ${progress.epoch || 0}/${progress.total_epochs || 0}</div>
      <div class="list-meta">elapsed: ${formatDuration(progress.elapsed_seconds)} · epoch time: ${formatDuration(progress.epoch_seconds)} · ETA: ${formatDuration(progress.eta_seconds)}</div>
      ${metrics}
      ${runtime}
    </div>
  `;
}

function renderLogTail(lines) {
  if (!lines || !lines.length) {
    return "";
  }
  return `<pre class="job-log">${escapeHtml(lines.join("\n"))}</pre>`;
}

function formatJobDate(value) {
  return new Date(value).toLocaleString("uk-UA");
}

function formatDuration(value) {
  if (value === null || value === undefined) {
    return "-";
  }
  const seconds = Number(value);
  if (!Number.isFinite(seconds)) {
    return "-";
  }
  if (seconds < 60) {
    return `${seconds.toFixed(1)}s`;
  }
  const minutes = Math.floor(seconds / 60);
  const rest = Math.round(seconds % 60);
  return `${minutes}m ${rest}s`;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

trainingElements.refreshButton.addEventListener("click", () => {
  loadTrainingPage().catch(error => {
    trainingElements.apiState.textContent = error.message;
    trainingElements.apiPill.classList.remove("online");
  });
});

loadTrainingPage().catch(error => {
  trainingElements.apiState.textContent = error.message;
  trainingElements.apiPill.classList.remove("online");
});
setInterval(() => loadJobs().catch(() => {}), 8000);
