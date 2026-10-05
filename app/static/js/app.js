/* 智诊AI —— 多模块单页应用
 * 通过 hash 路由在「诊断任务 / 错题结构化 / 根因诊断 / 知识图谱回溯 /
 * 跨题聚合盲区 / 诊断过程 / 3D 干预展示」之间切换，每个模块独立成页。
 */

const PAGES = ["task", "ocr", "diagnosis", "graph", "history", "process", "viewer"];

const state = {
  hasRun: false,
  running: false,
  ocrResult: null,
  diagnosisResult: null,
  graphSuggestion: null,
  graphMutationRunning: false,
  graphPrompted: false,
  warnings: [],
  interventionResult: null,
  historySummary: null,
  processSteps: initialProcessSteps(),
  previewObjectUrl: null,
  previewName: "",
  errorMessage: null,
};

const viewerState = {
  scene: null,
  camera: null,
  renderer: null,
  controls: null,
  rootGroup: null,
  sceneType: null,
  stepIndex: 0,
  exploded: false,
  animatedParts: [],
  magnetArrows: [],
  initialized: false,
  ready: false,
};

const form = document.getElementById("pipeline-form");
const submitButton = document.getElementById("submit-button");
const imageInput = document.getElementById("image-input");
const imagePreview = document.getElementById("image-preview");
const imagePreviewName = document.getElementById("image-preview-name");

bindEvents();
loadHistorySummary();
renderPage(currentPage());

/* ---------- 路由 ---------- */

function currentPage() {
  const hash = window.location.hash.replace(/^#\/?/, "");
  return PAGES.includes(hash) ? hash : "task";
}

function navigate(page) {
  if (page === currentPage()) {
    renderPage(page);
    return;
  }
  window.location.hash = "#/" + page;
}

function renderPage(page) {
  document.querySelectorAll(".page").forEach((section) => {
    section.classList.toggle("active", section.dataset.page === page);
  });
  document.querySelectorAll(".nav-item").forEach((item) => {
    item.classList.toggle("active", item.dataset.page === page);
  });
  document.title = pageTitle(page);
  refreshReadiness();

  if (page === "task") refreshTask();
  if (page === "ocr") refreshOCR();
  if (page === "diagnosis") refreshDiagnosis();
  if (page === "graph") refreshGraph();
  if (page === "history") refreshHistory();
  if (page === "process") refreshProcess();
  if (page === "viewer") {
    refreshViewer();
    requestAnimationFrame(() => resizeViewer());
  }
}

function pageTitle(page) {
  const map = {
    task: "诊断任务",
    ocr: "错题结构化",
    diagnosis: "根因诊断",
    graph: "知识图谱回溯",
    history: "跨题聚合盲区",
    process: "诊断过程",
    viewer: "3D 干预展示",
  };
  return `${map[page] || "智诊AI"} · 智诊AI`;
}

function refreshReadiness() {
  document.querySelectorAll(".nav-item").forEach((item) => {
    item.classList.toggle("ready", isReady(item.dataset.page));
  });
}

function isReady(page) {
  switch (page) {
    case "ocr":
      return !!state.ocrResult;
    case "diagnosis":
      return !!state.diagnosisResult;
    case "graph":
      return !!(
        state.diagnosisResult &&
        ((state.diagnosisResult.graph_snapshot && state.diagnosisResult.graph_snapshot.nodes?.length) ||
          state.diagnosisResult.graph_paths?.length)
      );
    case "history":
      return !!(state.historySummary && state.historySummary.total_records > 0);
    case "process":
      return state.hasRun;
    case "viewer":
      return !!state.interventionResult;
    default:
      return false;
  }
}

/* ---------- 事件绑定 ---------- */

function bindEvents() {
  form.addEventListener("submit", handleSubmit);
  imageInput.addEventListener("change", handleImagePreview);
  window.addEventListener("hashchange", () => renderPage(currentPage()));
  window.addEventListener("resize", resizeViewer);
  document.addEventListener("click", handleGraphAction);
  document.querySelectorAll("[data-control]").forEach((button) => {
    button.addEventListener("click", () => handleViewerControl(button.dataset.control));
  });
  const llmProvider = document.getElementById("llm-provider");
  const llmModel = document.getElementById("llm-model");
  llmProvider.addEventListener("change", () => {
    llmModel.value = llmProvider.value === "deepseek" ? "deepseek-chat" : "qwen2.5:7b";
  });
}

/* ---------- 任务提交 ---------- */

async function handleSubmit(event) {
  event.preventDefault();
  if (state.running) {
    return;
  }
  const formData = new FormData(form);
  state.hasRun = true;
  state.running = true;
  state.errorMessage = null;
  state.graphSuggestion = null;
  state.graphPrompted = false;
  state.processSteps = runningProcessSteps();
  setStatus("处理中", "正在执行 OCR、知识图谱回溯、大模型诊断与 3D 干预生成。");
  submitButton.disabled = true;
  submitButton.textContent = "诊断中…";
  navigate("process");

  try {
    const response = await fetch("/api/pipeline/stream", { method: "POST", body: formData });
    if (!response.ok) {
      throw new Error("请求失败，状态码 " + response.status);
    }
    if (!response.body) {
      throw new Error("当前浏览器不支持流式返回。");
    }
    await consumeStreamResponse(response.body);
    setStatus("诊断完成", "已完成从 OCR 到 3D 干预展示的完整闭环。");
  } catch (error) {
    state.errorMessage = error.message;
    state.processSteps = failedProcessSteps(error.message);
    setStatus("执行失败", error.message);
  } finally {
    state.running = false;
    submitButton.disabled = false;
    submitButton.textContent = "开始诊断";
    renderPage(currentPage());
  }
}

async function consumeStreamResponse(body) {
  const reader = body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) {
      break;
    }
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";
    for (const line of lines) {
      if (line.trim()) {
        handleStreamEvent(JSON.parse(line));
      }
    }
  }
  const tail = buffer.trim();
  if (tail) {
    handleStreamEvent(JSON.parse(tail));
  }
}

function handleStreamEvent(event) {
  switch (event.type) {
    case "process_update":
      state.processSteps = event.process_steps || [];
      setStatus("处理中", "已上传图片，正在进入 OCR 结构化流程。");
      break;
    case "ocr_result":
      state.ocrResult = event.ocr_result;
      state.processSteps = event.process_steps || [];
      setStatus("OCR 完成", "已完成图片结构化识别，正在回溯知识图谱。");
      break;
    case "diagnosis_result":
      state.diagnosisResult = event.diagnostic_result;
      state.warnings = event.warnings || [];
      state.processSteps = event.process_steps || [];
      setStatus("诊断中", "已定位前置漏洞，正在生成 3D 干预内容。");
      break;
    case "scene_result":
      state.interventionResult = event.intervention_result;
      state.processSteps = event.process_steps || [];
      setStatus("3D 生成中", "已生成 3D 干预内容，正在整理最终结果。");
      break;
    case "complete":
      applyPayload(event.payload);
      setStatus("诊断完成", "已完成从 OCR 到 3D 干预展示的完整闭环。");
      break;
    case "error":
      throw new Error(event.message || "流式任务执行失败。");
    default:
      break;
  }
  renderPage(currentPage());
}

function applyPayload(payload) {
  state.ocrResult = payload.ocr_result;
  state.diagnosisResult = payload.diagnostic_result;
  state.graphSuggestion = null;
  state.graphPrompted = false;
  state.warnings = payload.warnings || [];
  state.interventionResult = payload.intervention_result;
  state.historySummary = payload.history_summary;
  state.processSteps = payload.process_steps || [];
}

/* ---------- 状态栏 ---------- */

function setStatus(label, hint) {
  document.getElementById("system-status").textContent = label;
  document.getElementById("status-hint").textContent = hint;
  const card = document.querySelector(".status-card");
  if (card) {
    const text = label || "";
    let mode = "idle";
    if (text.includes("失败") || text.includes("错误")) mode = "error";
    else if (text.includes("完成")) mode = "done";
    else if (text.includes("中") || text.includes("处理")) mode = "running";
    card.dataset.state = mode;
  }
}

/* ---------- 各模块渲染 ---------- */

function refreshTask() {
  submitButton.disabled = state.running;
  submitButton.textContent = state.running ? "诊断中…" : "开始诊断";
}

function refreshOCR() {
  const tag = document.getElementById("ocr-provider-tag");
  const el = document.getElementById("ocr-result");
  const summary = document.getElementById("analysis-summary");
  if (state.ocrResult) {
    tag.textContent = state.ocrResult.provider;
    renderOCR(state.ocrResult);
    summary.textContent = state.ocrResult.raw_summary || "暂无摘要。";
  } else {
    tag.textContent = "OCR";
    el.innerHTML = `<div class="empty-state">尚未上传作业图片，请前往「诊断任务」提交。</div>`;
    summary.textContent = "等待开始识图分析。";
  }
  renderOCRImage();
}

function renderOCRImage() {
  const box = document.getElementById("ocr-image");
  if (state.previewObjectUrl) {
    box.innerHTML = `<img src="${state.previewObjectUrl}" alt="上传预览">`;
    box.classList.remove("empty-state");
  } else {
    box.innerHTML = "暂无图片";
    box.classList.add("empty-state");
  }
}

function refreshDiagnosis() {
  const tag = document.getElementById("llm-provider-tag");
  const el = document.getElementById("diagnosis-result");
  if (state.diagnosisResult) {
    tag.textContent = state.diagnosisResult.provider;
    renderDiagnosis(state.diagnosisResult, state.warnings);
  } else {
    tag.textContent = "LLM";
    el.innerHTML = `<div class="empty-state">完成诊断后，这里将展示错误层级、前置漏洞和证据链。</div>`;
  }
}

function refreshGraph() {
  const el = document.getElementById("graph-result");
  if (
    state.diagnosisResult &&
    ((state.diagnosisResult.graph_snapshot && state.diagnosisResult.graph_snapshot.nodes?.length) ||
      state.diagnosisResult.graph_paths?.length)
  ) {
    renderGraph(state.diagnosisResult.graph_snapshot, state.diagnosisResult.graph_paths || []);
  } else {
    queueMissingGraphPrompt();
    el.innerHTML = renderMissingGraphPanel();
  }
}

function refreshHistory() {
  const el = document.getElementById("history-result");
  if (state.historySummary && state.historySummary.top_blind_spots?.length) {
    renderHistory(state.historySummary);
  } else {
    el.innerHTML = `<div class="empty-state">至少完成一次诊断后，这里会展示累计盲区。</div>`;
  }
}

function refreshProcess() {
  renderProcess(state.processSteps);
}

function refreshViewer() {
  const title = document.getElementById("scene-title");
  const description = document.getElementById("scene-description");
  const prompt = document.getElementById("scene-prompt");
  const subtitle = document.getElementById("viewer-subtitle");
  const steps = document.getElementById("scene-steps");
  const onlineResult = document.getElementById("scene-online-result");
  const modelLink = document.getElementById("scene-model-link");
  const previewWrap = document.getElementById("scene-preview-wrap");
  const previewImage = document.getElementById("scene-preview-image");

  if (!state.interventionResult) {
    title.textContent = "等待生成模型";
    description.textContent = "完成诊断后，这里会展示 3D 干预说明。";
    prompt.textContent = "暂无";
    subtitle.textContent = "诊断完成后，这里会展示对应的 3D 干预模型与讲解步骤。";
    steps.innerHTML = "";
    onlineResult.classList.add("hidden");
    modelLink.classList.add("hidden");
    modelLink.removeAttribute("href");
    previewWrap.classList.add("hidden");
    previewImage.removeAttribute("src");
    return;
  }

  const recipe = state.interventionResult.scene_recipe;
  title.textContent = recipe.title;
  description.textContent = recipe.description;
  prompt.textContent = state.interventionResult.prompt;
  subtitle.textContent = state.interventionResult.generation_status;
  if (state.interventionResult.model_download_url || state.interventionResult.preview_image_url) {
    onlineResult.classList.remove("hidden");
    if (state.interventionResult.model_download_url) {
      modelLink.classList.remove("hidden");
      modelLink.href = state.interventionResult.model_download_url;
    } else {
      modelLink.classList.add("hidden");
      modelLink.removeAttribute("href");
    }
    if (state.interventionResult.preview_image_url) {
      previewWrap.classList.remove("hidden");
      previewImage.src = state.interventionResult.preview_image_url;
    } else {
      previewWrap.classList.add("hidden");
      previewImage.removeAttribute("src");
    }
  } else {
    onlineResult.classList.add("hidden");
    modelLink.classList.add("hidden");
    modelLink.removeAttribute("href");
    previewWrap.classList.add("hidden");
    previewImage.removeAttribute("src");
  }
  viewerState.stepIndex = 0;
  steps.innerHTML = (recipe.steps || [])
    .map((item, index) => `<div class="step-item ${index === 0 ? "active" : ""}">${index + 1}. ${escapeHtml(item)}</div>`)
    .join("");

  if (document.querySelector('.page[data-page="viewer"]').classList.contains("active")) {
    ensureViewer();
    if (viewerState.ready) {
      buildScene(recipe);
    }
  }
}

/* ---------- 结果渲染 ---------- */

function renderOCR(result) {
  const el = document.getElementById("ocr-result");
  const question = result.questions && result.questions[0];
  if (!question) {
    el.innerHTML = `<div class="empty-state">未提取到题目结构。</div>`;
    return;
  }
  el.innerHTML = `
    <div class="result-card">
      <div class="metric"><span>题目文本</span><strong>${escapeHtml(question.question_text)}</strong></div>
      <div class="metric"><span>学生答案</span><strong>${escapeHtml(question.student_answer)}</strong></div>
      <div class="metric"><span>正确答案</span><strong>${escapeHtml(question.correct_answer)}</strong></div>
      <div class="metric"><span>知识点标签</span>
        <div class="tag-row">${(question.knowledge_points || []).map((item) => `<span class="tag">${escapeHtml(item)}</span>`).join("")}</div>
      </div>
      <div class="metric"><span>错误步骤定位</span>
        <div class="tag-row">${(question.error_steps || []).map((item) => `<span class="tag">${escapeHtml(item)}</span>`).join("")}</div>
      </div>
    </div>
  `;
}

function renderDiagnosis(result, warnings) {
  const el = document.getElementById("diagnosis-result");
  const warningHtml = (warnings || [])
    .map((item) => `<div class="warning-card">${escapeHtml(item)}</div>`)
    .join("");
  const confidenceHtml = (result.confidence_distribution || [])
    .map((item) => {
      const percent = Math.round((item.confidence || 0) * 100);
      return `
        <div class="result-card confidence-item">
          <strong>${escapeHtml(item.knowledge_point)}</strong>
          <div>${escapeHtml(item.reason)}</div>
          <div class="confidence-bar"><span style="width:${percent}%"></span></div>
          <small>置信度 ${percent}%</small>
        </div>
      `;
    })
    .join("");
  const evidenceHtml = (result.evidence || [])
    .map((item) => `<li>${escapeHtml(item)}</li>`)
    .join("");
  el.innerHTML = `
    ${warningHtml}
    <div class="result-card">
      <div class="metric"><span>错误层级</span><strong>${escapeHtml(result.error_level)}</strong></div>
      <div class="metric"><span>前置漏洞</span><strong>${escapeHtml(result.prerequisite_gap)}</strong></div>
      <div class="metric"><span>补救建议</span><strong>${escapeHtml(result.advice)}</strong></div>
    </div>
    <div class="confidence-list">${confidenceHtml}</div>
    <ul class="evidence-list">${evidenceHtml}</ul>
  `;
}

function renderGraph(snapshot, paths) {
  const el = document.getElementById("graph-result");
  if (!snapshot || !snapshot.nodes || !snapshot.nodes.length) {
    el.innerHTML = renderGraphPaths(paths || []);
    return;
  }

  const summaryItems = [
    { tone: "focus", label: "当前知识点", value: snapshot.focus_node || "未识别" },
    { tone: "gap", label: "前置漏洞", value: snapshot.prerequisite_gap || "未定位" },
    { tone: "nodes", label: "节点数", value: snapshot.node_count || 0 },
    { tone: "edges", label: "关系边", value: snapshot.edge_count || 0 },
  ];
  const summaryHtml = summaryItems
    .map((item) => `
      <div class="result-card graph-summary-card tone-${item.tone}">
        <span>${escapeHtml(item.label)}</span>
        <strong>${escapeHtml(String(item.value))}</strong>
      </div>
    `)
    .join("");

  const graphHtml = renderGraphSnapshot(snapshot, paths || []);
  const pathHtml = renderGraphPaths(paths || []);
  el.innerHTML = `
    <div class="graph-summary-grid">${summaryHtml}</div>
    ${graphHtml}
    <div class="graph-path-stack">
      <div class="panel-header graph-subhead">
        <h3>依赖路径</h3>
        <span class="badge badge-soft">Paths</span>
      </div>
      ${pathHtml}
    </div>
  `;
}

function renderMissingGraphPanel() {
  if (!state.ocrResult?.questions?.length) {
    return `<div class="empty-state">这里会展示从当前知识点回溯到前置漏洞的路径。</div>`;
  }
  const knowledgePoints = state.ocrResult.questions[0].knowledge_points || [];
  const missingInfo = state.graphSuggestion
    ? renderGraphSuggestion(state.graphSuggestion)
    : `
      <div class="graph-missing-actions">
        <button class="action-link" data-graph-action="suggest">将该知识点加入现有图谱</button>
      </div>
    `;
  return `
    <div class="result-card graph-missing-card">
      <div class="panel-header graph-subhead">
        <div>
          <strong>当前题目暂未命中可展示的 DAG</strong>
          <div class="panel-sub">可以把这次识别出的新知识点补充到现有 K12-KGraph 合并图谱中，后续相关题目就能继续回溯。</div>
        </div>
        <span class="badge badge-soft">待补图谱</span>
      </div>
      <div class="graph-missing-tags">
        ${(knowledgePoints.length ? knowledgePoints : ["未提取到知识点"]).map((item) => `<span class="tag">${escapeHtml(item)}</span>`).join("")}
      </div>
      ${missingInfo}
    </div>
  `;
}

function queueMissingGraphPrompt() {
  if (state.graphPrompted || state.graphMutationRunning || !state.ocrResult?.questions?.length) {
    return;
  }
  state.graphPrompted = true;
  window.setTimeout(() => {
    const shouldSuggest = window.confirm("当前知识点还没有可展示的 DAG，是否尝试把它加入现有知识图谱？");
    if (shouldSuggest) {
      requestGraphSuggestion();
    } else {
      refreshGraph();
    }
  }, 80);
}

function renderGraphSuggestion(suggestionPayload) {
  const suggestions = suggestionPayload.suggestions || [];
  const cards = suggestions.length
    ? suggestions.map((item) => `
        <div class="graph-suggest-card">
          <div class="graph-suggest-head">
            <strong>${escapeHtml(item.name)}</strong>
            <span class="badge badge-soft">${escapeHtml(item.source || "候选")}</span>
          </div>
          <div class="graph-suggest-desc">${escapeHtml(item.description || "暂无说明")}</div>
          ${renderSuggestionField("别名", item.aliases)}
          ${renderSuggestionField("前置依赖", item.prerequisites)}
          ${renderSuggestionField("相关连接", item.related_to)}
          ${renderSuggestionField("从属关系", item.is_a)}
          ${renderSuggestionField("相近已有节点", item.similar_existing_nodes)}
        </div>
      `).join("")
    : `<div class="empty-state">还没生成候选补图方案。</div>`;
  const loadingText = state.graphMutationRunning ? "处理中…" : "确认加入图谱";
  return `
    <div class="graph-suggest-wrap">
      <div class="panel-sub">${escapeHtml(suggestionPayload.message || "已生成候选补图方案，请确认是否写入图谱。")}</div>
      <div class="graph-suggest-list">${cards}</div>
      <div class="graph-missing-actions">
        <button class="action-link" data-graph-action="confirm-add" ${state.graphMutationRunning ? "disabled" : ""}>${loadingText}</button>
        <button class="ghost-button" data-graph-action="reset-suggest" ${state.graphMutationRunning ? "disabled" : ""}>重新生成</button>
      </div>
    </div>
  `;
}

function renderSuggestionField(label, items) {
  if (!items || !items.length) {
    return "";
  }
  return `
    <div class="graph-suggest-field">
      <span>${escapeHtml(label)}</span>
      <div class="graph-missing-tags">${items.map((item) => `<span class="tag">${escapeHtml(item)}</span>`).join("")}</div>
    </div>
  `;
}

async function handleGraphAction(event) {
  const button = event.target.closest("[data-graph-action]");
  if (!button) return;
  const action = button.dataset.graphAction;
  if (action === "suggest") {
    await requestGraphSuggestion();
  } else if (action === "confirm-add") {
    await confirmGraphAddition();
  } else if (action === "reset-suggest") {
    state.graphSuggestion = null;
    refreshGraph();
  }
}

async function requestGraphSuggestion() {
  if (!state.ocrResult?.questions?.length || state.graphMutationRunning) return;
  state.graphMutationRunning = true;
  refreshGraph();
  try {
    const question = state.ocrResult.questions[0];
    const response = await fetch("/api/graph/suggest", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question_text: question.question_text || "",
        subject: state.ocrResult.subject || "",
        grade: state.ocrResult.grade || "",
        knowledge_points: question.knowledge_points || [],
        llm_provider: document.getElementById("llm-provider").value || "",
        llm_model: document.getElementById("llm-model").value || "",
        ollama_base_url: document.getElementById("ollama-base-url").value || "",
        deepseek_base_url: document.getElementById("deepseek-base-url").value || "",
      }),
    });
    if (!response.ok) {
      throw new Error("生成补图建议失败，状态码 " + response.status);
    }
    state.graphSuggestion = await response.json();
  } catch (error) {
    state.graphSuggestion = {
      suggestions: [],
      message: error.message || "生成补图建议失败。",
    };
  } finally {
    state.graphMutationRunning = false;
    refreshGraph();
  }
}

async function confirmGraphAddition() {
  if (!state.graphSuggestion?.suggestions?.length || state.graphMutationRunning) return;
  state.graphMutationRunning = true;
  refreshGraph();
  try {
    const question = state.ocrResult.questions[0];
    const response = await fetch("/api/graph/add", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        suggestions: state.graphSuggestion.suggestions,
        focus_knowledge_points: question.knowledge_points || [],
        prerequisite_gap: state.diagnosisResult?.prerequisite_gap || "",
      }),
    });
    if (!response.ok) {
      throw new Error("写入图谱失败，状态码 " + response.status);
    }
    const payload = await response.json();
    state.diagnosisResult = state.diagnosisResult || {};
    state.diagnosisResult.graph_paths = payload.graph_paths || [];
    state.diagnosisResult.graph_snapshot = payload.graph_snapshot || {};
    state.graphSuggestion = {
      suggestions: [],
      message: payload.message || "已写入图谱。",
    };
  } catch (error) {
    state.graphSuggestion = {
      ...(state.graphSuggestion || { suggestions: [] }),
      message: error.message || "写入图谱失败。",
    };
  } finally {
    state.graphMutationRunning = false;
    refreshReadiness();
    refreshGraph();
  }
}

function renderGraphSnapshot(snapshot, paths) {
  const nodes = snapshot.nodes || [];
  const edges = snapshot.edges || [];
  const { layoutNodes, width, height } = buildGraphLayout(nodes);
  const nodeMap = new Map(layoutNodes.map((item) => [item.name, item]));
  const activeEdgeKeys = collectActiveEdgeKeys(paths);

  const markerDefs = `
    <defs>
      ${graphMarker("prerequisites_for", "#60a5fa")}
      ${graphMarker("relates_to", "#f59e0b")}
      ${graphMarker("is_a", "#22c55e")}
    </defs>
  `;

  const edgeHtml = edges
    .map((edge) => {
      const source = nodeMap.get(edge.source);
      const target = nodeMap.get(edge.target);
      if (!source || !target) {
        return "";
      }
      const relation = edge.relation || "relates_to";
      const activeClass = activeEdgeKeys.has(edge.source + "->" + edge.target) ? " edge-active" : "";
      const startX = source.x + source.width;
      const startY = source.y + source.height / 2;
      const endX = target.x;
      const endY = target.y + target.height / 2;
      const midX = startX + (endX - startX) / 2;
      return `
        <path class="graph-edge edge-${escapeHtml(relation)}${activeClass}"
              d="M ${startX} ${startY} C ${midX} ${startY}, ${midX} ${endY}, ${endX} ${endY}"
              marker-end="url(#arrow-${escapeHtml(relation)})" />
      `;
    })
    .join("");

  const nodeHtml = layoutNodes
    .map((node, index) => `
      <div class="graph-node role-${escapeHtml(node.role)}"
           style="left:${node.x}px;top:${node.y}px;width:${node.width}px;height:${node.height}px;animation-delay:${index * 70}ms;">
        <div class="graph-node-top">
          <span class="graph-role">${escapeHtml(graphRoleLabel(node.role))}</span>
          <span class="graph-type">${escapeHtml(node.node_type || "知识点")}</span>
        </div>
        <strong class="graph-node-name">${escapeHtml(node.name)}</strong>
        <small class="graph-node-subject">${escapeHtml(node.subject || "K12-KGraph")}</small>
        <div class="graph-node-desc">${escapeHtml(truncateText(node.description || "暂无说明。", 58))}</div>
      </div>
    `)
    .join("");

  const edgeLegend = [
    { relation: "prerequisites_for", label: "前置依赖" },
    { relation: "relates_to", label: "相关概念" },
    { relation: "is_a", label: "从属关系" },
  ]
    .map((item) => `<span class="graph-legend-item"><i class="legend-line edge-${escapeHtml(item.relation)}"></i>${escapeHtml(item.label)}</span>`)
    .join("");

  return `
    <div class="graph-board-card result-card">
      <div class="graph-board-head">
        <div>
          <strong>回溯子图</strong>
          <div class="panel-sub">除了前置链，还展示当前知识点的直接后继和关联节点。</div>
        </div>
        <div class="graph-legend">${edgeLegend}</div>
      </div>
      <div class="graph-axis">
        <span>前置基础</span>
        <span class="graph-axis-arrow">依赖方向 →</span>
        <span>当前知识点</span>
      </div>
      <div class="graph-stage-wrap">
        <div class="graph-stage" style="width:${width}px;height:${height}px;">
          <svg class="graph-svg" viewBox="0 0 ${width} ${height}" preserveAspectRatio="none">
            ${markerDefs}
            ${edgeHtml}
          </svg>
          ${nodeHtml}
        </div>
      </div>
    </div>
  `;
}

function graphMarker(id, color) {
  return `<marker id="arrow-${id}" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M 0 0 L 10 5 L 0 10 z" fill="${color}"/>
  </marker>`;
}

function collectActiveEdgeKeys(paths) {
  const keys = new Set();
  (paths || []).forEach((path) => {
    for (let index = 0; index < path.length - 1; index += 1) {
      keys.add(path[index] + "->" + path[index + 1]);
      keys.add(path[index + 1] + "->" + path[index]);
    }
  });
  return keys;
}

function renderGraphPaths(paths) {
  if (!paths || !paths.length) {
    return `<div class="empty-state">暂无可展示的依赖路径。</div>`;
  }
  return paths
    .map((path) => {
      const nodes = path.map((item) => `<span class="tag">${escapeHtml(item)}</span>`).join('<span class="arrow">→</span>');
      return `<div class="path-card">${nodes}</div>`;
    })
    .join("");
}

function buildGraphLayout(nodes) {
  const laneMap = new Map();
  const orderedLayers = Array.from(new Set(nodes.map((item) => item.layer || 0))).sort((a, b) => a - b);
  orderedLayers.forEach((layer) => laneMap.set(layer, []));
  nodes.forEach((node) => {
    laneMap.get(node.layer || 0).push(node);
  });
  laneMap.forEach((items) => {
    items.sort((a, b) => graphRoleWeight(a.role) - graphRoleWeight(b.role) || a.name.localeCompare(b.name, "zh-Hans-CN"));
  });

  const nodeWidth = 210;
  const nodeHeight = 124;
  const laneGap = 60;
  const rowGap = 28;
  const padding = 24;
  const maxRows = Math.max(...Array.from(laneMap.values()).map((items) => items.length), 1);
  const width = padding * 2 + orderedLayers.length * nodeWidth + Math.max(orderedLayers.length - 1, 0) * laneGap;
  const height = padding * 2 + maxRows * nodeHeight + Math.max(maxRows - 1, 0) * rowGap;

  const layoutNodes = [];
  orderedLayers.forEach((layer, laneIndex) => {
    const items = laneMap.get(layer) || [];
    items.forEach((node, rowIndex) => {
      layoutNodes.push({
        ...node,
        width: nodeWidth,
        height: nodeHeight,
        x: padding + laneIndex * (nodeWidth + laneGap),
        y: padding + rowIndex * (nodeHeight + rowGap),
      });
    });
  });
  return { layoutNodes, width, height };
}

function graphRoleWeight(role) {
  const order = { gap: 0, focus: 1, path: 2, next: 3, related: 4 };
  return order[role] ?? 99;
}

function graphRoleLabel(role) {
  const map = {
    gap: "前置漏洞",
    focus: "当前知识点",
    path: "依赖链节点",
    next: "后续节点",
    related: "关联节点",
  };
  return map[role] || "知识点";
}

function truncateText(text, maxLength) {
  if (!text || text.length <= maxLength) {
    return text || "";
  }
  return text.slice(0, maxLength) + "…";
}

function renderHistory(summary) {
  const el = document.getElementById("history-result");
  const items = (summary.top_blind_spots || [])
    .map((item) => `<div class="history-card"><strong>${escapeHtml(item.knowledge_point)}</strong><div>出现次数：${item.count}</div></div>`)
    .join("");
  el.innerHTML = `
    <div class="result-card"><div class="metric"><span>累计记录</span><strong>${summary.total_records} 次</strong></div></div>
    ${items}
  `;
}

function renderProcess(steps) {
  const el = document.getElementById("process-result");
  if (!steps || !steps.length) {
    el.innerHTML = `<div class="empty-state">暂无过程数据。</div>`;
    return;
  }
  el.innerHTML = `
    <div class="process-list">
      ${steps.map((step) => `
        <div class="process-card is-${escapeHtml(step.status)}">
          <div class="process-title">
            <strong>${escapeHtml(step.label)}</strong>
            <span class="process-status ${escapeHtml(step.status)}">${processLabel(step.status)}</span>
          </div>
          <div>${escapeHtml(step.detail || "")}</div>
        </div>
      `).join("")}
    </div>
  `;
}

/* ---------- 图片预览 ---------- */

function handleImagePreview() {
  const file = imageInput.files && imageInput.files[0];
  if (state.previewObjectUrl) {
    URL.revokeObjectURL(state.previewObjectUrl);
    state.previewObjectUrl = null;
  }
  if (!file) {
    state.previewName = "";
    imagePreviewName.textContent = "未选择文件";
    imagePreview.innerHTML = "选择图片后将在这里预览。";
    imagePreview.classList.add("empty-state");
    return;
  }
  state.previewName = file.name;
  imagePreviewName.textContent = `${file.name} · ${formatFileSize(file.size)}`;
  state.previewObjectUrl = URL.createObjectURL(file);
  imagePreview.innerHTML = `<img src="${state.previewObjectUrl}" alt="上传预览">`;
  imagePreview.classList.remove("empty-state");
}

/* ---------- 历史摘要预加载 ---------- */

async function loadHistorySummary() {
  try {
    const response = await fetch("/api/history/summary");
    if (response.ok) {
      state.historySummary = await response.json();
      refreshReadiness();
    }
  } catch (error) {
    /* 忽略预加载失败，历史页会在诊断完成后更新 */
  }
}

/* ---------- 3D 查看器 ---------- */

function ensureViewer() {
  if (viewerState.initialized) {
    resizeViewer();
    return;
  }
  try {
    initializeViewer();
    viewerState.initialized = true;
    viewerState.ready = true;
  } catch (error) {
    viewerState.initialized = true;
    viewerState.ready = false;
    const container = document.getElementById("viewer");
    if (container) {
      container.innerHTML = `<div class="empty-state">3D 组件加载失败，不影响 OCR 与诊断结果展示。</div>`;
    }
    document.getElementById("viewer-subtitle").textContent = "3D 组件初始化失败，已切换为结果展示模式。";
    console.error(error);
  }
}

function initializeViewer() {
  const container = document.getElementById("viewer");
  if (!window.THREE) {
    throw new Error("Three.js 未加载成功。");
  }
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x081121);
  const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 1000);
  camera.position.set(8, 7, 9);

  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  container.appendChild(renderer.domElement);

  const controls = createControls(camera, renderer.domElement);
  controls.enableDamping = true;

  const ambientLight = new THREE.AmbientLight(0xffffff, 1.2);
  const directionalLight = new THREE.DirectionalLight(0xffffff, 1.4);
  directionalLight.position.set(8, 10, 12);
  const fillLight = new THREE.DirectionalLight(0x93c5fd, 0.8);
  fillLight.position.set(-6, 5, -4);
  scene.add(ambientLight, directionalLight, fillLight);

  const grid = new THREE.GridHelper(22, 22, 0x1d4ed8, 0x334155);
  grid.position.y = -2.5;
  scene.add(grid);

  viewerState.scene = scene;
  viewerState.camera = camera;
  viewerState.renderer = renderer;
  viewerState.controls = controls;
  resizeViewer();
  animate();
}

function animate() {
  requestAnimationFrame(animate);
  updateAnimation();
  if (viewerState.controls) {
    viewerState.controls.update();
  }
  if (viewerState.renderer && viewerState.scene && viewerState.camera) {
    viewerState.renderer.render(viewerState.scene, viewerState.camera);
  }
}

function updateAnimation() {
  if (viewerState.sceneType === "magnet_field") {
    viewerState.magnetArrows.forEach((arrow, index) => {
      arrow.rotation.z += 0.004 + index * 0.0003;
    });
  }
}

function buildScene(recipe) {
  if (viewerState.rootGroup) {
    viewerState.scene.remove(viewerState.rootGroup);
  }
  viewerState.rootGroup = new THREE.Group();
  viewerState.animatedParts = [];
  viewerState.magnetArrows = [];
  viewerState.exploded = false;
  viewerState.sceneType = recipe.scene_type;

  if (recipe.scene_type === "cylinder_slices") {
    buildCylinderScene(recipe.payload);
  } else if (recipe.scene_type === "magnet_field") {
    buildMagnetScene(recipe.payload);
  } else {
    buildFractionScene(recipe.payload);
  }
  viewerState.scene.add(viewerState.rootGroup);
  setCameraView("default");
}

function buildCylinderScene(payload) {
  const sliceCount = payload.sliceCount || 10;
  const radius = payload.radius || 2.4;
  const height = payload.height || 4.8;
  const sliceHeight = height / sliceCount;
  const material = new THREE.MeshStandardMaterial({ color: 0x60a5fa, transparent: true, opacity: 0.95 });

  for (let index = 0; index < sliceCount; index += 1) {
    const geometry = new THREE.CylinderGeometry(radius, radius, sliceHeight * 0.94, 48, 1, false);
    const mesh = new THREE.Mesh(geometry, material.clone());
    mesh.position.y = -2 + sliceHeight * index + sliceHeight / 2;
    mesh.userData.basePosition = mesh.position.clone();
    viewerState.animatedParts.push(mesh);
    viewerState.rootGroup.add(mesh);
  }
}

function buildMagnetScene(payload) {
  const magnet = new THREE.Group();
  const bodyGeometry = new THREE.BoxGeometry(5, 1.5, 1.5);
  const north = new THREE.Mesh(bodyGeometry, new THREE.MeshStandardMaterial({ color: 0xef4444 }));
  north.position.x = -1.25;
  const south = new THREE.Mesh(bodyGeometry, new THREE.MeshStandardMaterial({ color: 0x3b82f6 }));
  south.position.x = 1.25;
  magnet.add(north, south);
  viewerState.rootGroup.add(magnet);

  const arrowCount = payload.arrowCount || 8;
  for (let index = 0; index < arrowCount; index += 1) {
    const arrow = new THREE.Group();
    const shaft = new THREE.Mesh(
      new THREE.CylinderGeometry(0.06, 0.06, 1.1, 16),
      new THREE.MeshStandardMaterial({ color: 0x93c5fd })
    );
    shaft.rotation.z = Math.PI / 2;
    const head = new THREE.Mesh(
      new THREE.ConeGeometry(0.16, 0.4, 16),
      new THREE.MeshStandardMaterial({ color: 0xf8fafc })
    );
    head.position.x = 0.6;
    head.rotation.z = -Math.PI / 2;
    arrow.add(shaft, head);
    const angle = (Math.PI * 2 * index) / arrowCount;
    const radius = 4;
    arrow.position.set(Math.cos(angle) * radius, Math.sin(angle) * 2.2, Math.sin(angle) * radius * 0.25);
    arrow.lookAt(0, 0, 0);
    arrow.userData.basePosition = arrow.position.clone();
    viewerState.magnetArrows.push(arrow);
    viewerState.animatedParts.push(arrow);
    viewerState.rootGroup.add(arrow);
  }
}

function buildFractionScene(payload) {
  const group = new THREE.Group();
  const createStrip = (segments, color, y) => {
    for (let index = 0; index < segments; index += 1) {
      const width = 6 / segments;
      const mesh = new THREE.Mesh(
        new THREE.BoxGeometry(width * 0.96, 0.7, 1.2),
        new THREE.MeshStandardMaterial({ color })
      );
      mesh.position.set(-3 + width * index + width / 2, y, 0);
      mesh.userData.basePosition = mesh.position.clone();
      viewerState.animatedParts.push(mesh);
      group.add(mesh);
    }
  };
  createStrip(payload.segmentsA || 3, 0x60a5fa, -0.5);
  createStrip(payload.segmentsB || 6, 0x22c55e, 1.0);
  viewerState.rootGroup.add(group);
}

function handleViewerControl(control) {
  if (!viewerState.ready || !viewerState.rootGroup) {
    return;
  }
  if (control === "explode") {
    viewerState.exploded = !viewerState.exploded;
    applyExplodeState();
    return;
  }
  if (control === "step") {
    viewerState.stepIndex += 1;
    updateStepHighlight();
    return;
  }
  setCameraView(control);
}

function applyExplodeState() {
  viewerState.animatedParts.forEach((part, index) => {
    const base = part.userData.basePosition || part.position.clone();
    if (viewerState.sceneType === "cylinder_slices") {
      part.position.set(base.x, base.y + (viewerState.exploded ? index * 0.16 : 0), base.z);
    } else if (viewerState.sceneType === "magnet_field") {
      const scale = viewerState.exploded ? 1.2 : 1.0;
      part.position.copy(base.clone().multiplyScalar(scale));
    } else {
      part.position.set(base.x, base.y + (viewerState.exploded ? (index % 2 === 0 ? -0.45 : 0.45) : 0), base.z);
    }
  });
}

function updateStepHighlight() {
  const steps = [...document.querySelectorAll("#scene-steps .step-item")];
  if (!steps.length) {
    return;
  }
  viewerState.stepIndex %= steps.length;
  steps.forEach((item, index) => {
    item.classList.toggle("active", index === viewerState.stepIndex);
  });
}

function setCameraView(control) {
  const camera = viewerState.camera;
  if (!camera || !viewerState.controls) {
    return;
  }
  if (control === "top") {
    camera.position.set(0, 12, 0.1);
  } else if (control === "side") {
    camera.position.set(12, 2.8, 0);
  } else if (control === "section") {
    camera.position.set(7, 4, 0.2);
  } else {
    camera.position.set(8, 7, 9);
  }
  viewerState.controls.target.set(0, 0, 0);
  viewerState.controls.update();
}

function createControls(camera, domElement) {
  if (window.THREE && typeof window.THREE.OrbitControls === "function") {
    return new THREE.OrbitControls(camera, domElement);
  }
  return {
    enableDamping: false,
    target: new THREE.Vector3(0, 0, 0),
    update() {},
  };
}

function resizeViewer() {
  const container = document.getElementById("viewer");
  if (!container || !viewerState.renderer || !viewerState.camera) {
    return;
  }
  const width = container.clientWidth || 640;
  const height = container.clientHeight || 460;
  viewerState.renderer.setSize(width, height);
  viewerState.camera.aspect = width / height;
  viewerState.camera.updateProjectionMatrix();
}

/* ---------- 工具函数 ---------- */

function initialProcessSteps() {
  return [
    { key: "upload", label: "图片上传", status: "pending", detail: "等待上传作业图片。" },
    { key: "ocr", label: "OCR 结构化", status: "pending", detail: "等待提取题目、答案和知识点。" },
    { key: "graph", label: "知识图谱回溯", status: "pending", detail: "等待回溯前置依赖路径。" },
    { key: "diagnosis", label: "根因诊断", status: "pending", detail: "等待输出错误层级与前置漏洞。" },
    { key: "scene", label: "3D 干预生成", status: "pending", detail: "等待生成 3D 干预内容。" },
  ];
}

function runningProcessSteps() {
  return [
    { key: "upload", label: "图片上传", status: "running", detail: "正在上传作业图片。" },
    { key: "ocr", label: "OCR 结构化", status: "pending", detail: "等待提取题目、答案和知识点。" },
    { key: "graph", label: "知识图谱回溯", status: "pending", detail: "等待回溯前置依赖路径。" },
    { key: "diagnosis", label: "根因诊断", status: "pending", detail: "等待输出错误层级与前置漏洞。" },
    { key: "scene", label: "3D 干预生成", status: "pending", detail: "等待生成 3D 干预内容。" },
  ];
}

function failedProcessSteps(message) {
  return [
    { key: "upload", label: "图片上传", status: "completed", detail: "请求已发出。" },
    { key: "ocr", label: "OCR 结构化", status: "failed", detail: message },
    { key: "graph", label: "知识图谱回溯", status: "pending", detail: "未执行。" },
    { key: "diagnosis", label: "根因诊断", status: "pending", detail: "未执行。" },
    { key: "scene", label: "3D 干预生成", status: "pending", detail: "未执行。" },
  ];
}

function processLabel(status) {
  if (status === "completed") return "已完成";
  if (status === "running") return "进行中";
  if (status === "failed") return "失败";
  return "待执行";
}

function formatFileSize(size) {
  if (size < 1024) return `${size} B`;
  if (size < 1024 * 1024) return `${(size / 1024).toFixed(1)} KB`;
  return `${(size / (1024 * 1024)).toFixed(2)} MB`;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}
