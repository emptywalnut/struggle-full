import {
  REGION_LABELS,
  REGION_SCORING,
  actionLabel,
  cardPlayContext,
  cardPlayModeEntries,
  cardSideClass,
  eventEntries,
  groupLegalActions,
  legalCardActionIndex,
  phaseLabel,
  plainText,
  policyMetadataText,
  preferredPanelTab,
  renderActionChoiceList,
  renderCountryComponent,
  sideLabel,
  selectablePolicies,
  summarizeRegions,
} from "./ui-core.mjs";
import { renderBoardStateMarkers } from "./board-layout.mjs";

const BOARD_WIDTH = 5100;
const BOARD_HEIGHT = 3300;
const requestedSession = new URLSearchParams(window.location.search).get("session");
const state = {
  game: null,
  sessionId: /^[a-f0-9]+$/.test(requestedSession || "")
    ? requestedSession
    : localStorage.getItem("ts_browser_session"),
  zoom: 0.22,
  busy: false,
  pollTimer: null,
  activeTab: "actions",
  renderedVersion: null,
  modelCatalog: null,
  registryLoaded: false,
  policyLoads: new Map(),
};

const $ = (selector) => document.querySelector(selector);
const root = $("#game-root");
const setupOverlay = $("#setup-overlay");
const setupForm = $("#setup-form");
const setupError = $("#setup-error");
const boardViewport = $("#board-viewport");
const boardScale = $("#board-scale");
const board = $("#game-board");

async function api(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: { "content-type": "application/json", ...(options.headers || {}) },
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    const error = new Error(payload.error || `Request failed (${response.status})`);
    error.status = response.status;
    error.state = payload.state;
    throw error;
  }
  return payload;
}

function registeredPolicy(id) {
  return state.modelCatalog?.policies?.find((policy) => policy.id === id) || null;
}

function aiControlledSetupSides() {
  const data = new FormData(setupForm);
  if (data.get("mode") === "ai-ai") return ["us", "ussr"];
  return [data.get("human_side") === "ussr" ? "us" : "ussr"];
}

function replaceRegisteredPolicy(entry) {
  const index = state.modelCatalog?.policies?.findIndex((policy) => policy.id === entry.id) ?? -1;
  if (index >= 0) state.modelCatalog.policies[index] = entry;
}

function updatePolicySetup() {
  for (const side of ["us", "ussr"]) {
    const select = $(`#${side}-policy`);
    const policy = registeredPolicy(select.value);
    $(`#${side}-policy-meta`).textContent = policyMetadataText(policy, side);
  }
  const registryStatus = $("#model-registry-status");
  const trained = state.modelCatalog?.policies?.filter((policy) => policy.kind === "model") || [];
  const rejected = state.modelCatalog?.rejected?.length || 0;
  if (!state.registryLoaded) {
    registryStatus.className = "model-registry-status";
    registryStatus.textContent = "Loading and validating the server-side checkpoint registry…";
  } else if (trained.length) {
    registryStatus.className = "model-registry-status ready";
    registryStatus.textContent = `${trained.length} compatible trained model packages registered · ${rejected} excluded artifacts retained in the audit response.`;
  } else {
    registryStatus.className = "model-registry-status warn";
    registryStatus.textContent = `No compatible trained checkpoint is available. Scripted baselines remain usable; ${rejected} artifacts were rejected by validation.`;
  }
  const selectedReady = aiControlledSetupSides().every((side) => {
    const policy = registeredPolicy($(`#${side}-policy`).value);
    return policy?.readiness?.[side]?.status === "ready";
  });
  $("#start-button").disabled = state.busy || !state.registryLoaded || state.policyLoads.size > 0 || !selectedReady;
}

function renderPolicySelectors() {
  for (const side of ["us", "ussr"]) {
    const select = $(`#${side}-policy`);
    const previous = select.value;
    select.replaceChildren();
    const policies = selectablePolicies(state.modelCatalog, side);
    for (const kind of ["baseline", "model"]) {
      const groupPolicies = policies.filter((policy) => policy.kind === kind);
      if (!groupPolicies.length) continue;
      const group = document.createElement("optgroup");
      group.label = kind === "baseline" ? "Baselines" : "Trained repository models";
      for (const policy of groupPolicies) {
        const option = document.createElement("option");
        option.value = policy.id;
        option.textContent = policy.label;
        option.disabled = policy.readiness?.[side]?.status === "error";
        group.append(option);
      }
      select.append(group);
    }
    select.disabled = false;
    select.value = policies.some((policy) => policy.id === previous) ? previous : "heuristic";
  }
  updatePolicySetup();
}

async function loadModelCatalog() {
  try {
    state.modelCatalog = await api("/api/models");
    state.registryLoaded = true;
    renderPolicySelectors();
  } catch (error) {
    state.registryLoaded = false;
    setupError.textContent = `Model registry unavailable: ${error.message}`;
    const status = $("#model-registry-status");
    status.className = "model-registry-status error";
    status.textContent = "The server-side AI registry could not be validated. Check the configured training Python environment.";
    updatePolicySetup();
  }
}

async function ensurePolicyReady(side) {
  const select = $(`#${side}-policy`);
  const policy = registeredPolicy(select.value);
  if (!policy) throw new Error(`No registered ${side.toUpperCase()} policy is selected.`);
  if (policy.readiness?.[side]?.status === "ready") return policy;
  const key = `${policy.id}:${side}`;
  if (state.policyLoads.has(key)) return state.policyLoads.get(key);
  policy.readiness[side] = { ...policy.readiness[side], status: "loading", error: null };
  updatePolicySetup();
  const promise = api(`/api/models/${encodeURIComponent(policy.id)}/load`, {
    method: "POST",
    body: JSON.stringify({ side }),
  }).then((entry) => {
    replaceRegisteredPolicy(entry);
    return entry;
  }).catch((error) => {
    policy.readiness[side] = { ...policy.readiness[side], status: "error", error: error.message };
    throw new Error(`${policy.label} failed to load for ${side.toUpperCase()}: ${error.message}`);
  }).finally(() => {
    state.policyLoads.delete(key);
    renderPolicySelectors();
  });
  state.policyLoads.set(key, promise);
  updatePolicySetup();
  return promise;
}

async function ensureSelectedPoliciesReady() {
  await Promise.all(aiControlledSetupSides().map((side) => ensurePolicyReady(side)));
}

function setConnection(text, kind = "") {
  const element = $("#connection-status");
  element.textContent = text;
  element.className = `connection-pill ${kind}`;
}

function toast(message, kind = "") {
  const element = $("#toast");
  element.textContent = message;
  element.className = `toast show ${kind}`;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => { element.className = "toast"; }, 3000);
}

function showGame(game) {
  state.game = game;
  state.sessionId = game.id;
  localStorage.setItem("ts_browser_session", game.id);
  const url = new URL(window.location.href);
  url.searchParams.set("session", game.id);
  window.history.replaceState(null, "", url);
  setupOverlay.hidden = true;
  root.hidden = false;
  $("#new-game-button").hidden = false;
  setConnection("Connected", "ok");
  render(true);
  schedulePoll();
}

async function startGame(event) {
  event.preventDefault();
  if (state.busy || !state.registryLoaded) return;
  state.busy = true;
  setupError.textContent = "";
  updatePolicySetup();
  try {
    $("#start-button").textContent = "Loading selected AI…";
    await ensureSelectedPoliciesReady();
    $("#start-button").textContent = "Starting Saito engine…";
    const data = new FormData(setupForm);
    const body = {
      mode: data.get("mode"),
      human_side: data.get("human_side"),
      policies: { us: data.get("us_policy"), ussr: data.get("ussr_policy") },
      seed: Number(data.get("seed")),
      ownership_confirmed: data.get("ownership") === "on",
    };
    showGame(await api("/api/sessions", { method: "POST", body: JSON.stringify(body) }));
  } catch (error) {
    setupError.textContent = error.message;
  } finally {
    state.busy = false;
    $("#start-button").textContent = "Start game";
    updatePolicySetup();
  }
}

function schedulePoll() {
  clearTimeout(state.pollTimer);
  if (!state.sessionId) return;
  state.pollTimer = setTimeout(poll, 350);
}

async function poll() {
  if (!state.sessionId) return;
  try {
    const game = await api(`/api/sessions/${state.sessionId}`);
    const versionChanged = game.version !== state.game?.version;
    state.game = game;
    setConnection("Connected", "ok");
    render(versionChanged);
  } catch (error) {
    if (error.status === 404) {
      localStorage.removeItem("ts_browser_session");
      state.sessionId = null;
      setupOverlay.hidden = false;
      root.hidden = true;
      setConnection("Session expired", "warn");
      return;
    }
    setConnection("Reconnecting…", "warn");
  }
  schedulePoll();
}

function render(versionChanged = true) {
  if (!state.game) return;
  state.activeTab = preferredPanelTab(
    state.activeTab,
    versionChanged,
    state.game.can_act,
    state.game.observation.legal_actions || [],
  );
  renderStatus();
  renderDecision();
  renderAiControls();
  if (versionChanged || state.renderedVersion !== state.game.version) {
    const scroll = { left: boardViewport.scrollLeft, top: boardViewport.scrollTop };
    renderBoard();
    renderRegions();
    renderHands();
    renderPanels();
    boardViewport.scrollLeft = scroll.left;
    boardViewport.scrollTop = scroll.top;
    state.renderedVersion = state.game.version;
  }
}

function renderStatus() {
  const obs = state.game.observation;
  const vpSide = obs.vp > 0 ? "us" : obs.vp < 0 ? "ussr" : "neutral";
  const status = [
    ["Turn", `${obs.turn} / 10`],
    ["Action round", String(obs.action_round)],
    ["Phase", phaseLabel(obs.phase)],
    ["Active", sideLabel(obs.current_player)],
    ["Victory points", obs.vp === 0 ? "0 · tied" : `${Math.abs(obs.vp)} · ${sideLabel(vpSide)}`],
    ["DEFCON", String(obs.defcon)],
    ["Mil Ops", `US ${obs.milops.us} · USSR ${obs.milops.ussr}`],
    ["Space", `US ${obs.space.us} · USSR ${obs.space.ussr}`],
    ["China Card", sideLabel(obs.china_owner)],
  ];
  $("#status-ribbon").innerHTML = status.map(([label, value]) => `
    <div class="status-item ${label === "DEFCON" ? `defcon-${obs.defcon}` : ""}">
      <span>${label}</span><strong>${value}</strong>
    </div>`).join("");
}

function renderDecision() {
  const obs = state.game.observation;
  const count = obs.legal_actions?.length || 0;
  $("#decision-eyebrow").textContent = obs.terminal
    ? "GAME OVER"
    : `${phaseLabel(obs.phase)} · ${sideLabel(obs.current_player)} · ${count} legal ${count === 1 ? "choice" : "choices"}`;
  $("#decision-prompt").textContent = obs.terminal
    ? `${sideLabel(obs.winner)} wins`
    : plainText(obs.prompt) || `${sideLabel(obs.current_player)} to act`;
  const ai = state.game.ai;
  $("#ai-progress").innerHTML = state.game.config.controllers[obs.current_player] === "ai"
    ? `<span class="thinking-dot ${ai.status === "thinking" ? "animate" : ""}"></span><span><strong>${ai.status}</strong>${ai.detail ? `<small>${ai.detail}</small>` : ""}</span>`
    : `<span class="human-indicator"></span><span><strong>Human decision</strong><small>Select a highlighted map space, card, or listed action.</small></span>`;
  const immediate = $("#decision-actions");
  const modes = state.game.can_act ? cardPlayModeEntries(obs.legal_actions || [], obs, state.game.cards) : [];
  immediate.hidden = !modes.length;
  immediate.replaceChildren();
  if (modes.length) {
    immediate.append(renderActionChoiceList(document, modes, obs, state.game.cards, {
      onAction: (index) => submitAction(index),
    }, true));
  }
}

function renderBoard() {
  const obs = state.game.observation;
  const legal = obs.legal_actions || [];
  board.replaceChildren();
  for (const country of obs.countries) {
    board.append(renderCountryComponent(document, country, legal, {
      onAction: (index) => submitAction(index),
      onIllegal: (target, isLegal) => {
        if (isLegal && !state.game.can_act) toast(`${sideLabel(obs.current_player)} is AI-controlled. Pause or step from the observer controls.`, "warn");
        else toast(`${target.name} is not legal for the current decision.`, "error");
      },
    }, state.game.can_act));
  }
  board.append(renderBoardStateMarkers(document, obs));
  applyZoom();
}

function renderRegions() {
  const summaries = summarizeRegions(state.game.observation.countries);
  const order = ["europe", "mideast", "asia", "seasia", "africa", "camerica", "samerica"];
  $("#region-summary").innerHTML = order.map((region) => {
    const summary = summaries[region] || { us: { countries: 0, battlegrounds: 0 }, ussr: { countries: 0, battlegrounds: 0 } };
    return `<article class="region-card region-${region}">
      <div><strong>${REGION_LABELS[region]}</strong><small>${region === "seasia" ? "special scoring" : "P / D / C"} · ${REGION_SCORING[region]}</small></div>
      <div class="region-control"><span class="us-text">US ${summary.us.countries} (${summary.us.battlegrounds} BG)</span><span class="ussr-text">USSR ${summary.ussr.countries} (${summary.ussr.battlegrounds} BG)</span></div>
    </article>`;
  }).join("");
}

function cardHtml(cardId, sideHeading = "") {
  const card = state.game.cards.find((candidate) => candidate.id === cardId);
  if (!card) return "";
  const actionIndex = legalCardActionIndex(state.game.observation.legal_actions || [], cardId);
  const legal = actionIndex >= 0;
  const inPlay = cardPlayContext(state.game.observation, state.game.cards)?.card.id === cardId;
  return `<button type="button" class="game-card ${cardSideClass(card)}${legal ? " is-legal" : ""}${inPlay ? " is-in-play" : ""}" data-card="${cardId}" data-action-index="${actionIndex}" ${legal && state.game.can_act ? "" : "aria-disabled=\"true\""}>
    ${sideHeading ? `<span class="card-owner">${sideHeading}</span>` : ""}
    <span class="ops">${card.scoring ? "S" : card.ops}</span>
    <strong>${card.name}</strong>
    <span class="card-meta">${card.scoring ? "SCORING · MUST PLAY" : `${card.era} · ${sideLabel(card.side)}`}${card.recurring ? " · recurring" : " · remove after event"}</span>
  </button>`;
}

function renderHands() {
  const obs = state.game.observation;
  const panel = $("#hand-panel");
  if (state.game.config.mode === "ai-ai" && !state.game.observer_hands) {
    panel.innerHTML = `<div class="hand-heading"><div><span class="eyebrow">OBSERVER INFORMATION</span><h3>AI hands hidden</h3></div><span>US ${obs.hand_count.us} cards · USSR ${obs.hand_count.ussr} cards</span></div>`;
    return;
  }
  const hands = state.game.config.mode === "ai-ai"
    ? [
      { label: "US AI", cards: state.game.observer_hands.us },
      { label: "USSR AI", cards: state.game.observer_hands.ussr },
    ]
    : [{ label: `${sideLabel(obs.side)} hand`, cards: obs.hand }];
  panel.innerHTML = `<div class="hand-heading"><div><span class="eyebrow">PRIVATE HAND</span><h3>${hands.map((hand) => `${hand.label} · ${hand.cards.length}`).join(" / ")}</h3></div><span>Deck ${obs.deck_count} · Discard ${obs.discard.length}</span></div>
    <div class="card-row">${hands.flatMap((hand) => hand.cards.map((card) => cardHtml(card, state.game.config.mode === "ai-ai" ? hand.label : ""))).join("")}</div>`;
  panel.querySelectorAll(".game-card").forEach((element) => {
    element.addEventListener("click", () => {
      const index = Number(element.dataset.actionIndex);
      if (index >= 0 && state.game.can_act) submitAction(index);
      else toast("That card is not a legal choice for the current prompt.", "error");
    });
  });
}

function renderActions() {
  const legal = state.game.observation.legal_actions || [];
  const groups = groupLegalActions(legal);
  const cardMode = cardPlayContext(state.game.observation, state.game.cards);
  const sections = [
    ["Cards", groups.cards],
    [cardMode ? `Use ${cardMode.card.name}` : "Options", groups.options],
    ["Country targets", groups.countries],
    ["Other", groups.other],
  ].filter(([, entries]) => entries.length);
  const panel = $("#panel-actions");
  panel.replaceChildren();
  if (!sections.length) {
    const empty = document.createElement("div");
    empty.className = "empty-panel";
    empty.textContent = `No legal choices. ${state.game.observation.terminal ? "The game is complete." : "The engine is advancing the queue."}`;
    panel.append(empty);
    return;
  }
  for (const [sectionLabel, entries] of sections) {
    const section = document.createElement("section");
    section.className = "action-group";
    const heading = document.createElement("h3");
    heading.textContent = sectionLabel;
    const count = document.createElement("span");
    count.textContent = String(entries.length);
    heading.append(count);
    section.append(heading, renderActionChoiceList(document, entries, state.game.observation, state.game.cards, {
      onAction: (index) => submitAction(index),
    }, state.game.can_act));
    panel.append(section);
  }
}

function renderHistory() {
  const entries = [...state.game.history].reverse();
  $("#panel-history").innerHTML = entries.length ? `<ol class="timeline">${entries.map((entry) => `
    <li class="${entry.side}">
      <span class="timeline-number">${entry.number}</span>
      <div><strong>${sideLabel(entry.side)} · ${actionLabel(entry.action)}</strong><small>Turn ${entry.turn}, AR ${entry.action_round} · ${phaseLabel(entry.phase)} · ${entry.source}${entry.rolls.length ? ` · rolls ${entry.rolls.join(", ")}` : ""}</small></div>
    </li>`).join("")}</ol>` : `<div class="empty-panel">Actions will appear here as the game advances.</div>`;
}

function renderLog() {
  const logs = [...state.game.logs].reverse();
  $("#panel-log").innerHTML = logs.length ? `<ol class="log-list">${logs.map((entry) => `<li>${entry}</li>`).join("")}</ol>` : `<div class="empty-panel">The Saito rules log is empty.</div>`;
}

function renderEffects() {
  const obs = state.game.observation;
  const events = eventEntries(obs.events);
  $("#panel-effects").innerHTML = `
    <section class="effect-section"><h3>Events in effect <span>${events.length}</span></h3>${events.length ? `<dl class="effect-list">${events.map(([name, value]) => `<div><dt>${name.replaceAll("_", " ")}</dt><dd>${Array.isArray(value) ? value.join(", ") : typeof value === "object" ? JSON.stringify(value) : String(value)}</dd></div>`).join("")}</dl>` : `<p>None exposed by the current state.</p>`}</section>
    <section class="effect-section"><h3>Public piles</h3><p><strong>Discard (${obs.discard.length})</strong><br>${obs.discard.map(cardName).join(", ") || "Empty"}</p><p><strong>Removed (${obs.removed.length})</strong><br>${obs.removed.map(cardName).join(", ") || "Empty"}</p></section>
    <section class="effect-section"><h3>Headline</h3><p>US: ${cardName(obs.headline.us) || "not selected"}<br>USSR: ${cardName(obs.headline.ussr) || "not selected"}</p></section>`;
}

function cardName(id) {
  return state.game.cards.find((card) => card.id === id)?.name || id || "";
}

function renderPanels() {
  renderActions();
  renderHistory();
  renderLog();
  renderEffects();
  setActiveTab(state.activeTab);
}

function renderAiControls() {
  const element = $("#ai-controls");
  const hasAi = Object.values(state.game.config.controllers).includes("ai");
  element.hidden = !hasAi;
  if (!hasAi) return;
  const aiAi = state.game.config.mode === "ai-ai";
  element.innerHTML = `<div class="control-heading"><div><span class="eyebrow">AI CONTROL</span><h3>${aiAi ? "Observer console" : "Opponent pacing"}</h3></div><span class="status-chip ${state.game.paused ? "paused" : "running"}">${state.game.paused ? "Paused" : "Running"}</span></div>
    <div class="control-buttons">
      <button type="button" data-control="${state.game.paused ? "resume" : "pause"}" class="button">${state.game.paused ? "Resume" : "Pause"}</button>
      <button type="button" data-control="step" class="button" ${state.game.config.controllers[state.game.observation.current_player] === "ai" ? "" : "disabled"}>Step one decision</button>
    </div>
    <label class="speed-control">Speed <select data-speed><option value="100">Very fast</option><option value="300">Fast</option><option value="650">Normal</option><option value="1200">Slow</option><option value="2500">Study</option></select></label>
    ${aiAi ? `<label class="toggle-control"><input type="checkbox" data-reveal ${state.game.reveal_ai_hands ? "checked" : ""}> Reveal both AI hands locally</label>` : ""}`;
  const select = element.querySelector("[data-speed]");
  select.value = String(state.game.speed_ms);
  element.querySelectorAll("[data-control]").forEach((button) => button.addEventListener("click", () => sendControl(button.dataset.control)));
  select.addEventListener("change", () => sendControl("speed", { speed_ms: Number(select.value) }));
  element.querySelector("[data-reveal]")?.addEventListener("change", (event) => sendControl("reveal", { reveal: event.target.checked }));
}

async function submitAction(actionIndex) {
  if (state.busy || !state.game?.can_act) return;
  state.busy = true;
  setConnection("Submitting…", "working");
  try {
    const game = await api(`/api/sessions/${state.sessionId}/action`, {
      method: "POST",
      body: JSON.stringify({ version: state.game.version, action_index: actionIndex }),
    });
    state.game = game;
    render(true);
    setConnection("Connected", "ok");
  } catch (error) {
    if (error.state) state.game = error.state;
    toast(error.message, "error");
    render(true);
  } finally {
    state.busy = false;
    schedulePoll();
  }
}

async function sendControl(command, extra = {}) {
  if (state.busy) return;
  state.busy = true;
  try {
    state.game = await api(`/api/sessions/${state.sessionId}/control`, {
      method: "POST",
      body: JSON.stringify({ command, ...extra }),
    });
    render(true);
  } catch (error) {
    toast(error.message, "error");
  } finally {
    state.busy = false;
    schedulePoll();
  }
}

function applyZoom() {
  state.zoom = Math.max(0.1, Math.min(0.8, state.zoom));
  board.style.transform = `scale(${state.zoom})`;
  boardScale.style.width = `${BOARD_WIDTH * state.zoom}px`;
  boardScale.style.height = `${BOARD_HEIGHT * state.zoom}px`;
  $("#zoom-readout").textContent = `${Math.round(state.zoom * 100)}%`;
}

function fitBoard() {
  const availableWidth = Math.max(300, boardViewport.clientWidth - 20);
  const availableHeight = Math.max(260, boardViewport.clientHeight - 20);
  state.zoom = Math.min(availableWidth / BOARD_WIDTH, availableHeight / BOARD_HEIGHT);
  applyZoom();
  boardViewport.scrollTo({ left: 0, top: 0 });
}

function setActiveTab(tab) {
  state.activeTab = tab;
  document.querySelectorAll(".tab").forEach((button) => button.classList.toggle("active", button.dataset.tab === tab));
  document.querySelectorAll(".panel-view").forEach((panel) => panel.classList.toggle("active", panel.id === `panel-${tab}`));
}

async function newGame() {
  clearTimeout(state.pollTimer);
  if (state.sessionId) await api(`/api/sessions/${state.sessionId}`, { method: "DELETE" }).catch(() => {});
  localStorage.removeItem("ts_browser_session");
  const url = new URL(window.location.href);
  url.searchParams.delete("session");
  window.history.replaceState(null, "", url);
  state.sessionId = null;
  state.game = null;
  state.renderedVersion = null;
  state.activeTab = "actions";
  root.hidden = true;
  setupOverlay.hidden = false;
  $("#new-game-button").hidden = true;
  setConnection("Ready");
  updatePolicySetup();
}

setupForm.addEventListener("submit", startGame);
$("#mode-input").addEventListener("change", (event) => {
  $("#human-side-field").hidden = event.target.value === "ai-ai";
  ensureSelectedPoliciesReady().catch((error) => { setupError.textContent = error.message; });
  updatePolicySetup();
});
setupForm.elements.human_side.addEventListener("change", () => {
  ensureSelectedPoliciesReady().catch((error) => { setupError.textContent = error.message; });
  updatePolicySetup();
});
for (const side of ["us", "ussr"]) {
  $(`#${side}-policy`).addEventListener("change", () => {
    if (aiControlledSetupSides().includes(side)) {
      ensurePolicyReady(side).catch((error) => { setupError.textContent = error.message; });
    }
    updatePolicySetup();
  });
}
$("#new-game-button").addEventListener("click", newGame);
$("#zoom-in").addEventListener("click", () => { state.zoom *= 1.18; applyZoom(); });
$("#zoom-out").addEventListener("click", () => { state.zoom /= 1.18; applyZoom(); });
$("#zoom-fit").addEventListener("click", fitBoard);
document.querySelectorAll(".tab").forEach((button) => button.addEventListener("click", () => setActiveTab(button.dataset.tab)));

window.addEventListener("resize", () => { if (state.game && state.zoom < 0.3) fitBoard(); });

(async function bootstrap() {
  await loadModelCatalog();
  if (state.sessionId) {
    setConnection("Restoring…", "working");
    try {
      showGame(await api(`/api/sessions/${state.sessionId}`));
    } catch {
      localStorage.removeItem("ts_browser_session");
      state.sessionId = null;
      setConnection("Ready");
    }
  }
})();
