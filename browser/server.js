#!/usr/bin/env node
"use strict";

const http = require("node:http");
const fs = require("node:fs");
const path = require("node:path");
const crypto = require("node:crypto");
const os = require("node:os");
const { spawn } = require("node:child_process");
const readline = require("node:readline");

const ROOT = path.resolve(__dirname, "..");
const BRIDGE_PATH = path.join(ROOT, "bridge", "saito_bridge.js");
const DEV_STATIC = path.join(ROOT, "web");
const DIST_STATIC = path.join(ROOT, "dist", "browser");
const BOARD_SOURCE = path.join(ROOT, "reference", "map-assets", "saito", "TS Map-11.jpg");
const MAX_BODY_BYTES = 1024 * 1024;
const SESSION_LIMIT = 24;
const BASELINE_POLICIES = new Set(["heuristic", "random"]);

function resolvePolicyPython() {
  const candidates = [
    process.env.STRUGGLE_AI_PYTHON,
    path.join(ROOT, ".venv", "bin", "python"),
    path.resolve(ROOT, "../../../miniconda3/envs/struggle-ppo/bin/python"),
    path.join(os.homedir(), "miniconda3", "envs", "struggle-ppo", "bin", "python"),
  ].filter(Boolean);
  return candidates.find((candidate) => fs.existsSync(candidate)) || "python3";
}

function parseArgs(argv) {
  const options = { host: "127.0.0.1", port: 41783, production: false };
  for (let i = 0; i < argv.length; i += 1) {
    if (argv[i] === "--host") options.host = argv[++i];
    else if (argv[i] === "--port") options.port = Number(argv[++i]);
    else if (argv[i] === "--production") options.production = true;
    else throw new Error(`unknown argument: ${argv[i]}`);
  }
  if (!Number.isInteger(options.port) || options.port < 1024 || options.port > 65535) {
    throw new Error(`invalid port: ${options.port}`);
  }
  return options;
}

class JsonLineProcess {
  constructor(command, args, label) {
    this.label = label;
    this.nextId = 1;
    this.pending = new Map();
    this.stderr = "";
    this.proc = spawn(command, args, {
      cwd: ROOT,
      stdio: ["pipe", "pipe", "pipe"],
      env: { ...process.env, PYTHONUNBUFFERED: "1" },
    });
    this.proc.stderr.setEncoding("utf8");
    this.proc.stderr.on("data", (chunk) => {
      this.stderr = (this.stderr + chunk).slice(-12000);
    });
    const lines = readline.createInterface({ input: this.proc.stdout, crlfDelay: Infinity });
    lines.on("line", (line) => this.onLine(line));
    this.proc.on("exit", (code, signal) => {
      const error = new Error(`${label} exited (${code ?? signal})${this.stderr ? `: ${this.stderr}` : ""}`);
      for (const { reject } of this.pending.values()) reject(error);
      this.pending.clear();
    });
  }

  onLine(line) {
    let message;
    try {
      message = JSON.parse(line);
    } catch (error) {
      return;
    }
    const pending = this.pending.get(message.id);
    if (!pending) return;
    this.pending.delete(message.id);
    if (message.ok) pending.resolve(message.result);
    else pending.reject(new Error(message.error || `${this.label} request failed`));
  }

  call(cmd, payload = {}) {
    if (this.proc.exitCode !== null || this.proc.killed) {
      return Promise.reject(new Error(`${this.label} is not running`));
    }
    const id = this.nextId++;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.proc.stdin.write(`${JSON.stringify({ id, cmd, ...payload })}\n`);
    });
  }

  close() {
    if (this.proc.exitCode === null && !this.proc.killed) this.proc.kill("SIGTERM");
  }
}

class PolicyClient extends JsonLineProcess {
  constructor() {
    super(resolvePolicyPython(), ["-m", "struggle_ai.browser_ai"], "browser AI adapter");
  }

  models() {
    return this.call("models");
  }

  loadModel(modelId, side) {
    if (BASELINE_POLICIES.has(modelId)) return Promise.resolve(null);
    return this.call("load", { model_id: modelId, side });
  }

  choose(session, policy, observation, legalActions) {
    return this.call("choose", {
      session_id: session.id,
      seed: session.seed,
      policy,
      cards: session.cards,
      observation,
      legal_actions: legalActions,
    });
  }

  record(session, before, after, legalActions, action, actionIndex, side) {
    return this.call("record", {
      session_id: session.id,
      cards: session.cards,
      before,
      after,
      legal_actions: legalActions,
      action,
      action_index: actionIndex,
      side,
    });
  }
}

function publicConfig(config) {
  return {
    mode: config.mode,
    human_side: config.humanSide,
    controllers: { ...config.controllers },
    policies: { ...config.policies },
    ownership_confirmed: config.ownershipConfirmed,
  };
}

function validateConfig(input = {}, allowedPolicies = BASELINE_POLICIES) {
  const mode = input.mode === "ai-ai" ? "ai-ai" : "human-ai";
  const humanSide = input.human_side === "ussr" ? "ussr" : "us";
  const policies = { us: input.policies?.us || "heuristic", ussr: input.policies?.ussr || "heuristic" };
  for (const [side, policy] of Object.entries(policies)) {
    if (!allowedPolicies.has(policy)) {
      const error = new Error(`Unknown registered ${side.toUpperCase()} policy ID: ${policy}`);
      error.statusCode = 422;
      throw error;
    }
  }
  const seed = Number.isInteger(Number(input.seed)) ? Math.max(1, Math.min(2147483647, Number(input.seed))) : 1;
  if (input.ownership_confirmed !== true) {
    const error = new Error("Confirm the GMT/Saito ownership and personal non-commercial use condition before starting.");
    error.statusCode = 422;
    throw error;
  }
  return {
    mode,
    humanSide: mode === "human-ai" ? humanSide : null,
    controllers: mode === "ai-ai"
      ? { us: "ai", ussr: "ai" }
      : { [humanSide]: "human", [humanSide === "us" ? "ussr" : "us"]: "ai" },
    policies,
    seed,
    ownershipConfirmed: true,
  };
}

class GameSession {
  constructor(id, config, policyClient) {
    this.id = id;
    this.config = config;
    this.seed = config.seed;
    this.policyClient = policyClient;
    this.bridge = new JsonLineProcess(process.execPath, [BRIDGE_PATH], `Saito bridge ${id}`);
    this.actorObservation = null;
    this.cards = [];
    this.fullLog = null;
    this.history = [];
    this.auditLog = [];
    this.stateVersion = 0;
    this.paused = config.mode === "ai-ai";
    this.speedMs = 650;
    this.revealAiHands = false;
    this.ai = { status: "idle", detail: "", last_action: null };
    this.lastError = null;
    this.publicState = null;
    this.timer = null;
    this.queue = Promise.resolve();
    this.createdAt = Date.now();
    this.lastAccess = Date.now();
    this.closed = false;
  }

  async initialize() {
    this.cards = await this.bridge.call("cards");
    this.actorObservation = await this.bridge.call("reset", { seed: this.seed });
    await this.refreshPublicState();
    this.schedule();
    return this.getState();
  }

  enqueue(task) {
    const run = this.queue.then(task, task);
    this.queue = run.catch(() => {});
    return run;
  }

  isAiTurn() {
    return Boolean(
      this.actorObservation
      && !this.actorObservation.terminal
      && this.config.controllers[this.actorObservation.current_player] === "ai"
    );
  }

  async refreshPublicState() {
    this.fullLog = await this.bridge.call("log");
    let viewerObservation = this.actorObservation;
    if (this.config.humanSide && this.config.humanSide !== this.actorObservation.current_player) {
      viewerObservation = await this.bridge.call("observe", { side: this.config.humanSide });
    }
    const observation = JSON.parse(JSON.stringify(viewerObservation));
    const headlineSelectionPrivate = ["headline_ussr", "headline_us"].includes(observation.phase);
    if (headlineSelectionPrivate) {
      // Saito's headless kernel selects headlines sequentially, but the second
      // human must not learn the first card through the browser projection.
      // This redaction changes presentation only; the bridge and AI contract
      // retain their existing semantics.
      observation.headline = { us: null, ussr: null };
      observation.log_tail = (observation.log_tail || []).filter((entry) => !/selects headline/i.test(entry));
    }
    if (this.config.mode === "ai-ai") {
      observation.side = "observer";
      observation.hand = [];
      observation.legal_actions = JSON.parse(JSON.stringify(this.actorObservation.legal_actions || []));
    }
    this.stateVersion += 1;
    this.publicState = {
      id: this.id,
      version: this.stateVersion,
      seed: this.seed,
      config: publicConfig(this.config),
      paused: this.paused,
      speed_ms: this.speedMs,
      reveal_ai_hands: this.revealAiHands,
      ai: { ...this.ai },
      error: this.lastError,
      can_act: this.config.controllers[this.actorObservation.current_player] === "human" && !observation.terminal,
      observation,
      cards: this.cards,
      history: this.history.slice(-160),
      logs: [...(this.fullLog.log || []), ...this.auditLog]
        .filter((entry) => !headlineSelectionPrivate || !/selects headline/i.test(entry))
        .slice(-240),
      observer_hands: this.config.mode === "ai-ai" && this.revealAiHands
        ? JSON.parse(JSON.stringify(this.fullLog.hands))
        : null,
      asset_notice: {
        board: "Saito TS Map-11.jpg",
        owner: "GMT Games and original artists",
        use: "Personal, non-commercial use; at least one participant must own Twilight Struggle.",
      },
    };
  }

  getState() {
    this.lastAccess = Date.now();
    return {
      ...this.publicState,
      paused: this.paused,
      speed_ms: this.speedMs,
      reveal_ai_hands: this.revealAiHands,
      ai: { ...this.ai },
      error: this.lastError,
      observer_hands: this.config.mode === "ai-ai" && this.revealAiHands
        ? JSON.parse(JSON.stringify(this.fullLog?.hands || {}))
        : null,
    };
  }

  schedule() {
    clearTimeout(this.timer);
    this.timer = null;
    if (this.closed || this.paused || !this.isAiTurn()) return;
    this.timer = setTimeout(() => {
      this.enqueue(() => this.runAiStep()).catch((error) => this.failAi(error));
    }, this.speedMs);
  }

  failAi(error) {
    this.paused = true;
    this.lastError = error.message;
    this.ai = { ...this.ai, status: "error", detail: error.message };
  }

  async runAiStep() {
    if (!this.isAiTurn() || this.actorObservation.terminal) return this.getState();
    const side = this.actorObservation.current_player;
    const policy = this.config.policies[side];
    const legal = this.actorObservation.legal_actions || [];
    if (!legal.length) throw new Error(`AI ${side} has no legal action in ${this.actorObservation.phase}`);
    this.ai = { status: "thinking", detail: `${side.toUpperCase()} ${policy} policy is evaluating ${legal.length} legal choices`, last_action: this.ai.last_action };
    const choice = await this.policyClient.choose(this, policy, this.actorObservation, legal);
    const selected = legal[choice.action_index];
    if (!selected) throw new Error(`AI policy returned illegal index ${choice.action_index}`);
    const source = choice.source || (BASELINE_POLICIES.has(policy) ? `ai:${policy}` : `ai:model:${policy}`);
    const device = choice.device ? ` on ${choice.device}` : "";
    this.ai = { status: "acting", detail: `${source}${device} selected ${selected.label || selected.value}`, last_action: selected.label || selected.value };
    await this.applyAction(selected, side, source, choice.action_index);
    this.ai = { status: this.actorObservation.terminal ? "complete" : "idle", detail: this.actorObservation.terminal ? "Game complete" : "Waiting for the next decision", last_action: selected.label || selected.value };
    await this.refreshPublicState();
    this.schedule();
    return this.getState();
  }

  async applyAction(action, side, source, actionIndex) {
    const before = this.actorObservation;
    const legalBefore = JSON.parse(JSON.stringify(before.legal_actions || []));
    const result = await this.bridge.call("step", { action });
    this.actorObservation = result.observation;
    await this.policyClient.record(this, before, this.actorObservation, legalBefore, action, actionIndex, side);
    this.history.push({
      number: this.history.length + 1,
      side,
      source,
      turn: before.turn,
      action_round: before.action_round,
      phase: before.phase,
      prompt: before.prompt,
      action: JSON.parse(JSON.stringify(action)),
      rolls: result.info?.chance_rolls || [],
      at: new Date().toISOString(),
    });
    if (source.startsWith("ai:")) {
      this.auditLog.push(`[${source}] ${side.toUpperCase()} selected ${action.label || action.value || action.decision}`);
      this.auditLog = this.auditLog.slice(-160);
    }
  }

  humanAction(body) {
    return this.enqueue(async () => {
      if (Number(body.version) !== this.stateVersion) {
        const error = new Error("The game advanced before that choice was submitted. Choose from the current legal actions.");
        error.statusCode = 409;
        error.state = this.getState();
        throw error;
      }
      const side = this.actorObservation.current_player;
      if (this.config.controllers[side] !== "human") {
        const error = new Error(`It is ${side.toUpperCase()}'s AI-controlled decision.`);
        error.statusCode = 409;
        throw error;
      }
      const index = Number(body.action_index);
      const legal = this.actorObservation.legal_actions || [];
      if (!Number.isInteger(index) || index < 0 || index >= legal.length) {
        const error = new Error(`Illegal action index ${body.action_index}; ${legal.length} legal choices are available.`);
        error.statusCode = 422;
        throw error;
      }
      clearTimeout(this.timer);
      const selected = legal[index];
      await this.applyAction(selected, side, "human", index);
      await this.refreshPublicState();
      this.schedule();
      return this.getState();
    });
  }

  control(body) {
    return this.enqueue(async () => {
      const command = body.command;
      if (command === "pause") this.paused = true;
      else if (command === "resume") this.paused = false;
      else if (command === "speed") {
        const speed = Number(body.speed_ms);
        if (!Number.isInteger(speed) || speed < 80 || speed > 10000) {
          const error = new Error("AI speed must be between 80 and 10000 ms.");
          error.statusCode = 422;
          throw error;
        }
        this.speedMs = speed;
      } else if (command === "reveal") {
        this.revealAiHands = Boolean(body.reveal);
      } else if (command === "step") {
        this.paused = true;
        clearTimeout(this.timer);
        return this.runAiStep();
      } else {
        const error = new Error(`Unknown control command: ${command}`);
        error.statusCode = 422;
        throw error;
      }
      clearTimeout(this.timer);
      this.schedule();
      return this.getState();
    });
  }

  close() {
    this.closed = true;
    clearTimeout(this.timer);
    this.bridge.close();
    this.policyClient.call("forget", { session_id: this.id }).catch(() => {});
  }
}

class SessionManager {
  constructor() {
    this.sessions = new Map();
    this.policyClient = new PolicyClient();
    this.modelCatalog = null;
  }

  async models(refresh = true) {
    if (refresh || !this.modelCatalog) this.modelCatalog = await this.policyClient.models();
    return this.modelCatalog;
  }

  async loadModel(modelId, side) {
    if (!BASELINE_POLICIES.has(modelId)) await this.models(false);
    const entry = await this.policyClient.loadModel(modelId, side);
    this.modelCatalog = await this.policyClient.models();
    return entry || this.modelCatalog.policies.find((policy) => policy.id === modelId);
  }

  async allowedPolicies(input) {
    const requested = [input.policies?.us || "heuristic", input.policies?.ussr || "heuristic"];
    if (requested.every((policy) => BASELINE_POLICIES.has(policy))) return BASELINE_POLICIES;
    const catalog = await this.models(false);
    return new Set(catalog.policies.map((policy) => policy.id));
  }

  async create(input) {
    if (this.sessions.size >= SESSION_LIMIT) {
      const oldest = [...this.sessions.values()].sort((a, b) => a.lastAccess - b.lastAccess)[0];
      this.delete(oldest.id);
    }
    const config = validateConfig(input, await this.allowedPolicies(input));
    for (const side of ["us", "ussr"]) {
      if (config.controllers[side] === "ai") {
        try {
          await this.loadModel(config.policies[side], side);
        } catch (error) {
          error.statusCode = error.statusCode || 503;
          throw error;
        }
      }
    }
    const id = crypto.randomBytes(10).toString("hex");
    const session = new GameSession(id, config, this.policyClient);
    this.sessions.set(id, session);
    try {
      return await session.initialize();
    } catch (error) {
      this.delete(id);
      throw error;
    }
  }

  get(id) {
    const session = this.sessions.get(id);
    if (!session) {
      const error = new Error("Game session not found or expired.");
      error.statusCode = 404;
      throw error;
    }
    return session;
  }

  delete(id) {
    const session = this.sessions.get(id);
    if (!session) return false;
    session.close();
    this.sessions.delete(id);
    return true;
  }

  close() {
    for (const id of [...this.sessions.keys()]) this.delete(id);
    this.policyClient.close();
  }
}

function readJson(req) {
  return new Promise((resolve, reject) => {
    let body = "";
    req.setEncoding("utf8");
    req.on("data", (chunk) => {
      body += chunk;
      if (body.length > MAX_BODY_BYTES) reject(Object.assign(new Error("Request body too large."), { statusCode: 413 }));
    });
    req.on("end", () => {
      try {
        resolve(body ? JSON.parse(body) : {});
      } catch (error) {
        reject(Object.assign(new Error("Invalid JSON request body."), { statusCode: 400 }));
      }
    });
    req.on("error", reject);
  });
}

function sendJson(res, status, payload) {
  const body = JSON.stringify(payload);
  res.writeHead(status, {
    "content-type": "application/json; charset=utf-8",
    "content-length": Buffer.byteLength(body),
    "cache-control": "no-store",
    "x-content-type-options": "nosniff",
  });
  res.end(body);
}

const MIME = {
  ".html": "text/html; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".mjs": "text/javascript; charset=utf-8",
  ".jpg": "image/jpeg",
  ".png": "image/png",
  ".svg": "image/svg+xml",
};

function sendFile(res, file, cache = false) {
  const stat = fs.statSync(file);
  res.writeHead(200, {
    "content-type": MIME[path.extname(file).toLowerCase()] || "application/octet-stream",
    "content-length": stat.size,
    "cache-control": cache ? "private, max-age=86400" : "no-cache",
    "x-content-type-options": "nosniff",
    "content-security-policy": "default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; script-src 'self'; connect-src 'self'",
  });
  fs.createReadStream(file).pipe(res);
}

function staticFile(staticRoot, pathname) {
  const relative = pathname === "/" ? "index.html" : decodeURIComponent(pathname.slice(1));
  const candidate = path.resolve(staticRoot, relative);
  if (!candidate.startsWith(`${path.resolve(staticRoot)}${path.sep}`) && candidate !== path.join(path.resolve(staticRoot), "index.html")) return null;
  return fs.existsSync(candidate) && fs.statSync(candidate).isFile() ? candidate : null;
}

function createRequestHandler(manager, options) {
  const staticRoot = options.production ? DIST_STATIC : DEV_STATIC;
  return async (req, res) => {
    const url = new URL(req.url, `http://${req.headers.host || "localhost"}`);
    try {
      if (req.method === "GET" && url.pathname === "/api/health") {
        return sendJson(res, 200, {
          ok: true,
          service: "twilight-struggle-browser",
          backend: "saito",
          active_sessions: manager.sessions.size,
          board_asset: fs.existsSync(BOARD_SOURCE),
        });
      }
      if (req.method === "GET" && url.pathname === "/api/models") {
        return sendJson(res, 200, await manager.models());
      }
      const modelMatch = url.pathname.match(/^\/api\/models\/([a-z0-9-]+)\/load$/);
      if (req.method === "POST" && modelMatch) {
        const body = await readJson(req);
        if (!['us', 'ussr'].includes(body.side)) {
          throw Object.assign(new Error("Model loading requires side 'us' or 'ussr'."), { statusCode: 422 });
        }
        try {
          return sendJson(res, 200, await manager.loadModel(modelMatch[1], body.side));
        } catch (error) {
          error.statusCode = error.statusCode || 503;
          throw error;
        }
      }
      if (req.method === "POST" && url.pathname === "/api/sessions") {
        return sendJson(res, 201, await manager.create(await readJson(req)));
      }
      const match = url.pathname.match(/^\/api\/sessions\/([a-f0-9]+)(?:\/(action|control))?$/);
      if (match) {
        const session = manager.get(match[1]);
        if (req.method === "GET" && !match[2]) return sendJson(res, 200, session.getState());
        if (req.method === "POST" && match[2] === "action") return sendJson(res, 200, await session.humanAction(await readJson(req)));
        if (req.method === "POST" && match[2] === "control") return sendJson(res, 200, await session.control(await readJson(req)));
        if (req.method === "DELETE" && !match[2]) return sendJson(res, 200, { deleted: manager.delete(match[1]) });
      }
      if (req.method === "GET" && url.pathname === "/assets/saito-board.jpg") {
        if (!fs.existsSync(BOARD_SOURCE)) throw Object.assign(new Error("Supplied Saito board asset is missing."), { statusCode: 503 });
        return sendFile(res, BOARD_SOURCE, true);
      }
      if (req.method === "GET" || req.method === "HEAD") {
        const file = staticFile(staticRoot, url.pathname);
        if (file) return sendFile(res, file, false);
      }
      return sendJson(res, 404, { error: "Not found" });
    } catch (error) {
      const status = error.statusCode || 500;
      return sendJson(res, status, { error: error.message, state: error.state || undefined });
    }
  };
}

async function startServer(options = parseArgs(process.argv.slice(2))) {
  const manager = new SessionManager();
  const server = http.createServer(createRequestHandler(manager, options));
  await new Promise((resolve, reject) => {
    server.once("error", reject);
    server.listen(options.port, options.host, resolve);
  });
  return { server, manager, options };
}

if (require.main === module) {
  const options = parseArgs(process.argv.slice(2));
  startServer(options).then(({ server, manager }) => {
    console.log(`Twilight Struggle browser listening on http://${options.host}:${options.port}`);
    const shutdown = () => {
      manager.close();
      server.close(() => process.exit(0));
      setTimeout(() => process.exit(1), 2000).unref();
    };
    process.on("SIGINT", shutdown);
    process.on("SIGTERM", shutdown);
  }).catch((error) => {
    console.error(error.stack || error.message);
    process.exit(1);
  });
}

module.exports = {
  GameSession,
  JsonLineProcess,
  PolicyClient,
  SessionManager,
  createRequestHandler,
  parseArgs,
  resolvePolicyPython,
  startServer,
  validateConfig,
};
