#!/usr/bin/env node

import crypto from "node:crypto";
import fs from "node:fs/promises";
import path from "node:path";
import process from "node:process";

function options(argv) {
  const parsed = {
    base: "http://127.0.0.1:41783",
    us: "phase3-u75-parent",
    ussr: "phase3-composed-p30u25",
    seed: 20260831,
    maxDecisions: 2500,
    requestTimeoutMs: 120000,
    output: `artifacts/browser-full-game-${new Date().toISOString().replaceAll(":", "").replaceAll(".", "")}.json`,
    deleteSession: false,
  };
  for (let index = 0; index < argv.length; index += 1) {
    const flag = argv[index];
    if (flag === "--delete-session") parsed.deleteSession = true;
    else if (flag === "--base") parsed.base = argv[++index];
    else if (flag === "--us") parsed.us = argv[++index];
    else if (flag === "--ussr") parsed.ussr = argv[++index];
    else if (flag === "--seed") parsed.seed = Number(argv[++index]);
    else if (flag === "--max-decisions") parsed.maxDecisions = Number(argv[++index]);
    else if (flag === "--request-timeout-ms") parsed.requestTimeoutMs = Number(argv[++index]);
    else if (flag === "--output") parsed.output = argv[++index];
    else throw new Error(`Unknown argument: ${flag}`);
  }
  if (!Number.isInteger(parsed.seed) || parsed.seed < 1) throw new Error("--seed must be a positive integer");
  if (!Number.isInteger(parsed.maxDecisions) || parsed.maxDecisions < 1) throw new Error("--max-decisions must be positive");
  return parsed;
}

function invariant(condition, message) {
  if (!condition) throw new Error(message);
}

async function request(url, init = {}, timeoutMs = 30000) {
  const started = performance.now();
  const response = await fetch(url, { ...init, signal: AbortSignal.timeout(timeoutMs) });
  const text = await response.text();
  let body;
  try { body = JSON.parse(text); } catch { body = { raw: text }; }
  if (!response.ok) {
    throw new Error(`${init.method || "GET"} ${url} returned ${response.status}: ${body.error || text}`);
  }
  return { body, latencyMs: performance.now() - started };
}

function actionKey(action) {
  return JSON.stringify([
    action?.type,
    action?.decision,
    action?.value,
    action?.label,
    action?.card,
    action?.country,
  ]);
}

function influenceDigest(countries) {
  const compact = countries.map((country) => `${country.id}:${country.us}:${country.ussr}`).join("|");
  return crypto.createHash("sha256").update(compact).digest("hex");
}

function visibleState(observation) {
  return {
    turn: observation.turn,
    action_round: observation.action_round,
    phase: observation.phase,
    active_side: observation.current_player,
    vp: observation.vp,
    defcon: observation.defcon,
    milops: { ...observation.milops },
    space: { ...observation.space },
    influence_digest: influenceDigest(observation.countries),
  };
}

function validateBrowserProjection(game) {
  const observation = game.observation;
  invariant(game.id && game.version > 0, "browser session/version projection is missing");
  invariant(Number.isInteger(observation.turn), "turn projection is missing");
  invariant(Number.isInteger(observation.action_round), "action-round projection is missing");
  invariant(typeof observation.phase === "string" && observation.phase.length > 0, "phase projection is missing");
  invariant(Array.isArray(observation.countries) && observation.countries.length === 84, "board country projection is incomplete");
  invariant(observation.countries.every((country) => Number.isFinite(country.us) && Number.isFinite(country.ussr)), "board influence projection is invalid");
  invariant(Number.isInteger(observation.defcon) && observation.defcon >= 1 && observation.defcon <= 5, "DEFCON projection is invalid");
  invariant(Number.isFinite(observation.vp), "VP projection is invalid");
  invariant(["us", "ussr"].every((side) => Number.isFinite(observation.milops?.[side])), "MilOps projection is invalid");
  invariant(["us", "ussr"].every((side) => Number.isFinite(observation.space?.[side])), "Space Race projection is invalid");
  invariant(Array.isArray(game.history) && Array.isArray(game.logs), "history/log projection is invalid");
  invariant(game.ai && typeof game.ai.status === "string", "AI progress projection is invalid");
}

function recordDistinct(distinct, snapshot) {
  distinct.turns.add(snapshot.turn);
  distinct.actionRounds.add(`${snapshot.turn}:${snapshot.action_round}`);
  distinct.phases.add(snapshot.phase);
  distinct.vp.add(snapshot.vp);
  distinct.defcon.add(snapshot.defcon);
  distinct.milops.add(JSON.stringify(snapshot.milops));
  distinct.space.add(JSON.stringify(snapshot.space));
  distinct.influence.add(snapshot.influence_digest);
}

async function main() {
  const config = options(process.argv.slice(2));
  const base = config.base.replace(/\/$/, "");
  const startedAt = new Date();
  const startedClock = performance.now();
  const allHistory = [];
  const transitions = [];
  const checkpoints = [];
  const polling = { requests: 0, failures: 0, max_latency_ms: 0, progress_statuses: new Set() };
  const distinct = {
    turns: new Set(), actionRounds: new Set(), phases: new Set(), vp: new Set(),
    defcon: new Set(), milops: new Set(), space: new Set(), influence: new Set(),
  };
  let sessionId = null;
  let game = null;
  let failure = null;

  try {
    const health = await request(`${base}/api/health`, {}, 10000);
    invariant(health.body.ok && health.body.backend === "saito", "production Saito health check failed");
    const page = await fetch(`${base}/`, { signal: AbortSignal.timeout(10000) });
    invariant(page.ok && (await page.text()).includes("Begin a new struggle"), "production browser page is unavailable");

    const catalog = (await request(`${base}/api/models`, {}, config.requestTimeoutMs)).body;
    for (const [side, modelId] of [["us", config.us], ["ussr", config.ussr]]) {
      const entry = catalog.policies.find((policy) => policy.id === modelId);
      invariant(entry?.kind === "model", `${side.toUpperCase()} policy ${modelId} is not a registered trained model`);
    }

    game = (await request(`${base}/api/sessions`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        mode: "ai-ai",
        seed: config.seed,
        ownership_confirmed: true,
        policies: { us: config.us, ussr: config.ussr },
      }),
    }, config.requestTimeoutMs)).body;
    sessionId = game.id;
    process.stdout.write(`${JSON.stringify({ event: "session_created", session: sessionId, inspect_url: `${base}/?session=${sessionId}` })}\n`);
    invariant(game.paused === true, "AI-vs-AI session did not start paused");
    invariant(game.config.policies.us === config.us && game.config.policies.ussr === config.ussr, "session policy IDs changed during creation");

    game = (await request(`${base}/api/sessions/${sessionId}/control`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ command: "speed", speed_ms: 80 }),
    }, config.requestTimeoutMs)).body;
    invariant(game.speed_ms === 80, "fastest supported production pacing was not accepted");

    validateBrowserProjection(game);
    polling.progress_statuses.add(game.ai.status);
    recordDistinct(distinct, visibleState(game.observation));

    for (let decision = 1; decision <= config.maxDecisions && !game.observation.terminal; decision += 1) {
      validateBrowserProjection(game);
      invariant(game.error === null, `server reported an AI error before decision ${decision}: ${game.error}`);
      const before = game.observation;
      const beforeVisible = visibleState(before);
      const legal = before.legal_actions || [];
      invariant(legal.length > 0, `unhandled prompt at decision ${decision}: ${before.phase} / ${before.prompt}`);
      const expectedSource = `ai:model:${before.current_player === "us" ? config.us : config.ussr}`;
      const legalKeys = new Set(legal.map(actionKey));

      let settled = false;
      const stepRequest = request(`${base}/api/sessions/${sessionId}/control`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ command: "step" }),
      }, config.requestTimeoutMs).finally(() => { settled = true; });

      while (!settled) {
        try {
          const poll = await request(`${base}/api/sessions/${sessionId}`, {}, 15000);
          polling.requests += 1;
          polling.max_latency_ms = Math.max(polling.max_latency_ms, poll.latencyMs);
          polling.progress_statuses.add(poll.body.ai.status);
          validateBrowserProjection(poll.body);
        } catch (error) {
          polling.failures += 1;
          throw new Error(`browser polling failed during decision ${decision}: ${error.message}`);
        }
        if (!settled) await new Promise((resolve) => setTimeout(resolve, 5));
      }

      const stepped = await stepRequest;
      game = stepped.body;
      validateBrowserProjection(game);
      polling.progress_statuses.add(game.ai.status);
      invariant(game.error === null, `server reported an AI error at decision ${decision}: ${game.error}`);
      invariant(game.paused === true, `step control unexpectedly resumed the queue at decision ${decision}`);
      const entry = game.history.at(-1);
      invariant(entry?.number === decision, `history sequence stalled or retried at decision ${decision}; got ${entry?.number}`);
      invariant(entry.side === before.current_player, `history acting side changed at decision ${decision}`);
      invariant(entry.source === expectedSource, `policy fallback/cross-attribution at decision ${decision}: ${entry.source}, expected ${expectedSource}`);
      invariant(legalKeys.has(actionKey(entry.action)), `history action was not authoritative-legal at decision ${decision}`);
      invariant(!/ai:(heuristic|random)/.test(entry.source), `silent baseline fallback at decision ${decision}`);
      invariant(game.logs.some((line) => line.includes(`[${expectedSource}]`)), `audited source missing from browser logs at decision ${decision}`);
      allHistory.push(entry);

      transitions.push({
        decision,
        before: beforeVisible,
        legal_action_count: legal.length,
        action: entry.action,
        source: entry.source,
        action_was_authoritative_legal: legalKeys.has(actionKey(entry.action)),
        after: visibleState(game.observation),
        terminal: Boolean(game.observation.terminal),
        winner: game.observation.winner || null,
        terminal_reason: game.observation.terminal_reason || null,
      });

      const snapshot = visibleState(game.observation);
      recordDistinct(distinct, snapshot);
      if (decision === 1 || decision % 25 === 0 || game.observation.terminal) {
        checkpoints.push({ decision, version: game.version, ai: { ...game.ai }, ...snapshot });
        process.stdout.write(`${JSON.stringify(checkpoints.at(-1))}\n`);
      }
    }

    invariant(game.observation.terminal === true, `game did not reach terminal state within ${config.maxDecisions} decisions`);
    invariant(["us", "ussr", "tie"].includes(game.observation.winner), `authoritative winner is invalid: ${game.observation.winner}`);
    invariant(typeof game.observation.terminal_reason === "string" && game.observation.terminal_reason.length > 0, "terminal reason is missing");
    invariant(allHistory.length === game.history.at(-1)?.number, "collected and authoritative history counts differ");
    invariant(polling.failures === 0 && polling.requests >= allHistory.length, "browser polling acceptance failed");
    invariant(polling.progress_statuses.has("thinking") && polling.progress_statuses.has("complete"), "browser AI progress did not expose thinking and complete states");

    const readyCatalog = (await request(`${base}/api/models`, {}, config.requestTimeoutMs)).body;
    const models = Object.fromEntries([config.us, config.ussr].map((id) => [id, readyCatalog.policies.find((policy) => policy.id === id)]));
    for (const [side, modelId] of [["us", config.us], ["ussr", config.ussr]]) {
      invariant(models[modelId]?.readiness?.[side]?.status === "ready", `${modelId} is not ready for ${side} after the game`);
    }

    const endedAt = new Date();
    const final = visibleState(game.observation);
    const artifact = {
      schema: "twilight-browser-full-game-validation/v1",
      status: "passed",
      production: { base_url: base, health: health.body, speed_ms: game.speed_ms },
      session: {
        id: sessionId,
        inspect_url: `${base}/?session=${sessionId}`,
        retained: !config.deleteSession,
        seed: config.seed,
        policies: { us: config.us, ussr: config.ussr },
      },
      timing: {
        started_at: startedAt.toISOString(),
        ended_at: endedAt.toISOString(),
        elapsed_seconds: Number(((performance.now() - startedClock) / 1000).toFixed(3)),
      },
      outcome: {
        winner: game.observation.winner,
        terminal_reason: game.observation.terminal_reason,
        total_decisions: allHistory.length,
        max_turn: Math.max(...allHistory.map((entry) => entry.turn)),
        max_action_round: Math.max(...allHistory.map((entry) => entry.action_round)),
        final,
      },
      assertions: {
        authoritative_legal_history: true,
        exact_model_attribution: true,
        no_baseline_fallback: true,
        no_invalid_action_retry: true,
        no_stalled_queue: true,
        no_unhandled_prompt: true,
        no_cross_attribution: true,
        no_server_error: true,
        no_browser_polling_failure: true,
      },
      browser_projection: {
        polling_requests: polling.requests,
        polling_failures: polling.failures,
        max_poll_latency_ms: Number(polling.max_latency_ms.toFixed(3)),
        progress_statuses: [...polling.progress_statuses].sort(),
        distinct_values: Object.fromEntries(Object.entries(distinct).map(([key, values]) => [key, values.size])),
        checkpoints,
        transitions,
      },
      model_registry_after_game: models,
      history: allHistory,
    };
    await fs.mkdir(path.dirname(config.output), { recursive: true });
    await fs.writeFile(config.output, `${JSON.stringify(artifact, null, 2)}\n`, "utf8");
    process.stdout.write(`${JSON.stringify({ result: "passed", output: config.output, session: sessionId, outcome: artifact.outcome, elapsed_seconds: artifact.timing.elapsed_seconds })}\n`);
  } catch (error) {
    failure = error;
    throw error;
  } finally {
    if (sessionId && config.deleteSession) {
      await request(`${base}/api/sessions/${sessionId}`, { method: "DELETE" }, 10000).catch(() => {});
    }
    if (failure) process.stderr.write(`Full-game acceptance failed: ${failure.stack || failure.message}\n`);
  }
}

await main();
