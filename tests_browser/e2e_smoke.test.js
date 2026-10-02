"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const crypto = require("node:crypto");
const { startServer } = require("../browser/server.js");

async function json(url, options = {}) {
  const response = await fetch(url, {
    ...options,
    headers: { "content-type": "application/json", ...(options.headers || {}) },
  });
  return { response, body: await response.json() };
}

const sleep = (milliseconds) => new Promise((resolve) => setTimeout(resolve, milliseconds));

test("browser stack starts and advances real Saito legal actions", async (t) => {
  const { server, manager } = await startServer({ host: "127.0.0.1", port: 0, production: false });
  t.after(async () => {
    manager.close();
    await new Promise((resolve) => server.close(resolve));
  });
  const address = server.address();
  const base = `http://127.0.0.1:${address.port}`;

  const health = await json(`${base}/api/health`);
  assert.equal(health.response.status, 200);
  assert.equal(health.body.backend, "saito");
  assert.equal(health.body.board_asset, true);

  const page = await fetch(base);
  assert.equal(page.status, 200);
  assert.match(await page.text(), /Begin a new struggle/);

  const board = await fetch(`${base}/assets/saito-board.jpg`);
  assert.equal(board.status, 200);
  const boardBuffer = Buffer.from(await board.arrayBuffer());
  assert.equal(boardBuffer.length, 1619682);
  assert.equal(crypto.createHash("sha256").update(boardBuffer).digest("hex"), "8f21635165f354214fa0b7329b3f3e9f2c2769be76ba2dafd192252591f53217");

  const created = await json(`${base}/api/sessions`, {
    method: "POST",
    body: JSON.stringify({
      mode: "human-ai", human_side: "ussr", seed: 1,
      ownership_confirmed: true,
      policies: { us: "heuristic", ussr: "heuristic" },
    }),
  });
  assert.equal(created.response.status, 201);
  assert.equal(created.body.observation.phase, "setup_ussr");
  assert.equal(created.body.observation.countries.length, 84);
  assert.equal(created.body.observation.legal_actions.length, 9);
  assert.equal(created.body.can_act, true);
  const canada = created.body.observation.countries.find((country) => country.id === "canada");
  assert.deepEqual({ top: canada.top, left: canada.left, stability: canada.stability }, { top: 752, left: 842, stability: 4 });

  const advanced = await json(`${base}/api/sessions/${created.body.id}/action`, {
    method: "POST",
    body: JSON.stringify({ version: created.body.version, action_index: 0 }),
  });
  assert.equal(advanced.response.status, 200);
  assert.equal(advanced.body.history.length, 1);
  assert.equal(advanced.body.history[0].source, "human");
  assert.equal(advanced.body.observation.phase, "setup_ussr");
  assert.notEqual(advanced.body.version, created.body.version);
  assert.match(advanced.body.logs.at(-1), /USSR initial placement/);

  const stale = await json(`${base}/api/sessions/${created.body.id}/action`, {
    method: "POST",
    body: JSON.stringify({ version: created.body.version, action_index: 0 }),
  });
  assert.equal(stale.response.status, 409);
  assert.match(stale.body.error, /game advanced/);
});

test("AI versus AI exposes pause and deterministic one-decision stepping", async (t) => {
  const { server, manager } = await startServer({ host: "127.0.0.1", port: 0, production: false });
  t.after(async () => {
    manager.close();
    await new Promise((resolve) => server.close(resolve));
  });
  const base = `http://127.0.0.1:${server.address().port}`;
  const created = await json(`${base}/api/sessions`, {
    method: "POST",
    body: JSON.stringify({ mode: "ai-ai", seed: 3, ownership_confirmed: true, policies: { us: "heuristic", ussr: "random" } }),
  });
  assert.equal(created.body.paused, true);
  assert.equal(created.body.observation.side, "observer");
  assert.deepEqual(created.body.observation.hand, []);

  const stepped = await json(`${base}/api/sessions/${created.body.id}/control`, {
    method: "POST", body: JSON.stringify({ command: "step" }),
  });
  assert.equal(stepped.response.status, 200);
  assert.equal(stepped.body.paused, true);
  assert.equal(stepped.body.history.length, 1);
  assert.equal(stepped.body.history[0].source, "ai:random");

  const revealed = await json(`${base}/api/sessions/${created.body.id}/control`, {
    method: "POST", body: JSON.stringify({ command: "reveal", reveal: true }),
  });
  assert.ok(revealed.body.observer_hands.us.length > 0);
  assert.ok(revealed.body.observer_hands.ussr.length > 0);
});

test("registered trained models independently make audited legal Saito decisions", async (t) => {
  const priorDevice = process.env.STRUGGLE_AI_DEVICE;
  process.env.STRUGGLE_AI_DEVICE = "cpu";
  const { server, manager } = await startServer({ host: "127.0.0.1", port: 0, production: false });
  t.after(async () => {
    if (priorDevice === undefined) delete process.env.STRUGGLE_AI_DEVICE;
    else process.env.STRUGGLE_AI_DEVICE = priorDevice;
    manager.close();
    await new Promise((resolve) => server.close(resolve));
  });
  const base = `http://127.0.0.1:${server.address().port}`;

  const catalog = await json(`${base}/api/models`);
  assert.equal(catalog.response.status, 200);
  const modelIds = catalog.body.policies.filter((policy) => policy.kind === "model").map((policy) => policy.id);
  assert.ok(modelIds.includes("phase3-u75-parent"));
  assert.ok(modelIds.includes("phase3-composed-p30u25"));
  assert.ok(catalog.body.policies.some((policy) => policy.architecture === "feedforward"));
  assert.ok(catalog.body.policies.some((policy) => policy.architecture === "transformer_history"));

  const invalid = await json(`${base}/api/sessions`, {
    method: "POST",
    body: JSON.stringify({
      mode: "ai-ai", seed: 31, ownership_confirmed: true,
      policies: { us: "../../arbitrary-checkpoint", ussr: "random" },
    }),
  });
  assert.equal(invalid.response.status, 422);
  assert.match(invalid.body.error, /Unknown registered US policy ID/);

  const created = await json(`${base}/api/sessions`, {
    method: "POST",
    body: JSON.stringify({
      mode: "ai-ai", seed: 37, ownership_confirmed: true,
      policies: { us: "phase3-u75-parent", ussr: "phase3-composed-p30u25" },
    }),
  });
  assert.equal(created.response.status, 201);
  let game = created.body;
  assert.equal(game.paused, true);
  for (let decision = 0; decision < 18; decision += 1) {
    const stepped = await json(`${base}/api/sessions/${game.id}/control`, {
      method: "POST", body: JSON.stringify({ command: "step" }),
    });
    assert.equal(stepped.response.status, 200);
    game = stepped.body;
    assert.equal(game.error, null);
    if (new Set(game.history.map((entry) => entry.source)).size >= 2) break;
  }
  const sources = new Set(game.history.map((entry) => entry.source));
  assert.ok(game.history.length >= 7, "trained model should make multiple setup decisions");
  assert.ok(sources.has("ai:model:phase3-u75-parent"));
  assert.ok(sources.has("ai:model:phase3-composed-p30u25"));
  assert.ok(game.logs.some((entry) => entry.includes("[ai:model:phase3-u75-parent]")));
  assert.ok(game.logs.some((entry) => entry.includes("[ai:model:phase3-composed-p30u25]")));

  const readyCatalog = await json(`${base}/api/models`);
  const parent = readyCatalog.body.policies.find((policy) => policy.id === "phase3-u75-parent");
  const composed = readyCatalog.body.policies.find((policy) => policy.id === "phase3-composed-p30u25");
  assert.equal(parent.readiness.us.status, "ready");
  assert.equal(parent.readiness.us.device, "cpu");
  assert.equal(composed.readiness.ussr.status, "ready");
  assert.equal(composed.readiness.ussr.device, "cpu");
});

test("human US can select Duck and Cover, see all modes, and enter Ops targeting", async (t) => {
  const { server, manager } = await startServer({ host: "127.0.0.1", port: 0, production: false });
  t.after(async () => {
    manager.close();
    await new Promise((resolve) => server.close(resolve));
  });
  const base = `http://127.0.0.1:${server.address().port}`;
  const created = await json(`${base}/api/sessions`, {
    method: "POST",
    body: JSON.stringify({
      mode: "human-ai", human_side: "us", seed: 2,
      ownership_confirmed: true,
      policies: { us: "heuristic", ussr: "heuristic" },
    }),
  });
  assert.equal(created.response.status, 201);
  let game = created.body;
  const spedUp = await json(`${base}/api/sessions/${game.id}/control`, {
    method: "POST", body: JSON.stringify({ command: "speed", speed_ms: 80 }),
  });
  game = spedUp.body;

  let selectedDuck = false;
  for (let decision = 0; decision < 160; decision += 1) {
    assert.equal(game.observation.terminal, false, "game ended before the Duck and Cover choice");
    if (!game.can_act) {
      await sleep(90);
      game = (await json(`${base}/api/sessions/${game.id}`)).body;
      continue;
    }
    const legal = game.observation.legal_actions;
    const duckIndex = legal.findIndex((action) => action.decision === "list" && action.value === "duckandcover");
    if (duckIndex >= 0 && /pick a card/i.test(game.observation.prompt)) {
      const selected = await json(`${base}/api/sessions/${game.id}/action`, {
        method: "POST",
        body: JSON.stringify({ version: game.version, action_index: duckIndex }),
      });
      assert.equal(selected.response.status, 200);
      game = selected.body;
      selectedDuck = true;
      break;
    }

    let actionIndex = 0;
    if (/headline/i.test(game.observation.phase)) {
      actionIndex = legal.findIndex((action) => action.value === "containment");
      if (actionIndex < 0) actionIndex = legal.findIndex((action) => action.value !== "duckandcover");
      if (actionIndex < 0) actionIndex = 0;
    }
    game = (await json(`${base}/api/sessions/${game.id}/action`, {
      method: "POST",
      body: JSON.stringify({ version: game.version, action_index: actionIndex }),
    })).body;
  }

  assert.equal(selectedDuck, true);
  assert.match(game.observation.prompt, /US playing <span>Duck and Cover<\/span>/);
  assert.equal(game.can_act, true);
  assert.deepEqual(game.observation.legal_actions.map((action) => action.value), ["event", "ops", "space"]);
  assert.ok(game.observation.legal_actions.every((action) => action.decision === "option"));

  const opsIndex = game.observation.legal_actions.findIndex((action) => action.value === "ops");
  const choseOps = await json(`${base}/api/sessions/${game.id}/action`, {
    method: "POST",
    body: JSON.stringify({ version: game.version, action_index: opsIndex }),
  });
  assert.equal(choseOps.response.status, 200);
  game = choseOps.body;
  assert.match(game.observation.prompt, /US plays 3 OPS/i);
  assert.equal(game.observation.legal_actions.some((action) => action.decision.startsWith("country_")), false);

  const placeIndex = game.observation.legal_actions.findIndex((action) => action.value === "place");
  assert.ok(placeIndex >= 0);
  const chosePlace = await json(`${base}/api/sessions/${game.id}/action`, {
    method: "POST",
    body: JSON.stringify({ version: game.version, action_index: placeIndex }),
  });
  assert.equal(chosePlace.response.status, 200);
  assert.ok(chosePlace.body.observation.legal_actions.some((action) => action.decision.startsWith("country_")));
});
