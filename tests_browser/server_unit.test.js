"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const { parseArgs, resolvePolicyPython, validateConfig } = require("../browser/server.js");

test("browser server parses an explicit remote host and high port", () => {
  assert.deepEqual(parseArgs(["--host", "0.0.0.0", "--port", "41783", "--production"]), {
    host: "0.0.0.0", port: 41783, production: true,
  });
});

test("asset ownership acknowledgement is mandatory", () => {
  assert.throws(() => validateConfig({ mode: "human-ai" }), /Confirm the GMT\/Saito ownership/);
  const config = validateConfig({
    mode: "human-ai", human_side: "ussr", seed: 9, ownership_confirmed: true,
    policies: { us: "heuristic", ussr: "random" },
  });
  assert.deepEqual(config.controllers, { ussr: "human", us: "ai" });
  assert.deepEqual(config.policies, { us: "heuristic", ussr: "random" });
});

test("AI versus AI starts paused and keeps both sides AI-controlled", () => {
  const config = validateConfig({ mode: "ai-ai", ownership_confirmed: true });
  assert.equal(config.humanSide, null);
  assert.deepEqual(config.controllers, { us: "ai", ussr: "ai" });
});

test("model policy IDs require explicit registry membership", () => {
  const allowed = new Set(["heuristic", "random", "phase3-u75-parent"]);
  const config = validateConfig({
    mode: "ai-ai",
    ownership_confirmed: true,
    policies: { us: "phase3-u75-parent", ussr: "random" },
  }, allowed);
  assert.equal(config.policies.us, "phase3-u75-parent");
  assert.throws(() => validateConfig({
    mode: "ai-ai",
    ownership_confirmed: true,
    policies: { us: "../../checkpoint", ussr: "random" },
  }, allowed), /Unknown registered US policy ID/);
  assert.match(resolvePolicyPython(), /struggle-ppo\/bin\/python$|python3$/);
});
