"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const { SaitoTwilightKernel } = require("../bridge/saito_bridge.js");

function fresh(side = "ussr") {
  const env = new SaitoTwilightKernel();
  env.reset(1);
  env.currentPlayer = side;
  env.twilight.game.player = side === "ussr" ? 1 : 2;
  env.phase = "action_round";
  env.adapter.reset();
  env.twilight.game.deck[0].cards = { ...env.cardsById };
  for (const countries of [env.twilight.countries, env.twilight.game.countries]) {
    for (const country of Object.values(countries || {})) {
      country.us = 0;
      country.ussr = 0;
      country.place = 0;
    }
  }
  return env;
}

function setInfluence(env, id, us, ussr) {
  for (const countries of [env.twilight.countries, env.twilight.game.countries]) {
    countries[id].us = us;
    countries[id].ussr = ussr;
  }
}

function values(env) {
  return (env.adapter.pending?.choices || []).map((choice) => choice.value);
}

function choice(env, value) {
  const selected = env.adapter.pending?.choices.find((item) => item.value === value);
  assert.ok(selected, `missing ${value} in ${env.adapter.lastStatus}: ${values(env).join(",")}`);
  return selected;
}

function enableBonus(env, kind) {
  if (kind === "vietnam" || kind === "both") {
    env.twilight.game.state.events.vietnam_revolts = 1;
    env.twilight.game.state.events.vietnam_revolts_eligible = 1;
  }
  if (kind === "china" || kind === "both") {
    env.twilight.game.state.events.china_card_in_play = 1;
    env.twilight.game.state.events.china_card_eligible = 1;
  }
}

async function startOps({ kind, card, ops, target = "vietnam", us = 0, ussr = 2, redPurge = false }) {
  const env = fresh();
  setInfluence(env, "vietnam", 0, 2);
  if (target !== "vietnam" || us !== 0 || ussr !== 2) setInfluence(env, target, us, ussr);
  enableBonus(env, kind);
  if (redPurge) env.twilight.game.state.events.redscare_player1 = 1;
  env.twilight.game.queue = [`ops\tussr\t${card}\t${ops}`];
  await env.drainUntilDecisionOrTurnEnd();
  return env;
}

async function spendPlacementUntilBonus(env, target) {
  await env.stepAsync(choice(env, "place"));
  let placements = 0;
  while (!env.adapter.normalizeStatus().startsWith("extra 1 op available for")) {
    if (!values(env).includes(target)) break;
    await env.stepAsync(choice(env, target));
    placements += 1;
    assert.ok(placements < 12, "placement loop did not reach a bonus/turn boundary");
  }
  return placements;
}

test("Saito bounded events allow the first and second country selections, then reject the third", async () => {
  const cases = [
    { card: "socgov", target: "italy", us: 3 },
    { card: "liberation", target: "mexico" },
    { card: "reformer", target: "westgermany", vp: 0 },
    { card: "warsawpact", target: "eastgermany", option: "add" },
  ];
  for (const item of cases) {
    const env = fresh();
    setInfluence(env, item.target, item.us || 0, 0);
    env.twilight.game.state.vp = item.vp || 0;
    env.twilight.playEvent("ussr", item.card);
    if (item.option) await env.adapter.choose(choice(env, item.option));
    env.adapter.refreshAnyDomPending();
    assert.ok(values(env).includes(item.target), `${item.card}: first selection`);
    await env.adapter.choose(choice(env, item.target));
    env.adapter.refreshAnyDomPending();
    assert.ok(values(env).includes(item.target), `${item.card}: second selection`);
    await env.adapter.choose(choice(env, item.target));
    env.adapter.refreshAnyDomPending();
    assert.ok(!values(env).includes(item.target), `${item.card}: third selection rejected`);
  }
});

test("Vietnam and China placement bonuses are one restricted Op and allow repeat-country placement", async () => {
  for (const item of [
    { kind: "vietnam", card: "redscare", region: new Set(["seasia"]) },
    { kind: "china", card: "china", region: new Set(["asia", "seasia"]) },
  ]) {
    const env = await startOps({ ...item, ops: 4 });
    const basePlacements = await spendPlacementUntilBonus(env, "vietnam");
    assert.equal(basePlacements, 4);
    assert.match(env.adapter.normalizeStatus(), /extra 1 op available/);
    assert.ok(values(env).includes("vietnam"), "the same country remains a legal bonus target");
    for (const action of env.adapter.pending.choices) {
      assert.ok(item.region.has(env.twilight.countries[action.value].region), `${item.kind}: ${action.value} is region legal`);
    }
    await env.stepAsync(choice(env, "vietnam"));
    assert.equal(env.twilight.countries.vietnam.ussr, 7);
    assert.doesNotMatch(env.adapter.normalizeStatus(), /extra 1 op available/);
  }
});

test("regional placement bonus preserves opponent-control cost, combines once per source, and honors Red Purge", async () => {
  const controlled = await startOps({ kind: "vietnam", card: "redscare", ops: 4, target: "thailand", us: 2, ussr: 0 });
  const paidPlacements = await spendPlacementUntilBonus(controlled, "thailand");
  assert.equal(paidPlacements, 3, "four printed Ops buy three placements while breaking US control");
  await controlled.stepAsync(choice(controlled, "thailand"));
  assert.equal(controlled.twilight.countries.thailand.ussr, 4, "the bonus buys exactly one further placement");

  const combined = await startOps({ kind: "both", card: "china", ops: 4 });
  assert.equal(await spendPlacementUntilBonus(combined, "vietnam"), 4);
  assert.match(combined.adapter.normalizeStatus(), /southeast asia/);
  await combined.stepAsync(choice(combined, "vietnam"));
  assert.match(combined.adapter.normalizeStatus(), /extra 1 op available for asia/);
  await combined.stepAsync(choice(combined, "vietnam"));
  assert.equal(combined.twilight.countries.vietnam.ussr, 8, "four printed plus Vietnam plus China");

  const purged = await startOps({ kind: "vietnam", card: "romanianab", ops: 1, redPurge: true });
  await purged.stepAsync(choice(purged, "place"));
  await purged.stepAsync(choice(purged, "vietnam"));
  assert.equal(purged.twilight.countries.vietnam.ussr, 3);
  assert.doesNotMatch(purged.adapter.normalizeStatus(), /extra 1 op available/);
});

test("regional coup bonuses modify coup strength exactly once per applicable source", async () => {
  const cases = [
    { kind: "vietnam", card: "redscare", expected: 5 },
    { kind: "china", card: "china", expected: 5 },
    { kind: "both", card: "china", expected: 6 },
  ];
  for (const item of cases) {
    const env = await startOps({ ...item, ops: 4, target: "thailand", us: 2, ussr: 0 });
    await env.stepAsync(choice(env, "coup"));
    env.twilight.endTurn = () => {};
    await env.adapter.choose(choice(env, "thailand"));
    const coupMove = env.twilight.moves.find((move) => move.startsWith("coup\tussr\tthailand\t"));
    assert.equal(Number(coupMove.split("\t")[3]), item.expected, item.kind);
  }
  const purged = await startOps({ kind: "vietnam", card: "romanianab", ops: 1, target: "thailand", us: 2, ussr: 0, redPurge: true });
  await purged.stepAsync(choice(purged, "coup"));
  purged.twilight.endTurn = () => {};
  await purged.adapter.choose(choice(purged, "thailand"));
  const purgedMove = purged.twilight.moves.find((move) => move.startsWith("coup\tussr\tthailand\t"));
  assert.equal(Number(purgedMove.split("\t")[3]), 1);
});

test("Vietnam and China realignment bonuses expose one region-restricted repeat roll", async () => {
  for (const item of [
    { kind: "vietnam", card: "olympic", ops: 2, region: new Set(["seasia"]) },
    { kind: "china", card: "china", ops: 4, region: new Set(["asia", "seasia"]) },
  ]) {
    const env = await startOps({ ...item, target: "vietnam", us: 1, ussr: 1 });
    await env.stepAsync(choice(env, "realign"));
    for (let roll = 0; roll < item.ops; roll += 1) {
      await env.stepAsync(choice(env, "vietnam"), [3, 3]);
    }
    assert.match(env.adapter.normalizeStatus(), /extra 1 op available/);
    assert.ok(values(env).includes("vietnam"));
    for (const action of env.adapter.pending.choices.filter((entry) => env.twilight.countries[entry.value])) {
      assert.ok(item.region.has(env.twilight.countries[action.value].region), `${item.kind}: ${action.value} is region legal`);
    }
    await env.stepAsync(choice(env, "vietnam"), [3, 3]);
    assert.doesNotMatch(env.adapter.normalizeStatus(), /extra 1 op available/);
  }
});
