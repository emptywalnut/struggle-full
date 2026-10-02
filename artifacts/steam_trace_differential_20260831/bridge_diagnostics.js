#!/usr/bin/env node
"use strict";

// Fixed, read-only/minimized bridge probes for trace-backed rule differences.
// This script mutates only fresh in-memory kernels and never production state.

const { SaitoTwilightKernel } = require("../../bridge/saito_bridge.js");

function fresh(side = "ussr") {
  const env = new SaitoTwilightKernel();
  env.reset(1);
  env.currentPlayer = side;
  env.twilight.game.player = side === "ussr" ? 1 : 2;
  env.phase = "action_round";
  env.adapter.reset();
  // reset() starts before era dealing; install the canonical card definitions
  // solely so isolated event handlers can be invoked without a synthetic game.
  env.twilight.game.deck[0].cards = { ...env.cardsById };
  for (const country of Object.values(env.twilight.countries)) {
    country.us = 0;
    country.ussr = 0;
    country.place = 0;
  }
  for (const country of Object.values(env.twilight.game.countries || {})) {
    country.us = 0;
    country.ussr = 0;
    country.place = 0;
  }
  return env;
}

function setInfluence(env, id, us, ussr) {
  for (const countries of [env.twilight.countries, env.twilight.game.countries]) {
    countries[id].us = us;
    countries[id].ussr = ussr;
  }
}

function pendingView(env) {
  const pending = env.adapter.pending;
  return {
    prompt: env.adapter.normalizeStatus(env.adapter.lastStatus),
    choices: (pending?.choices || []).map((choice) => ({
      type: choice.type,
      value: choice.value,
      selector: choice.selector || null,
      event: choice.event || null,
    })),
  };
}

function findChoice(env, value) {
  const choice = env.adapter.pending?.choices.find((item) => item.value === value);
  if (!choice) {
    throw new Error(`missing choice ${value}: ${JSON.stringify(pendingView(env))}`);
  }
  return choice;
}

async function boundedEvent({ card, target, us = 0, ussr = 0, vp = 0, option = null }) {
  const env = fresh("ussr");
  env.twilight.game.state.vp = vp;
  setInfluence(env, target, us, ussr);
  const returnCode = env.twilight.playEvent("ussr", card);
  if (option) await env.adapter.choose(findChoice(env, option));
  env.adapter.refreshAnyDomPending();
  const before = pendingView(env);
  await env.adapter.choose(findChoice(env, target));
  env.adapter.refreshAnyDomPending();
  const after = pendingView(env);
  const influenceAfterFirst = {
    us: env.twilight.countries[target].us,
    ussr: env.twilight.countries[target].ussr,
  };
  const legalAfterFirst = after.choices.some((choice) => choice.value === target);
  if (legalAfterFirst) await env.adapter.choose(findChoice(env, target));
  env.adapter.refreshAnyDomPending();
  const afterSecond = pendingView(env);
  return {
    card,
    target,
    event_return: returnCode,
    before,
    after_first_selection: after,
    target_still_legal: legalAfterFirst,
    after_second_selection: afterSecond,
    target_legal_after_second: afterSecond.choices.some((choice) => choice.value === target),
    influence_after_first: influenceAfterFirst,
    bounded_selection_count: legalAfterFirst ? 2 : 1,
    target_place_flag: env.twilight.countries[target].place,
  };
}

async function regionalBonus({ kind, card, printedOps, target }) {
  const env = fresh("ussr");
  setInfluence(env, "vietnam", 0, 2);
  if (kind === "vietnam") {
    env.twilight.game.state.events.vietnam_revolts = 1;
    env.twilight.game.state.events.vietnam_revolts_eligible = 1;
  }
  if (kind === "china") {
    env.twilight.game.state.events.china_card_in_play = 1;
    env.twilight.game.state.events.china_card_eligible = 1;
  }
  env.twilight.game.queue = [`ops\tussr\t${card}\t${printedOps}`];
  await env.drainUntilDecisionOrTurnEnd();
  const initial = pendingView(env);
  await env.stepAsync(findChoice(env, "place"));
  const placements = [];
  for (let index = 0; index < printedOps; index += 1) {
    const before = pendingView(env);
    const action = findChoice(env, target);
    await env.stepAsync(action);
    placements.push({ index: index + 1, before, after: pendingView(env) });
  }
  const afterPrintedOps = pendingView(env);
  const influenceAfterPrintedOps = {
    us: env.twilight.countries[target].us,
    ussr: env.twilight.countries[target].ussr,
  };
  let extraPlacementsCompleted = 0;
  if (afterPrintedOps.choices.some((choice) => choice.value === target)) {
    await env.stepAsync(findChoice(env, target));
    extraPlacementsCompleted = 1;
  }
  const afterBonus = pendingView(env);
  return {
    kind,
    card,
    printed_ops: printedOps,
    target,
    initial,
    placements,
    after_printed_ops: afterPrintedOps,
    incorrectly_reentered_full_ops_menu:
      afterPrintedOps.prompt === `ussr plays ${printedOps} ops:` &&
      afterPrintedOps.choices.some((choice) => choice.value === "place"),
    extra_placements_completed: extraPlacementsCompleted,
    total_placements_completed: printedOps + extraPlacementsCompleted,
    after_bonus: afterBonus,
    bonus_prompt_consumed_once: !afterBonus.prompt.includes("extra 1 op available for"),
    influence_after_printed_ops: {
      ...influenceAfterPrintedOps,
    },
    influence_after_replayed_ops: {
      us: env.twilight.countries[target].us,
      ussr: env.twilight.countries[target].ussr,
    },
  };
}

async function regionalSaitoBoundary({ kind, card, printedOps, target }) {
  const env = fresh("ussr");
  setInfluence(env, "vietnam", 0, 2);
  if (kind === "vietnam") {
    env.twilight.game.state.events.vietnam_revolts = 1;
    env.twilight.game.state.events.vietnam_revolts_eligible = 1;
  } else {
    env.twilight.game.state.events.china_card_in_play = 1;
    env.twilight.game.state.events.china_card_eligible = 1;
  }
  env.twilight.game.queue = [`ops\tussr\t${card}\t${printedOps}`];
  await env.drainUntilDecisionOrTurnEnd();
  await env.stepAsync(findChoice(env, "place"));
  for (let index = 0; index < printedOps - 1; index += 1) {
    await env.stepAsync(findChoice(env, target));
  }
  // Invoke the original Saito callback at the exact base-Ops boundary, but do
  // not let stepAsync's queue adapter special-case drain the new prompt.
  await env.adapter.choose(findChoice(env, target));
  env.adapter.refreshAnyDomPending();
  const boundary = pendingView(env);
  return {
    kind,
    prompt_after_original_saito_callback: boundary.prompt,
    choices_after_original_saito_callback: boundary.choices,
    exactly_one_restricted_bonus_prompt:
      boundary.prompt === `extra 1 op available for ${kind === "vietnam" ? "southeast asia" : "asia"}` &&
      boundary.choices.some((choice) => choice.value === target),
  };
}

async function main() {
  const bounded = [];
  bounded.push(await boundedEvent({ card: "socgov", target: "italy", us: 2 }));
  bounded.push(await boundedEvent({ card: "liberation", target: "mexico" }));
  bounded.push(await boundedEvent({ card: "reformer", target: "westgermany", vp: 0 }));
  bounded.push(await boundedEvent({ card: "warsawpact", target: "eastgermany", option: "add" }));
  const regional = [];
  const regionalInputs = [
    { kind: "vietnam", card: "redscare", printedOps: 4, target: "vietnam" },
    { kind: "china", card: "china", printedOps: 4, target: "vietnam" },
  ];
  for (const input of regionalInputs) regional.push(await regionalBonus(input));
  const regional_saito_boundary = [];
  for (const input of regionalInputs) regional_saito_boundary.push(await regionalSaitoBoundary(input));
  process.stdout.write(`${JSON.stringify({ bounded, regional, regional_saito_boundary }, null, 2)}\n`);
}

main().catch((error) => {
  console.error(error.stack || error.message);
  process.exit(1);
});
