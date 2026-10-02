#!/usr/bin/env node
"use strict";

// Read-only differential helper. Input is JSON-lines on stdin. Trace strings are
// data only and are never evaluated or used as selectors or filesystem paths.

const readline = require("readline");
const { SaitoTwilightKernel } = require("../../bridge/saito_bridge.js");

function kernel() {
  const env = new SaitoTwilightKernel();
  env.reset(1);
  env.twilight.browser_active = 0;
  return env;
}

function applyCountries(env, countries = {}) {
  for (const [id, influence] of Object.entries(countries)) {
    if (!env.twilight.countries[id]) continue;
    env.twilight.countries[id].us = Number(influence.us || 0);
    env.twilight.countries[id].ussr = Number(influence.ussr || 0);
    if (env.twilight.game.countries?.[id]) {
      env.twilight.game.countries[id].us = Number(influence.us || 0);
      env.twilight.game.countries[id].ussr = Number(influence.ussr || 0);
    }
  }
}

function applyEvents(env, events = {}) {
  for (const [key, value] of Object.entries(events)) {
    // Keys are emitted by the fixed local parser, never taken from trace text.
    env.twilight.game.state.events[key] = value;
  }
}

function countryState(env, id) {
  const country = env.twilight.countries[id];
  return { us: Number(country.us || 0), ussr: Number(country.ussr || 0) };
}

function handle(request) {
  if (request.kind === "meta") {
    const env = kernel();
    const cards = env.cardsArray().map((card) => ({
      id: card.id,
      name: card.name,
      ops: card.ops,
      scoring: card.scoring,
      recurring: Boolean(env.cardsById?.[card.id]?.recurring),
    }));
    const countries = Object.entries(env.twilight.countries).map(([id, country]) => ({
      id,
      name: country.name,
      stability: Number(country.control),
      region: country.region,
      bg: Number(country.bg || 0),
      neighbours: [...country.neighbours],
      us: Number(country.us || 0),
      ussr: Number(country.ussr || 0),
    }));
    return { cards, countries };
  }

  const env = kernel();
  applyCountries(env, request.countries);
  applyEvents(env, request.events);
  if (request.defcon != null) env.twilight.game.state.defcon = Number(request.defcon);
  if (request.lower_defcon_on_coup != null) {
    env.twilight.game.state.lower_defcon_on_coup = Number(request.lower_defcon_on_coup);
  }

  if (request.kind === "influence") {
    const id = request.country;
    if (!env.twilight.countries[id]) throw new Error(`unknown country ID: ${id}`);
    if (request.delta > 0) env.twilight.placeInfluence(id, Number(request.delta), request.side);
    else env.twilight.removeInfluence(id, Math.abs(Number(request.delta)), request.side);
    return { country: countryState(env, id) };
  }

  if (request.kind === "coup") {
    env.forcedRolls = [Number(request.roll)];
    env.lastRolls = [];
    env.twilight.playCoup(request.side, request.country, Number(request.ops));
    return {
      country: countryState(env, request.country),
      defcon: Number(env.twilight.game.state.defcon),
      vp: Number(env.twilight.game.state.vp),
      rolls: [...env.lastRolls],
    };
  }

  if (request.kind === "realign") {
    // Saito consumes the US die first even though Steam prints USSR first.
    env.forcedRolls = [Number(request.us_roll), Number(request.ussr_roll)];
    env.lastRolls = [];
    env.twilight.playRealign(request.side, request.country);
    return {
      country: countryState(env, request.country),
      rolls: [...env.lastRolls],
      log: env.log.slice(-4),
    };
  }

  if (request.kind === "score") {
    if (request.region === "seasia") {
      const before = Number(env.twilight.game.state.vp || 0);
      env.twilight.playEvent(request.side || "ussr", "seasia");
      return { scoring: null, vp_delta: Number(env.twilight.game.state.vp || 0) - before };
    }
    const scoring = env.twilight.calculateScoring(request.region, 0);
    return {
      scoring,
      vp_delta: Number(scoring.us.vp || 0) - Number(scoring.ussr.vp || 0),
    };
  }

  if (request.kind === "space") {
    env.twilight.game.state.space_race_us = Number(request.space_us || 0);
    env.twilight.game.state.space_race_ussr = Number(request.space_ussr || 0);
    env.twilight.game.state.vp = 0;
    env.forcedRolls = [Number(request.roll)];
    env.lastRolls = [];
    env.twilight.playerSpaceCard(request.card, request.side);
    return {
      space_us: Number(env.twilight.game.state.space_race_us),
      space_ussr: Number(env.twilight.game.state.space_race_ussr),
      vp_delta: Number(env.twilight.game.state.vp),
      rolls: [...env.lastRolls],
    };
  }

  throw new Error(`unknown request kind: ${request.kind}`);
}

const lines = readline.createInterface({ input: process.stdin, crlfDelay: Infinity });
lines.on("line", (line) => {
  if (!line.trim()) return;
  let request;
  try {
    request = JSON.parse(line);
    const result = handle(request);
    process.stdout.write(`${JSON.stringify({ id: request.id, ok: true, result })}\n`);
  } catch (error) {
    process.stdout.write(`${JSON.stringify({ id: request?.id, ok: false, error: error.message })}\n`);
  }
});
