"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");

class FakeElement {
  constructor(tagName) {
    this.tagName = tagName.toUpperCase();
    this.className = "";
    this.dataset = {};
    this.style = {};
    this.attributes = {};
    this.children = [];
    this.listeners = {};
    this.textContent = "";
  }
  setAttribute(name, value) { this.attributes[name] = String(value); }
  append(...children) { this.children.push(...children); }
  addEventListener(name, handler) { this.listeners[name] = handler; }
  click() { this.listeners.click?.(); }
}

const fakeDocument = { createElement: (tagName) => new FakeElement(tagName) };

test("influence markers exactly occupy Saito's original 100px country panes", () => {
  const styles = fs.readFileSync(path.join(__dirname, "..", "web", "styles.css"), "utf8");
  const layer = styles.match(/\.influence-markers\s*\{([^}]+)\}/)?.[1] || "";
  const marker = styles.match(/\.influence-marker\s*\{([^}]+)\}/)?.[1] || "";
  const us = styles.match(/\.influence-marker\.us\s*\{([^}]+)\}/)?.[1] || "";
  const ussr = styles.match(/\.influence-marker\.ussr\s*\{([^}]+)\}/)?.[1] || "";

  // The transparent border is cancelled, so country.top is the pane origin.
  assert.match(layer, /top:\s*-9px/);
  assert.match(layer, /left:\s*-9px/);
  assert.match(layer, /width:\s*202px/);
  assert.match(layer, /height:\s*100px/);
  assert.match(marker, /top:\s*0/);
  assert.match(marker, /bottom:\s*auto/);
  assert.match(marker, /width:\s*100px/);
  assert.match(marker, /height:\s*100px/);
  assert.match(us, /left:\s*0/);
  assert.match(ussr, /left:\s*100px/);
});

test("country component aligns to Saito coordinates and submits its legal action", async () => {
  const { renderCountryComponent } = await import("../web/ui-core.mjs");
  const country = {
    id: "poland", name: "Poland", top: 580, left: 2386, region: "europe",
    stability: 3, bg: 1, us: 0, ussr: 4, control: "ussr",
  };
  const legal = [{ value: "poland", label: "Poland", decision: "country_click", type: "saito_dom" }];
  let submitted = null;
  const component = renderCountryComponent(fakeDocument, country, legal, {
    onAction: (index, selected) => { submitted = { index, selected }; },
  }, true);

  assert.equal(component.style.top, "580px");
  assert.equal(component.style.left, "2386px");
  assert.match(component.className, /is-legal/);
  assert.match(component.className, /battleground/);
  assert.match(component.attributes["aria-label"], /stability 3, battleground/);
  component.click();
  assert.deepEqual(submitted, { index: 0, selected: country });
});

test("illegal country component rejects locally without inventing an engine action", async () => {
  const { renderCountryComponent } = await import("../web/ui-core.mjs");
  const country = {
    id: "france", name: "France", top: 906, left: 1820, region: "europe",
    stability: 3, bg: 1, us: 0, ussr: 0, control: "none",
  };
  let rejected = null;
  const component = renderCountryComponent(fakeDocument, country, [], {
    onIllegal: (selected, engineLegal) => { rejected = { selected, engineLegal }; },
  }, true);
  component.click();
  assert.deepEqual(rejected, { selected: country, engineLegal: false });
  assert.equal(component.dataset.actionIndex, "-1");
});

test("country UI follows repeat-target catalogs without imposing a client-side click cap", async () => {
  const { renderCountryComponent } = await import("../web/ui-core.mjs");
  const country = {
    id: "italy", name: "Italy", top: 1060, left: 1995, region: "europe",
    stability: 2, bg: 1, us: 2, ussr: 0, control: "us",
  };
  const repeatAction = { value: "italy", label: "Italy", decision: "country_click", type: "saito_dom" };
  const submitted = [];
  for (const legal of [[repeatAction], [repeatAction]]) {
    const component = renderCountryComponent(fakeDocument, country, legal, {
      onAction: (index, selected) => submitted.push({ index, id: selected.id }),
    }, true);
    assert.match(component.className, /is-legal/);
    component.click();
  }
  const capped = renderCountryComponent(fakeDocument, country, [], {}, true);
  assert.doesNotMatch(capped.className, /is-legal/);
  assert.deepEqual(submitted, [{ index: 0, id: "italy" }, { index: 0, id: "italy" }]);
});

test("legal actions remain exhaustively grouped and region counts use engine control", async () => {
  const { groupLegalActions, summarizeRegions } = await import("../web/ui-core.mjs");
  const actions = [
    { decision: "list", value: "europe" },
    { decision: "option", value: "ops" },
    { decision: "country_click", value: "france" },
    { decision: "back", value: "__back__" },
    { decision: "custom", value: "x" },
  ];
  const groups = groupLegalActions(actions);
  assert.equal(Object.values(groups).flat().length, actions.length);
  assert.equal(groups.cards.length, 1);
  assert.equal(groups.options.length, 2);
  assert.equal(groups.countries.length, 1);
  assert.equal(groups.other.length, 1);

  const regions = summarizeRegions([
    { region: "europe", control: "us", bg: 1 },
    { region: "europe", control: "us", bg: 0 },
    { region: "europe", control: "ussr", bg: 1 },
    { region: "europe", control: "none", bg: 1 },
  ]);
  assert.deepEqual(regions.europe.us, { countries: 2, battlegrounds: 1 });
  assert.deepEqual(regions.europe.ussr, { countries: 1, battlegrounds: 1 });
});

test("human sees and can submit every Duck and Cover card-use mode", async () => {
  const {
    cardPlayContext,
    cardPlayModeEntries,
    groupLegalActions,
    preferredPanelTab,
    renderActionChoiceList,
  } = await import("../web/ui-core.mjs");
  const observation = {
    phase: "action",
    current_player: "us",
    prompt: "US playing <span>Duck and Cover</span>",
  };
  const cards = [{ id: "duckandcover", name: "Duck and Cover", ops: 3, scoring: false }];
  const legal = [
    { type: "saito_choice", decision: "option", value: "event", label: "event", prompt: observation.prompt },
    { type: "saito_choice", decision: "option", value: "ops", label: "ops", prompt: observation.prompt },
    { type: "saito_choice", decision: "option", value: "space", label: "space", prompt: observation.prompt },
  ];

  assert.equal(cardPlayContext(observation, cards).card.id, "duckandcover");
  assert.equal(groupLegalActions(legal).countries.length, 0);
  assert.equal(preferredPanelTab("history", true, true, legal), "actions");

  const entries = cardPlayModeEntries(legal, observation, cards);
  const submitted = [];
  const list = renderActionChoiceList(fakeDocument, entries, observation, cards, {
    onAction: (index, action) => submitted.push({ index, value: action.value }),
  }, true);

  assert.equal(list.children.length, 3);
  assert.deepEqual(list.children.map((button) => button.disabled), [false, false, false]);
  assert.equal(list.children[0].children[0].textContent, "Play Duck and Cover as its event");
  assert.equal(list.children[1].children[0].textContent, "Use Duck and Cover for Operations (3 Ops)");
  assert.equal(list.children[2].children[0].textContent, "Send Duck and Cover to the Space Race (3 Ops)");
  list.children.forEach((button) => button.click());
  assert.deepEqual(submitted, [
    { index: 0, value: "event" },
    { index: 1, value: "ops" },
    { index: 2, value: "space" },
  ]);
});

test("trained policy metadata and readiness remain side-specific", async () => {
  const { policyMetadataText, selectablePolicies } = await import("../web/ui-core.mjs");
  const model = {
    id: "trained",
    kind: "model",
    architecture: "transformer_history",
    compatibility: { us: "native", ussr: "canonical_widening" },
    inference_environment: {
      us: { defcon_suicide_mode: "hard_filter" },
      ussr: { defcon_suicide_mode: "none" },
    },
    training: { us: { steps: 12650523, episodes: 57005 } },
    readiness: {
      us: { status: "ready", device: "cuda:3" },
      ussr: { status: "available", device: null },
    },
  };
  assert.match(policyMetadataText(model, "us"), /transformer history · native schema/);
  assert.match(policyMetadataText(model, "us"), /DEFCON suicide filter on/);
  assert.match(policyMetadataText(model, "us"), /12,650,523 steps · 57,005 episodes/);
  assert.match(policyMetadataText(model, "us"), /ready on cuda:3/);
  assert.match(policyMetadataText(model, "ussr"), /canonical schema migration/);
  assert.match(policyMetadataText(model, "ussr"), /available to load/);
  assert.match(policyMetadataText(model, "ussr"), /DEFCON suicide filter off/);
  assert.deepEqual(selectablePolicies({ policies: [model] }, "us"), [model]);
});
