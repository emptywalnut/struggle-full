"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");

test("every DEFCON state maps to the exact Saito cell", async () => {
  const { DEFCON_TRACK, defconTrackPosition } = await import("../web/board-layout.mjs");
  const expected = {
    5: { top: 2592, left: 1526 },
    4: { top: 2592, left: 1682 },
    3: { top: 2592, left: 1838 },
    2: { top: 2592, left: 1994 },
    1: { top: 2592, left: 2150 },
  };
  assert.equal(DEFCON_TRACK.length, 5);
  for (const [level, position] of Object.entries(expected)) {
    const actual = defconTrackPosition(Number(level));
    assert.deepEqual({ top: actual.top, left: actual.left }, position);
  }
});

test("all -20 through +20 VP states preserve Saito's USSR-to-US direction", async () => {
  const { VP_TRACK, victoryPointTrackPosition } = await import("../web/board-layout.mjs");
  assert.equal(VP_TRACK.length, 41);
  for (let vp = -20; vp <= 20; vp += 1) {
    const actual = victoryPointTrackPosition(vp);
    assert.equal(actual.value, vp);
    assert.deepEqual(actual, VP_TRACK[vp + 20]);
  }
  assert.deepEqual(victoryPointTrackPosition(-20), { value: -20, top: 2466, left: 3108 });
  assert.deepEqual(victoryPointTrackPosition(0), { value: 0, top: 2748, left: 3572 });
  assert.deepEqual(victoryPointTrackPosition(20), { value: 20, top: 3030, left: 4042 });
});

test("every MilOps state maps to its cell and coincident counters remain distinct", async () => {
  const { MILOPS_TRACK, militaryOperationsTrackPositions } = await import("../web/board-layout.mjs");
  const expectedLeft = [1520, 1675, 1830, 1985, 2150, 2305];
  assert.equal(MILOPS_TRACK.length, 6);
  for (let value = 0; value <= 5; value += 1) {
    const actual = militaryOperationsTrackPositions({ us: value, ussr: value });
    assert.equal(actual.coincident, true);
    assert.deepEqual(
      { top: actual.us.cellTop, left: actual.us.cellLeft },
      { top: 2940, left: expectedLeft[value] },
    );
    assert.deepEqual(
      { top: actual.ussr.cellTop, left: actual.ussr.cellLeft },
      { top: 2940, left: expectedLeft[value] },
    );
    assert.notDeepEqual(
      { top: actual.us.top, left: actual.us.left },
      { top: actual.ussr.top, left: actual.ussr.left },
    );
    assert.deepEqual(
      { top: actual.ussr.top, left: actual.ussr.left },
      { top: 2980, left: expectedLeft[value] + 10 },
    );
  }
});

test("every Space Race state maps to Saito's exact cell and preserves its Soviet offset", async () => {
  const { SPACE_RACE_TRACK, spaceRaceTrackPositions } = await import("../web/board-layout.mjs");
  const expectedLeft = [3465, 3638, 3810, 3980, 4150, 4320, 4490, 4660, 4830];
  assert.equal(SPACE_RACE_TRACK.length, 9);
  for (let value = 0; value <= 8; value += 1) {
    const actual = spaceRaceTrackPositions({ us: value, ussr: value });
    assert.equal(actual.coincident, true);
    assert.deepEqual(
      { top: actual.us.cellTop, left: actual.us.cellLeft },
      { top: 510, left: expectedLeft[value] },
    );
    assert.deepEqual(
      { top: actual.ussr.cellTop, left: actual.ussr.cellLeft },
      { top: 510, left: expectedLeft[value] },
    );
    assert.deepEqual(
      { top: actual.ussr.top, left: actual.ussr.left },
      { top: 550, left: expectedLeft[value] + 10 },
    );
    assert.notDeepEqual(
      { top: actual.us.top, left: actual.us.left },
      { top: actual.ussr.top, left: actual.ussr.left },
    );
  }
  assert.equal(spaceRaceTrackPositions({ us: -2, ussr: 99 }).us.value, 0);
  assert.equal(spaceRaceTrackPositions({ us: -2, ussr: 99 }).ussr.value, 8);
});

test("Space Race component renders both accessible counters in every coincident cell", async () => {
  const { SPACE_RACE_TRACK, renderBoardStateMarkers } = await import("../web/board-layout.mjs");
  class Element {
    constructor() { this.className = ""; this.style = {}; this.attributes = {}; this.children = []; this.textContent = ""; }
    setAttribute(name, value) { this.attributes[name] = String(value); }
    append(...children) { this.children.push(...children); }
  }
  const document = { createElement: () => new Element() };
  for (let value = 0; value <= 8; value += 1) {
    const layer = renderBoardStateMarkers(document, {
      defcon: 5,
      vp: 0,
      milops: { us: 0, ussr: 0 },
      space: { us: value, ussr: value },
    });
    const us = layer.children[4];
    const ussr = layer.children[5];
    assert.match(us.className, /track-space us coincident/);
    assert.match(ussr.className, /track-space ussr coincident/);
    assert.equal(us.attributes["aria-label"], `US Space Race position ${value} on the Space Race track`);
    assert.equal(ussr.attributes["aria-label"], `USSR Space Race position ${value} on the Space Race track`);
    assert.equal(us.style.top, `${SPACE_RACE_TRACK[value].top}px`);
    assert.equal(us.style.left, `${SPACE_RACE_TRACK[value].left}px`);
    assert.equal(ussr.style.top, `${SPACE_RACE_TRACK[value].top + 40}px`);
    assert.equal(ussr.style.left, `${SPACE_RACE_TRACK[value].left + 10}px`);
  }
});

test("rendered board markers expose authoritative values accessibly", async () => {
  const { renderBoardStateMarkers } = await import("../web/board-layout.mjs");
  class Element {
    constructor() { this.className = ""; this.style = {}; this.attributes = {}; this.children = []; this.textContent = ""; }
    setAttribute(name, value) { this.attributes[name] = String(value); }
    append(...children) { this.children.push(...children); }
  }
  const document = { createElement: () => new Element() };
  const layer = renderBoardStateMarkers(document, {
    defcon: 3,
    vp: 0,
    milops: { us: 2, ussr: 2 },
    space: { us: 4, ussr: 4 },
  });
  assert.equal(layer.children.length, 6);
  assert.equal(layer.attributes.role, "group");
  assert.match(layer.children[0].attributes["aria-label"], /DEFCON 3/);
  assert.match(layer.children[1].attributes["aria-label"], /Victory Points 0, neutral/);
  assert.match(layer.children[2].attributes["aria-label"], /US Military Operations 2/);
  assert.match(layer.children[3].attributes["aria-label"], /USSR Military Operations 2/);
  assert.notEqual(layer.children[2].style.top, layer.children[3].style.top);
  assert.match(layer.children[4].attributes["aria-label"], /US Space Race position 4/);
  assert.match(layer.children[5].attributes["aria-label"], /USSR Space Race position 4/);
  assert.equal(layer.children[4].style.top, "510px");
  assert.equal(layer.children[4].style.left, "4150px");
  assert.equal(layer.children[5].style.top, "550px");
  assert.equal(layer.children[5].style.left, "4160px");
});
