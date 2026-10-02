// Native 5100x3300 board coordinates copied from the materialized Saito
// Twilight implementation's state.*_ps tables. These are top-left positions
// for Saito's 120px DEFCON/VP tiles and 100px MilOps/Space Race tiles.

export const DEFCON_TRACK = Object.freeze([
  Object.freeze({ level: 5, top: 2592, left: 1526 }),
  Object.freeze({ level: 4, top: 2592, left: 1682 }),
  Object.freeze({ level: 3, top: 2592, left: 1838 }),
  Object.freeze({ level: 2, top: 2592, left: 1994 }),
  Object.freeze({ level: 1, top: 2592, left: 2150 }),
]);

export const MILOPS_TRACK = Object.freeze([
  Object.freeze({ value: 0, top: 2940, left: 1520 }),
  Object.freeze({ value: 1, top: 2940, left: 1675 }),
  Object.freeze({ value: 2, top: 2940, left: 1830 }),
  Object.freeze({ value: 3, top: 2940, left: 1985 }),
  Object.freeze({ value: 4, top: 2940, left: 2150 }),
  Object.freeze({ value: 5, top: 2940, left: 2305 }),
]);

export const SPACE_RACE_TRACK = Object.freeze([
  Object.freeze({ value: 0, top: 510, left: 3465 }),
  Object.freeze({ value: 1, top: 510, left: 3638 }),
  Object.freeze({ value: 2, top: 510, left: 3810 }),
  Object.freeze({ value: 3, top: 510, left: 3980 }),
  Object.freeze({ value: 4, top: 510, left: 4150 }),
  Object.freeze({ value: 5, top: 510, left: 4320 }),
  Object.freeze({ value: 6, top: 510, left: 4490 }),
  Object.freeze({ value: 7, top: 510, left: 4660 }),
  Object.freeze({ value: 8, top: 510, left: 4830 }),
]);

export const VP_TRACK = Object.freeze([
  Object.freeze({ value: -20, top: 2466, left: 3108 }),
  Object.freeze({ value: -19, top: 2466, left: 3304 }),
  Object.freeze({ value: -18, top: 2466, left: 3440 }),
  Object.freeze({ value: -17, top: 2466, left: 3572 }),
  Object.freeze({ value: -16, top: 2466, left: 3708 }),
  Object.freeze({ value: -15, top: 2466, left: 3840 }),
  Object.freeze({ value: -14, top: 2466, left: 3972 }),
  Object.freeze({ value: -13, top: 2466, left: 4108 }),
  Object.freeze({ value: -12, top: 2608, left: 3036 }),
  Object.freeze({ value: -11, top: 2608, left: 3172 }),
  Object.freeze({ value: -10, top: 2608, left: 3304 }),
  Object.freeze({ value: -9, top: 2608, left: 3440 }),
  Object.freeze({ value: -8, top: 2608, left: 3572 }),
  Object.freeze({ value: -7, top: 2608, left: 3708 }),
  Object.freeze({ value: -6, top: 2608, left: 3840 }),
  Object.freeze({ value: -5, top: 2608, left: 3972 }),
  Object.freeze({ value: -4, top: 2608, left: 4108 }),
  Object.freeze({ value: -3, top: 2748, left: 3036 }),
  Object.freeze({ value: -2, top: 2748, left: 3172 }),
  Object.freeze({ value: -1, top: 2748, left: 3304 }),
  Object.freeze({ value: 0, top: 2748, left: 3572 }),
  Object.freeze({ value: 1, top: 2748, left: 3840 }),
  Object.freeze({ value: 2, top: 2748, left: 3972 }),
  Object.freeze({ value: 3, top: 2748, left: 4108 }),
  Object.freeze({ value: 4, top: 2888, left: 3036 }),
  Object.freeze({ value: 5, top: 2888, left: 3172 }),
  Object.freeze({ value: 6, top: 2888, left: 3304 }),
  Object.freeze({ value: 7, top: 2888, left: 3440 }),
  Object.freeze({ value: 8, top: 2888, left: 3572 }),
  Object.freeze({ value: 9, top: 2888, left: 3708 }),
  Object.freeze({ value: 10, top: 2888, left: 3840 }),
  Object.freeze({ value: 11, top: 2888, left: 3972 }),
  Object.freeze({ value: 12, top: 2888, left: 4108 }),
  Object.freeze({ value: 13, top: 3030, left: 3036 }),
  Object.freeze({ value: 14, top: 3030, left: 3172 }),
  Object.freeze({ value: 15, top: 3030, left: 3304 }),
  Object.freeze({ value: 16, top: 3030, left: 3440 }),
  Object.freeze({ value: 17, top: 3030, left: 3572 }),
  Object.freeze({ value: 18, top: 3030, left: 3708 }),
  Object.freeze({ value: 19, top: 3030, left: 3840 }),
  Object.freeze({ value: 20, top: 3030, left: 4042 }),
]);

function boundedInteger(value, minimum, maximum) {
  const numeric = Number(value);
  const integer = Number.isFinite(numeric) ? Math.trunc(numeric) : minimum;
  return Math.max(minimum, Math.min(maximum, integer));
}

export function defconTrackPosition(value) {
  const level = boundedInteger(value, 1, 5);
  return DEFCON_TRACK.find((cell) => cell.level === level);
}

export function victoryPointTrackPosition(value) {
  const vp = boundedInteger(value, -20, 20);
  return VP_TRACK[vp + 20];
}

export function militaryOperationsTrackPositions(milops = {}) {
  const usValue = boundedInteger(milops.us, 0, 5);
  const ussrValue = boundedInteger(milops.ussr, 0, 5);
  const usCell = MILOPS_TRACK[usValue];
  const ussrCell = MILOPS_TRACK[ussrValue];
  const coincident = usValue === ussrValue;

  return {
    coincident,
    us: { ...usCell, side: "us", cellTop: usCell.top, cellLeft: usCell.left },
    // Saito deliberately shifts the Soviet tile 10px right and 40px down.
    // The underlying cell coordinates remain available for exact semantics.
    ussr: {
      ...ussrCell,
      side: "ussr",
      cellTop: ussrCell.top,
      cellLeft: ussrCell.left,
      top: ussrCell.top + 40,
      left: ussrCell.left + 10,
    },
  };
}

export function spaceRaceTrackPositions(space = {}) {
  const usValue = boundedInteger(space.us, 0, 8);
  const ussrValue = boundedInteger(space.ussr, 0, 8);
  const usCell = SPACE_RACE_TRACK[usValue];
  const ussrCell = SPACE_RACE_TRACK[ussrValue];
  const coincident = usValue === ussrValue;

  return {
    coincident,
    us: { ...usCell, side: "us", cellTop: usCell.top, cellLeft: usCell.left },
    // Saito's updateSpaceRace always offsets the Soviet tile 10px right and
    // 40px down. This is only visual separation; cellTop/cellLeft preserve
    // the authoritative 0..8 track position.
    ussr: {
      ...ussrCell,
      side: "ussr",
      cellTop: ussrCell.top,
      cellLeft: ussrCell.left,
      top: ussrCell.top + 40,
      left: ussrCell.left + 10,
    },
  };
}

function marker(document, className, label, text, position) {
  const element = document.createElement("span");
  element.className = `board-state-marker ${className}`;
  element.style.top = `${position.top}px`;
  element.style.left = `${position.left}px`;
  element.textContent = text;
  element.setAttribute("role", "img");
  element.setAttribute("aria-label", label);
  return element;
}

export function renderBoardStateMarkers(document, observation) {
  const layer = document.createElement("div");
  layer.className = "board-state-markers";
  layer.setAttribute("role", "group");
  layer.setAttribute("aria-label", "Live board state tracks");

  const defcon = defconTrackPosition(observation.defcon);
  layer.append(marker(
    document,
    "track-defcon",
    `DEFCON ${defcon.level} on the DEFCON status track`,
    `D${defcon.level}`,
    defcon,
  ));

  const vp = victoryPointTrackPosition(observation.vp);
  const vpOwner = vp.value > 0 ? "US" : vp.value < 0 ? "USSR" : "neutral";
  const vpText = vp.value > 0 ? `+${vp.value}` : String(vp.value);
  layer.append(marker(
    document,
    `track-vp ${vpOwner.toLowerCase()}`,
    `Victory Points ${vp.value}, ${vpOwner}, on the victory point track`,
    vpText,
    vp,
  ));

  const military = militaryOperationsTrackPositions(observation.milops);
  layer.append(marker(
    document,
    `track-milops us${military.coincident ? " coincident" : ""}`,
    `US Military Operations ${military.us.value} on the required military operations track`,
    "US",
    military.us,
  ));
  layer.append(marker(
    document,
    `track-milops ussr${military.coincident ? " coincident" : ""}`,
    `USSR Military Operations ${military.ussr.value} on the required military operations track`,
    "SU",
    military.ussr,
  ));

  const space = spaceRaceTrackPositions(observation.space);
  layer.append(marker(
    document,
    `track-space us${space.coincident ? " coincident" : ""}`,
    `US Space Race position ${space.us.value} on the Space Race track`,
    "US",
    space.us,
  ));
  layer.append(marker(
    document,
    `track-space ussr${space.coincident ? " coincident" : ""}`,
    `USSR Space Race position ${space.ussr.value} on the Space Race track`,
    "SU",
    space.ussr,
  ));

  return layer;
}
