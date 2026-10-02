export const SIDE_LABELS = { us: "United States", ussr: "Soviet Union", observer: "Observer" };

export const PHASE_LABELS = {
  boot: "Preparing round",
  setup_ussr: "USSR setup",
  setup_us: "US setup",
  setup_us_bonus: "US bonus setup",
  headline_ussr: "USSR headline",
  headline_us: "US headline",
  headline_resolve: "Headline resolution",
  action: "Action rounds",
};

export const REGION_LABELS = {
  europe: "Europe",
  asia: "Asia",
  seasia: "Southeast Asia",
  mideast: "Middle East",
  africa: "Africa",
  camerica: "Central America",
  samerica: "South America",
};

export const REGION_SCORING = {
  europe: "3 / 7 / auto-win",
  asia: "3 / 7 / 9",
  mideast: "3 / 5 / 7",
  africa: "1 / 4 / 6",
  camerica: "1 / 3 / 5",
  samerica: "2 / 5 / 6",
  seasia: "1 each · Thailand 2",
};

export function phaseLabel(phase) {
  return PHASE_LABELS[phase] || String(phase || "Unknown phase").replaceAll("_", " ");
}

export function sideLabel(side) {
  return SIDE_LABELS[side] || String(side || "Unknown").toUpperCase();
}

export function actionLabel(action) {
  if (!action) return "Unknown action";
  const label = action.label || action.value || action.type || "Action";
  const prefixes = {
    country_click: "Target",
    country_mouseup: "Target",
    option: "Choose",
    list: "Play",
    back: "Return",
  };
  const prefix = prefixes[action.decision];
  return prefix ? `${prefix}: ${label}` : String(label);
}

export function plainText(value) {
  return String(value || "")
    .replace(/<[^>]+>/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function normalizedName(value) {
  return plainText(value).toLowerCase().replace(/[^a-z0-9]+/g, "");
}

export function cardPlayContext(observation = {}, cards = []) {
  const prompt = plainText(observation.prompt);
  const match = /^(?:(?:US|USSR)\s+playing|Playing)\s+(.+?):?$/i.exec(prompt);
  if (!match) return null;
  const requestedName = match[1].replace(/:$/, "").trim();
  const requestedKey = normalizedName(requestedName);
  const card = cards.find((candidate) => normalizedName(candidate.name) === requestedKey)
    || cards.find((candidate) => requestedKey.includes(normalizedName(candidate.name)));
  return card ? { card, prompt } : null;
}

function humanizeOption(value) {
  return String(value || "option")
    .replaceAll("_", " ")
    .replace(/\b\w/g, (character) => character.toUpperCase());
}

export function actionPresentation(action, observation = {}, cards = []) {
  const context = cardPlayContext(observation, cards);
  if (!context || action?.decision !== "option") {
    return { label: actionLabel(action), meta: action?.decision || action?.type || "action" };
  }
  const card = context.card;
  const ops = Number(card.ops || 0);
  const modes = {
    event: { label: card.scoring ? `Score ${card.name}` : `Play ${card.name} as its event`, meta: card.scoring ? "Required scoring" : "Card event" },
    ops: { label: `Use ${card.name} for Operations (${ops} Ops)`, meta: `${ops} printed Ops` },
    space: { label: `Send ${card.name} to the Space Race (${ops} Ops)`, meta: `Space Race · ${ops} printed Ops` },
    cancel_cmc: { label: `Cancel the Cuban Missile Crisis before playing ${card.name}`, meta: "Special card mode" },
    before_ops: { label: `Resolve ${card.name} before Operations`, meta: "Event timing" },
    after_ops: { label: `Resolve ${card.name} after Operations`, meta: "Event timing" },
  };
  return modes[action.value] || {
    label: `${humanizeOption(action.label || action.value)} — ${card.name}`,
    meta: "Special card mode",
  };
}

export function cardPlayModeEntries(legalActions = [], observation = {}, cards = []) {
  if (!cardPlayContext(observation, cards)) return [];
  return legalActions
    .map((action, index) => ({ action, index }))
    .filter(({ action }) => !["list", "country_click", "country_mouseup"].includes(action.decision));
}

export function renderActionChoiceList(document, entries, observation, cards, handlers = {}, canAct = true) {
  const list = document.createElement("div");
  list.className = "action-list";
  for (const entry of entries) {
    const presentation = actionPresentation(entry.action, observation, cards);
    const button = document.createElement("button");
    button.type = "button";
    button.dataset.actionIndex = String(entry.index);
    button.disabled = !canAct;
    const label = document.createElement("span");
    label.textContent = presentation.label;
    const meta = document.createElement("small");
    meta.textContent = presentation.meta;
    button.append(label, meta);
    button.addEventListener("click", () => {
      if (canAct) handlers.onAction?.(entry.index, entry.action);
    });
    list.append(button);
  }
  return list;
}

export function preferredPanelTab(currentTab, decisionChanged, canAct, legalActions = []) {
  return decisionChanged && canAct && legalActions.length ? "actions" : currentTab;
}

export function policyMetadataText(policy, side) {
  if (!policy) return "Policy metadata unavailable";
  const readiness = policy.readiness?.[side] || {};
  const status = readiness.status === "ready"
    ? `ready on ${readiness.device || "device unknown"}`
    : readiness.status === "available"
      ? "available to load"
      : readiness.status || "unavailable";
  const error = readiness.error ? `error: ${readiness.error}` : "";
  if (policy.kind === "baseline") return [policy.architecture, status, error].filter(Boolean).join(" · ");
  const compatibility = policy.compatibility?.[side] === "native" ? "native schema" : "canonical schema migration";
  const defconMode = policy.inference_environment?.[side]?.defcon_suicide_mode;
  const defconSafety = defconMode === "hard_filter"
    ? "DEFCON suicide filter on"
    : defconMode === "penalty_only"
      ? "DEFCON suicide penalty only"
      : defconMode === "none"
        ? "DEFCON suicide filter off"
        : "";
  const sideTraining = policy.training?.[side] || policy.training || {};
  const training = [
    sideTraining.steps ? `${Number(sideTraining.steps).toLocaleString("en-US")} steps` : "",
    sideTraining.episodes ? `${Number(sideTraining.episodes).toLocaleString("en-US")} episodes` : "",
    sideTraining.updates ? `${Number(sideTraining.updates).toLocaleString("en-US")} updates` : "",
  ].filter(Boolean).join(" · ");
  return [policy.architecture?.replaceAll("_", " "), compatibility, defconSafety, training, status, error].filter(Boolean).join(" · ");
}

export function selectablePolicies(catalog, side) {
  return (catalog?.policies || []).filter((policy) => policy.readiness?.[side]?.status !== "incompatible");
}

export function legalCountryActionIndex(legalActions, countryId) {
  return legalActions.findIndex((action) => (
    action.value === countryId
    && ["country_click", "country_mouseup"].includes(action.decision)
  ));
}

export function legalCardActionIndex(legalActions, cardId) {
  return legalActions.findIndex((action) => action.value === cardId && action.decision === "list");
}

export function groupLegalActions(legalActions = []) {
  const groups = { cards: [], countries: [], options: [], other: [] };
  legalActions.forEach((action, index) => {
    const entry = { action, index };
    if (action.decision === "list") groups.cards.push(entry);
    else if (["country_click", "country_mouseup"].includes(action.decision)) groups.countries.push(entry);
    else if (["option", "back"].includes(action.decision)) groups.options.push(entry);
    else groups.other.push(entry);
  });
  return groups;
}

export function summarizeRegions(countries = []) {
  const summary = {};
  for (const country of countries) {
    const region = country.region;
    if (!summary[region]) {
      summary[region] = {
        region,
        us: { countries: 0, battlegrounds: 0 },
        ussr: { countries: 0, battlegrounds: 0 },
      };
    }
    if (country.control === "us" || country.control === "ussr") {
      const side = summary[region][country.control];
      side.countries += 1;
      side.battlegrounds += Number(Boolean(country.bg));
    }
  }
  return summary;
}

export function cardSideClass(card) {
  if (!card) return "both";
  return card.side === "us" ? "us" : card.side === "ussr" ? "ussr" : "both";
}

export function countryAccessibleLabel(country, isLegal) {
  const control = country.control === "none" ? "uncontrolled" : `${sideLabel(country.control)} controlled`;
  return `${country.name}, ${REGION_LABELS[country.region] || country.region}, stability ${country.stability}${country.bg ? ", battleground" : ""}, US ${country.us}, USSR ${country.ussr}, ${control}${isLegal ? ", legal target" : ""}`;
}

export function renderCountryComponent(document, country, legalActions, handlers = {}, canAct = true) {
  const button = document.createElement("button");
  const actionIndex = legalCountryActionIndex(legalActions, country.id);
  const isLegal = actionIndex >= 0;
  button.type = "button";
  button.className = `country-hotspot region-${country.region}${country.bg ? " battleground" : ""}${isLegal ? " is-legal" : ""}${country.control !== "none" ? ` controlled-${country.control}` : ""}`;
  button.dataset.country = country.id;
  button.dataset.actionIndex = String(actionIndex);
  button.style.left = `${country.left}px`;
  button.style.top = `${country.top}px`;
  button.setAttribute("aria-label", countryAccessibleLabel(country, isLegal));
  button.title = countryAccessibleLabel(country, isLegal);

  const markers = document.createElement("span");
  markers.className = "influence-markers";
  for (const side of ["us", "ussr"]) {
    if (!country[side]) continue;
    const marker = document.createElement("span");
    marker.className = `influence-marker ${side}${country.control === side ? " controlled" : ""}`;
    marker.textContent = String(country[side]);
    marker.setAttribute("aria-hidden", "true");
    markers.append(marker);
  }
  button.append(markers);
  button.addEventListener("click", () => {
    if (isLegal && canAct) handlers.onAction?.(actionIndex, country);
    else handlers.onIllegal?.(country, isLegal);
  });
  return button;
}

export function eventEntries(events = {}) {
  return Object.entries(events)
    .filter(([, value]) => value !== 0 && value !== "" && value !== null && value !== false && !(Array.isArray(value) && value.length === 0))
    .sort(([a], [b]) => a.localeCompare(b));
}
