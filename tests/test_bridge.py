import json
import copy
import subprocess

from struggle_ai.bridge_client import TwilightBridgeClient


def finish_setup(bridge: TwilightBridgeClient, obs: dict) -> dict:
    while obs["phase"].startswith("setup_"):
        obs = bridge.step(obs["legal_actions"][0])["observation"]
    return obs


def test_trace_minimized_bridge_catalog_defects_are_resolved():
    result = subprocess.run(
        ["node", "artifacts/steam_trace_differential_20260831/bridge_diagnostics.js"],
        cwd=".", text=True, capture_output=True, check=True,
    )
    diagnostics = json.loads(result.stdout)
    assert {item["card"] for item in diagnostics["bounded"]} == {
        "socgov", "liberation", "reformer", "warsawpact",
    }
    for item in diagnostics["bounded"]:
        assert item["target_still_legal"], item["card"]
        assert not item["target_legal_after_second"], item["card"]
        assert item["bounded_selection_count"] == 2
    for item in diagnostics["regional"]:
        assert not item["incorrectly_reentered_full_ops_menu"], item["kind"]
        assert item["extra_placements_completed"] == 1
        assert item["total_placements_completed"] == item["printed_ops"] + 1
        assert item["bonus_prompt_consumed_once"]


def test_preset_and_card_scope():
    with TwilightBridgeClient() as bridge:
        preset = bridge.preset()
        assert preset["id"] == "optional_us_plus_2"
        assert preset["deck"] == "optional"
        assert preset["usbonus"] == 2
        assert preset["backend"] == "saito"
        card_ids = {card["id"] for card in bridge.cards()}
        for card_id in ["muslimrevolution", "cambridge", "defectors", "che", "tehran", "iraniraq", "yuri", "awacs"]:
            assert card_id in card_ids
        for card_id in ["berlinagreement", "pinochet", "perestroika", "gouzenkoaffair", "communistrevolution"]:
            assert card_id not in card_ids


def test_country_metadata_exposes_native_saito_board_coordinates():
    with TwilightBridgeClient() as bridge:
        countries = {country["id"]: country for country in bridge.countries()}
        assert len(countries) == 84
        assert countries["canada"]["top"] == 752
        assert countries["canada"]["left"] == 842
        assert countries["poland"]["top"] == 580
        assert countries["poland"]["left"] == 2386
        assert countries["poland"]["stability"] == 3
        assert countries["poland"]["bg"] == 1
        assert countries["poland"]["neighbours"] == ["eastgermany", "czechoslovakia"]


def test_hidden_hand_observation_and_legal_actions():
    with TwilightBridgeClient() as bridge:
        obs = bridge.reset(3)
        assert obs["side"] == obs["current_player"] == "ussr"
        assert obs["phase"] == "setup_ussr"
        assert obs["hand_count"]["us"] == 8
        assert obs["hand_count"]["ussr"] == 9
        assert "china" in obs["hand"]
        us_obs = bridge.observe("us")
        assert "hand" in us_obs
        assert us_obs["hand"] != obs["hand"]
        assert us_obs["legal_actions"] == []
        assert len(obs["legal_actions"]) > 0


def test_initial_placement_is_wired_before_headline():
    with TwilightBridgeClient() as bridge:
        obs = bridge.reset(1)
        assert obs["phase"] == "setup_ussr"
        assert "Eastern Europe" in obs["prompt"]
        for _ in range(6):
            obs = bridge.step(next(action for action in obs["legal_actions"] if action["value"] == "poland"))["observation"]
        assert obs["phase"] == "setup_us"
        assert "Western Europe" in obs["prompt"]
        for _ in range(7):
            obs = bridge.step(next(action for action in obs["legal_actions"] if action["value"] == "westgermany"))["observation"]
        assert obs["phase"] == "setup_us_bonus"
        assert "+2" in obs["prompt"]
        for _ in range(2):
            obs = bridge.step(next(action for action in obs["legal_actions"] if action["value"] == "iran"))["observation"]
        assert obs["phase"] == "headline_ussr"
        countries = {country["id"]: country for country in obs["countries"]}
        assert countries["poland"]["ussr"] == 6
        assert countries["westgermany"]["us"] == 7
        assert countries["iran"]["us"] == 3


def test_headline_phase_is_wired_before_ar1():
    with TwilightBridgeClient() as bridge:
        obs = finish_setup(bridge, bridge.reset(1))
        assert obs["phase"] == "headline_ussr"
        assert "headline" in obs["prompt"].lower()
        assert obs["legal_actions"]
        ussr_headline = obs["legal_actions"][0]["value"]

        obs = bridge.step(obs["legal_actions"][0])["observation"]
        assert obs["phase"] == "headline_us"
        assert obs["current_player"] == "us"
        assert obs["headline"]["ussr"] is None
        assert not any("USSR selects headline " in entry for entry in obs["log_tail"])
        assert "headline" in obs["prompt"].lower()
        us_headline = obs["legal_actions"][0]["value"]

        for _ in range(40):
            if obs["phase"] == "action" and obs["action_round"] == 0 and obs["current_player"] == "ussr":
                break
            assert obs["legal_actions"]
            obs = bridge.step(obs["legal_actions"][0])["observation"]

        log = bridge.log()
        assert log["headline"]["ussr"] == ussr_headline
        assert log["headline"]["us"] == us_headline
        assert any("headlines" in entry for entry in log["log"])
        assert obs["phase"] == "action"
        assert obs["action_round"] == 0


def test_initial_deal_contains_only_early_war_cards():
    with TwilightBridgeClient() as bridge:
        cards = {card["id"]: card for card in bridge.cards()}
        for seed in range(1, 21):
            ussr_obs = bridge.reset(seed)
            us_obs = bridge.observe("us")
            initial_hands = ussr_obs["hand"] + us_obs["hand"]
            assert initial_hands
            assert all(cards[card_id]["era"] == "early" or card_id == "china" for card_id in initial_hands)


def test_headline_submit_status_does_not_advance_action_round():
    with TwilightBridgeClient() as bridge:
        obs = finish_setup(bridge, bridge.reset(1))
        for value in ["decolonization", "arabisraeli"]:
            action = next(action for action in obs["legal_actions"] if action["value"] == value)
            obs = bridge.step(action)["observation"]

        assert "Africa or Southeast Asia" in obs["prompt"]
        first = "bulgaria"
        first = "burma"
        action = next(action for action in obs["legal_actions"] if action["value"] == first)
        obs = bridge.step(action)["observation"]
        assert first not in {action["value"] for action in obs["legal_actions"]}

        for value in ["laos", "thailand", "vietnam"]:
            action = next(action for action in obs["legal_actions"] if action["value"] == value)
            obs = bridge.step(action)["observation"]

        assert obs["phase"] == "action"
        assert obs["action_round"] == 0
        assert obs["current_player"] == "ussr"
        assert obs["legal_actions"]


def test_random_legal_self_play_reaches_terminal_or_progresses():
    with TwilightBridgeClient() as bridge:
        obs = bridge.reset(5)
        for _ in range(300):
            if obs["terminal"]:
                break
            legal = obs["legal_actions"]
            assert legal
            result = bridge.step(legal[0])
            obs = result["observation"]
            assert obs["side"] == obs["current_player"]
        assert obs["turn"] >= 1


def test_forced_response_observation_uses_post_drain_actor_atomically():
    values = [
        "poland", "poland", "poland", "poland", "eastgermany", "poland",
        "italy", "italy", "france", "france", "france", "westgermany", "westgermany",
        "iran", "westgermany", "blockade", "olympic",
    ]
    with TwilightBridgeClient() as bridge:
        obs = bridge.reset(8_808_001)
        for value in values:
            action = next(action for action in obs["legal_actions"] if action["value"] == value)
            obs = bridge.step(action)["observation"]
            assert obs["side"] == obs["current_player"]

        assert "Olympic Games" in obs["prompt"]
        assert obs["side"] == "ussr"
        participate = next(action for action in obs["legal_actions"] if action["value"] == "participate")
        obs = bridge.step(participate)["observation"]
        assert obs["side"] == obs["current_player"]
        assert "Blockade" in obs["prompt"]
        assert obs["side"] == "us"
        assert "blockade" not in obs["hand"]
        snapshot = bridge.snapshot()
        assert bridge.restore(snapshot)["observation"] == obs


def test_replay_snapshot_restores_exact_decision_state():
    with TwilightBridgeClient() as bridge:
        obs = bridge.reset(17)
        for _ in range(35):
            assert not obs["terminal"]
            obs = bridge.step(obs["legal_actions"][0])["observation"]

        snapshot = bridge.snapshot()
        before = bridge.log()
        action = obs["legal_actions"][0]
        first = bridge.step(action)
        first_log = bridge.log()

        restored = bridge.restore(snapshot)["observation"]
        assert bridge.log() == before
        assert restored == obs

        second = bridge.step(action)
        assert second == first
        assert bridge.log() == first_log


def test_runtime_checkpoint_matches_replay_across_repeated_branches():
    with TwilightBridgeClient() as fast, TwilightBridgeClient() as replay:
        fast_obs = fast.reset(29)
        replay_obs = replay.reset(29)
        for _ in range(55):
            action = fast_obs["legal_actions"][0]
            fast_obs = fast.step(action)["observation"]
            replay_obs = replay.step(action)["observation"]
            assert fast_obs == replay_obs

        fast_snapshot = fast.snapshot()
        replay_snapshot = replay.snapshot()
        replay_snapshot.pop("checkpoint_namespace", None)
        replay_snapshot.pop("checkpoint_token", None)
        actions = copy.deepcopy(fast_obs["legal_actions"])
        for action in (actions[0], actions[-1]):
            fast.restore(fast_snapshot)
            fast_result = fast.step(action, forced_rolls=[0.17] * 16)
            fast_log = fast.log()
            replay.restore(replay_snapshot)
            replay_result = replay.step(action, forced_rolls=[0.17] * 16)
            assert fast_result == replay_result
            assert fast_log == replay.log()


def test_runtime_checkpoint_namespace_is_process_local():
    with TwilightBridgeClient() as source, TwilightBridgeClient() as target:
        snapshot = source.snapshot()
        target.restore(snapshot)
        metrics = target.checkpoint_metrics()
        assert metrics["fast"] == 0
        assert metrics["replay"] == 1


def test_forced_dice_are_validated_and_reported():
    script = r"""
const { SaitoTwilightKernel } = require('./bridge/saito_bridge.js');
const env = new SaitoTwilightKernel();
env.forcedRolls = [2, 6, 0.0, 0.999999, 0.5];
env.lastRolls = [];
const rolls = [
  env.twilight.rollDice(6),
  env.twilight.rollDice(6),
  env.twilight.rollDice(1),
  env.twilight.rollDice(8),
  env.twilight.rollDice(3),
];
const sides = [...env.lastRollSides];
let error = null;
env.forcedRolls = [7];
try { env.twilight.rollDice(6); } catch (err) { error = err.message; }
console.log(JSON.stringify({ rolls, recorded: env.lastRolls, sides, error }));
"""
    result = subprocess.run(["node", "-e", script], cwd=".", text=True, capture_output=True, check=True)
    payload = json.loads(result.stdout)
    assert payload["rolls"] == [2, 6, 1, 8, 2]
    assert payload["recorded"] == [2, 6, 1, 8, 2]
    assert payload["sides"] == [6, 6, 1, 8, 3]
    assert "neither in 1..6 nor a quantile" in payload["error"]


def test_terminal_check_preserves_saito_nuclear_winner():
    script = r"""
const { SaitoTwilightKernel } = require('./bridge/saito_bridge.js');
const env = new SaitoTwilightKernel();
env.currentPlayer = 'us';
env.twilight.game.state.defcon = 1;
env.winner = 'us';
env.terminalReason = 'thermonuclear war';
env.checkTerminal();
const preserved = { winner: env.winner, reason: env.terminalReason };
env.winner = null;
env.terminalReason = null;
env.checkTerminal();
const fallback = { winner: env.winner, reason: env.terminalReason };
console.log(JSON.stringify({ preserved, fallback }));
"""
    result = subprocess.run(["node", "-e", script], cwd=".", text=True, capture_output=True, check=True)
    payload = json.loads(result.stdout)
    assert payload["preserved"] == {"winner": "us", "reason": "thermonuclear war"}
    assert payload["fallback"] == {"winner": "ussr", "reason": "nuclear_war"}


def test_brush_war_excludes_nato_protected_us_controlled_targets():
    script = r"""
const { SaitoTwilightKernel } = require('./bridge/saito_bridge.js');
const env = new SaitoTwilightKernel();
env.reset(1);
env.adapter.lastStatus = 'Pick target for Brush War';
env.twilight.game.state.events.nato = 1;
for (const id of ['italy', 'greece', 'spain', 'turkey']) {
  env.twilight.countries[id].us = 3;
  env.twilight.countries[id].ussr = 0;
}
env.twilight.countries.iran.us = 3;
env.twilight.countries.iran.ussr = 0;
const filtered = env.adapter.filterIdsForContext(
  ['italy', 'greece', 'spain', 'turkey', 'iran'], '.country', 'click'
);
console.log(JSON.stringify(filtered));
"""
    result = subprocess.run(["node", "-e", script], cwd=".", text=True, capture_output=True, check=True)
    assert json.loads(result.stdout) == ["iran"]


def test_async_headline_drain_preserves_ops_priority_before_second_event():
    script = r"""
const { SaitoTwilightKernel } = require('./bridge/saito_bridge.js');
const env = new SaitoTwilightKernel();
env.reset(1);
env.phase = 'headline_resolve';
env.headlineCards = { us: 'duckandcover', ussr: 'junta' };
env.headlineOrder = ['ussr'];
env.resolvingHeadline = { side: 'us', card: 'duckandcover' };
env.currentPlayer = 'us';
env.twilight.game.state.defcon = 3;
env.twilight.game.queue = ['event\tus\tduckandcover'];
env.drainUntilDecisionOrTurnEnd().then(() => {
  console.log(JSON.stringify({
    defcon: env.twilight.game.state.defcon,
    log: env.log,
  }));
}).catch((err) => {
  console.error(err.stack || err.message);
  process.exit(1);
});
"""
    result = subprocess.run(["node", "-e", script], cwd=".", text=True, capture_output=True, check=True)
    payload = json.loads(result.stdout)
    assert payload["defcon"] == 2
    defcon_index = next(i for i, entry in enumerate(payload["log"]) if "DEFCON falls to 2" in entry)
    second_headline_index = next(
        i for i, entry in enumerate(payload["log"]) if "Resolving USSR headline: Junta" in entry
    )
    assert defcon_index < second_headline_index


def test_redeterminization_preserves_observer_information_and_hidden_pool():
    with TwilightBridgeClient() as bridge:
        obs = bridge.reset(31)
        snapshot = bridge.snapshot()
        full_before = bridge.log()
        own_hand = list(full_before["hands"][obs["side"]])
        hidden_side = "us" if obs["side"] == "ussr" else "ussr"
        hidden_pool = sorted(full_before["hands"][hidden_side] + full_before["deck"])

        bridge.redeterminize(obs["side"], 1001)
        first = bridge.log()
        assert first["hands"][obs["side"]] == own_hand
        assert sorted(first["hands"][hidden_side] + first["deck"]) == hidden_pool

        bridge.restore(snapshot)
        bridge.redeterminize(obs["side"], 1001)
        assert bridge.log()["hands"] == first["hands"]
        assert bridge.log()["deck"] == first["deck"]


def test_snapshot_restores_the_same_redeterminized_particle():
    with TwilightBridgeClient() as bridge:
        obs = bridge.reset(37)
        observer = obs["side"]
        hidden_side = "us" if observer == "ussr" else "ussr"
        original = bridge.log()

        bridge.redeterminize(observer, 2003)
        particle = bridge.log()
        assert sorted(particle["hands"][hidden_side] + particle["deck"]) == sorted(
            original["hands"][hidden_side] + original["deck"]
        )

        particle_snapshot = bridge.snapshot()
        bridge.restore(particle_snapshot)
        restored = bridge.log()
        assert restored["hands"] == particle["hands"]
        assert restored["deck"] == particle["deck"]


def test_redeterminization_samples_future_rng_deterministically():
    script = r"""
const { SaitoTwilightKernel } = require('./bridge/saito_bridge.js');
const env = new SaitoTwilightKernel();
env.reset(41);
const snapshot = env.snapshot(false, true);
const observer = env.currentPlayer;
env.applyRedeterminization(observer, 3001);
const first = env.rand.getState();
env.restore(snapshot).then(() => {
  env.applyRedeterminization(observer, 3001);
  const repeated = env.rand.getState();
  return env.restore(snapshot).then(() => {
    env.applyRedeterminization(observer, 3002);
    console.log(JSON.stringify({ first, repeated, different: env.rand.getState() }));
  });
});
"""
    result = subprocess.run(["node", "-e", script], cwd=".", text=True, capture_output=True, check=True)
    payload = json.loads(result.stdout)
    assert payload["first"] == payload["repeated"]
    assert payload["first"] != payload["different"]


def test_ops_influence_placement_consumes_ops():
    with TwilightBridgeClient() as bridge:
        obs = finish_setup(bridge, bridge.reset(1))
        for value in ["decolonization", "arabisraeli", "burma", "laos", "thailand", "vietnam",
                      "suezcrisis", "event", "formosan", "event",
                      "naziscientist", "event", "degaulle", "ops", "before_ops", "place"]:
            action = next(action for action in obs["legal_actions"] if action["value"] == value)
            obs = bridge.step(action)["observation"]

        assert obs["prompt"] == "Place 3 influence"
        assert all(action["decision"] == "country_mouseup" for action in obs["legal_actions"])
        action = next(action for action in obs["legal_actions"] if action["value"] == "iran")
        obs = bridge.step(action)["observation"]
        assert obs["prompt"] != "Place 3 influence"


def test_one_per_country_event_placements_are_not_reoffered():
    with TwilightBridgeClient() as bridge:
        obs = finish_setup(bridge, bridge.reset(1))
        for value in ["decolonization", "arabisraeli"]:
            action = next(action for action in obs["legal_actions"] if action["value"] == value)
            obs = bridge.step(action)["observation"]

        assert "Africa or Southeast Asia" in obs["prompt"]
        first = "sudan"
        action = next(action for action in obs["legal_actions"] if action["value"] == first)
        obs = bridge.step(action)["observation"]
        assert first not in {action["value"] for action in obs["legal_actions"]}


def test_headless_submit_status_advances_action_round():
    with TwilightBridgeClient() as bridge:
        obs = finish_setup(bridge, bridge.reset(1))
        for _ in range(300):
            if obs["terminal"] or obs["action_round"] > 2:
                break
            obs = bridge.step(obs["legal_actions"][0])["observation"]

        assert "submitting moves" not in obs["prompt"].lower()
        assert obs["action_round"] > 0 or obs["terminal"]


def test_terminal_reason_is_exposed_in_authoritative_observation():
    script = r"""
const { SaitoTwilightKernel } = require('./bridge/saito_bridge.js');
const env = new SaitoTwilightKernel();
env.reset(1);
env.winner = 'us';
env.terminalReason = 'vp_threshold';
console.log(JSON.stringify(env.observe(env.currentPlayer)));
"""
    result = subprocess.run(["node", "-e", script], cwd=".", text=True, capture_output=True, check=True)
    observation = json.loads(result.stdout)
    assert observation["terminal"] is True
    assert observation["winner"] == "us"
    assert observation["terminal_reason"] == "vp_threshold"


def test_ops_menu_hides_targetless_coup_and_realign_options():
    script = r"""
const { SaitoTwilightKernel } = require('./bridge/saito_bridge.js');
const env = new SaitoTwilightKernel();
env.reset(1);
env.phase = 'action';
env.currentPlayer = 'us';
env.twilight.game.state.defcon = 5;
for (const country of Object.values(env.twilight.countries)) {
  country.us = 0;
  country.ussr = 0;
}
env.adapter.setOptions(
  'US plays 3 OPS:',
  '<ul><li class="option" id="place">place influence</li><li class="option" id="coup">launch coup</li><li class="option" id="realign">realign country</li></ul>',
  () => {}
);
console.log(JSON.stringify(env.adapter.pending.choices.map((choice) => choice.value)));
"""
    result = subprocess.run(["node", "-e", script], cwd=".", text=True, capture_output=True, check=True)
    assert json.loads(result.stdout) == ["place"]


def test_targetless_junta_free_ops_exposes_narrow_skip_and_clears_limits():
    script = r"""
const { SaitoTwilightKernel } = require('./bridge/saito_bridge.js');
const env = new SaitoTwilightKernel();
env.reset(1);
env.phase = 'action';
env.currentPlayer = 'us';
env.twilight.game.player = 2;
env.twilight.game.state.events.junta = 1;
env.twilight.game.state.limit_placement = 1;
env.twilight.game.state.limit_milops = 1;
env.twilight.game.state.limit_region = ['europe', 'africa', 'mideast', 'asia', 'seasia'];
env.twilight.game.queue = ['ops\tus\tjunta\t2'];
for (const country of Object.values(env.twilight.countries)) {
  country.us = 0;
  country.ussr = 0;
}
env.adapter.setOptions(
  'US plays 2 OPS:',
  '<ul><li class="option" id="coup">launch coup</li><li class="option" id="realign">realign country</li></ul>',
  () => {}
);
const choices = env.adapter.pending.choices.map((choice) => choice.value);
env.stepAsync(env.adapter.pending.choices[0]).then(() => {
  console.log(JSON.stringify({
    choices,
    nextChoices: (env.adapter.pending && env.adapter.pending.choices || []).map((choice) => choice.value),
    limitRegion: env.twilight.game.state.limit_region,
    limitPlacement: env.twilight.game.state.limit_placement,
    junta: env.twilight.game.state.events.junta,
    logTail: env.log.slice(-3),
  }));
}).catch((err) => {
  console.error(err.stack || err.message);
  process.exit(1);
});
"""
    result = subprocess.run(["node", "-e", script], cwd=".", text=True, capture_output=True, check=True)
    payload = json.loads(result.stdout)
    assert payload["choices"] == ["__skip_targetless_junta_ops__"]
    assert payload["nextChoices"]
    assert payload["limitRegion"] == []
    assert payload["limitPlacement"] == 0
    assert payload["junta"] == 0
    assert any("free OPS skipped" in entry for entry in payload["logTail"])


def test_targetless_non_junta_ops_exposes_generic_skip():
    script = r"""
const { SaitoTwilightKernel } = require('./bridge/saito_bridge.js');
const env = new SaitoTwilightKernel();
env.reset(1);
env.phase = 'action';
env.currentPlayer = 'ussr';
env.twilight.game.player = 1;
env.twilight.game.queue = ['ops\tussr\tnuclearsubs\t2'];
env.isLegalCoupTarget = () => false;
env.isLegalRealignTarget = () => false;
env.adapter.setOptions(
  'USSR plays 2 OPS:',
  '<ul><li class="option" id="coup">launch coup</li><li class="option" id="realign">realign country</li></ul>',
  () => {}
);
console.log(JSON.stringify(env.adapter.pending.choices.map((choice) => [choice.decision, choice.value])));
"""
    result = subprocess.run(["node", "-e", script], cwd=".", text=True, capture_output=True, check=True)
    assert json.loads(result.stdout) == [["skip_targetless_ops", "__skip_targetless_ops__"]]


def test_targetless_non_junta_ops_skip_resolves_ops_marker():
    script = r"""
const { SaitoTwilightKernel } = require('./bridge/saito_bridge.js');
(async () => {
  const env = new SaitoTwilightKernel();
  env.reset(1);
  env.phase = 'action';
  env.currentPlayer = 'ussr';
  env.twilight.game.player = 1;
  env.twilight.game.queue = ['ops\tussr\tnuclearsubs\t2'];
  env.isLegalCoupTarget = () => false;
  env.isLegalRealignTarget = () => false;
  env.adapter.setOptions(
    'USSR plays 2 OPS:',
    '<ul><li class="option" id="coup">launch coup</li><li class="option" id="realign">realign country</li></ul>',
    () => {}
  );
  await env.stepAsync({type: 'saito_bridge', value: '__skip_targetless_ops__'});
  console.log(JSON.stringify({
    queue: env.twilight.game.queue,
    logTail: env.log.slice(-2),
  }));
})().catch((err) => {
  console.error(err && err.stack || err);
  process.exit(1);
});
"""
    result = subprocess.run(["node", "-e", script], cwd=".", text=True, capture_output=True, check=True)
    payload = json.loads(result.stdout)
    assert any("no legal OPS target mode" in entry for entry in payload["logTail"])
    assert payload["queue"]


def test_pinned_runtime_checkpoint_survives_eviction() -> None:
    script = r"""
const { SaitoTwilightKernel } = require('./bridge/saito_bridge.js');
(async () => {
  const env = new SaitoTwilightKernel();
  env.maxRuntimeCheckpoints = 2;
  const root = env.snapshot(true, false);
  env.pinRuntimeCheckpoint(root, true);
  for (let i = 0; i < 6; i++) env.snapshot(true, false);
  await env.restore(root);
  console.log(JSON.stringify({
    rootPresent: env.runtimeCheckpoints.has(root.checkpoint_token),
    rootPinned: env.pinnedRuntimeCheckpoints.has(root.checkpoint_token),
    checkpointCount: env.runtimeCheckpoints.size,
  }));
})().catch((err) => {
  console.error(err && err.stack || err);
  process.exit(1);
});
"""
    result = subprocess.run(["node", "-e", script], cwd=".", text=True, capture_output=True, check=True)
    payload = json.loads(result.stdout)
    assert payload == {"rootPresent": True, "rootPinned": True, "checkpointCount": 3}
