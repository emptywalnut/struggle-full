from struggle_ai.train_rllib import (
    SHARED_POLICY_ID,
    checkpoint_elo_ratings,
    eval_only_elo_seed_ratings,
    EvalPolicySwitcher,
    US_POLICY_ID,
    USSR_POLICY_ID,
    eval_games_per_side,
    eval_benchmark_policy_id,
    initialize_elo_state,
    metric_label,
    new_elo_state,
    parse_policies_to_train,
    restore_policy_model_modes,
    seed_eval_elo_ratings,
    set_policy_models_evaluation_mode,
    update_elo_match,
)


def test_eval_games_per_side_enforces_minimum():
    assert eval_games_per_side(5, 10) == 10
    assert eval_games_per_side(12, 10) == 12


def test_metric_label_sanitizes_checkpoint_names():
    assert metric_label("steps-0000010000-episodes-00000100") == "steps-0000010000-episodes-00000100"
    assert metric_label("foo/bar:baz") == "foo_bar_baz"


def test_update_elo_match_updates_both_side_board_entries():
    state = new_elo_state(k_factor=32.0)

    player_rating, opponent_rating = update_elo_match(
        state,
        "us",
        "checkpoint-1:us_policy",
        "initial:ussr_policy",
        1.0,
        k_factor=32.0,
        meta={"terminal_reason": "final_scoring"},
    )

    assert player_rating > 1500.0
    assert opponent_rating < 1500.0
    assert state["leaderboards"]["us"]["games"]["checkpoint-1:us_policy"] == 1
    assert state["leaderboards"]["us"]["games"]["initial:ussr_policy"] == 1
    assert state["leaderboards"]["ussr"]["ratings"] == {}
    assert state["matches"][0]["terminal_reason"] == "final_scoring"


def test_update_elo_match_supports_unified_bot_board():
    state = new_elo_state(k_factor=32.0)

    player_rating, opponent_rating = update_elo_match(
        state,
        "bot",
        f"checkpoint-1:{SHARED_POLICY_ID}",
        f"initial:{SHARED_POLICY_ID}",
        0.75,
        k_factor=32.0,
        meta={"games": 20, "aggregate": True},
    )

    assert player_rating > 1500.0
    assert opponent_rating < 1500.0
    assert state["leaderboards"]["bot"]["games"][f"checkpoint-1:{SHARED_POLICY_ID}"] == 1
    assert state["leaderboards"]["us"]["ratings"] == {}
    assert state["matches"][0]["aggregate"] is True


def test_seed_eval_elo_ratings_inherits_current_and_resets_fixed_anchor():
    state = new_elo_state(k_factor=32.0)
    state["leaderboards"]["us"]["ratings"]["best_anchor:ussr_policy"] = 1500.0
    state["leaderboards"]["ussr"]["ratings"]["best_anchor:us_policy"] = 1500.0
    state["leaderboards"]["bot"]["ratings"][f"best_anchor:{SHARED_POLICY_ID}"] = 1500.0

    seed_eval_elo_ratings(
        state,
        eval_label="candidate",
        unified_policy_eval=True,
        current_ratings={"us": 1742.0, "ussr": 1751.0, "bot": 1690.0},
        opponent_ratings={
            "best_anchor": {"us": 1742.0, "ussr": 1751.0, "bot": 1690.0}
        },
    )

    assert state["leaderboards"]["us"]["ratings"]["candidate:us_policy"] == 1742.0
    assert state["leaderboards"]["ussr"]["ratings"]["candidate:ussr_policy"] == 1751.0
    assert state["leaderboards"]["bot"]["ratings"][f"candidate:{SHARED_POLICY_ID}"] == 1690.0
    assert state["leaderboards"]["us"]["ratings"]["best_anchor:ussr_policy"] == 1751.0
    assert state["leaderboards"]["ussr"]["ratings"]["best_anchor:us_policy"] == 1742.0
    assert state["leaderboards"]["bot"]["ratings"][f"best_anchor:{SHARED_POLICY_ID}"] == 1690.0


def test_seed_eval_elo_ratings_does_not_reset_existing_candidate():
    state = new_elo_state(k_factor=32.0)
    state["leaderboards"]["us"]["ratings"]["candidate:us_policy"] = 1760.0

    seed_eval_elo_ratings(
        state,
        eval_label="candidate",
        unified_policy_eval=False,
        current_ratings={"us": 1742.0},
    )

    assert state["leaderboards"]["us"]["ratings"]["candidate:us_policy"] == 1760.0


def test_checkpoint_elo_ratings_recovers_parent_from_checkpoint_label(tmp_path):
    state = new_elo_state(k_factor=32.0)
    label = "steps-0000133130-episodes-00000660"
    state["leaderboards"]["us"]["ratings"][f"{label}:us_policy"] = 1742.0
    state["leaderboards"]["ussr"]["ratings"][f"{label}:ussr_policy"] = 1751.0
    state["leaderboards"]["bot"]["ratings"][f"{label}:{SHARED_POLICY_ID}"] = 1690.0

    assert checkpoint_elo_ratings(state, tmp_path / label) == {
        "us": 1742.0,
        "ussr": 1751.0,
        "bot": 1690.0,
    }


def test_eval_only_elo_seed_ratings_inherits_missing_sides_from_parent(tmp_path):
    state = new_elo_state(k_factor=32.0)
    parent = "parent"
    candidate = "candidate"
    state["leaderboards"]["us"]["ratings"][f"{parent}:us_policy"] = 1742.0
    state["leaderboards"]["ussr"]["ratings"][f"{parent}:ussr_policy"] = 1751.0
    state["leaderboards"]["bot"]["ratings"][f"{parent}:{SHARED_POLICY_ID}"] = 1690.0
    state["leaderboards"]["us"]["ratings"][f"{candidate}:us_policy"] = 1760.0

    assert eval_only_elo_seed_ratings(
        state,
        tmp_path / candidate,
        tmp_path / parent,
    ) == {
        "us": 1760.0,
        "ussr": 1751.0,
        "bot": 1690.0,
    }


def test_initialize_elo_state_seeds_only_a_new_destination(tmp_path):
    seed_path = tmp_path / "seed.json"
    destination = tmp_path / "active.json"
    seed = new_elo_state(k_factor=16.0)
    seed["matches"].append({"side": "us"})
    seed_path.write_text(__import__("json").dumps(seed), encoding="utf-8")

    state, source = initialize_elo_state(destination, seed_path, k_factor=32.0)
    assert source == "seeded"
    assert state["matches"] == [{"side": "us"}]
    assert state["k_factor"] == 32.0

    destination.write_text(__import__("json").dumps(new_elo_state(8.0)), encoding="utf-8")
    state, source = initialize_elo_state(destination, seed_path, k_factor=64.0)
    assert source == "existing"
    assert state["matches"] == []
    assert state["k_factor"] == 64.0


def test_parse_policies_to_train_unified_collapses_side_aliases():
    assert parse_policies_to_train(None, multi_agent=True, policy_sharing="unified") == [SHARED_POLICY_ID]
    assert parse_policies_to_train("us,ussr", multi_agent=True, policy_sharing="unified") == [SHARED_POLICY_ID]


def test_eval_benchmark_policy_id_prefers_split_pair_over_shared_policy():
    class Switcher:
        weights = {
            "split_best": {
                US_POLICY_ID: {"w": 1},
                USSR_POLICY_ID: {"w": 2},
                SHARED_POLICY_ID: {"w": 3},
            }
        }

    side_policy_ids = {"us": SHARED_POLICY_ID, "ussr": SHARED_POLICY_ID}

    assert eval_benchmark_policy_id("split_best", "us", side_policy_ids, Switcher()) == US_POLICY_ID
    assert eval_benchmark_policy_id("split_best", "ussr", side_policy_ids, Switcher()) == USSR_POLICY_ID


def test_eval_policy_switcher_caches_weights_per_policy_id():
    class Policy:
        def __init__(self):
            self.set_calls = 0

        def set_weights(self, _weights):
            self.set_calls += 1

    class Algo:
        def __init__(self):
            self.policies = {"us_policy": Policy(), "ussr_policy": Policy()}

        def get_policy(self, policy_id):
            return self.policies[policy_id]

        def compute_single_action(self, _obs, policy_id=None, explore=False):
            return 0

    algo = Algo()
    switcher = EvalPolicySwitcher(
        algo,
        current_weights={"us_policy": {"w": 1}, "ussr_policy": {"w": 2}},
        benchmark_weights={"initial": {"us_policy": {"w": 3}, "ussr_policy": {"w": 4}}},
    )
    obs = {"action_mask": [1]}

    for _ in range(3):
        switcher.action("us_policy", "current", obs)
        switcher.action("ussr_policy", "initial", obs)

    assert algo.policies["us_policy"].set_calls == 1
    assert algo.policies["ussr_policy"].set_calls == 1

    switcher.action("us_policy", "initial", obs)
    assert algo.policies["us_policy"].set_calls == 2


def test_evaluation_mode_disables_dropout_and_restores_prior_modes():
    class Model:
        def __init__(self, training):
            self.training = training
            self.calls = []

        def eval(self):
            self.calls.append(False)
            self.training = False

        def train(self, mode=True):
            self.calls.append(mode)
            self.training = mode

    class Policy:
        def __init__(self, model):
            self.model = model

    training_model = Model(True)
    eval_model = Model(False)

    class Algo:
        policies = {
            "us_policy": Policy(training_model),
            "ussr_policy": Policy(eval_model),
            "duplicate": Policy(training_model),
        }

        def get_policy(self, policy_id):
            return self.policies[policy_id]

    prior_modes = set_policy_models_evaluation_mode(
        Algo(),
        ["us_policy", "ussr_policy", "duplicate"],
    )
    assert not training_model.training
    assert not eval_model.training
    assert len(prior_modes) == 2

    restore_policy_model_modes(prior_modes)
    assert training_model.training
    assert not eval_model.training
    assert training_model.calls == [False, True]
    assert eval_model.calls == [False, False]
