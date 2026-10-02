# Struggle AI

PPO-ready Twilight Struggle environment using Saito Twilight as the headless rules runner.

This source snapshot includes the browser UI, AI training and inference code,
the materialized Saito rules module, and the board asset with its license notice.
Model weights, checkpoints, training runs, private game logs, and external
Steam reverse-engineering inputs are not included. The browser can run with
the heuristic or random bot; trained-model play requires separate checkpoints.

## Rules Preset

The v1 preset is `optional_us_plus_2`:

- Saito `deck = "optional"`.
- Saito `usbonus = 2`.
- Core base cards plus official optional/expanded cards:
  `Defectors`, `The Cambridge Five`, `Special Relationship`, `NORAD`,
  `Che`, `Our Man in Tehran`, `Iran-Iraq War`, `Yuri and Samantha`, and `AWACS`.
- Normal core cards such as `Muslim Revolution` are included.
- Saito/community/End of History/Cold War Crazies extra cards are excluded.

## Current Implementation

This repository contains a runnable Saito-backed wire:

- `bridge/saito_bridge.js` exposes a JSON-lines headless game bridge backed by Saito Twilight source.
- `struggle_ai/env.py` wraps that bridge in a Gymnasium/RLlib-compatible environment.
- `struggle_ai/train_rllib.py` contains the PPO entrypoint.
- `struggle_ai/mcts.py` contains selective belief-aware exact-chance PUCT.
- `struggle_ai/search_replay.py` and `struggle_ai/search_distill.py` provide
  targeted search supervision without replacing PPO rollouts.
- `struggle_ai/batched_inference.py` and
  `scripts/collect_puct_selfplay_parallel.py` provide shared dual-GPU inference
  for many concurrent complete-game PUCT workers.
- `struggle_ai/play.py` provides a CLI human-vs-bot loop.
- `browser/server.js` and `web/` provide a private playable browser frontend
  for human-vs-AI and observable AI-vs-AI games, including validated checkpoint
  discovery and real Python model inference.

The bridge loads Saito's real Twilight module headlessly, intercepts the browser/UI decision callbacks, exposes them as structured legal actions, and then feeds selected actions back into the Saito handlers. Card play, event queues, scoring, ops, influence placement, coups, realignments, Space Race choices, and event-specific browser choices now run through the Saito queue rather than Python-side rule code.

## Setup

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e '.[test]'
npm install
pytest
npm run smoke
```

For RLlib training:

```bash
pip install -e '.[train,test]'
python -m struggle_ai.train_rllib --stop-iters 1 --num-env-runners 0
```

For the staged PPO diagnostics, seed from the saved mixed best policies:

```bash
python -m struggle_ai.league init --league-dir runs/ppo/league_stage_a
python -m struggle_ai.league launch --league-dir runs/ppo/league_stage_a --role main --branch hard_filter --episodes 1000 --dry-run
python -m struggle_ai.eval_rllib \
  --us-policy runs/ppo/selected_policies/best_us57005_ussr66001/policies/us_policy \
  --ussr-policy runs/ppo/selected_policies/best_us57005_ussr66001/policies/ussr_policy \
  --games 100 \
  --opponent mixed \
  --out-dir runs/ppo/eval_mixed/stage_a_sanity
```

League launch scripts default to GPU 2 by exporting `CUDA_VISIBLE_DEVICES=2`
and passing `--num-gpus 1.0`. They also pass `--league-manifest` into
`train_rllib.py`, so rollout episodes sample frozen current/best/history
opponents plus random/legal and heuristic baselines instead of plain fixed
self-play.

Use `--defcon-suicide-mode hard_filter` for the hard-filter branch and
`--defcon-suicide-mode penalty_only --terminal-reward-scale 2.0
--nuke-death-penalty 1.0` for the penalty-only branch. Transformer-history
training is enabled with `--model-arch transformer_history`; matching encoder
weights are partially warm-started by default and the skipped/new keys are
written to `load_report.json`.

Auxiliary reward shaping can be held fixed with `--reward-shaping-scale` or
annealed with `--reward-shaping-final-scale`,
`--reward-shaping-phaseout-start-episodes`, and
`--reward-shaping-phaseout-end-episodes`. This scales VP/control/coup/headline
shaping only; terminal win/loss and nuclear-loss penalties are unchanged.

For CLI play against a heuristic bot:

```bash
python -m struggle_ai.play --human us --bot heuristic --seed 1
```

For the private browser frontend:

```bash
npm run build
npm run browser -- --host 0.0.0.0 --port 41783 --production
```

The bounded real-model terminal-game gate is available as
`npm run validate:browser-full-game`; see `docs/browser_frontend.md` for its
reproducible production-stack invocation and retained-session inspection URL.

Then forward the remote port over SSH and open `http://127.0.0.1:41783`.
The browser uses the supplied Saito board under GMT's personal,
non-commercial game-module conditions, so its start screen requires an
ownership acknowledgement. See `docs/browser_frontend.md` and
`reference/map-assets/saito/LICENSE.txt`.

## Saito Source

Fetch the upstream Saito implementation with:

```bash
scripts/fetch_saito.sh
```

The bridge is intentionally the integration boundary so the Python PPO stack does not depend on browser UI code. The current materialized Saito source lives under `third_party/saito-lite-rust-materialized`.

Selective PUCT configuration, dataset generation, distillation, evaluation, and
measured limitations are documented in `docs/selective_puct.md`.
