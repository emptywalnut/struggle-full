# PUCT-in-the-Loop Self-Play

## Training boundary

Search behavior and PPO behavior are deliberately separate:

- PUCT games use `NN -> selective PUCT -> visit policy -> environment`.
- PUCT actions are never passed through PPO importance ratios.
- Complete PUCT games train policy imitation, terminal value, and optionally search-Q regression.
- PPO stabilization runs in a separate RLlib stage using genuine policy-generated trajectories.

Both US and USSR use the same checkpoint and search configuration. Each side has an independent planner runtime so continuation state and search counters cannot cross sides.

## Replay guarantees

Search replay version 2 stores:

- masked information-state observations
- legal actions and raw network policy
- visit counts, normalized search policy, and action Q values
- hierarchical card/mode/Ops/allocation targets
- selected behavior action and behavior temperature
- trigger, depth, budget, and horizon metadata
- complete-game winner, terminal reason, and acting-side value target

Examples remain in memory until the game reaches a real terminal state. A truncated game or missing winner invalidates the whole game transaction. An individual search with an atomic/node-budget cutoff contributes no teacher label, while other complete searches from that finished game remain valid. Sampled hidden hands remain internal to search and are never written into observations.

## Collection

```bash
python scripts/collect_puct_selfplay_parallel.py \
  --checkpoint CHECKPOINT \
  --output-dir runs/puct_selfplay/collection \
  --games 64 \
  --workers 32 \
  --gpus 4,5 \
  --inference-replicas-per-gpu 2 \
  --max-inference-batch-size 32 \
  --meaningful-depth 2 \
  --initial-simulations 32 \
  --medium-simulations 32 \
  --max-simulations 32 \
  --search-temperature 1.0 \
  --behavior-temperature 0.75
```

`search_temperature` controls the visit-distribution teacher target. `behavior_temperature` controls sampling of the real self-play action from visits. Evaluation keeps behavior temperature at zero.

The parallel collector runs many resident Saito/V8 games on CPU and sends policy/value requests through one global dynamic queue. Multiple resident inference replicas consume that queue across both GPUs, so a temporarily busy GPU does not own or stall a fixed subset of games. Telemetry in `throughput.jsonl` separates project-owned GPU processes from external processes and reports queue depth, batch size, simulations/second, and complete games/hour. Replay shards remain worker-local until each game reaches a real terminal state, then a merged manifest is created without loading all observations into memory.

## Policy/value learning

```bash
CUDA_VISIBLE_DEVICES=4 python scripts/distill_search_replay.py \
  --restore-from CHECKPOINT \
  --search-replay runs/puct_selfplay/collection/search_replay \
  --output-dir runs/puct_selfplay/student \
  --updates 200 \
  --batch-size 128 \
  --parameter-scope policy_value_heads \
  --lambda-search 1.0 \
  --search-policy-mix 0.5 \
  --lambda-value 0.1 \
  --lambda-search-q 0.02 \
  --lambda-anchor 0.3 \
  --lambda-value-anchor 1.0 \
  --confidence-weight-floor 0.1 \
  --min-q-advantage 0.02 \
  --min-teacher-visit-share 0.20 \
  --min-value-examples-per-batch 32 \
  --separate-value-batch-size 64 \
  --require-complete-game \
  --require-flip
```

Near-ties receive less policy weight. High-regret flips and hierarchical allocation corrections retain higher priority. The critic target is in the acting side's perspective: win `+1`, loss `-1`, draw `0`. `lambda-value-anchor` constrains values to the frozen parent on replay states while terminal and search-Q targets still update the critic. This protects PUCT action ranking from critic drift that may not appear in aggregate terminal MSE.

With `--require-flip --separate-value-batch-size 64`, actor supervision is
drawn only from decisions actually changed by PUCT, while critic terminal
targets come from an independent, side-balanced pool of all complete-game
root states. This avoids letting near-identity search policies dominate actor
updates without discarding broad complete-game value supervision.
The default iteration gate also rejects policy labels whose PUCT improvement
is below `0.02` or whose teacher action receives less than 20% of root visits;
these thresholds remain CLI-configurable rather than game-specific rules.
`--search-policy-mix 0.5` uses an equally weighted convex target of PUCT visits
and the parent policy from the same information state. This local trust region
preserves the direction of search corrections while reducing unrelated raw
policy drift. Search-Q regression continues to use the unblended PUCT visits.
The same trust-region target is available during PPO stabilization as
`--search-distill-policy-mix`; this keeps standalone and in-loop search
distillation semantics aligned.
For joint in-loop updates, use `--search-distill-lambda-value`,
`--search-distill-lambda-search-q`, and
`--search-distill-min-value-examples-per-batch`. Terminal-value examples come
from a separate pool of complete root states before actor-quality filters are
applied; confident flip filters therefore cannot starve critic supervision.
Both pools are filtered from replay metadata before large observation tensors
are copied, which keeps PPO startup memory proportional to the selected
policy/value populations rather than the entire replay archive.

`policy_value_heads` is the conservative promotion scope: it freezes the
shared state and action encoders while updating the side-specific policy
queries/biases and the value head. Use `actor_value` only for deliberate
full-encoder experiments. `--us-policy-weight` and `--ussr-policy-weight`
default to `1.0`; side balancing is applied first and these weights then alter
the effective policy-loss share without changing value sampling.

## Mixed curriculum

`run_puct_selfplay_iteration.py` uses complete games as the mixture unit. For example, `--total-games 80 --search-fraction 0.75` collects 60 PUCT games and then performs PPO stabilization for 20 complete PPO games. Supported fractions are `0.5`, `0.75`, and `1.0`; `0.75` is the default.

```bash
python scripts/run_puct_selfplay_iteration.py \
  --parent-checkpoint CHECKPOINT \
  --output-dir runs/puct_selfplay/iteration-001 \
  --total-games 80 \
  --search-fraction 0.75 \
  --parallel-inference-gpus 4,5 \
  --parallel-search-workers 32 \
  --inference-replicas-per-gpu 2 \
  --eval-inference-replicas-per-checkpoint 1 \
  --distill-parameter-scope policy_value_heads \
  --distill-us-policy-weight 1.0 \
  --distill-ussr-policy-weight 1.0 \
  --gpu 4 \
  --diagnostic-pass-file runs/puct_selfplay/iteration-001/frozen_diagnostics/pass.json
```

Stages are resumable through `.STAGE.complete` markers. The runner evaluates child versus immediate parent in matched, side-swapped games for:

- raw versus raw
- D2/32 versus D2/32
- D4/128 versus D4/128

Promotion requires the raw overall gate, both side floors, the cheap-search gate, and an explicit frozen-diagnostic approval artifact. The approval JSON must contain `passed: true` and a `child_checkpoint` matching the evaluated child. Omitting it deliberately leaves `frozen_diagnostics=false`, so a metrics-only run can never promote accidentally. Results are written to `promotion_report.json`; a failed child is retained for audit but is not designated as promoted.

Parallel paired evaluation writes `progress.json` atomically after every
completed game. It is always marked `status: partial` and is not promotion
evidence; only a completed `results.json` is final. For large gates, increasing
the number of concurrent seed pairs generally improves inference batching more
than adding replicas. The iteration runner therefore uses one inference server
per distinct evaluation checkpoint by default while retaining two replicas per
GPU for single-checkpoint self-play generation. Too many evaluation replicas
split a sparse request stream and reduce average batch size.
New progress snapshots include an exact checkpoint/search/seed identity.
`eval_puct_agent_pair_parallel.py` resumes identity-matching completed
seed/orientation tasks by default and refuses missing, mismatched, duplicate,
or out-of-range records. Older progress files without identity remain
deliberately non-resumable.

## Multi-day campaign

`run_puct_multiday_campaign.py` repeats complete generation, broad joint
policy/value learning, PPO stabilization, and matched side-swapped gates. The
default long-run launcher uses 96 D2/32 search games and 32 genuine PPO games
per 128-game cycle. Broad replay includes every complete-game, cutoff-free
PUCT label; action flips and high-Q corrections receive extra weight but are
not the only policy targets. Minibatches are side-balanced and capped across
source game, Turn, card, target region/country, and trigger category.

The collector writes one atomic `completed_games/game-NNNNNN.json` after each
terminal game. A restart verifies checkpoint/search/seed identity and queues
only unfinished games. Each student checkpoint carries search-optimizer state;
campaign cycles and stages use atomic state plus completion markers.

```bash
tmux new-session -d -s ts-puct-multiday \
  'bash scripts/launch_puct_multiday_gpu45.sh'
```

The campaign distinguishes the exploratory `working_checkpoint` from the
audited `accepted_parent`. A child may become the next cycle's working model
after cheap raw and D2 side gates, but the accepted parent is never replaced
automatically. Larger paired gates create promotion-candidate reports for
explicit review. This prevents a noisy tournament from silently becoming the
new baseline.

## Deep adjudication

The default iteration first escalates up to 16 high-value roots to parallel D4/128. It then sends at most two roots to D8/2048, and only when D2 and D4 choose different actions or D4 still has a small root-Q gap. Configure these stages with `--d4-adjudication-states`, `--d4-adjudication-workers`, and `--deep-adjudication-states`. Both stages use the same shared dual-GPU inference service design through `generate_search_replay_parallel.py`. Deep replay is mixed into the same policy/value learner only after complete-tree and complete-game checks pass.

Every iteration must regenerate PUCT trajectories from its own parent checkpoint. Reusing an older teacher dataset indefinitely is not an iterative policy-improvement cycle.

## Auditable head interpolation

If a search student improves D2 play but narrowly regresses as a raw policy,
screen smaller checkpoint deltas before collecting another teacher set:

```bash
python scripts/interpolate_policy_checkpoint.py \
  --parent-checkpoint PARENT \
  --student-checkpoint STUDENT \
  --output-dir runs/puct_selfplay/interpolation \
  --label alpha-050 \
  --alpha 0.50 \
  --scope policy_value_heads
```

This starts from the parent and interpolates only the selected head tensors.
The generated manifest records source and output hashes, tensor deltas, and a
post-save exact verification. It is an experimental trust-region screen, not
a promotion shortcut: raw, searched, side, and diagnostic gates still apply.
