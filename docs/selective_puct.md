# Selective PUCT and Search Distillation

This document describes the selective planner layered on the existing Saito-backed
policy/value agent. It does not replace PPO and does not hard-code strategic card
choices.

## Components

- `struggle_ai/mcts.py`
  - information-state PUCT
  - legal-consistent hidden-hand root sampling
  - exact small chance nodes
  - progressive widening
  - configurable search triggers and adaptive budgets
- `struggle_ai/search_replay.py`
  - versioned compressed search replay shards
- `struggle_ai/search_distill.py`
  - side/trigger-aware replay sampling
  - actor auxiliary cross-entropy updates
- `scripts/generate_search_replay.py`
  - replays complete Saito game logs and labels selected states
- `scripts/distill_search_replay.py`
  - standalone actor distillation and checkpoint export
- `scripts/eval_puct_agent_pair.py`
  - resumable matched-seed, side-swapped raw/search evaluation
- `scripts/run_puct_pair_shards.py`
  - bounded multi-GPU shard orchestration and per-shard search replay
- `scripts/summarize_puct_tournament.py`
  - paired-bootstrap Elo, compute, terminal, and trigger aggregation

## Search Semantics

The network receives only the normal acting-player observation. Before every
evaluation, the planner verifies that the absolute opponent-hand tensor is zero.
The sampled hidden hand exists only inside the Saito search state and is never
copied into the network observation.

Each PUCT simulation uses one of `K=32` legal-consistent hidden-hand particles.
Particles are sampled uniformly from the opponent-hand/deck pool while preserving
public zones, hand sizes, the observer hand, and China-card ownership.

Search values are stored internally from the absolute US perspective. Selection
converts them to the side-to-act perspective, so US maximizes `Q_US` and USSR
maximizes `-Q_US`. Terminal values are `+1` for a US win and `-1` for a USSR win.

Known small random outcomes are exact expectation nodes. The bridge reports the
number of sides for every consumed forced roll; the planner enumerates the finite
conditional roll tree and backs up probability-weighted values. Olympic tie rerolls
are reduced to the two observation-distinct winner branches with their exact
conditional probabilities. The production exact bound is 36 leaves, covering 1d6,
2d6, wars, and Olympic outcomes. Larger auditable
chance trees use one independent deterministic sample per PUCT simulation and are
reported separately as `sampled_large_chance_nodes`; unknown chance arity remains a
hard error.

## Runtime State Checkpoints

The production bridge can capture the complete live JS simulator graph with V8
serialization, including queue state, pending adapter callbacks, action history,
logs, and RNG state. A portable replay snapshot remains the cross-process fallback,
but each search root is reconstructed only once. Search branches then use local
runtime checkpoint tokens:

```text
portable root replay -> pinned local root -> local branch clones/restores
```

Pinned roots and cached exact-chance outcomes are protected from ordinary checkpoint
eviction and are released together after the planner call. Search-mode environment
steps omit rollout logging/reward bookkeeping and reuse encoded observations where
the state is unchanged. Differential regression tests compare portable replay,
local restore, and lean/full step results.

The root exposes every legal action. Non-root nodes begin with the highest-prior
children and widen according to:

```text
active_children ~= c * visit_count ** alpha
```

The default limits are 4 initial and 24 maximum children. Production search also
caps distinct expanded information-state nodes at 512. Reaching that independent
tree budget leaf-evaluates only new frontier states and increments
`node_budget_cutoffs`; exact chance probabilities already being backed up are not
changed. This avoids exhaustive complete allocation/chance-response trees while
preserving policy-guided alternatives.

## Selective Budgets

The default adaptive schedule is `32 -> 128 -> 512` simulations. Search starts only
when one of the enabled generic trigger categories is present:

- `forced_response`
- `opponent_event`
- `influence_allocation`
- `scoring_card`
- `chance_heavy`
- `low_confidence`

Escalation uses configurable root visit share, root Q gap, raw/search disagreement,
and high-value trigger categories. Ordinary strategic decisions use two meaningful
decision layers by default. A forced response can use one meaningful layer because
that layer still resolves all atomic response/event/free-Ops actions before the leaf.

Planner telemetry includes the searched-decision fraction, simulations per real
action, latency, raw/search disagreement, action flips, root Q gap, Q advantage over
the raw action, exact/sampled chance counts, atomic-horizon cutoffs, Q-gate rejections,
and trigger counts. The paired full-game evaluator adds side-swapped score, paired
bootstrap confidence intervals, Elo delta, terminal reasons, and search compute.

## Hierarchical Actions

Every tree edge is classified as one of:

- `card`
- `mode`
- `ops_action_type`
- `influence_allocation`
- `coup_target`
- `realignment_target`
- `forced_response`

Search replay stores the most visited hierarchical path. Dataset generation also
searches logged downstream states independently, so allocation supervision remains
an allocation target rather than incorrectly relabeling a retained root card.

## Action Representation Fix

The legacy 45 action features represented broad classes but aliased free-form
responses such as `boycott` and `participate`. The encoder now appends 16 stable,
signed semantic-hash features over generic action fields and prompt context. There is
no Olympic-specific feature or policy rule.

Old checkpoints are migrated by copying the original 45 input columns and zeroing
the 16 new columns. Therefore their initial behavior is preserved exactly until the
new representation is trained. Frozen Olympic states still start at `0.5/0.5`, which
is an explicit regression check against accidentally injecting a heuristic.

## Search Replay Format

Each `.npz` example contains:

- all information-state observation tensors
- padded raw policy
- padded PUCT visits and normalized search policy
- padded root action Q values
- side and trigger categories
- serialized legal action set
- raw and search action indices
- meaningful depth and simulation budget
- root Q gap and action-flip flag
- hierarchical action path
- optional search value target
- source game and source action index

The manifest is versioned and lists every compressed shard. Rare trigger categories
and action-flip examples receive higher sampling priority in the dedicated replay
sampler.

Generate exact labels from complete JSONL logs:

```bash
CUDA_VISIBLE_DEVICES=3 conda run -n struggle-ppo python scripts/generate_search_replay.py \
  --checkpoint /path/to/checkpoint \
  --game-log /path/to/complete-games.jsonl \
  --output-dir runs/search_replay/example \
  --initial-simulations 32 \
  --medium-simulations 128 \
  --max-simulations 512 \
  --belief-particles 32 \
  --meaningful-depth 2
```

For a reproducible diagnostic root, use the zero-based action index:

```text
--position 8808001:19
```

Explicit positions force search even if a trigger classifier does not select them.

## Distillation

The target is normalized PUCT visits:

```text
pi_search(a | I) proportional to N(I, a) ** (1 / tau)
L_search = cross_entropy(pi_search, pi_actor)
```

Standalone distillation:

```bash
CUDA_VISIBLE_DEVICES=3 conda run -n struggle-ppo python scripts/distill_search_replay.py \
  --restore-from /path/to/checkpoint \
  --search-replay-root runs/eval/tournament-example \
  --output-dir runs/distill/example \
  --updates 100 \
  --batch-size 32 \
  --lr 1e-5 \
  --max-examples-per-source-root 8
```

`--search-replay` may still be repeated for individual manifests. Tournament-root
discovery consumes `shard-*/search_replay/manifest.json`, and the standalone trainer
uses the same side/rare-trigger/action-flip priority sampler as PPO integration.
The optional per-source-root cap first retains the strongest label from each
hierarchy level, then fills the quota by Q/visit quality. This avoids allowing one
large deep tree to dominate a small targeted replay set.

PPO integration keeps environment rollouts on-policy and applies bounded auxiliary
actor updates after each normal PPO iteration:

```text
--search-replay runs/search_replay/example
--search-distill-lambda 0.05
--search-distill-updates-per-iter 1
--search-distill-batch-size 32
--search-distill-lr 1e-5
--search-distill-max-examples-per-source-root 8
```

The search optimizer is checkpointed as `search_distill_optimizers.pt`. Metrics are
printed and forwarded to W&B with the `search_distill/` prefix. Critic-head parameters
are excluded from this optimizer; the shared state encoder remains trainable, so
general evaluation is still mandatory after distillation.

## Verified Diagnostics

Using the frozen `decisions-0000023522-games-0000054` checkpoint after zero-column
migration:

- Olympic T1/T3/T5: raw actor is exactly `0.5/0.5`; production 32-sim search selects
  `participate` in all three states.
- The three Olympic root Q advantages for `participate` are approximately `0.1104`,
  `0.0250`, and `0.0717`.
- Destalinization T1: production depth-2, 32-sim search changes the root from
  `Destalinization` to `The Cambridge Five`; Q changes from about `-0.2580` to
  `+0.2489`.
- Three-state Olympic distillation changes raw `participate` probabilities from
  `0.5` to approximately `0.590`, `0.544`, and `0.561` without a strategic rule.
- A 10-game same-seed smoke comparison has the same side-conditioned aggregate as
  the migrated parent: US `0/5`, USSR `4/5`. This is evidence of no observed aggregate
  regression at that sample size, not evidence of an Elo improvement.
- A paired two-side full-game selective smoke searched 5 of 499 controlled-side
  decisions (1.00%), used 160 total simulations, and averaged 24.44 seconds per
  searched action. Raw and selective corresponding-side results were both `1/2`;
  paired changes were `improved=0`, `regressed=0`. The Wilson interval was
  `[0.095, 0.905]`, so this is an end-to-end mechanics/cost check, not a playing-
  strength claim.

The full frozen diagnostic report remains under
`runs/diagnostics/frozen-mcts23522-k32-cachefixed-20260808/`.

## Current Throughput

With native runtime checkpoints, measured local captures/restores are approximately
`0.7-2.9 ms` / `0.9-3.5 ms` over the tested T1-T5 roots. The optimized 32-simulation
Olympic regressions take `3.09-5.19 s` (`6.17-10.37 sims/s`), versus the previous
reported `8-113 s` range and the immediately preceding approximately `24 s/search`
baseline. Deep card/allocation states remain substantially more expensive because
exact chance branches and depth-2 opponent responses dominate rather than restore.
For example, the 128-simulation Mexico allocation regression takes `71.76 s` and
changes the raw Mexico placement to Romania with estimated Q advantage `+0.2795`.

A statistically measurable full-game Elo improvement was treated as an empirical
gate, not an implementation assumption. Use `scripts/run_puct_pair_shards.py` for
resumable matched-seed side-swapped shards and
`scripts/summarize_puct_tournament.py` for paired bootstrap Elo/compute/trigger
aggregation.

## Scaling Evaluation Result

The 2026-08-11 tuned selective-32 evaluation reached the empirical gate against the
frozen parent: 17 complete matched, side-swapped seed blocks (34 games), 67.65%
score, Elo `+128.13`, paired-bootstrap 95% CI `[+20.46, +267.60]`. Search averaged
47.24 roots, 1,511.5 simulations, and 550.2 search seconds per game. The result is
side-skewed: US scored 8/17 while USSR scored 15/17, so the aggregate improvement
must not be described as a proven two-sided improvement.

The accepted action-head distilled checkpoint scored 55% without search over 20
games (no detected regression) and 72.22% with tuned search over 18 games, Elo
`+165.99`, paired-bootstrap CI `[+38.76, +361.24]`. Its search flip rate was 19.04%,
versus 20.30% for the parent search agent. An adaptive 32-to-128 pilot escalated
86.1% of searched roots, used 6,592 simulations and 1,397.6 search seconds per game,
and was stopped as compute-dominated.

See `docs/selective_puct_scaling_report_20260811.md` and the generated four-agent
summary under `runs/eval/four-agent-selective-puct-summary-20260811.json` for exact
confidence intervals, terminal distributions, trigger metrics, and caveats.

## Cross-Turn Deep Search

`meaningful_depth` now counts strategic macro decisions rather than Saito UI steps.
The counted boundaries are card/headline choice, event/Ops/space mode, Ops subtype,
complete influence allocation, coup/realignment target, and forced strategic response.
Repeated country clicks in one influence allocation share one macro identity;
deterministic resolution, exact chance outcomes, and bookkeeping do not consume depth.
End-of-Turn is not a boundary.

Deep-search telemetry includes:

- `max_meaningful_depth_reached`
- `max_turn_delta`
- `cross_turn_leaf_evaluations`
- atomic and expanded-node budget cutoffs

The replayable position suite is
`config/cross_turn_puct_positions.json`. It stores source log, seed, side-swapped game
occurrence, and action index without storing an expected action. Run an independent
depth/budget matrix with:

```bash
CUDA_VISIBLE_DEVICES=4 conda run -n struggle-ppo python scripts/sweep_puct_horizon.py \
  --checkpoint /path/to/checkpoint \
  --position-manifest config/cross_turn_puct_positions.json \
  --config 2:32 --config 4:128 --config 6:512 --config 8:512 \
  --belief-particles 32 \
  --max-atomic-steps 512 \
  --max-expanded-nodes 8192 \
  --output runs/deep_puct/horizon-sweep.json \
  --num-gpus 1
```

The output records root action/Q agreement between every configuration. Simulation
budget and meaningful depth are independent; a deeper result is not accepted merely
because its leaf Q is numerically larger.

Search replay now contains a root example plus separate information-safe examples for
visited mode/Ops/allocation nodes along the selected path. Every downstream example
has its own masked observation, legal actions, policy prior, visits, Q values, and
search policy. This lets distillation repair an allocation without labeling the root
card as bad. `hierarchy_level` and `hierarchy_index` distinguish these examples in a
shard, and Q-weighted distillation gives larger updates to high-regret local flips.

Generate a fixed strong teacher from the same suite with:

```bash
CUDA_VISIBLE_DEVICES=5 conda run -n struggle-ppo python scripts/generate_search_replay.py \
  --checkpoint /path/to/nn-k \
  --position-manifest config/cross_turn_puct_positions.json \
  --output-dir runs/deep_puct/iteration-k/search-replay \
  --initial-simulations 2048 \
  --medium-simulations 2048 \
  --max-simulations 2048 \
  --meaningful-depth 6 \
  --forced-response-depth 6 \
  --belief-particles 32 \
  --max-atomic-steps 512 \
  --max-expanded-nodes 16384
```

After distillation, regenerate teacher replay from the new checkpoint. Reusing labels
generated by `NN_k` for every later student is not iterative policy improvement.
`scripts/run_iterative_puct_distill.py` enforces this checkpoint chain and writes an
`iterative_manifest.json` after every phase. Set `--ppo-stabilization-episodes` to
retain ordinary on-policy PPO between teacher iterations; zero keeps the loop as
offline actor-only distillation.

When an iteration includes paired evaluation, the launcher accepts the student only
if its raw policy meets both the configured overall score and US/USSR side floors
against `NN_k`. Rejected checkpoints remain auditable in the manifest but do not
become the parent for the next teacher generation.

Search correction batches are deliberately narrow, so the anchor loss is sampled
from every cutoff-free replay example discovered in the teacher shards, not only
from the high-regret examples selected for correction. This preserves the parent's
raw action distribution over a broader set of searched information states while the
Q-weighted teacher loss changes the selected mistakes. Configure this with
`--anchor-from-all-safe-replay`, `--anchor-replay-batch-size`, and
`--max-examples-per-source-root`. The source-root cap applies only to correction
labels; it does not reduce the broad anchor pool.

For a unified US/USSR policy, `--balance-sides` normalizes correction weights so US
and USSR each contribute half of the search loss even when deep-search hierarchy
produces more qualified labels for one side. This prevents label count alone from
steering the shared action scorer. It cannot compensate for missing strategic
coverage, so paired evaluation still enforces a minimum score for each side.
