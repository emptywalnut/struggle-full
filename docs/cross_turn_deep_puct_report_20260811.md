# Cross-Turn Deep PUCT Scaling

Date: 2026-08-11

## Implementation

- Search horizon is configurable independently of simulation budget.
- Meaningful decisions include card/headline, mode, Ops subtype, complete influence
  allocation, coup/realignment target, and forced strategic response.
- Repeated country clicks in one allocation are coalesced. Deterministic event
  resolution, chance outcomes, and bookkeeping do not consume meaningful depth.
- End-of-Turn is traversed normally. Telemetry records maximum reached depth,
  maximum Turn delta, and cross-Turn leaf evaluations.
- Root hidden hands use legal-consistent particles. Future deck order and draws are
  sampled inside those particles. Each particle also receives a deterministic,
  independently sampled future RNG stream for post-root reshuffles and random
  events that are not handled by an immediate exact chance node. Every actor/leaf
  observation passes the normal opponent-hand mask assertion.
- Small chance nodes remain exactly enumerated.
- Search replay emits root and downstream hierarchical targets with separate masked
  observations. Distillation can therefore change an allocation without relabeling
  the selected card.
- Distillation weights local corrections by Q advantage and can reject labels with
  inadequate local visit count/share.
- Iterative distillation can cap qualified labels per source root. The cap retains
  the strongest representative of each hierarchy level before filling remaining
  slots by Q/visit quality, preventing one deep root from dominating the student.
- Search now fails closed when any atomic-horizon or expanded-node cutoff occurs:
  the real action remains the raw actor action, hierarchical continuation and
  teacher targets are cleared, and telemetry records the rejection. Partial trees
  can therefore be profiled but cannot control a game or enter distillation.
- `scripts/build_puct_teacher_manifest.py` selects balanced US/USSR high-regret
  roots from paired games, reserves a configurable cross-Turn quota, and encodes no
  preferred strategic action. A deeper search must solve every selected root again.

Regression status: `151 passed, 2 skipped` after the macro-depth, hierarchical replay, complete-
game teacher input, future-RNG particle, transfer-evaluator, and training-metric changes.

## Real Cross-Turn Gate

At seed `82440006`, occurrence 0, action 78:

| Configuration | Max Turn delta | Cross-Turn leaves | Time | Sims/s |
|---|---:|---:|---:|---:|
| D2/8 | 0 | 0 | 0.50 s | 16.10 |
| D4/8 | 0 | 0 | 0.69 s | 11.62 |
| D6/8 | 1 | 12 | 2.05 s | 3.90 |
| D8/8 | 1 | 16 | 2.91 s | 2.75 |

This gate traversed Turn 1 end effects, Turn 2 draw/headline transitions, and a
forced Olympic response without an information-mask or atomic-horizon failure.

## Budget Stability Finding

The informative USSR late-AR root is seed `82440006`, occurrence 0, action 72.
Its selected action is non-monotonic at insufficient budgets:

| Configuration | Action | Root visits/Q | Time |
|---|---|---|---:|
| D6/512 | Truman | 421 / 0.479 | 80.4 s |
| D6/2048 | Formosan Resolution | 1640 / 0.621 | 300.1 s |
| D8/512 | Formosan Resolution | 486 / 0.544 | 117.0 s |
| D8/2048 | Formosan Resolution | 1992 / 0.545 | 400.3 s |

D8/512 and D8/2048 agree, with root-Q `L_inf` difference approximately `0.0115`.
D6/512 and D6/2048 disagree because progressive widening discovers a stronger
Formosan branch late. Therefore D8/2048 is used as the high-quality offline teacher;
512-simulation labels are not assumed converged merely because their visit margin is
large.

The aggregate machine-readable matrix is
`runs/deep_puct/horizon-scaling-summary-20260811.json`.

## Refined-RNG D8 Teacher Evidence

After future RNG was independently seeded per hidden-state particle, four useful
D8/2048 roots completed without atomic or node-budget cutoffs:

| Seed / action | Expanded | Exact chance | Cross-Turn leaves | Time | Accepted high-Q corrections |
|---|---:|---:|---:|---:|---:|
| `82440006 / 72` | 5,600 | 123 | 556 | 379.4 s | 1 |
| `82440006 / 78` | 6,984 | 2,290 | 4,371 | 607.8 s | 2 |
| `82440000 / 643` | 9,145 | 3,537 | 10,627 | 1,408.0 s | 2 |
| `82440003 / 714` | 142 | 2,132 | 0 | 680.0 s | 0 |

The accepted corrections use `Q advantage >= 0.003`, at least four teacher visits,
teacher visit share at least 0.5, and require an action flip. One additional root,
`82440007 / 619`, reached the 16,384-node cap and was rejected in full. Distillation
never consumes a partially searched shard.

The five accepted hierarchical labels include two complete influence-allocation
corrections, one forced response, one card choice, and one coup-target correction.
They are intentionally sparse: quality gates take precedence over filling a replay
quota.

## Iteration 0 Transfer

Standalone Q-weighted action-head distillation used only the five zero-cutoff labels:

| Checkpoint | Teacher top-1 | Mean teacher probability | Q-weighted teacher probability |
|---|---:|---:|---:|
| NN0 | 0% | 0.170 | 0.114 |
| Distilled NN | 20% | 0.281 | 0.247 |

This standalone checkpoint became NN1 after a matched raw-policy safety gate against
NN0: 13/20 (65%, approximately `+107.5 Elo`, bootstrap 95% interval
`[-34.9, +301.3]`), with US 6/10 and USSR 7/10. This passed the configured
50%-overall/40%-per-side promotion rule, but its confidence interval crosses zero;
it is an accepted non-regression candidate, not a statistically proven raw strength
gain.

The subsequent 100-game PPO-stabilized candidate retained 20% top-1 transfer and
raised mean teacher probability to 0.338 on these five labels, but failed the
whole-game acceptance gate: against NN0 raw it scored 4/20 (20%, approximately
`-241 Elo`, bootstrap 95% interval `[-512, -108]`). Its US score was 4/10 and its
USSR score was 0/10. It is retained as a rejected diagnostic candidate and is not
used as the parent for the next deep teacher.

The evaluator is `scripts/evaluate_search_replay_checkpoint.py`; it reports the same
metrics by acting side, hierarchy level, and trigger. PPO stabilization must preserve
or improve these metrics before its checkpoint is accepted as NN1.

`run_iterative_puct_distill.py` now records both the distilled and PPO candidate
checkpoints and requires explicit `--ppo-student-selection {distilled,ppo}`. The safe
default is `distilled`; `ppo` should be selected only after paired evaluation rules
out a general regression.

When paired iteration evaluation is enabled, the launcher now applies an acceptance
gate to the raw student against its actual parent. Both overall score and a minimum
US/USSR side score are configurable. A rejected student is recorded but does not
become the next teacher parent, and the chain stops by default. This prevents an
apparently successful local transfer metric from silently advancing a weaker model.

The iterative launcher now always runs PPO stabilization as multi-agent unified-policy
training. This is required because search loads `shared_policy`; a single-agent PPO
run would update RLlib's `default_policy` and leave the searched policy stale. The
launcher also disables redundant in-training 50-game checkpoint evaluations and can
optionally run the matched raw/cheap/deep three-curve evaluation after each iteration.

Actor-only distillation now separates the correction batch from its preservation
anchor. Correction labels still pass the flip, Q, visit, cutoff, hierarchy, and
per-source-root quality gates. The anchor batch is sampled independently from all
cutoff-free replay examples, using each parent's stored raw policy as the target.
This addresses a failure observed in the second teacher cycle: fitting a concentrated
set of deep-search corrections can improve teacher-action probability while moving
the shared action scorer enough to regress unrelated whole-game decisions.

## Active Evaluations

The practical whole-game D4/128 screen is complete. It used ten matched seeds with
side swapping against the exact same NN0 raw checkpoint:

| Agent | Score | Elo delta | Bootstrap 95% CI | US | USSR |
|---|---:|---:|---:|---:|---:|
| NN0 + D4/128 | 15/20 (75%) | +190.8 | `[+70.4, +381.7]` | 9/10 | 6/10 |
| NN0 raw | 5/20 (25%) | reference | - | 4/10 | 1/10 |

Terminal reasons were 12 VP-threshold, seven final-scoring, and one nuclear ending.
This is statistically supported whole-game evidence that deeper search improves the
frozen agent, not merely the tactical diagnostic suite.

The gain is too expensive for default inference: D4/128 averaged 68.25 searches and
8,736 simulations per game, 2,727 search-seconds per game, 25.3-second median root
latency, 121.5-second p95 root latency, and one 960.6-second maximum. Search crossed a
Turn boundary in 12.2% of roots and flipped 30.2% of searched actions. It is therefore
a strong offline-teacher configuration, while D2/32 remains the practical inference
candidate.

Three of 1,365 roots contained one atomic branch cutoff each. All three selected the
same root action as the raw policy, so they did not change the recorded games. They
are excluded from teacher selection; subsequent runs also use the fail-closed planner.
The authoritative result is under
`runs/deep_puct/tournament-nn0-d4-128-vs-raw-10pairs-gpu4-refinedrng-node8192-workers8-20260811/`.

- The next teacher manifest contains 12 high-regret roots selected from that
  tournament: six US, six USSR, and six with observed cross-Turn leaves. It is
  `config/cross_turn_puct_teacher_positions_cycle2.json`.
- Iteration 0 teacher/student chain: D8/2048, Q/visit-filtered hierarchical
  distillation, then 100-game PPO stabilization, under
  `runs/deep_puct/iterative-d8-2048-nn0-qualityfilter-v4-refinedrng-sharded-gpu5-20260811/`.

The authoritative PPO stabilization directory is
`iteration-00/ppo-stabilization-zero-cutoff-shared-runners4/`. Earlier stabilization
attempts under `ppo-stabilization-zero-cutoff/` and
`ppo-stabilization-zero-cutoff-fastresume/` are rejected because they updated
`default_policy`, not the `shared_policy` consumed by PUCT. The one-runner shared
policy attempt is valid but superseded before checkpointing by the equivalent
four-runner, 4,096-step-batch launch.

The older `workers4` and `v2` roots were stopped before producing complete artifacts
because their already-running Node processes predated future-RNG particle sampling.
The first refined four-worker root was also superseded by the otherwise identical
eight-worker launch before any paired block completed. The serial refined teacher
was superseded before its first root completed by the equivalent six-root parallel
teacher. The first eight-worker tournament was then rejected after telemetry exposed
node-budget cutoffs at chance-heavy roots; the valid relaunch raises the cap from
2,048 to 8,192. None of the superseded roots are valid inputs to the final refined
evaluation.

These jobs are not counted as successful until complete result/checkpoint artifacts
exist. Process liveness alone is not evidence.

After each student is produced, the matched-seed three-curve evaluator is:

```bash
python scripts/evaluate_puct_iteration.py \
  --student-checkpoint /path/to/NN_k_plus_1 \
  --parent-checkpoint /path/to/NN_k \
  --output-root runs/deep_puct/iteration-k-eval \
  --seed-pairs 10 \
  --gpus 4,5 \
  --cheap-depth 2 --cheap-simulations 32 \
  --deep-depth 4 --deep-simulations 128 \
  --capture-search-replay
```

It evaluates raw, cheap selective-32, and practical deep-search curves against the
same parent and writes one resumable shard tree plus `iteration_evaluation.json`.
The D8/2048 teacher is deliberately not used for every full-game action because the
measured 300-400 second root cost makes that comparison economically misleading.

## Iteration 1 Teacher and Rejected Students

The second D8/2048 teacher regenerated search from the accepted standalone NN1
checkpoint, rather than reusing NN0 labels. Twelve balanced source positions produced
24,576 simulations and 763 hierarchical examples. Search averaged 659.5 seconds per
root, reached depth 8 and the following Turn, and generated 34 cutoff-free action-flip
labels after the Q/visit gates. The raw qualified set was nevertheless concentrated:
19 labels came from one root and ten from another. After an eight-label source-root
cap, the correction batch still contained 6 US and 15 USSR labels.

The uncapped 200-update student fit the teacher strongly (79.4% top-1 on 34 labels),
but scored only 50% against NN0 raw and was not promoted. A capped 100-update student
scored 45% against NN0 raw. More importantly, immediate-parent gates against NN1
showed the remaining side skew:

| NN1-derived candidate vs NN1 raw | Score | Elo delta / 95% CI | US | USSR | Accepted |
|---|---:|---:|---:|---:|---|
| broad anchor, 100 updates | 12/20 | `+70.4 [-70.4, +240.8]` | 3/10 | 9/10 | no |
| broad anchor + side balance, 100 | 8/20 | `-70.4 [-240.8, +70.4]` | 1/10 | 7/10 | no |
| broad anchor + side balance, 50 | 11/20 | `+34.9 [-107.5, +190.8]` | 3/10 | 8/10 | no |

The gate requires at least 50% overall and 40% on each side. Therefore the accepted
chain stops at NN1. Side-balanced loss correctly assigns 50% correction weight to
each side, but cannot manufacture strategic coverage absent from the teacher data.
The next teacher manifest must add diverse high-regret US roots before another
distillation iteration is eligible for promotion.

The practical-search curve also failed to improve. In a direct ten-seed,
side-swapped comparison, NN1+D2/32 scored 8/20 against NN0+D2/32: 5/10 as US and
3/10 as USSR, Elo `-70.4` with bootstrap 95% interval `[-240.8, +70.4]`. Both agents
searched, averaging 5,200 simulations and 752.8 search-seconds per game combined.
Twelve low-confidence roots reached the 16,384-node cap and failed closed to their
raw actions; there were no atomic-horizon cutoffs. Thus the accepted raw NN1 gain did
not produce the required `NN_(k+1)+selective32` improvement.

Search demand did not decline after distillation. NN0 searched 1,613 roots with a
16.43% flip rate; NN1 searched 1,637 roots with a 16.68% flip rate. Mean root latency
was 4.52 seconds for NN0 and 4.74 seconds for NN1. The evidence therefore rejects a
claim that the first distillation cycle internalized enough corrections to make
selective search cheaper.

Distillation preservation now uses an independent batch sampled from all 760
cutoff-free replay examples. Search correction loss can additionally normalize total
US and USSR weight with `--balance-sides`. Replay-root discovery accepts both the
iterative `shard-*/manifest.json` and tournament
`shard-*/search_replay/manifest.json` layouts.
