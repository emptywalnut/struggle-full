# Selective PUCT Scaling Report (2026-08-11)

## Scope

This report evaluates selective, information-set-aware PUCT on top of the frozen
`decisions-0000023522-games-0000054` policy/value checkpoint. PPO self-play was not
replaced, and no Twilight-Struggle-specific strategic decision rule was added.

The four evaluated agent forms are:

1. frozen parent policy without search;
2. frozen parent policy with selective PUCT;
3. action-head-distilled policy without search;
4. action-head-distilled policy with selective PUCT.

All full-game comparisons use matched seeds/deals and both candidate-side
orientations. A seed contributes to aggregate statistics only after both its US and
USSR games complete.

## Runtime Optimization

The Saito bridge now supports process-local V8 runtime checkpoints. A portable
history snapshot is used to reconstruct each search root once; descendants capture
and restore the complete live JS simulator state, including queue, pending adapter
callbacks, RNG state, action history, and logs.

```text
portable root replay -> pinned local root -> local branch capture/restore
```

The portable form remains the cross-process and persisted fallback. Differential
tests compare portable replay with local restore and compare lean search steps with
full environment steps.

Measured on a deep, 305-action Five Year Plan state:

| Operation | Mean time |
|---|---:|
| Root reconstruction | 520.4 ms |
| Local snapshot capture | 2.00 ms |
| Local snapshot restore | 2.17 ms |
| Cold NN evaluation | 9.26 ms |
| Cached NN evaluation | 0.74 ms |

Root reconstruction across the diagnostic suite was approximately 0.04-0.60 s.
The three 32-simulation Olympic searches took 3.09-5.19 s, or 6.17-10.37
simulations/s. At the T3 Olympic root, the previous measured implementation took
12.71 s and the optimized implementation takes 3.09 s, a 4.1x reduction.

Deep allocation branches remain more expensive than forced responses. A 128-sim
Mexico allocation search took 71.76 s (1.78 simulations/s). At that point, exact
chance expansion, legal-action generation, and depth-2 opponent continuations, not
root reconstruction, dominate runtime.

## Search Semantics

- The actor receives only its normal masked observation. A runtime assertion rejects
  a nonzero opponent-hand tensor.
- Hidden opponent hands use 32 uniformly sampled, legal-consistent root particles.
- Values are stored in absolute US perspective; selection converts them to the side
  to act before PUCT maximization.
- Auditable chance trees up to 36 leaves are exactly enumerated and probability
  weighted. This covers d6, 2d6, war, coup, realignment, and Olympic outcomes.
- Larger known chance trees use independent deterministic sampling and are counted
  separately. Unknown chance arity is a hard error.
- Search uses policy-guided progressive widening and a configurable information-state
  node budget. The tournament configuration uses 256 nodes and two meaningful
  decision layers.

## Trigger Tuning

The production selective-32 configuration searches forced responses, opponent-event
card/mode decisions, influence allocations, scoring-card states, and a 2% sample of
low-confidence states. Standalone `chance_heavy` triggering is disabled because its
early tournament flip rate did not justify its cost. Opponent-event context no
longer retriggers search on each downstream country click.

On the first complete 10-pair selective-32 tournament:

| Trigger | Occurrences | Flip rate | Mean Q advantage | Mean latency |
|---|---:|---:|---:|---:|
| Forced response | 60 | 15.0% | 0.0606 | 5.32 s |
| Influence allocation | 312 | 21.5% | 0.0365 | 11.22 s |
| Low confidence | 41 | 36.6% | 0.0572 | 14.92 s |
| Opponent event | 516 | 17.8% | 0.0410 | 12.74 s |
| Scoring card | 70 | 24.3% | 0.0684 | 17.24 s |

Trigger categories can overlap. Their downstream game scores are associations, not
causal estimates, so pruning decisions use cost, flip rate, and Q improvement rather
than winner correlation alone.

## Search Distillation

The dedicated search replay sampler prioritizes action flips, large Q disagreement,
hierarchical allocations, rare sides/triggers, and scoring/horizon-sensitive states.
The accepted distillation run used 274 flipped examples with Q advantage at least
0.03, 100 updates, the `action_head` parameter scope, and an anchor coefficient of
1.0. The value head was frozen.

| Metric on the 274 selected examples | Before | After |
|---|---:|---:|
| Teacher top-1 accuracy | 0.0% | 20.44% |
| Mean probability of teacher action | 0.1622 | 0.1819 |
| Raw parent action retention | 100% | 56.20% |

A broader all-actor distillation was rejected: its raw full-game score against the
parent was 25% over 20 games, Elo -190.85 with bootstrap 95% CI
[-381.70, -70.44]. A head-only variant preserved behavior but could not train the
new semantic action columns. The accepted action-head checkpoint scored 55% over 20
raw games, Elo +34.86 with 95% CI [-107.54, +190.85], so no general regression was
detected, but no strength gain was statistically established either.

The 61-D action-representation audit covered 2,146 states, 38,692 legal actions, and
927,753 action pairs. The legacy 45-D representation had 29 aliases; the 16 semantic
columns resolved all 29 and the complete 61-D representation had zero observed
aliases. This included Blockade discard/remove, Warsaw Pact add/remove, DEFCON
choices, Tear Down this Wall responses, and Olympic participate/boycott.

At the accepted distilled Olympic T1 root, the raw actor now assigns:

```text
P(participate) = 0.504656
P(boycott)     = 0.495344
```

The old migrated model was exactly 0.5/0.5. Depth-2 32-sim PUCT still selects
`participate` (17 visits versus 15; Q -0.3383 versus -0.3717).

## Regression Status

The frozen diagnostic suite remains reproducible:

- Olympic T1/T3/T5: 32-sim search selects `participate` in all three states.
- Destalinization: depth-2 search flips the known bad root action.
- Europe scoring: the depth-2 allocation test preserves scoring-card horizon.
- Mexico T5.04/T5.05: deeper allocation search escapes repeated Mexico stacking.
- Iran 1-Ops coup: chance value is the exact mean of rolls 1 through 6.
- T10 final scoring: terminal search reaches simulator final scoring rather than
  trusting the critic.

The complete automated suite passes: `132 passed`.

## Full-Game Results

The parent selective tournament stopped after the paired confidence interval became
informative. It contains 17 complete seed blocks (34 games); three partially
completed blocks are listed in the JSON and excluded. The distilled-search
tournament contains 9 complete blocks (18 games); one incomplete block is excluded.

| Agent vs parent raw | Games | Score | Score 95% CI | Elo | Elo 95% CI |
|---|---:|---:|---:|---:|---:|
| Parent raw NN | baseline | 50% by definition | n/a | 0 | n/a |
| Parent + tuned selective 32 | 34 | 67.65% | [50.84%, 80.87%] | +128.13 | [+20.46, +267.60] |
| Distilled raw NN | 20 | 55.00% | [34.21%, 74.18%] | +34.86 | [-107.54, +190.85] |
| Distilled + tuned selective 32 | 18 | 72.22% | [49.13%, 87.50%] | +165.99 | [+38.76, +361.24] |

Elo intervals use paired seed-block bootstrap. Score intervals are Wilson intervals
over games. The two intervals answer related but different questions; the
distilled-search score interval narrowly includes 50%, while its paired-block Elo
interval is positive.

The gain is not side-balanced:

| Candidate | US score | USSR score |
|---|---:|---:|
| Parent + tuned selective 32 | 8/17 (47.06%) | 15/17 (88.24%) |
| Distilled raw | 3/10 (30.00%) | 8/10 (80.00%) |
| Distilled + tuned selective 32 | 4/9 (44.44%) | 9/9 (100.00%) |

Thus selective PUCT has a statistically supported aggregate improvement against the
parent, but the evidence does not establish a US-side improvement. Most measured
gain comes from stronger USSR play. This is the principal remaining playing-strength
limitation.

Terminal reasons for parent selective versus parent raw were: VP threshold 16,
thermonuclear war 8, final scoring 6, Wargames 2, tie 1, and victory points 1. For
distilled selective versus parent raw they were: VP threshold 8, thermonuclear war
5, and final scoring 5. Search therefore did not eliminate nuclear endings.

## Compute-Strength Scaling

| Configuration | Strength result | Searches/game | Sims/game | Search s/game | Wall s/game | Flip rate |
|---|---|---:|---:|---:|---:|---:|
| No search (distilled raw) | 55.0%, Elo +34.9 | 0 | 0 | 0 | 23.8 | 0% |
| Broad selective 32, 512-node tree | 38.9%, Elo -78.5 (9 pairs) | 77.3 | 2,472.9 | 1,358.7 | 1,447.3 | 23.3% |
| Tuned selective 32, 256-node tree | 67.6%, Elo +128.1 (17 pairs) | 47.2 | 1,511.5 | 550.2 | 645.7 | 20.3% |
| Distilled tuned selective 32 | 72.2%, Elo +166.0 (9 pairs) | 44.1 | 1,409.8 | 502.4 | 585.0 | 19.0% |
| Adaptive 32 to 128 pilot | 1-1, unpaired cost pilot | 57.5 | 6,592.0 | 1,397.6 | 1,508.6 | 20.0% |

The adaptive pilot escalated 99 of 115 searches (86.1%) to 128 simulations. It used
4.36x the simulations and 2.54x the search time of tuned selective 32 without a
higher flip rate. Its second orientations were stopped at T2 after the two complete
first-orientation games split 1-1. This is a compute result, not a strength estimate.
The current escalation OR-conditions are therefore too permissive for whole-game
use. A 512-simulation full-game configuration was not run after the 128 pilot was
already dominated; 32/128/512 behavior remains covered by the frozen tactical suite.

A literal forced-32 search on every atomic decision was also not run: tuned selective
32 already spends roughly ten minutes per game, and broad selective 32 costs over
twenty minutes. The broad-selector run is the practical upper-bound comparison used
here, while forced tactical roots remain regression-tested.

## Distillation Effect on Search Demand

Compared with the parent selective tournament, accepted action-head distillation
reduced:

- searches/game from 47.24 to 44.06;
- simulations/game from 1,511.5 to 1,409.8;
- search time/game from 550.2 s to 502.4 s;
- search flip rate from 20.30% to 19.04%.

The flip reduction is 1.26 percentage points (6.2% relative). These tournaments use
different sample counts, so this is evidence consistent with partial internalization,
not a causal estimate. The actor-level Olympic distinction and the zero-alias 61-D
audit provide direct representation-level evidence that learning transferred beyond
a hard-coded response.

## Decision

Selective PUCT should remain available as an inference mode, using the tuned
32-simulation, 256-node configuration. It produces a statistically supported
aggregate improvement and is materially better than the broad selector in both
strength and compute. Adaptive escalation to 128/512 should not be enabled by
default until escalation criteria are made much stricter.

The next algorithmic priority is US-side search quality and latency, not deeper
search or a larger critic. Useful next experiments are a US-stratified search replay
buffer, side-specific Q-calibration diagnostics, and batching/caching legal-action
evaluation in high-branch allocation/event states.

## Artifacts

- `runs/eval/tournament-parent-tuned-selective32-vs-parent-raw-17pairs-summary-20260811.json`
- `runs/eval/tournament-distilled-actionhead-anchor1-tuned-selective32-vs-parent-raw-9pairs-summary-20260811.json`
- `runs/eval/four-agent-selective-puct-summary-20260811.json`
- `runs/distill/selective-puct-highq003-actionhead-anchor1-100updates-20260811/`
- `runs/diagnostics/frozen-mcts23522-k32-cachefixed-20260808/`
- `runs/diagnostics/action-representation-61d-audit-20260811.json`
- `runs/benchmarks/puct-olympics-t1-regression-20260811.json`
- `runs/benchmarks/puct-olympics-t3-regression-20260811.json`
- `runs/benchmarks/puct-olympics-t5-regression-20260811.json`

Confidence intervals are read from the generated artifacts, not inferred from point
estimates. Incomplete seed blocks remain auditable in each summary but do not affect
ratings, trigger statistics, terminal distributions, or compute aggregates.
