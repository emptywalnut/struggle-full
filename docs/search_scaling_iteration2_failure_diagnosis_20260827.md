# Search-Scaling Iteration 2 Diagnosis

## Decision

The retained general parent remains:

`runs/search_distill/root-mechanics-actor-correction-actionhead-u100-gpu4-20260819/checkpoints/updates-000075`

No iteration-2 child is promoted. The evidence does not establish the required
`Raw < D2 < D4 < deeper` scaling, and the best student failed the frozen anomaly
gate.

## Iteration-2 u15

The actor-only u15 checkpoint was the strongest student from the 32 complete-game
adaptive collection (3,621 teacher examples):

| Gate | Games | Overall | US | USSR | Result |
|---|---:|---:|---:|---:|---|
| Raw preliminary | 20 | 50% | 70% | 30% | ambiguous |
| D2/32 preliminary | 20 | 65% | 80% | 50% | promising |
| D4/128 preliminary | 20 | 55% | 70% | 40% | no scaling evidence |
| Raw confirmatory | 50 | 56% | 48% | 64% | CI overlaps parity |
| D2/32 confirmatory | 50 | 58% | 76% | 40% | side-imbalanced |

The four-root anomaly regression failed because US Italy overcontrol reappeared
in one D4/128 repeat. The other three suspicious targets were suppressed. This
checkpoint is retained for audit but is not a general parent.

## Search compatibility

The 96-root fixed panel is balanced 48/48 by side. Q scales remain close to one,
but action rankings change sharply with meaningful horizon:

| Pair | Comparable | Mean rank corr. | Same top | Stable | Median Q scale |
|---|---:|---:|---:|---:|---:|
| D2/32 -> D2/128 | 96 | 0.799 | 54.2% | 55.2% | 0.933 |
| D4/128 -> D4/512 | 96 | 0.808 | 61.5% | 68.8% | 0.943 |
| D2/128 -> D4/128 | 96 | 0.497 | 46.9% | 30.2% | 1.055 |
| D2/128 -> D4/512 | 96 | 0.492 | 46.9% | 27.1% | 1.012 |
| D2/128 -> D6/512 | 96 | 0.210 | 25.0% | 4.2% | 1.066 |
| D4/128 -> D6/512 | 96 | 0.369 | 32.3% | 16.7% | 0.909 |
| D4/512 -> D6/512 | 96 | 0.391 | 37.5% | 18.8% | 1.006 |
| D6/512 -> D6/2048 | 7 | 0.829 | 100% | 71.4% | 1.024 |

This is structural horizon disagreement, not a simple search-Q temperature or
scale mismatch. The seven-root D6/2048 result is encouraging but too small and
selection-biased to establish a strong teacher globally.

Historic matched cross-mode tournaments agree with the root analysis:

- parent D2/32 beat parent raw in a 20-game screen (65%) and a separate 50-game
  confirmation (72%);
- parent D4/128 scored 46% against parent D2/32 over 50 games;
- parent D4/512 scored 37.5% against D4/128 in an eight-game preliminary screen.

Therefore D4 is not currently a demonstrated stronger policy-improvement
operator, despite being more expensive.

## Actor absorption

Of 3,621 labels, 1,707 passed the conservative teacher filters. Mean probability
assigned to the teacher action changed only slightly:

| Checkpoint | Mean teacher probability | Delta vs parent | Top-1 changes to teacher |
|---|---:|---:|---:|
| Parent | 0.196181 | - | - |
| u5 | 0.196222 | +0.000041 | 0 |
| u10 | 0.196261 | +0.000080 | 0 |
| u15 | 0.196299 | +0.000118 | 0 |

Small actor-head changes nevertheless caused large D2 game-level changes, with
weaker transfer to D4. This explains why more SGD on the same replay is unsafe.

## Interpolation ablation

Interpolating only the u5 action-head delta back toward the parent did not
recover a robust candidate:

| Candidate | Raw | D2/32 | US D2 | USSR D2 | Status |
|---|---:|---:|---:|---:|---|
| 25% u5 delta | 50% (20) | 42.1% (19) | 70% | 11.1% | failed; last pathological game stopped |
| 50% u5 delta | 50% (20) | 35% (20) | 50% | 20% | failed |

The 25% gate was stopped at 19 complete games because even winning its final
game could reach only 45%, while the remaining worker had grown to about 30 GiB
RSS in one search root. Completed games remain durable; no D4 gate was started
after the D2 failure.

## Fail-closed adaptive search

A reusable `ConvergedAdaptivePuctPlanner` now performs paired-seed:

1. D2/32 base search;
2. D2/128 same-depth budget confirmation;
3. D4/128 only when the D2 ranking is unresolved;
4. D4 adoption only when D2/128 and D4/128 Q rankings converge;
5. fallback to D2/128 for near ties, cutoffs, rejected evidence, or unresolved
   depth disagreement.

The planner preserves masked observations, exact chance semantics, Q-aware root
selection, and hierarchical continuation. Collection has an explicit
`--adaptive-fail-closed-unresolved-to-budget` option so an inadmissible deep tree
cannot silently drive gameplay while being excluded only from teacher labels.

An offline 96-root replay of this rule accepted D4 on only 6.25% of roots and
fell back on 38.5%. Against D6/512 it agreed on only 25% of top actions, with
mean Q regret 0.104. This is a safety mechanism, not proof of strength.

The matched 20-game adaptive-versus-D2 screen completed at:

`runs/eval/search-scaling-phase3u75-adaptive-converged-vs-d2-20260827`

It used 20 concurrent complete games, K=32, exact small chance nodes, and the
same retained checkpoint on both sides. The adaptive candidate scored 70%
(US 80%, USSR 60%) for +147 Elo. Its paired-bootstrap 95% score interval was
55%--85%. Terminal reasons were 11 final scoring, six thermonuclear war, and
three VP threshold. There were no atomic or node-budget cutoffs.

The candidate used 13,406 simulations per game versus 2,794 for D2/32. Across
1,567 adaptive decisions, convergence status was 1,052 budget-converged, 49
horizon-converged, 370 near ties, 70 unresolved, and 26 unvalidated. It fell
back to D2/128 on 440 decisions (13.3%). This establishes a strength signal,
but not yet an economical deployment policy.

The paired-seed adaptive anomaly gate also passed all four roots:

| Root | Adaptive action | Status |
|---|---|---|
| USSR UK overcontrol | Saharan States | D2 budget convergence |
| US Italy overcontrol | Pakistan | D2 budget convergence |
| USSR Mexico realignment | Colombia | near-tie fallback |
| US Venezuela realignment | Cuba | near-tie fallback |

All four were cutoff-free and retained valid information masking. Notably,
D4 alone selected suspicious Venezuela in the last root; the adaptive wrapper
correctly rejected the unresolved depth change and used D2/128 instead.

Independent confirmatory shards and D2/128 attribution ablations used disjoint
seed blocks and were aggregated only after both side-swapped orientations of a
seed completed. Evaluation was expanded to about 81 concurrent game workers
after `vmstat` showed substantial real CPU idle time. The pool used all eight
A100s with centralized batched inference per shard.

`scripts/aggregate_paired_eval_shards.py` merges raw game records across
resumable shards, rejects identity mismatches and duplicate seed/orientation
records, and computes paired statistics from complete seed blocks. The final
results are:

| Comparison | Games | Overall | US | USSR | Elo | Paired score CI |
|---|---:|---:|---:|---:|---:|---:|
| Adaptive vs D2, independent confirmation | 54 | 44.4% | 63.0% | 25.9% | -39 | 31.5%--57.4% |
| Adaptive vs D2, screen + confirmation | 74 | 51.4% | 67.6% | 35.1% | +9 | 40.5%--62.2% |
| D2/128 vs D2/32 | 42 | 52.4% | 61.9% | 42.9% | +17 | 35.7%--69.1% |
| Adaptive vs D2/128 | 12 | 50.0% | 83.3% | 16.7% | 0 | preliminary |

All four comparisons had zero search cutoffs. The initial 70% screen was not
reproduced. Increasing search compute consistently shifted strength toward US
and away from USSR, so adaptive search is rejected as a general teacher.

A larger frozen-value audit used all 1,493 complete hierarchy roots available
from the prior 32-game search collection. Overall terminal-sign accuracy was
81.8%; US and USSR aggregates were 80.2% and 83.4%. Outcome stratification
exposed the material asymmetry hidden by those means:

| Acting side/outcome | Roots | Sign accuracy | Mean raw value | Target |
|---|---:|---:|---:|---:|
| US loss | 429 | 81.4% | -0.799 | -1 |
| US win | 339 | 78.8% | +0.718 | +1 |
| USSR loss | 314 | 72.6% | -0.538 | -1 |
| USSR win | 411 | 91.7% | +0.735 | +1 |

The critic is materially over-optimistic in USSR losing states. This is
consistent with additional leaf evaluation helping US while degrading USSR.
The audit artifact is:

`runs/diagnostics/search-scaling-phase3u75-full32-value-audit-20260827/value_audit.json`

## Value calibration ablations

A side-conditioned monotonic affine-tanh calibration was fitted with grouped
cross-validation and side/outcome-balanced weights. It selected an almost
identity transform and improved held-out MSE by only 0.0011, which is too small
to justify production integration.

Three value-head-only sweeps were then trained on a game-disjoint 16/8-game
split. Both raw-value MSE and calibrated-value MSE objectives failed: every
u5/u10/u15/u20 checkpoint had worse held-out calibrated MSE than the parent.
No game-search gate was run for dominated checkpoints.

Because the final value head cannot repair a state-dependent error encoded in a
shared state embedding, an optional critic-only residual adapter was added. It
is default-off and its final layer is zero-initialized. Migration was verified
on 128 held-out roots with exactly zero actor-logit and raw-value delta. A first
24-game adapter sweep also failed held-out validation.

The value corpus was then broadened by streaming root-only examples from four
complete-game campaigns with 32 parallel shard workers. The resulting corpus
contains 117 training games / 15,282 roots and 19 validation games / 2,224 roots,
with no game overlap and all four side/outcome strata represented.

The best broad adapter checkpoint is `j40`:

| Metric | Parent | j40 |
|---|---:|---:|
| Root-weighted calibrated MSE | 1.0126 | 0.9986 |
| Four-stratum macro MSE | 1.0055 | 0.9996 |
| US-loss MSE | 1.1417 | 1.0989 |
| US-win MSE | 1.0900 | 1.1067 |
| USSR-loss MSE | 0.8665 | 0.9015 |
| USSR-win MSE | 0.9240 | 0.8914 |

A 50,000-sample game-block bootstrap estimated game-macro child-minus-parent
MSE at -0.0250 with 95% CI [-0.0427, -0.0075]. The improvement is statistically
credible in aggregate, but the critical USSR-loss stratum regressed. Actor
teacher probabilities remained exactly unchanged, confirming that this is a
value-only effect.

## Decision gate

The retained general parent is unchanged. The adaptive teacher and all
value-head-only candidates are rejected. `j40` first received a 20-game D2
screen that scored 60% on both sides, but that run accidentally used the
evaluation script's 512-node debug default and incurred 226 node-budget
cutoffs. It is retained as contaminated preliminary evidence only.

The correct confirmation used the established D2 settings:

- D2/32, K=32;
- `max_atomic_steps=1536`;
- `max_expanded_nodes=32768`;
- 40 matched, side-swapped seed pairs across four disjoint shards;
- 80 concurrent complete games on all eight A100s.

| Gate | Games | Overall | US | USSR | Elo | Elo CI | Paired score CI |
|---|---:|---:|---:|---:|---:|---:|---:|
| Parent actor + j40 value vs parent D2 | 80 | 47.5% | 57.5% | 37.5% | -17 | -98--+61 | 36.3%--58.8% |

There were zero atomic or node-budget cutoffs. The candidate averaged 2,516
simulations/game and the parent 2,576. Terminal reasons were 34 final scoring,
23 thermonuclear war, and 23 VP threshold. Aggregate held-out value calibration
therefore did not transfer to playing strength and retained a material side
imbalance. `j40` is rejected.

No actor amortization or adaptive-search trajectory generation starts until a
search operator stronger than D2 is demonstrated in balanced whole-game play.
The requested planning/amortization loop is not established by this iteration:
deeper/adaptive search did not beat D2, and the only statistically credible
offline value-calibration gain failed the cutoff-free game gate. All current
changes pass the complete test suite: 301 tests.

## Critic-specific value tower follow-up

The next controlled ablation added an optional critic-specific state tower. The
actor, shared policy representation, and action heads remain byte-equivalent to
the Phase-3 u75 parent; only the cloned critic state encoder and value head are
trainable. Side/outcome-stratified loss weights were added so that the known
USSR-loss calibration defect can be targeted without forcing actor changes.

The retained value candidate is `p30u25`:

`runs/search_distill/search-scaling-phase3u75-valuetower-weighted-p30-u25-gpu2-20260827/checkpoints/updates-000025`

Its held-out root-weighted MSE is 1.0066 versus 1.0126 for the parent, and its
four-stratum macro MSE is 1.0022. The game-block bootstrap child-minus-parent
MSE interval is [-0.0151, -0.0019]. Actor logits remain unchanged.

Cutoff-free game gates use the Phase-3 u75 policy checkpoint for both agents
and substitute `p30u25` only as the candidate value checkpoint:

| Gate | Games | Overall | US | USSR | Elo | Paired score CI |
|---|---:|---:|---:|---:|---:|---:|
| p30u25 D2/32 vs parent D2/32 | 44 | 63.6% | 63.6% | 63.6% | +97 | 50.0%--77.3% |
| p30u25 D4/128 vs parent D4/128 | 30 | 60.0% | 60.0% | 60.0% | +70 | 46.7%--73.3% |
| p30u25 D4/128 vs p30u25 D2/32 | 50 | 60.0% | 64.0% | 56.0% | +70 | 46.0%--74.0% |

All reported games have zero atomic-horizon and node-budget cutoffs. The
cross-mode result is still confirmatory rather than final because its interval
crosses parity; disjoint, side-swapped shards are continuing toward about 100
games under the adaptive evaluation rule.

The four-root anomaly suite was also repeated with eight deterministic search
seeds per root. The Phase-3 parent selected the suspicious action on 4/32 runs;
`p30u25` did so on 3/32. Every run was cutoff-free. This rules out a regression
in the repaired root mechanics and shows that the earlier two-seed Venezuela
result was search variance around a near tie.

Full-game D6/512 is not currently an economical confirmation mode: a four-game
shard failed to complete a game in about one hour and approached the host-memory
guard with a 2,097,152-node limit. D6 remains useful on bounded diagnostic roots,
but broad D6 evaluation is deferred until its tree-memory representation is
reduced. No D6 result is being inferred from incomplete games.

Accordingly, the final search-scaling promotion chain requires matched evidence
for Raw -> D2/32 -> D4/128 -> D4/512. Full-game D6/512 is exposed only through
an explicit `--run-d6` option and is recorded as optional evidence when enabled;
it is not allowed to block a candidate that has already demonstrated the deeper
D4/512 scaling required by the stage specification.

### Composed search backbone and convergence matrix

The checkpoint composer was extended to include both `value.*` and
`value_state_encoder.*` tensors. It now fails closed on an architecture mismatch
and verifies every actor tensor remains unchanged. The standalone search artifact
is:

`runs/search_distill/search-scaling-phase3u75-parentactor-p30u25-composed-20260828/checkpoints/composed`

Its saved full-weight hash equals the gated `p30u25` checkpoint hash, and a
fixed-root GPU smoke produced one complete 32-simulation root with zero cutoffs.
The complete test suite after this wiring change reports 307 passing tests.

The original 18-root Phase-3 convergence panel was then rescored with paired root
seeds and the composed model. Every completed row below is balanced 9 US / 9 USSR
unless marked selective:

| Configuration | Roots | Cutoffs | Wall time | Simulations/sec | Examples |
|---|---:|---:|---:|---:|---:|
| D2/32 | 18 | 0 | 117 s | 4.93 | 137 |
| D2/128 | 18 | 0 | 195 s | 11.82 | 284 |
| D4/128 | 18 | 0 | 400 s | 5.76 | 1,300 |
| D4/512 | 18 | 0 | 1,867 s | 4.94 | 1,484 |
| D6/512, unresolved roots only | 10 | 0 | 2,329 s | 2.20 | 1,105 |

Root-level parallelism used 8 workers for D2/32 and D4/128, 16 for D2/128,
and four memory-bounded workers for D4/512 and D6/512. The D4/512 maximum root
latency was 924 seconds; the D6/512 maximum was 1,728 seconds. This heavy tail
rules out uniform deep search as a practical behavior policy even though aggregate
cross-root throughput is usable.

Classification through D4/512 found 2 horizon-converged roots, 6 near ties, and
10 unresolved roots. Selective D6/512 changed one unresolved root into a near tie
but produced no additional high-confidence teacher. The paired D6/2048 pass then
completed all 9 unresolved roots with zero cutoffs: 18,432 simulations and 1,067
examples in 9,142 seconds, or 2.02 simulations/second. The longest root, a large
realignment decision, took 8,163 seconds. Classification through D6/2048 yielded
5 horizon-converged, 8 near-tie, and 5 unresolved roots, with 5 usable teacher
roots in total.

Selective D8/2048 was therefore limited to the two high-value Turn-10 unresolved
roots with the largest Q gaps. Both completed cutoff-free: 4,096 simulations and
33 examples in 2,599 seconds. The headline root retained its action but not a
stable full Q ranking; the allocation root became a near tie. D8 added no teacher,
leaving the final panel at 5 horizon-converged, 9 near-tie, and 4 unresolved roots.
This is sufficient to reject routine D6/D8 teaching while retaining them as rare,
fail-closed adjudicators.

The expanded whole-game p30u25 D4/128 versus p30u25 D2/32 gate stopped
adaptively after 144 complete matched games (72 side-swapped seed pairs). D4
scores 60.42% overall, 61.11% as US, and 59.72% as USSR, with Elo +73.46. The
Elo 95% interval is +24.17 to +125.87 and the paired bootstrap score interval is
53.47%--67.36%. All games are cutoff-free. This establishes D4/128 as a
statistically credible, side-balanced improvement over D2/32 on the composed
p30u25 backbone; it justifies using adaptive D4 as the practical policy-improvement
operator while reserving D6/D8 for rare adjudication.

### Adaptive complete-game generation status (2026-08-28 18:40 CST)

Four disjoint 32-game shards are running on the
composed p30u25 backbone. All real actions use D4/128, with selective D4/512 and
D6/512 adjudication under the convergence gates. The original three shards use
16 concurrent games each and the fourth uses eight, for 56 game workers and
eight resident inference servers. Recent telemetry sustains approximately
96--98% utilization on both A100s with near-zero queue depth. GPU memory is only
about 2.3 GiB per card, but host RSS, rather than accelerator memory, is the
binding resource, so no further workers are admitted while compute remains
saturated.

An attempted 64-worker topology exposed the operational limit. Search-worker
RSS growth caused 24 user-cgroup OOM kills; dead inference replicas left request
queues stalled, and terminating collector parents without their process groups
orphaned V8/search children. After terminating the three stale process groups,
user RSS fell from approximately 454 GiB to 181 GiB and available host memory
recovered to approximately 666 GiB. The durable six completed games were not
lost. The fourth shard was reduced to eight workers; after all four shards were
restored, user RSS remained below 280 GiB and the OOM-kill counter stayed at 24.

Future collector starts now take an OS-held single-writer lock on their output
directory, fail immediately if an inference replica exits, and avoid blocking
on shutdown sentinels for dead replicas. Supervisor-launched collectors receive
their own process session so an entire shard can be terminated without leaving
unrelated or orphaned workers. The focused lock, liveness, resume, replay-prune,
and adaptive-collection suite passes 24 tests.

The resource watcher keeps the 56-worker topology while host available memory
stays healthy. If availability remains below 300 GiB for three consecutive
one-minute polls, it atomically records an eight-worker override for one
16-worker shard, terminates that shard's complete process group, and lets the
continuation supervisor resume it from durable games. It never changes search,
seed, or replay identity, and it does not scale back up automatically.

A read-only sample of 956 staged replay shards contained 30,592 hierarchical
examples: 30,379 D4 budget-converged labels and 213 selectively escalated D6
horizon-converged labels. No sampled label reported an atomic-horizon or
node-budget cutoff. The sample included card, mode, Ops subtype, influence,
coup, realignment, and forced-response hierarchy levels; 1,644 labels had a root
Q gap of at least 0.01. These figures establish that generation is producing
usable search evidence rather than stalling, but they are not training data yet.
Every staged transaction remains inadmissible until its source game terminates,
receives a terminal outcome, and commits atomically.

The continuation supervisor now watches each shard independently. If a collector
is absent for three consecutive polls, it restarts only that shard with its
original output identity, seed base, and search configuration, reusing completed
games. Resume startup atomically prunes worker replay directories to labels whose
source game, worker, terminal winner/reason, and counts match durable completed-game
records; stale staging and orphan labels from an interrupted terminal commit are
removed, and missing committed labels fail closed. After all 128 games commit,
the four converged-Q conversions run in
parallel, followed by actor-head-only u5/u10/u15/u20/u25 training. Frozen encoder
and critic tensors are compared exactly against the composed parent before any
checkpoint is allowed into Raw/D2/D4/deeper promotion gates.

At 17:18 CST, the `gpu45b` shard exposed a runtime-depth failure after one
durable game: a worker reached Python's default recursion limit while decoding
a normal bridge response near the end of a long recursive atomic search path.
This was not a simulator, CUDA, memory-guard, or replay-commit failure. The
supervisor detected the absent collector after its three-poll audit and resumed
only that shard with the same identity and seed base. Its one committed game was
retained; stale in-flight replay was pruned before new workers started.

Search workers now reserve an interpreter recursion limit derived from the
configured atomic horizon (`4 * max_atomic_steps + 2048`, with a floor of 4096),
leaving serialization headroom without changing the search horizon or result.
Worker failures also preserve the complete traceback and effective recursion
limit in the parent log. This patch becomes active for any subsequent supervised
resume. Reloading the continuation supervisor also terminated the collector it
had launched directly, so the `gpu45b` shard was immediately resumed a second
time with the patched worker code. Its durable game remained intact; roughly 20
minutes of uncommitted in-flight branch work was discarded and replay-pruned.
The other two collectors were not interrupted.

The promotion-gate scheduler was also changed from serial execution to bounded
two-way process parallelism. Independent Raw/D2 preliminary gates, D4 candidate
gates, and confirmatory gates retain separate matched-seed output directories
and resumability, while using up to 48 evaluation workers across GPUs 4--5.
Raw/D2/D4 confirmatory modes now use the same seed/deal blocks rather than
mode-specific seed bases. The same-checkpoint search-scaling scheduler now also
runs independent D2-versus-Raw, D4/128-versus-D2/32, and
D4/512-versus-D4/128 comparisons with bounded two-way parallelism; D6 remains
optional and gate-dependent. Converged-teacher promotion additionally requires each
side to provide at least 40% of accepted labels; the first four committed games
were balanced at 46.6% US / 53.4% USSR labels.
At the time of the scale-up, 6/128 games had committed atomically across the
original three shards. The fourth shard uses a disjoint seed base and the same
transactional identity/search configuration. All committed games have a valid
winner and terminal target and report zero discarded games, partial labels,
atomic cutoffs, and node cutoffs. Recursion
headroom is enforced both at parallel-worker startup and at the core
`SelectivePuctPlanner.plan()` entry, so diagnostic and single-process search
paths receive the same protection. After making D4/512 the required deeper
full-game gate and D6/512 optional, the complete suite previously reported 330
passing tests. The subsequent operational guard changes pass their focused
24-test suite while collection remains active.

The active collection excludes `NEAR_TIE`, `UNRESOLVED`, and `UNVALIDATED`
roots from teacher replay. Its behavior policy still follows the selective D6
action on an unresolved adjudication (roughly 1--4 decisions in each of the first
completed games); changing that fallback in place would violate the durable
collection identity and mix policies across one dataset. This iteration therefore
keeps the declared behavior fixed while admitting only converged labels. A later
regenerated iteration may explicitly enable same-depth D4/512 budget fallback
for unresolved roots under a new collection identity.

The final evaluation continuation now emits `completion_audit.json` and
`completion_audit.md`. The audit remains fail-closed unless the convergence
panel, balanced D4 operator gate, 128-game teacher corpus, frozen actor sweep,
Raw/D2/D4 parent gates, four-root anomaly suite, same-checkpoint positive search
scaling, and promotion decision all pass. A rejected candidate is therefore an
auditable completed iteration, but it cannot be reported as completion of the
search-scaling policy-iteration goal.

At 19:07 CST, all 56 current game workers and eight inference replicas remained
live. GPU 4 and GPU 5 were at 92% and 91% utilization, respectively; the session
used approximately 368 GiB RSS, and the OOM-kill counter remained unchanged at
24. Durable progress was still 6/128 because all currently in-flight games were
in Turn 1 or Turn 2, but worker heartbeats, staging transactions, and inference
request counters were advancing without errors.

Final replay merging is now independent of the worker count used by the last
resume. It discovers every worker replay manifest after resume pruning and
atomically rebuilds the merged replay directory. This preserves valid completed
games owned by inactive worker IDs if the resource watcher reduces a shard from
16 workers to eight. Stale target shards are removed, while partial staging data
remains excluded. The resume-prune, operational-guard, continuation, resource,
completion-audit, and new resume-merge focused suite passes 22 tests.

The resource watcher now also performs one-way elastic rebalancing after a
shard finishes. While A/B/C are all active, D remains at eight workers and the
campaign stays at the measured-safe 56-worker topology. If one 16-worker shard
finishes while host available memory remains at least 400 GiB, D may resume at
24 workers; after another finishes it may rise to 32. The total requested
concurrency never exceeds 56, and D is not restarted when its remaining games
already fit in its current final wave. This avoids leaving both A100s underused
while the four-wave D shard becomes the tail-latency bottleneck. Resume pruning
and worker-count-independent final merging preserve only atomically committed
games across the rebalance. The expanded focused suite passes 25 tests.

The confirmatory same-checkpoint search-scaling gates now share one matched
seed/deal block across Raw-to-D2, D2-to-D4/128, and D4/128-to-D4/512 edges.
Expensive comparisons use a shorter prefix of that same block instead of a
mode-specific seed offset. This aligns the implementation with the documented
variance-control protocol and makes the monotonic scaling chain directly
comparable. The focused promotion, cross-mode, continuation, resource, replay,
and completion-audit suite passes 24 tests after this correction.

Confirmatory search scaling now also fails closed on side imbalance. The
50-game Raw-to-D2 and D2-to-D4/128 comparisons require at least 40% score on
both US and USSR; the smaller eight-pair D4/128-to-D4/512 comparison uses a 25%
side floor to avoid treating its higher sampling noise as a precise side
estimate. Aggregate score and confidence interval alone can no longer promote
a model whose test-time search gain comes from one side while the other
collapses. The expanded focused suite passes 26 tests.

The standalone completion audit independently requires the confirmatory
side-balance flag and records the per-edge US/USSR floors; it no longer trusts
the aggregate scaling flag as sufficient indirect evidence. Thus an old,
malformed, or partially generated cross-mode summary cannot satisfy completion.
The combined collection, amortization, promotion, cross-mode, resource, and
completion-audit suite passes 37 focused tests.

The queued distillation supplies broad anchor manifests, `lambda_anchor=4.0`,
and a 512-example anchor batch drawn from a stable 25% sample of complete-game
anchor replay. Inspection confirmed that `--anchor-from-all-safe-replay`
already defaults to true, so broad anchoring was behaviorally active before the
flag was made explicit; the explicit option prevents a future default change
from silently weakening this campaign. Provenance is now also fail-closed: all
four new anchor shards must name the composed parent, the historical
72,991-example broad corpus must name the actor checkpoint in the verified
composition manifest, and the final completion audit records this as an
independent requirement. The composed parent changes only the value tower, so
the historical Phase-3 u75 actor targets are exact for actor anchoring. The
expanded focused suite passes 65 tests.
