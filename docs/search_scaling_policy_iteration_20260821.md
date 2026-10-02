# Search-Scaling Policy Iteration Diagnostic (2026-08-21)

## Decision

The retained Phase-3 u75 parent remains the general parent:

`runs/search_distill/root-mechanics-actor-correction-actionhead-u100-gpu4-20260819/checkpoints/updates-000075`

The original single-particle 96-root sweep was not sufficient for actor training. A
follow-up replicated panel separated belief-particle variance from genuine horizon
changes. Diverse adaptive-search generation, teacher construction, and actor-only
amortization are complete. No child has yet been promoted because whole-game deeper
search gates are still in progress.

Current campaign state:

- 18 stratified roots, three independent K=32 streams at D4/512 and D6/512;
- D6/2048 adjudication on all ten material D4/D6 disagreements;
- 9 converged roots, 4 meaningful-Q near ties, and 5 particle-variance roots;
- 32/32 adaptive D4/D6 complete games and 179,547 replay examples committed;
- 6,438 cutoff-free, particle-safe teacher examples retained from all 32 games;
- actor-only u5/u10/u15/u20/u25 checkpoints trained with encoder and value frozen;
- u5 is the only candidate that clears the preliminary parent-relative Raw/D2/D4 floor;
- same-checkpoint D2 gains are confirmed, while D4/128 is currently a balanced near tie
  against D2/32; D4/512 and D6/512 gates remain active.

The first complete adaptive game (`seed=97300000`) finished after 5.23 hours:

- USSR win by VP threshold at 229 environment steps;
- 120 meaningful searches, balanced 66 US / 54 USSR;
- 58,880 total simulations, 3.83x the D4/128-only budget;
- 40 budget-converged, 1 horizon-converged, 37 near-tie, 2 unresolved,
  and 40 unvalidated roots;
- 3,758 complete-game hierarchical examples committed, no partial rejection;
- approximately 21 converged roots remain after the final `Q gap >= 0.02` gate.

This confirms end-to-end transaction correctness and adequate teacher yield. It also
quantifies the cost: 80/120 roots escalated beyond D4/128, so this campaign is an
offline strong-teacher run rather than a deployment-latency configuration.

## Whole-game evidence

All completed entries below are matched and side-swapped with zero atomic-horizon or
node-budget cutoffs. Scores in the first table are against the Phase-3 u75 parent under
the same inference mode.

| Candidate | Raw | D2/32 | D4/128 | Status |
|---|---:|---:|---:|---|
| u5 | 50% (US 70 / USSR 30) | 50% (50 / 50) | **65% (60 / 70)** | only general candidate |
| u10 | 45% (60 / 30) | 45% (70 / 20) | 50% (50 / 50) | dominated by u5 |
| u15 | 35% (50 / 20) | 50% (70 / 30) | not run | raw collapse |
| u20 | 35% (50 / 20) | 55% (70 / 40) | not run | D2 specialization |
| u25 | 40% (50 / 30) | 50% (50 / 50) | 50% (60 / 40) | raw regression |

Direct same-checkpoint search scaling for u5 currently shows:

| Comparison | Games | Score | US | USSR | Paired bootstrap 95% CI |
|---|---:|---:|---:|---:|---:|
| D2/32 vs Raw, screen | 20 | 75% | 70% | 80% | 55%-95% |
| D2/32 vs Raw, confirmatory | 50 | **72%** | **84%** | **60%** | **60%-84%** |
| D4/128 vs D2/32, screen | 20 | 65% | 70% | 60% | 45%-85% |
| D4/128 vs D2/32, confirmatory | 50 | **46%** | **48%** | **44%** | **36%-56%** |

The independent D4 samples combine to 36/70 (51.4%). The supported conclusion is a
balanced D4/D2 near tie, not a positive D4 gain and not a search-regime collapse. The
Phase-3 parent itself scored 65% (US 60 / USSR 70, CI 50%-80%) in a 20-game D2-vs-Raw
baseline, so repaired cheap search is already useful before this amortization cycle.

The u5 frozen anomaly gate has also completed. Across D2/32, D4/128, and D6/512 repeats,
the suspicious UK overcontrol, Italy overcontrol, Mexico realignment, and Venezuela
realignment actions were selected 0/20 times with zero cutoffs. Formal summary emission
waits for the parent-relative D4/512 gate so the automated promotion transaction remains
fail-closed.

## Fixed panel

The panel contains 96 exact complete-game roots:

- US / USSR: 48 / 48
- Early / Mid / Late: 31 / 33 / 32
- Low / Medium / High branching: 26 / 34 / 36
- Coup / influence / opponent-event / other / realignment / scoring: 26 / 18 / 15 / 15 / 11 / 11

All configurations use the same per-root RNG seed, normal information masking, K=32
legal-consistent particles, exact small chance handling, and the repaired root mechanics.

## Search sweep

| Config | Roots | Simulations | Wall time | Sims/s | Replay examples |
|---|---:|---:|---:|---:|---:|
| D2/32 | 96 | 3,072 | 4.2 min | 12.16 | 917 |
| D2/128 | 96 | 12,288 | 13.0 min | 15.78 | 1,895 |
| D4/128 | 96 | 12,288 | 23.1 min | 8.88 | 5,345 |
| D4/512 | 96 | 49,152 | 81.4 min | 10.06 | 7,245 |
| D6/512 | 96 | 49,152 | 158.6 min | 5.17 | 10,464 |
| D6/2048 | 7/8 | 14,336 | 1 root rejected | n/a | 263 |

D4/512's worst root took 33.9 minutes. D6/512's worst root took 77.9 minutes.
The omitted D6/2048 root was stopped fail-closed after roughly 36 CPU minutes and 46 GiB
RSS; it is not present in teacher replay.

## Convergence result

Thresholds: rank correlation >= 0.70 and meaningful Q gap >= 0.01.

| Classification | Roots | US | USSR |
|---|---:|---:|---:|
| Budget convergence | 2 | 1 | 1 |
| Horizon convergence | 6 | 1 | 5 |
| Near tie | 27 | 16 | 11 |
| Unresolved | 61 | 30 | 31 |

Teacher configuration counts:

- D6/512: 5 roots
- D6/2048: 3 roots

The eight stable roots cover early/mid/late and several action types, but are only 8.3%
of the panel and are USSR-heavy (6/2). Sixty-one roots still have materially inconsistent
rankings at the strongest available cutoff-free search.

## Interpretation

The original 96-root result mixed at least two effects: genuine horizon changes and
variation between legal-consistent hidden-hand particle sets. Matching one RNG stream
across depth does not establish stability across belief samples. The replicated panel
shows that deeper search does converge on a material subset, while five roots remain too
particle-sensitive for strong labels.

The fixed roots remain diagnostic evidence rather than the actor training distribution.
Broad actor amortization now uses adaptive-search states from complete games, with
convergence and cutoff metadata preserved per decision.

## Replicated variance and horizon adjudication

The follow-up panel contains 18 roots balanced 9/9 by side and 6/6/6 by era. It excludes
the pathological high-branching roots that previously dominated runtime. Three independent
K=32 streams were run at D4/512 and D6/512. Direct or leave-one-replicate-out ensemble
stability classified:

| Initial replicated status | Roots | US | USSR |
|---|---:|---:|---:|
| Ensemble convergence | 2 | 1 | 1 |
| D4/D6 horizon disagreement | 10 | 6 | 4 |
| Particle variance | 5 | 2 | 3 |
| Near tie | 1 | 0 | 1 |

All ten horizon-disagreement roots then received three cutoff-free D6/2048 replicates.
Seven stabilized as horizon convergence (US 5, USSR 2); three became deep near ties
(US 1, USSR 2). None remained unresolved. The final 18-root attribution is therefore:

| Final status | Roots |
|---|---:|
| Converged | 9 |
| Near tie | 4 |
| Particle variance | 5 |

This establishes positive high-budget convergence evidence, but not yet whole-game
strength improvement. Only complete-game generation, amortization, and matched promotion
gates can establish the latter.

## Implemented safeguards

- Fixed, stratified convergence manifest generation.
- Per-root paired RNG across budget/depth sweeps.
- Q/ranking-based convergence classification with cutoff fail-closed semantics.
- Correct horizon-convergence classification when D4->D6 changes and D6/512->D6/2048 stabilizes.
- Strategic/Q-gap prioritization for deep adjudication; fixed the prior nonexistent D8 Q-gap field.
- Adaptive complete-game planners now share root RNG without clearing committed hierarchical continuations.
- High branching alone no longer triggers expensive adaptive escalation.
- Complete-game generation can disable flip-only escalation with `--no-adaptive-on-flip`;
  D4/128 remains the behavior policy, while D4/512 and D6/512 are reserved for
  small-Q-gap and high-stakes roots.
- Convergence metadata is persisted in search replay.

## Completed generation and amortization

The adaptive complete-game collection finished all 32 scheduled games without a discarded
or partial game. It produced 179,547 replay examples and 8,242 searched decisions, balanced
4,180 US / 4,062 USSR. The behavior planner averaged 528.31 simulations per searched
decision. Terminal reasons were 17 VP threshold, eight final scoring, three thermonuclear
war, two Wargames, one scoring card held, and one tie.

The first capped deep-candidate manifest was found to be biased toward the first three
games because it sliced game-ordered records. No replay was lost: every completed-game
transaction retained its own candidates. Candidate capping and audit-panel construction
now stratify by side, era, and source game and read those durable transactions directly.

The replacement 18-root particle audit uses 18 distinct complete games and is balanced
9/9 by side and 6/6/6 by era. Three independent K=32 streams at D4/512 and D6/512 yielded:

| Particle-audit status | Roots |
|---|---:|
| Ensemble convergence | 8 |
| Horizon disagreement | 3 |
| Near tie | 3 |
| Particle variance | 4 |

Particle-variance roots were excluded by durable source-game/source-step identity. The
converged teacher retained 6,438 examples from all 32 games after complete-tree, cutoff,
visit, Q-gap, and particle-stability gates. Coverage was 2,974 US / 3,464 USSR, with 2,901
raw/search flips and 3,537 non-flips. Hierarchical coverage includes 608 influence-allocation,
136 coup-target, and seven realignment-target labels in addition to root/card/mode/Ops and
forced-response labels.

Actor-only amortization starts from Phase-3 u75. The encoder and value head remain frozen;
only `action_head` parameters are updated. Twenty-five low-rate updates produced durable
u5/u10/u15/u20/u25 checkpoints with parent anchoring, modest pairwise ranking pressure,
and no value or search-Q regression loss. The diversity/root caps reduced the 6,438 accepted
teacher examples to 4,559 training examples rather than repeatedly optimizing the same roots.

## Active promotion gates

1. Evaluate every u5/u10/u15/u20/u25 candidate against Phase-3 u75 under matched Raw and
   D2/32 games.
2. Run D4/128 for the two candidates with the strongest non-collapsing Raw/D2 floor.
3. Run a smaller D4/512 confirmation only for a candidate that clears all preliminary gates.
4. Re-run the four frozen anomaly probes before any promotion.
5. Measure direct same-checkpoint Raw -> D2 -> D4 -> deeper scaling for the anomaly-safe
   candidate. A failed gate retains Phase-3 u75 as the general parent.

The active complete-game collection is one independently resumable 32-game shard.
After disk-backed transactions and explicit heap reclamation were validated, the
primary shard was resumed with 32 workers and one shared inference server per GPU.
Its first preserved complete game remains committed, while the other 31 games run as
one wave.

The primary shard has a fail-safe memory watcher. After worker RSS plateaued and
the optional supplemental shard was quarantined, the takeover threshold was reduced
from 260 to 160 GiB available memory; three consecutive low samples are still
required. It retains committed complete
games and resumes the same identity with 8 workers and one inference replica per
GPU; it never converts partial games into teacher data.

A concurrent optional 32-worker supplemental shard was admitted briefly to test the
remaining host capacity. Aggregate inference throughput rose from roughly 1,050 to
1,970 requests/s, but available memory crossed its 300-GiB guard. The guard correctly
aborted and quarantined only that shard after three consecutive low samples
(285.2/292.1/246.1 GiB). It had committed zero games and reported zero inference
errors; the primary and its committed game were untouched. A later guarded 8-worker
attempt was likewise quarantined with zero complete games. The supplemental shard was
then removed from this iteration: the primary is projected to yield roughly 3k-4k
visit-qualified labels, while the broad anchor already contributes about 9k ordinary
states. Iterating to a new network is more useful than doubling old-parent search data.

The post-tournament anomaly gate is automated by
`scripts/run_search_scaling_anomaly_gate.py`. It selects only a candidate with
complete Raw/D2/D4 evidence, runs influence and realignment probes concurrently,
and rejects any checkpoint that selects a suspicious action again or introduces
an atomic/node-budget cutoff on the fixed roots.

Particle audit failures are also root-local: positions classified as
`PARTICLE_VARIANCE` are excluded from the converged teacher by durable
`(source_game, source_step)` identity, in addition to the campaign-wide variance
rate gate. This prevents a known unstable root from being accepted merely because
one common-particle D4/D6 comparison happened to agree.

Search scaling is measured with direct same-checkpoint, matched-seed comparisons:
`D2/32 vs Raw`, `D4/128 vs D2/32`, `D4/512 vs D4/128`, and
`D6/512 vs D4/512`. Candidate and opponent search configurations are resolved
independently and included in the resumable evaluation identity, so scores from
different parent baselines are never combined into a false scaling curve.

The first complete adaptive game required 229 environment steps, 120 searched
decisions, and 58,880 total simulations. It terminated by USSR VP threshold after
5.23 hours. The search work was 3.83x a pure D4/128 game because 80 roots received
budget/depth comparison. Its classifications were 40 budget-converged, one
horizon-converged, 37 near ties, two unresolved, and 40 unvalidated. This confirms
that broad complete-game generation is healthy but simulator/search bound. With 32
workers, batches average about 4.6 leaves and the two inference servers serve roughly
1,050 requests/s; GPU 4/5 utilization is usually 35-55% while V8 branch transitions
and branch-local state dominate wall time.

Long-game memory is now bounded for every resumed/new collector process:

- branch-local PUCT sessions explicitly release node, observation, trace, and chance caches;
- uncommitted complete-game replay is staged in small disk shards instead of retaining
  thousands of observation tensors in worker RAM;
- teacher and broad parent-anchor replay are separate fail-closed transactions;
- worker heartbeat files report game, step, Turn/AR, search latency, and RSS;
- workers periodically collect cycles and return free glibc pages.

The current primary workers use these safeguards. A separate memory watcher can still
perform an identity-preserving lower-concurrency resume if the primary crosses its
safety margin.

The same watcher now also handles a pathological collection tail. It does not interrupt
a broad active shard: takeover is eligible only when at most two games remain and every
remaining game heartbeat has been stale for at least twelve hours. The collector then
reuses the immutable collection identity and schedules only missing game indices, so
already committed games and their replay transactions are never repeated. This guard
was added after one deep root demonstrated that a single branch can occupy a worker for
many hours even though the other games and inference servers remain healthy.

Actor amortization uses converged roots only as teacher labels. Parent anchoring also
includes a deterministic 12.5% sample of the existing 72,991-example, 16-complete-game
Phase-4 replay generated from the same Phase-3 parent. This supplies roughly 9k broad
ordinary states without materializing the full replay in memory.

Convergence at a real root is not automatically treated as convergence for every
belief-tree hierarchy node. A one-game audit found 160/312 initially accepted labels
with fewer than eight total visits, including one-visit nodes with spuriously large Q
gaps. The production teacher now requires at least eight total visits and at least two
visits on the Q-best action. On the audited game this retained 134 labels (US 61 / USSR
73), including 25 influence-allocation labels and all 23 real-root labels.

Promotion remains two-stage. Preliminary matched gates use 20 games for Raw/D2/D4 and
8 games for deeper modes. A candidate with non-collapsing preliminary scaling then gets
50 games for D2-vs-Raw and D4-vs-D2, 16 games for D4/512-vs-D4/128, and 8 games for
D6/512-vs-D4/512. Point estimates and paired/bootstrap confidence intervals are retained;
the latest checkpoint is never promoted automatically.

Whole-game evaluation now records atomic-horizon and node-budget cutoffs separately for
candidate and opponent. Every gate fails closed if any cutoff is observed, so a truncated
deeper tree cannot masquerade as negative search scaling.

The first u5 D2 gate demonstrated that the original 4,096-node D2 ceiling was still
binding under K=32 belief search: 109 decision roots hit the ceiling. That result is
retained only as `u5/d2-cutoff4096-invalid` and is excluded from selection. Evaluation
ceilings are now 32,768 for D2/32, 262,144 for D4/128, 1,048,576 for D4/512, and
2,097,152 for D6/512. These are safety ceilings rather than preallocations; simulation
budgets and search semantics are unchanged. The matched u5 D2 replay with the corrected
ceiling completed 20/20 games with zero cutoff and changed the result from the invalid
45% to 50%, confirming that cutoff-contaminated gates cannot be treated as approximate
strength measurements.

## Artifacts

- Panel: `runs/search_scaling/phase3u75-convergence-20260821/positions.json`
- Matched sweeps: `runs/search_scaling/phase3u75-convergence-20260821/matched/`
- Final evidence: `runs/search_scaling/phase3u75-convergence-20260821/matched/classification-final-v2/`
- D6/2048 fail-closed status: `runs/search_scaling/phase3u75-convergence-20260821/matched/d6_2048/adjudication_status.json`
- Replicated ensemble evidence: `runs/search_scaling/phase3u75-variance-20260821/ensemble-analysis-v3/`
- Final high-budget horizon evidence: `runs/search_scaling/phase3u75-variance-20260821/horizon-adjudication/final-v3/`
- Active selective-adaptive collection: `runs/puct_selfplay/search-scaling-phase3u75-selective-adaptive-d4d6-32games-w12-gpu45-20260821/`
- Resumable amortization campaign: `runs/search_distill/search-scaling-phase3u75-amortization-u25-gpu4-20260821/`

The earlier `search-scaling-phase3u75-adaptive-d4d6-32games-gpu45-20260821`
collection was stopped with zero complete games after telemetry showed that actor flips
alone escalated too many roots. It is retained only as a superseded throughput artifact.

## Final promotion decision

The early-checkpoint sweep selected u5 as the only plausible child, but the full
search-scaling gate rejects it as a general parent. All formal games below are complete,
matched/side-swapped, information-masked, and free of atomic-horizon or node-budget
cutoffs.

### Child versus Phase-3 parent

| Child | Mode on both agents | Games | Overall | US | USSR | Elo delta | Paired/bootstrap 95% CI |
|---|---|---:|---:|---:|---:|---:|---:|
| u5 | Raw | 20 | 50% | 70% | 30% | 0.0 | 50%-50% |
| u5 | D2/32 | 20 | 50% | 50% | 50% | 0.0 | 35%-65% |
| u5 | D4/128 | 20 | 65% | 60% | 70% | +107.5 | 50%-80% |
| u5 | D4/512 | 8 | **37.5%** | **25%** | 50% | -88.7 | 12.5%-50% |

The u5 D4/512 result fails both the parent-strength and side-balance gates. u10 was the
other D4 candidate; it scored 45% Raw, 45% D2/32, and 50% D4/128, so it is also not a
promotion candidate. u15/u20/u25 had Raw scores of 35%/35%/40% and were eliminated before
deep evaluation.

### Same-checkpoint search scaling

| Marginal comparison | Formal games | Score of deeper mode | US | USSR | Elo delta | Paired/bootstrap 95% CI |
|---|---:|---:|---:|---:|---:|---:|
| u5 D2/32 vs u5 Raw | 20 | 75% | 70% | 80% | +190.8 | 55%-95% |
| u5 D4/128 vs u5 D2/32 | 20 | 65% | 70% | 60% | +107.5 | 45%-85% |
| u5 D4/512 vs u5 D4/128 | 8 | **37.5%** | **25%** | 50% | -88.7 | 0%-75% |

The larger confirmation sets refine the first two rows: D2/32 beat Raw 36/50 (72%,
CI 60%-84%), while D4/128 scored 23/50 against D2/32 (46%, CI 36%-56%). Combining the
independent 20- and 50-game D4/D2 seed blocks gives 36/70 (51.4%), so D4/128 is best
described as approximately tied with D2/32, not demonstrably stronger.

A supplemental D4/512 block was paused after 9/16 complete games once the formal
8-game gate failed. It scored 5/9 (55.6%, US 60%, USSR 50%); this is retained for audit
but is not substituted for the predeclared formal gate. D6/512 was likewise stopped
fail-closed after the upstream D4/512 gate failed. Its formal partial progress was 2/2
without a complete matched pair (one win on each side); a separate supplemental block
had one loss. These underpowered D6 observations are not a strength estimate.

The resulting tested curve is therefore:

`Raw < D2/32`, `D2/32 ~= D4/128`, and `D4/512` does not improve on `D4/128`.

This does not satisfy the requested positive scaling curve. More D6 games cannot repair
the already failed Raw/D2/D4/D4-512 promotion gate, so the adaptive evaluator stopped
them rather than spending tens of additional hours on a rejected child.

### Frozen anomaly regression

The u5 child passes the four-root fail-closed anomaly suite. Across two D2/32 repeats,
two D4/128 repeats, and one D6/512 run per root, the suspicious action was selected 0/20
times with zero cutoff:

| Root | Suspicious raw prior | Search selections |
|---|---:|---:|
| USSR UK overcontrol | 3.65% | 0/5 |
| US Italy overcontrol | 6.11% | 0/5 |
| USSR Mexico realignment | 17.24% | 0/5 |
| US Venezuela realignment | 19.37% | 0/5 |

Thus repaired root mechanics and local anomaly suppression remain intact. The rejection
is caused by whole-game search compatibility/generalization, not recurrence of the four
known root-search artifacts.

### Decision

- **Rejected general child:** u5
  `runs/search_distill/search-scaling-phase3u75-amortization-u25-gpu4-20260821/checkpoints/updates-000005`
- **Retained general parent:** Phase-3 u75
  `runs/search_distill/root-mechanics-actor-correction-actionhead-u100-gpu4-20260819/checkpoints/updates-000075`
- The shared encoder and value head remain frozen; no value-calibration experiment was
  justified because the frozen-root diagnostics did not establish stable deep-search
  contradiction of the critic.
- Do not iterate search-generated data from u5. The next experiment should first diagnose
  why D4/512 produces worse whole-game decisions than D4/128 despite zero cutoffs: compare
  root Q calibration, prior/visit concentration, and action-flip quality on complete-game
  roots from the failed D4/512 matches.

Final artifacts:

- Parent-relative gates: `runs/eval/search-scaling-phase3u75-amortization-gates-20260821/gate_summary.json`
- Same-checkpoint scaling: `runs/eval/search-scaling-phase3u75-amortization-cross-mode-20260821/summary.json`
- Frozen anomaly gate: `runs/diagnostics/search-scaling-phase3u75-amortization-anomaly-gate-20260821/summary.json`
- Promotion decision: `runs/search_distill/search-scaling-phase3u75-amortization-u25-gpu4-20260821/promotion_decision.json`

Validation: full project test suite passes, 258 tests.
