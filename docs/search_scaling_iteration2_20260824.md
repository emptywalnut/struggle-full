# Search-Scaling Policy Iteration 2

## Parent

The retained general parent is unchanged:

`runs/search_distill/root-mechanics-actor-correction-actionhead-u100-gpu4-20260819/checkpoints/updates-000075`

The rejected iteration-1 u5 child is not used as the parent or teacher backbone.

## Root-selection correction

Production root selection previously required a Q candidate to have both at
least three visits and at least 25% of the visit leader's count. This was
non-monotonic: increasing the simulation budget could increase the relative
threshold enough to make an already evaluated, higher-Q action ineligible.

On the 96 matched D4 roots, offline counterfactual selection found:

| Metric | Relative 25% floor | Fixed minimum-visit floor |
|---|---:|---:|
| Roots missing the best action with at least 3 visits | 12 | 5 |
| Mean positive root Q regret | 0.000966 | 0.000045 |
| Roots changed by the correction | 7 | 7 better / 0 worse |
| Worst corrected Q regret | 0.0317 | 0.0 |

`SelectivePuctConfig.root_selection_visit_fraction` now defaults to `0.0`.
The absolute three-visit reliability floor, Q-advantage threshold, Q-aware
visit-tie resolution, progressive widening, information masking, and exact
chance semantics remain unchanged. The relative fraction remains available
only for explicit ablations.

## Fixed-root convergence panel

The panel contains 96 roots and is exactly balanced by side:

- US / USSR: 48 / 48
- Early / Mid / Late War: 31 / 33 / 32
- Branching: low / medium / high: 26 / 34 / 36
- Coup / influence / opponent-event / other / realignment / scoring:
  26 / 18 / 15 / 15 / 11 / 11

Fresh common-root runs with the corrected production selector:

| Configuration | Complete roots | Partial roots admitted | Notes |
|---|---:|---:|---|
| D2/32 | 96/96 | 0 | 3,072 simulations; 22.10 sims/s |
| D4/128 | 96/96 | 0 | 12,288 simulations; 14.32 sims/s |
| D4/512 | 75/96 | 0 | 21 long-running roots fail-closed as unresolved |

The D4/512 unresolved roots are listed in:

`runs/search_scaling/phase3u75-convergence-rootselectfix-v2-20260824/d4_512/unresolved_positions.json`

No unfinished tree is present in its replay manifest.

Actual post-fix session outputs also confirm Q-to-action conversion:

| Configuration | Q-best selections among roots with >=3-visit actions | Mean Q regret | Max Q regret |
|---|---:|---:|---:|
| D2/32 | 92/95 | 0.000016 | 0.001157 |
| D4/128 | 94/96 | 0.000018 | 0.001654 |
| D4/512 | 74/75 | 0.000007 | 0.000493 |

Every non-Q-best selection is within the configured `0.002` material-Q
threshold; none is a relative-visit eligibility failure.

Single-particle classification remains deliberately conservative:

| Status | Roots |
|---|---:|
| Budget convergence | 1 |
| Horizon convergence | 4 |
| Near tie | 27 |
| Unresolved | 64 |

The replicated K=32 particle audit from the retained parent remains the better
variance estimate: 2 ensemble-converged roots, 7 deep horizon-converged roots,
4 near ties, and 5 particle-variance roots across its 18-root adjudication
panel. This supports adaptive deeper search, but not blind high-weight labels
from every high-budget root.

The complete pre-repair Q/ranking matrix on the same parent and same 96 matched
roots is also retained and reusable for the convergence diagnosis. Root Q and
rank telemetry is independent of the corrected final root tie-break:

| Configuration | Complete roots | Search throughput |
|---|---:|---:|
| D2/32 | 96/96 | 12.16 sims/s |
| D2/128 | 96/96 | 15.78 sims/s |
| D4/128 | 96/96 | 8.88 sims/s |
| D4/512 | 96/96 | 10.06 sims/s |
| D6/512 | 96/96 | 5.17 sims/s |

Selective D6/2048 adjudication completed 7/8 requested roots. The eighth root
was omitted fail-closed after reaching 46.7 GiB RSS and infeasible wall time;
it is absent from the replay manifest. The matrix classified 2 roots as budget
converged, 6 as horizon converged, 27 as near ties, and 61 as unresolved under
single-particle evidence. Replicated K=32 evidence is therefore required for
strong labels from unresolved roots, while near ties remain low-weight or
excluded.

## Value decision

The critic stays frozen. The evidence does not show a broad leaf-value failure:
the known marginal-value anomaly roots were already critic-correct, while much
of the remaining deep disagreement is attributable to horizon and particle/Q
ranking variance. Search Q is therefore used for conservative actor ranking,
not as a primary critic target.

## Iteration-2 generation and training

Active complete-game collection:

`runs/puct_selfplay/search-scaling-iteration2-phase3u75-adaptive-d4d6-32games-gpu45-20260824`

Configuration:

- 32 concurrent complete games, balanced US/USSR through both-side search
- base behavior D4/128
- D4/512 budget comparison for Q-gap <= 0.01 and high-stakes triggers
- selective D6/512 adjudication when D4 budgets do not converge
- K=32 information-safe belief particles
- exact small chance nodes
- fixed-minimum Q-aware root selection
- no PPO treatment of search behavior

Actor amortization will freeze encoder and value parameters and update only the
action encoder and policy/action heads. Checkpoints are saved at u5/u10/u15/u20/u25.
The objective uses low-weight Q-soft and pairwise ranking supervision, a strong
parent-policy anchor, balanced complete-game states, and no value loss.

Promotion requires parent-relative Raw/D2/D4 gates, same-checkpoint
Raw-vs-D2-vs-D4/deeper scaling, balanced sides, and all four frozen anomaly
roots. The newest update is never promoted automatically.

## Scoring-pressure horizon audit

The first two completed adaptive games both ended because US held a scoring
card. Replay inspection showed a concrete search-horizon failure rather than a
behavior-temperature deviation. In seed `97600018`, US held both Asia and
Europe Scoring. At T1 AR5 it selected Indo-Pakistani War while only two normal
US action rounds remained. D4/128 and D4/512 both preferred the non-scoring
card, after which only one scoring card could be played and the other caused an
automatic loss.

Two search-control defects were identified and corrected for subsequent
generation:

- same-depth D4/128 versus D4/512 agreement was being treated as sufficient
  evidence even for a horizon-sensitive scoring state;
- `scoring_card` escalation was capped at one search per side per Turn, so the
  later and more urgent card-choice roots became unvalidated D4-only actions.

The search trigger classifier now emits `scoring_pressure` when the number of
mandatory scoring cards in hand is at least the number of remaining normal
action rounds. Saito's zero-based AR index is handled explicitly. This trigger
does not prescribe an action; it requires deeper PUCT adjudication. Adaptive
collection exposes configurable `--adaptive-deep-always-trigger` categories,
and iteration 2 uses `scoring_pressure`.

Teacher construction is fail-closed: shallow-only `scoring_card` labels are
rejected unless their convergence status is `HORIZON_CONVERGENCE`. The count is
reported as `rejected:requires_horizon_validation`. Completed non-scoring
evidence from the in-flight collection remains usable; no partial tree is
admitted.

The exact T1 AR5 pressure root was replayed independently. D4/128, D4/512, and
D6/512 all selected Indo-Pakistani War instead of either scoring card. D6/512
completed 512 simulations in 241 seconds with zero node/atomic cutoffs, but its
hierarchical horizon was consumed before it could reliably back up the forced
end-of-Turn loss. The root is therefore `UNRESOLVED`, not horizon-converged.

A subsequent standalone D8/2048 run completed without node or atomic cutoffs.
It reached meaningful depth 8, crossed one Turn boundary, expanded 4,785 nodes,
and evaluated 2,616 cross-Turn leaves in 824 seconds. It strongly rejected the
non-scoring Indo-Pakistani War line (`Q=-0.8834`) and selected Europe Scoring
(`Q=-0.3330`) over Asia Scoring (`Q=-0.4651`). This establishes a concrete
horizon-convergence example: D4/D6 preserved the bad non-scoring action, while
D8 reversed it after crossing the Turn boundary.

That artifact predates explicit `terminal_evaluations` telemetry, so it is
diagnostic evidence rather than an admissible production teacher label. A
resumable post-collection D8 replay is scheduled to record terminal coverage
under the current schema. Future `scoring_pressure` teacher labels require both
depth 8 and at least one terminal branch evaluation.

Future adaptive generation also uses a configurable fail-closed behavior gate.
If a `scoring_pressure` search has not reached depth 8 with terminal coverage,
its search label is excluded and the real move falls back to the raw actor.
At the audited root this is the conservative behavior: the raw actor assigned
about 90% combined probability to the two scoring cards, while D4/D6 incorrectly
overrode it. The fallback is based on search-evidence quality and does not name
or force a Twilight Struggle action.

The same gate is implemented in the reusable `SelectivePuctPlanner`, not only
the collector. A fresh D4/128 replay of the exact AR5 root still computed the
old bad internal Q ranking, but deployment returned the raw actor's Asia
Scoring action with `evidence_gate_rejected=true`, zero backed Q advantage, no
action flip, and no teacher label. Thus shallow search cannot degrade this
state while a future cutoff-free D8 tree can still override raw if it reaches
terminal coverage.

## Campaign resilience

The 32-game collector writes each completed game atomically and resumes only
missing game indices under a strict collection-identity check. Its memory
watcher now also detects an absent collector process across three consecutive
polls, in addition to low-memory and stale-tail conditions. Any takeover reuses
the original seed/search identity and cannot replay durable completed games.
The duplicate gate watcher was removed; the primary iteration orchestrator is
the sole owner of promotion, anomaly, and cross-mode tournaments.

## In-flight side audit

The first nine completed adaptive games were all USSR wins: seven VP-threshold
wins at VP -20 to -27 and two held-scoring-card wins. None was a nuclear
terminal. This is a material side-balance warning, but generation/search effort
is balanced rather than side-starved:

- searched decisions: 467 US / 489 USSR;
- admissible budget-converged deep roots: 143 US / 144 USSR;
- admissible deep-root mean Q advantage over raw: 0.075 US / 0.066 USSR;
- actor top-1 agreement with the search teacher: about 38% on both sides.

A frozen critic audit over 288 balanced hierarchy-root examples from those
nine complete games found 80.9% overall terminal-sign accuracy. USSR-side roots
had 91.7% sign accuracy and calibrated mean +0.474 for target +1; US-side roots
had 69.9% and calibrated mean -0.408 for target -1. There is no value-perspective
sign inversion, but US losing-state calibration is materially weaker. Because
every audited game is an USSR win, this sample cannot establish general value
discrimination or justify value-head training by itself. The critic therefore
remains frozen, and the Raw/D2/D4 side gates must reject any child that preserves
or amplifies the observed imbalance.

Audit artifact:

`runs/search_scaling/iteration2-scoring-pressure-probe-20260824/nine-game-frozen-value-audit.json`

## Matched raw side-control and memory-safe resume

A 10-seed, side-swapped raw-vs-raw control was run with the retained Phase-3
u75 parent on the same seed range used by adaptive collection. Overall score is
necessarily 50% because the checkpoint plays itself, but the result by side is
strongly asymmetric: US won 2/10 and USSR won 8/10. Terminal outcomes were ten
VP-threshold, four final-scoring, four thermonuclear-war, and two ordinary
victory-point endings. This confirms that the parent itself is USSR-biased;
the early adaptive-search USSR sweep cannot be attributed to D4 alone.

Control artifact:

`runs/eval/search-scaling-iteration2-controls-20260825/parent-raw-selfplay-seeds97600000-10/results.json`

During collection, several long high-branching roots accumulated roughly
110/85/81 GiB worker RSS and reduced host-available memory below the configured
120 GiB guard for three consecutive samples. The memory watcher performed the
planned failover: it preserved the ten atomically completed games, stopped the
old 32-worker collector, and resumed only the missing game indices with 16
workers under the same strict collection identity. Available host memory
recovered from about 106 GiB to about 590 GiB. No completed game was replayed.

The resumed shard was subsequently moved from contended GPUs 4/5 to idle GPUs
0/1. GPU placement is intentionally excluded from collection identity because
it does not alter checkpoint, seed, search, belief, or sampling semantics. At
16 workers, aggregate policy/value throughput increased from roughly 203 to
604 requests/second. Increasing to 24 workers (22 actual missing games) raised
aggregate throughput to about 934 requests/second with 2.47 items/batch, while
host-available memory remained near 572 GiB. The low-memory takeover target
remains 16 workers, so a later pathological-root expansion will fail over to
the conservative concurrency without losing completed games.

The 24-worker setting was not retained. Over the following fourteen minutes,
available memory fell monotonically from roughly 555 to 482 GiB while sustained
throughput settled near 727 requests/second. Since this projected another
low-memory takeover before any new game completed, collection was proactively
returned to 16 workers. Available memory recovered immediately to about 609
GiB. The durable choice is therefore 16 workers on GPUs 0/1: lower peak
requests/second, but a materially better probability of completing full games
without discarding hours of in-flight search.

Even at 16 workers, four simultaneous pathological D4/D6 roots later grew to
roughly 100/83/78/42 GiB RSS and reduced host-available memory to about 164
GiB. The collector was stopped after 13 durable games, before the host-level
watcher or OOM killer fired. Search now has an operational
`max_process_rss_mib` guard; this campaign resumes with
`--max-search-worker-rss-mib 20480`. A root that hits the guard is marked with
a node-budget cutoff, returns to the raw behavior action, clears its
process-local snapshots, and is excluded by the existing fail-closed teacher
gates. This is an execution-safety boundary, not a strategic label or an
information shortcut.

The resumed tail also uses `--max-search-seconds 300`. This provides the same
fail-closed behavior for exact-chance roots that remain below the RSS ceiling
but consume pathological wall-clock time. A timed-out tree increments the
existing node/resource-cutoff telemetry, falls back to the raw behavior
action, and emits no hierarchical or teacher labels; completed games and all
other search semantics are unchanged.

## Fail-closed promotion protocol

The 10-seed-pair Raw/D2/D4 child-versus-parent gates are preliminary screens.
Only one candidate that clears those screens receives 25 matched, side-swapped
seed pairs for each of Raw, D2/32, and D4/128. Promotion is then decided by a
separate durable artifact rather than by checkpoint recency or training loss.

The candidate must satisfy all of the following:

- Raw versus parent: score at least 0.45 and paired-bootstrap lower bound at
  least 0.40;
- D2 versus parent: score above 0.50 and paired-bootstrap lower bound at least
  0.50;
- D4 versus parent: score at least 0.50 and paired-bootstrap lower bound at
  least 0.45;
- US and USSR score floor at least 0.35 across confirmatory Raw/D2/D4;
- all four frozen anomaly roots pass without cutoff or root-selection
  regression;
- confirmatory same-checkpoint marginal comparisons cover Raw-to-D2,
  D2-to-D4/128, D4/128-to-D4/512, and D4/512-to-D6/512 and support
  non-decreasing strength.

The final decision is written to
`runs/search_distill/search-scaling-iteration2-phase3u75-u25-gpu4-20260824/promotion_decision.json`.
The adjacent `GENERAL_PARENT` pointer names the child only when every criterion
passes; otherwise it retains the Phase-3 u75 parent and records every failed
criterion.
