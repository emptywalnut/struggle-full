# Root Search Mechanics: Phase 1-2 Validation

Date: 2026-08-19

## Scope

The accepted parent remained frozen:

`runs/puct_selfplay/cycle0-student-balanced2games-100updates-gpu4-20260812/checkpoints/distilled`

All roots use the validated exact snapshots from the prior D4 anomaly suite. Opponent
hands remain masked, hidden hands are legal-consistent root samples, and small chance
nodes are enumerated exactly. No reward, rules, actor, or critic weight changed during
these two phases.

## Root Mechanics Change

The old search activated every root action and force-visited each unvisited action. At
57-72-way placement roots, D2/32 therefore produced all-one-visit ties and selected by
array index.

The corrected root now uses:

- root-only progressive widening: initial 6, maximum 32, `c=1.5`, `alpha=0.5`;
- root prior temperature 2.0 and root `cpuct=0.9`;
- candidate admission ranked by 50% raw-prior rank and 50% one-step exact/critic-Q rank;
- two protected raw-prior candidates, so cheap-Q admission cannot hide the actor's bad
  favorite from evaluation;
- a 25% root prior mixture from cheap Q;
- final selection by visits, then Q/prior/label for exact ties, with Q override among
  actions having at least 3 visits and 25% of the visit leader;
- deterministic label-based residual tie handling, never action-index handling.

The root Q probe re-determinizes legal-consistently and sends only the normal masked
observation to the network. Exact chance enumeration is preserved.

## Frozen Results

Each cell lists two deterministic search seeds. All 24 searches had zero atomic-horizon
and zero node-budget cutoffs.

| Root | Raw suspicious action | D2/32 | D4/128 | D6/512 | Result |
|---|---|---|---|---|---|
| USSR UK overcontrol | UK | Saharan States / Romania | Algeria / South Africa | Zimbabwe / Zimbabwe | UK rejected 6/6 |
| US Italy overcontrol | Italy | Pakistan / North Korea | East Germany / UK | Libya / Libya | Italy rejected 6/6 |
| USSR Mexico target | Mexico | Brazil / Chile | Chile / Colombia | Venezuela / Peru | Mexico rejected 6/6 |
| US Venezuela target | Venezuela | Angola / Cuba | Cuba / Cuba | Nigeria / Nigeria | Venezuela rejected 6/6 |

### D2 large-root behavior

| Root | Legal | Visited | Maximum N | Suspicious N | Selected N |
|---|---:|---:|---:|---:|---:|
| UK seed 0 | 57 | 9 | 7 | 2 | 7 |
| UK seed 1 | 57 | 9 | 5 | 1 | 5 |
| Italy seed 0 | 72 | 9 | 5 | 4 | 5 |
| Italy seed 1 | 72 | 9 | 6 | 4 | 6 |

D2 now compares a budget-sized active set and revisits actions. It no longer spends all
32 simulations creating an uninformative one-visit tie. The protected actor favorites
UK and Italy are admitted and sampled, but better-Q actions can accumulate enough visits
to beat them.

### Q-to-visit conversion

Q-aware final selection was required in both Italy D4 seeds, both Mexico D4 seeds, both
Mexico D6 seeds, and most Venezuela D4/D6 seeds. This is intended: a materially
better-Q, sufficiently sampled action can now overcome an early prior-induced visit
lead. It cannot win from a one-sample estimate because the minimum is three visits.

Representative wall times were 4.8-6.6 seconds for large-root D2, 21.5-37.3 seconds for
large-root D4, 111-234 seconds for large-root D6, 36-43 seconds for target D2, 167-181
seconds for target D4, and 848-875 seconds for target D6.

## Ablations

- Widening alone removed the D2 tie artifact but remained prior-sensitive.
- Root-prior flattening improved diversity but did not consistently expose critic-ranked
  alternatives.
- Cheap-Q admission produced the strongest practical correction.
- Protecting the top two raw-prior actions was necessary: without it, cheap-Q admission
  could exclude UK/Italy entirely and make the diagnostic unable to measure whether
  search actually defeated the bad prior.
- A two-visit Q override was noisy on realignment roots; the final minimum is three.

## Gate Decision

Phase 2 passes. All four pathologies are corrected at D2/D4 and remain corrected at D6,
with no cutoff-limited teacher evidence. D8 is intentionally skipped: the prior anomaly
suite showed thousands of D8 cutoffs, while the practical budgets already satisfy the
diagnostic objective.

The next stage can therefore train downstream actor corrections. It must use corrected
Q-aware targets and exact/counterfactual rankings, not the old D4 visit distributions.

## Artifacts

- Implementation: `struggle_ai/mcts.py`
- Probe telemetry: `scripts/probe_d4_anomalies.py`
- Tests: `tests/test_selective_puct.py`
- Canonical results: `runs/diagnostics/root-mechanics-phase1-20260819/protected-prior/`
- Small-target D4 results: `runs/diagnostics/root-mechanics-phase1-20260819/default/d4/`
- Actor-correction replay builder: `scripts/build_anomaly_actor_replay.py`
- Final four-root replay: `runs/search_distill/root-mechanics-actor-correction-replay-v3-20260819/`
