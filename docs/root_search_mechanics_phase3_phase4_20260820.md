# Root Search Mechanics: Phase 3-4 Outcome

Date: 2026-08-20

## Decision

The Phase-3 actor-correction checkpoint remains the accepted general parent:

`runs/search_distill/root-mechanics-actor-correction-actionhead-u100-gpu4-20260819/checkpoints/updates-000075`

The best Phase-4 cheap-search branch is:

`runs/search_distill/rootfix-phase4-d4-snapshot9-actorvalueheads-u100-gpu4-20260819/checkpoints/updates-000075`

It is **not promoted as the general parent**. It beats the parent decisively under
D2/32, but regresses under raw play and D4/128. It is retained as a deployment/search
compatibility experiment and as evidence that corrected D2 targets can improve cheap
search.

## Phase 3

The downstream actor was trained from corrected Q-aware search targets, exact or
counterfactual rankings, pairwise ranking loss, and parent anchoring. The critic was
preserved. The selected u75 checkpoint scored 95% over 20 matched raw games in its
Phase-3 gate (US 100%, USSR 90%); u100 fell to 75%, so training was not continued to
the final update automatically.

On the two large influence roots, the suspicious raw prior remained rank 1 but moved
in the correct direction relative to the original frozen model:

| Action | Original probability | Phase-4 child probability | Change |
|---|---:|---:|---:|
| USSR place UK | 4.82% | 3.73% | -1.10 pp |
| US place Italy | 6.81% | 6.14% | -0.67 pp |

The actor has not fully internalized the correction; practical search still supplies
most of the protection.

## Phase 4 Generation and Training

D4/128 self-play from the Phase-3 parent completed 16/16 games with no discarded
games. It produced 72,991 cutoff-free examples from 2,356 searched decisions, balanced
between US 1,160 and USSR 1,196. Search used exactly 128 simulations on average, had a
63.37% flip rate, and rejected 93 partial labels fail-closed.

The joint actor/value-head student used the first nine complete games (33,804 examples),
with the shared encoder frozen, low learning rate, parent anchoring, pairwise Q ranking,
terminal value targets, and low-weight search-Q supervision. Checkpoints were retained
at u25/u50/u75/u100.

## Matched Gates

All entries use 10 matched seed pairs and side swapping against the Phase-3 parent with
the same inference mode on both agents.

| Candidate | Mode | Score | US | USSR | Elo delta | Paired 95% CI |
|---|---|---:|---:|---:|---:|---:|
| u50 | raw | 40% | 50% | 30% | -70.44 | 25%-50% |
| u75 | raw | 45% | 50% | 40% | -34.86 | 30%-60% |
| u100 | raw | 35% | 30% | 40% | -107.54 | 20%-50% |
| u25 | D2/32 | 50% | 70% | 30% | 0.00 | 25%-75% |
| u50 | D2/32 | 65% | 80% | 50% | +107.54 | 40%-90% |
| **u75** | **D2/32** | **75%** | **80%** | **70%** | **+190.85** | **60%-90%** |
| u50 | D4/128 | 40% | 50% | 30% | -70.44 | 25%-50% |
| u75 | D4/128 | 25% | 20% | 30% | -190.85 | 10%-40% |

The u75 D2 gate averaged 159.6 searches and 5,107 simulations per game. The u75 D4
gate averaged 153 searches and 19,584 simulations per game, with a 69.84% flip rate.
Its terminal distribution was 12 final scoring, 6 VP threshold, and 2 thermonuclear.

## Frozen Child Diagnostics

All 16 child searches below had zero atomic or node-budget cutoffs.

| Root | D2/32 (two seeds) | D4/128 (two seeds) | Suspicious action result |
|---|---|---|---|
| USSR UK overcontrol | Spain / Ecuador | Austria / South Africa | UK rejected 4/4 |
| US Italy overcontrol | Pakistan / North Korea | Brazil / Italy | Italy rejected 3/4 |
| USSR Mexico realignment | Brazil / Chile | Paraguay / Venezuela | Mexico rejected 4/4 |
| US Venezuela realignment | Angola / Cuba | Angola / Angola | Venezuela rejected 4/4 |

The single Italy D4 recurrence was a near tie: Italy Q=-0.94546 versus the best sampled
Q=-0.94400, below the configured 0.002 Q-override threshold. The counterfactual critic
still assigned negative marginal value to every additional Italy influence point.

## Interpretation

The root-search repair remains valid: D2 no longer degenerates into one-visit ties,
Q-aware alternatives defeat bad prior leaders, information masking remains intact, and
the anomaly roots are corrected in 15/16 child searches.

The Phase-4 update is depth-specific rather than generally compatible. Increasing
updates improves D2 from 50% to 65% to 75%, while raw strength and D4 strength decline.
This is consistent with policy/value-head co-adaptation to the small D4 replay, not a
failure of root widening or action generation.

The next iteration should start from the retained Phase-3 parent and isolate the source
of D4 incompatibility. The first experiment should update the actor only (or use a much
slower/frozen value head), strengthen parent anchoring, and stop at intermediate
checkpoints. It should require raw, D2, and D4 gates before promotion. More updates or
deeper search are not justified by these results.

## Artifacts

- Phase-4 replay: `runs/puct_selfplay/rootfix-phase4-d4-128-16games-w16-gpu45-20260819/`
- Student training: `runs/search_distill/rootfix-phase4-d4-snapshot9-actorvalueheads-u100-gpu4-20260819/`
- Matched gates: `runs/eval/rootfix-phase4-child-sweep-20260819/`
- Child anomaly probes: `runs/diagnostics/rootfix-phase4-child-u75-20260819/`
- Phase 1-2 report: `docs/root_search_mechanics_phase1_phase2_20260819.md`

## Verification

The complete test suite passes: 203 tests.
