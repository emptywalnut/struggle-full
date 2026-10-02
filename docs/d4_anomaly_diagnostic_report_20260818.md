# D4 Gameplay Anomaly Diagnostic

Date: 2026-08-18

## Scope

This is a frozen-model diagnostic. It uses the accepted-parent checkpoint:

`runs/puct_selfplay/cycle0-student-balanced2games-100updates-gpu4-20260812/checkpoints/distilled`

No training, reward change, model update, or production PUCT behavior change was performed.
All reported values are from the root player's perspective, so higher is better.

The roots were replayed exactly from the complete D4 logs, with assertions on turn,
AR, side, phase, prompt, VP, DEFCON, legal action order, and the logged action index.
Each local root has a portable compressed snapshot plus state and observation digests.
Round-trip restoration reproduced all four observation digests and legal action counts.
The opponent hand tensor was zero at every actor/search root.

## Summary

| Root | Raw actor evidence | Counterfactual value evidence | D2/32 | D4/128 | D6/512 | D8/2048 | Diagnosis |
|---|---|---|---|---|---|---|---|
| USSR UK overcontrol | UK rank 1, P=0.048; repeated UK probabilities rise to 0.077 | First UK point ranks 44/57, dV=-0.0014; all-UK V=0.792 vs all-Turkey V=0.807 | Canada* / Canada* | UK / Canada* | Algeria / Algeria | Sudan, 606 cutoffs | MIXED: ACTOR + SEARCH_COVERAGE + HORIZON |
| US Italy overcontrol | Italy rank 1, P=0.068; repeated Italy rises to 0.120 | First Italy point ranks 46/72, dV=-0.0071; all-Italy V=-0.945 vs North Korea V=-0.927 | Canada* / Canada* | Italy / Italy | Italy / Italy | Italy, 2,331 cutoffs | MIXED: ACTOR + SEARCH_COVERAGE |
| USSR Mexico realignment target | Mexico rank 1, P=0.337 | Exact immediate Q ranks Mexico 9/11; Argentina is +0.0062 better | Mexico / Mexico | Mexico / Mexico | Mexico / Mexico | Mexico, 4,154 cutoffs | MIXED: ACTOR + SEARCH_COVERAGE; value separation is weak |
| US Venezuela realignment target | Venezuela rank 1, P=0.322 | Exact immediate Q ranks Venezuela 5/7; Cuba is +0.0172 better | Venezuela / Venezuela | Venezuela / Venezuela | Venezuela / Venezuela | Cuba, 5,723 cutoffs | MIXED: ACTOR + SEARCH_COVERAGE + HORIZON |

`*` The selected action is not meaningful strength evidence. At D2, UK has 57 legal
actions and Italy has 72, but only 32 simulations. The root force-visits the 32 highest
prior actions once, then `argmax(visits)` resolves the all-one-visit tie by action index,
which selects Canada.

## Root Reconstruction

| Root | Source | Seed | Decision | Exact state |
|---|---|---:|---|---|
| USSR UK | `worker-001/games-3961640.jsonl` | 97000000 | T5 AR4, first NATO influence placement | UK US 5 / USSR 12; 57 legal countries |
| US Italy | `worker-001/games-3961640.jsonl` | 97000000 | T8 AR6, first China influence placement | Italy US 15 / USSR 2; 72 legal countries |
| USSR Mexico | `worker-003/games-3961642.jsonl` | 97000001 | T7 AR4, first Junta realignment target | 11 legal targets; US has about +2 adjacency |
| US Venezuela | `worker-001/games-3961640.jsonl` | 97000000 | T5 AR2, first realignment target | 7 legal targets; USSR has about +2 adjacency |

The UK move in the source game was made by the pre-campaign opponent model. The probe
still uses the requested accepted parent, which independently ranks UK first and can
select UK under D4, so the tendency is not confined to the older source policy.

## 1. USSR UK Overcontrol

The logged hierarchy is strongly self-reinforcing:

`NATO 0.723 -> ops 0.831 -> place 0.957 -> UK 0.048 -> 0.061 -> 0.064 -> 0.077`

The actor's country distribution is broad (normalized entropy 0.973), but UK is rank 1
at every click. The critic does not assign high marginal value to the stack. The first
three UK increments are `-0.0014, +0.0024, +0.0023`; after full macro resolution,
all-UK is 0.0153 below all-Turkey. First-click UK is critic rank 44/57.

D4 is unstable: one seed selects UK and one ends in a visit tie that selects Canada,
while several lower-prior actions have higher Q. K=64 does not repair this: both runs
select UK, with one again decided by a visit tie. Belief sampling changes estimates but
does not explain the strategic error.

The depth-budget cross-ablation separates the causes:

- D4/512: UK in both seeds.
- D6/128: UK in one seed and Saharan States in one; the UK result is another visit tie.
- D6/512: Algeria in both seeds, with no cutoff.

Depth exposes the downstream penalty, while enough visits are required to turn that Q
signal into a different visit winner. This is a mixed actor, coverage, and horizon issue.

## 2. US Italy Overcontrol

The logged hierarchy is:

`China 0.504 -> ops 0.788 -> place 0.470 -> Italy 0.068 -> 0.077 -> 0.082 -> 0.120`

The actor increasingly favors Italy as the allocation continues. The critic clearly
disagrees: the four marginal deltas are `-0.0071, -0.00035, -0.00016, -0.00041`.
All-Italy is 0.0179 below a complete North Korea allocation. First-click Italy is critic
rank 46/72.

D4, D6, and D8 all select Italy. This is not because alternatives are absent. At D6,
Brazil has Q=-0.9372 with 16 visits while Italy has Q=-0.9506 with 36 visits. At D8,
North Korea again has a better Q but far fewer visits. D4/512 and D6/128 also select
Italy in both seeds. D4 and deeper searches cross the T8-to-T9 boundary, so this is not
fixed by seeing the next Turn.

This is the clearest actor-prior plus finite-PUCT coverage failure. The value function
understands diminishing returns, although its near-certain-loss values are compressed.

## 3. USSR Mexico Realignment

At the Junta event root, raw policy selects event with P=0.573. Primary D4/D6/D8 instead
select Ops, so card-level search can avoid the line. Once the event line is entered:

- `conduct` has P=0.929 versus `skip` P=0.071.
- D2, D4, D6, and D8 all select conduct.
- In six of seven runs, skip has the better Q; D8 still gives conduct 1,873 visits
  versus 175 for skip. D8 has 3,002 node cutoffs.

At the exact target root, Mexico is actor rank 1 with P=0.337. Exact dice enumeration
ranks Argentina first at Q=-0.2257 and Mexico ninth at Q=-0.2320. No dice outcome is
sampled; all d6 pairs are enumerated and equivalent successor states are collapsed.

All D2/D4/D6 target runs select Mexico. At D6, Mexico receives 184 visits; better-Q
Brazil receives 68. D8 also selects Mexico, but has 4,154 node cutoffs. Deeper horizon
therefore does not provide a clean correction. The main cause is actor prior plus visit
allocation. Belief affects Q ordering at depth but never changes the selected action.

## 4. US Venezuela Realignment

This hierarchy differs from Mexico. At the parent Ops-type root, the accepted parent
already prefers `place` with P=0.922; the logged `realign` is rank 2 at P=0.078. D2,
D4, D6, and D8 all select place at that root. Thus the logged parent choice is not a
stable current-model failure under the fixed probe seeds.

If the line reaches target selection, however, Venezuela is rank 1 with P=0.322. Exact
enumeration ranks Cuba first at Q=-0.4732 and Venezuela fifth at Q=-0.4904. D4 gives
Venezuela 40 visits versus 8 for better-Q Cuba; D6 remains on Venezuela in both seeds.
D8 finally flips to Cuba, but only after reaching the node cap and recording 5,723
cutoff evaluations. The D8 direction supports a horizon/coverage correction, but that
tree is not cutoff-free and should not be used as a teacher label.

## Search Mechanics

- Configuration: K=32 legal-consistent root particles, `cpuct=1.5`, exact small chance
  nodes, child progressive widening, 1,024 atomic-step cap, 8,192 expanded-node cap.
- Roots are fully active and force unvisited actions. Important actions enter in prior
  order: UK/Italy/Mexico/Venezuela at simulation 1; Algeria at 18; Argentina at 4;
  Cuba at 7.
- D4 expands every root action in all four probes. The failure is not ACTION_GEN.
- For large allocation roots, D2 cannot cover the root and its visit-tie action is an
  implementation artifact.
- D8 is cutoff-free only in some small-card roots. The local D8 cutoff counts are UK
  606, Italy 2,331, Mexico conduct 3,002, Mexico target 4,154, and Venezuela target
  5,723. These results are diagnostic, not fail-closed teacher data.
- Representative wall times: influence D4 21-25 s, influence D6 113-195 s, target D4
  157-177 s, target D6 872-885 s, and target D8 3,287-4,174 s.

## Answers to the Critical Questions

1. **Diminishing marginal influence value:** Yes in the critic, strongly for Italy and
   weakly but directionally for UK. The actor has not internalized it.
2. **Overcontrol cause:** Primarily actor prior plus root visit allocation/coverage, not
   a critic that believes the stack is valuable.
3. **D4 behavior:** It reproduces Italy and both target-selection errors; UK is unstable;
   it corrects the parent Venezuela Ops-type choice and the primary Junta event choice.
4. **Does D6/D8 fix it:** D6 reliably fixes UK only with 512 visits. It does not fix Italy
   or either realignment target. D8 flips Venezuela but is heavily cutoff-limited; it
   does not fix Italy or Mexico.
5. **Realignment cause:** Exact expected values identify better targets. The error is
   actor prior plus insufficient Q-driven visit reallocation, not dice handling or
   missing candidates.
6. **Side symmetry:** Yes. The overcontrol and bad-target actor pathology exists for
   both US and USSR.
7. **Next priority:** Fix PUCT root coverage/tie behavior and improve hierarchical actor
   targets first; continue broader policy/value training second. Broadly increasing
   depth is too expensive and does not reliably fix these roots. Belief modeling is not
   the current priority.

## Recommended Next Changes

These are recommendations only; none were applied in this diagnostic.

1. Make root expansion budget-aware. Do not force more one-visit root actions than the
   simulation budget can compare, and do not resolve equal-visit roots by action index.
2. Add root progressive widening or policy/Q-guided candidate admission for 57-72 way
   allocation roots. Revisit `cpuct` or prior calibration at those roots.
3. Distill hierarchical search/counterfactual targets specifically at allocation and
   realignment-target levels. Do not punish the parent card when its downstream choice
   is the error.
4. Preserve the critic and add low-weight marginal/action-ranking supervision rather
   than redesigning it. Improve value discrimination in saturated win/loss states.
5. Use D6 selectively after root mechanics improve. Do not treat the cutoff-heavy D8
   rows as teacher targets.

## Artifacts

- Probe harness: `scripts/probe_d4_anomalies.py`
- Tests: `tests/test_probe_d4_anomalies.py`, `tests/test_selective_puct.py`
- Full JSON/search telemetry: `runs/diagnostics/d4-anomaly-probes-accepted-parent-20260818/`
- Source D4 logs: `runs/eval/best-d4-vs-d4-4games-gpu45-20260818/game_logs/`
- Snapshot validation: `runs/diagnostics/d4-anomaly-probes-accepted-parent-20260818/validation.json`

Verification: `26 passed` in the focused probe and selective-PUCT test suites.

The validation artifact is reproducibly generated by:

```bash
python scripts/validate_d4_anomaly_diagnostic.py
```

It independently restores every canonical decision snapshot through the Saito bridge,
rechecks observation and snapshot digests, compares legal actions with the fixture,
verifies opponent-hand masking, checks the source-log action, requires the complete
D2/D4/D6/D8 matrix and its N/Q/P/progressive-widening/chance/cutoff/timing telemetry,
and records SHA-256 hashes for the canonical and supporting result files.
