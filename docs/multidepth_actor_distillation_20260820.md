# Multi-Depth Actor Distillation Report

Date: 2026-08-20

## Decision

No child checkpoint is promoted. The retained general parent remains:

`runs/search_distill/root-mechanics-actor-correction-actionhead-u100-gpu4-20260819/checkpoints/updates-000075`

The u25 child is retained only as a D2-specialized experimental branch. It improved D2/32 in the matched gate, but failed D4/128 decisively. u50 and u75 did not beat the parent in the completed raw and D2 gates.

## Frozen Training Scope

Training started from the required Phase-3 u75 parent and used `parameter_scope=action_head`.

An exact tensor comparison against the parent found 12 changed tensors at u25, u50, and u75:

- `action_encoder.{0,2}.{weight,bias}`
- `policy_query_{us,ussr}.{weight,bias}`
- `action_bias_{us,ussr}.{weight,bias}`

No global/country/card/history/GNN encoder tensor changed. No value-head tensor changed. Representation and critic weights are bit-identical to the parent at every checkpoint.

Training configuration:

- updates: 75; checkpoints: u25/u50/u75
- learning rate: `2e-6`; batch size: 64
- search-policy loss: 1.0
- pairwise ranking loss: 0.5, up to 16 pairs/example
- parent-policy anchor: 1.0
- value and search-Q regression losses: 0
- balanced US/USSR sampling; complete-game roots only

## Multi-Depth Teacher

The source was 256 balanced roots sampled from complete Phase-4 D4 replay. All roots were rescored with repaired D2/32 search. D6/512 was used only on 32 high-conflict roots; 31 completed cutoff-free and one was excluded fail-closed.

Final consensus outcome:

| Outcome | Roots |
|---|---:|
| Accepted D2/D4 consensus | 118 |
| Accepted after D6 adjudication | 8 |
| D2/D4 disagreement unresolved without D6 | 95 |
| Still unresolved after D6 | 23 |
| Low consensus Q gap | 8 |
| Missing depth evidence | 4 |

D2 and D4 disagreed on 126 of 244 roots with usable, non-negligible shared evidence (51.6%). D6 resolved only 8 of the 31 selected conflicts: seven in favor of D4 and one in favor of D2. This is direct evidence that depth targets differ structurally; they are not merely noisy versions of one ordering.

The final teacher contains 126 roots: 61 US and 65 USSR. It stores legal actions plus P/Q/N for every available depth and uses Q/rank support rather than blindly averaging contradictory visit distributions.

## Search Compression

On 83 quality-filtered consensus examples, the actor's mean probability assigned to the consensus action changed as follows:

| Checkpoint | Mean probability | Q-weighted probability | Top-1 agreement |
|---|---:|---:|---:|
| Phase-3 parent | 6.72% | 6.22% | 0.00% |
| u25 | 6.82% | 6.34% | 1.20% |
| u50 | 6.92% | 6.47% | 1.20% |
| u75 | 7.03% | 6.59% | 1.20% |

The target probability moved in the intended direction, but only weakly. The final minibatch had 95.3% raw/teacher action flips, mean sample weight 8.77, policy CE 3.31, pairwise loss 0.913, and gradient norm 8.84. The selected replay therefore applies strong correction pressure to a narrow distribution despite the parent anchor.

## Mandatory Triple Gate

All completed gates use 20 matched, side-swapped games against the Phase-3 parent with seed base 98900000. Elo and confidence intervals are relative to that parent under the same inference mode.

| Candidate | Mode | Score | US | USSR | Elo delta | Elo 95% CI | Status |
|---|---|---:|---:|---:|---:|---:|---|
| u25 | raw | 55% | 30% | 80% | +34.9 | [-107.5, +190.8] | completed |
| u25 | D2/32 | **65%** | 70% | 60% | **+107.5** | [-107.5, +381.7] | completed; specialized gain |
| u25 | D4/128 | **25%** | 20% | 30% | **-190.8** | **[-381.7, -70.4]** | completed; decisive failure |
| u50 | raw | 40% | 20% | 60% | -70.4 | [-190.8, 0.0] | completed |
| u50 | D2/32 | 40% | 50% | 30% | -70.4 | [-190.8, 0.0] | completed |
| u75 | raw | 50% | 30% | 70% | 0.0 | [-147.2, +147.2] | completed |
| u75 | D2/32 | 45% | 50% | 40% | -34.9 | [-190.8, +107.5] | completed |

The u25 D4 gate used 17,568 simulations and 137.25 searches per game, with a 68.9% action-flip rate. Its paired bootstrap score interval was [10%, 40%], also entirely below parity.

u50 and u75 D4 were stopped after four durable games each because both checkpoints were already dominated by their completed raw and D2 gates. Their 4-game scores (75% and 100%) are preliminary noise and are not used for selection.

## Frozen Anomaly Regression

All D2/D4/D6 searches used K=32 information-safe particles and repaired root mechanics. Every listed search completed with zero atomic-horizon and node-budget cutoffs.

| Root | Parent suspicious P | u25 suspicious P/rank | D2 choices | D4 choices | D6 choice | Result |
|---|---:|---:|---|---|---|---|
| USSR UK overcontrol | 4.82% | 3.78%, rank 1 | South Africa / Uruguay | Saharan States / Algeria | Zimbabwe | bad UK placement suppressed at every depth |
| US Italy overcontrol | 6.81% | 6.06%, rank 1 | Pakistan / Pakistan | Libya / Brazil | India | bad Italy placement suppressed at every depth |
| USSR Mexico operation | conduct 92.92% | conduct 92.83%, rank 1 | conduct / conduct | skip / skip | skip | horizon correction remains intact |
| US Venezuela operation | realign 7.82% | realign 7.76%, rank 2 | place / place | place / place | place | bad operation suppressed at every depth |

Target-level probes expose the remaining actor problem:

- Mexico is still the raw actor's rank-1 realignment target at 34.25%. D2 chooses Chile/Brazil, D4 Paraguay/Colombia, and D6 Paraguay.
- Venezuela is still the raw actor's rank-1 realignment target at 31.72%. D2 chooses Cuba/Colombia, D4 Angola/Angola, and D6 Cuba.

Thus the repaired search mechanics and information masking pass regression, but the actor has only weakly internalized the corrected target rankings. Local search remains responsible for most corrections.

## Diagnosis

This experiment rules out actor/value co-adaptation as the cause: the critic and all shared representation tensors were frozen. The observed `D2 up, D4 down` behavior is actor-prior/search-regime specialization.

Three factors contribute:

1. D2 and D4 action orderings conflict on 51.6% of usable roots.
2. D6 usually does not provide a confident consensus: 23 of 31 adjudicated conflicts remained unresolved.
3. The accepted training distribution is unusually corrective (about 95% action flips and high Q-derived sample weights), so modest parent anchoring does not preserve broad search compatibility.

The loss is not simply under-training: u50/u75 increase consensus-action probability monotonically but lose the u25 D2 gain and do not improve raw strength. More SGD on the same replay is therefore not justified.

## Next Experiment

Do not unfreeze or jointly retrain the value head yet. The next controlled actor experiment should:

- use substantially more diverse complete-game roots;
- train mainly on roots where D2 and D4 rank agreement is high;
- retain separate calibrated D2 and D4 soft targets instead of collapsing them into one sharp target;
- cap Q/sample weights and reduce pairwise pressure;
- add stronger KL anchoring on broad, unselected parent states;
- sweep earlier checkpoints such as u10/u20 before u25;
- keep the same raw/D2/D4 promotion gate.

No production search setting was changed by this experiment.

## Artifacts

- Teacher evidence: `runs/search_distill/multidepth-actor-phase3u75-20260820/consensus-final-v2/multidepth_evidence.jsonl`
- Teacher replay: `runs/search_distill/multidepth-actor-phase3u75-20260820/consensus-final-v2/search_replay/manifest.json`
- Child checkpoints: `runs/search_distill/multidepth-consensus-actionhead-u75-gpu4-20260820/checkpoints`
- Triple gates: `runs/eval/multidepth-consensus-triple-gate-20260820`
- Anomaly probes: `runs/diagnostics/multidepth-u25-anomaly-regression-20260820`

Focused regression suite: 55 tests passed.
