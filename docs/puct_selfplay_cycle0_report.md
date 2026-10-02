# PUCT Self-Play Cycle 0 Report

Date: 2026-08-12

## Objective

Validate one complete search-driven policy-improvement cycle:

`NN0 -> D2/32 selective-PUCT self-play -> policy/value distillation -> NN1 -> paired evaluation`

This cycle is an integration and point-estimate experiment. Its tournament sample is too small for statistical promotion.

## Model identities

- Parent (`NN0`): `runs/distill/selective-puct-highq003-actionhead-anchor1-100updates-20260811/checkpoints/distilled`
- Student (`NN1`): `runs/puct_selfplay/cycle0-student-balanced2games-100updates-gpu4-20260812/checkpoints/distilled`

## Search-driven collection

Two complete games supplied the 903 examples used for training:

- `runs/puct_selfplay/cycle0-d2-32-games2-gpu4-20260812/search_replay`
  - 582 committed examples from its complete, valid first game.
  - Its second game completed, but was discarded by the older whole-game partial-tree guard. This exposed and motivated the per-label partial-tree rejection fix.
- `runs/puct_selfplay/cycle0-d2-32-uswin-seed86081201-gpu4-20260812/search_replay`
  - 321 committed examples.
  - 121 environment decisions, 34 searched decisions.
  - US and USSR each searched 17 decisions.
  - 14.7% search action-flip rate.
  - 38.2% behavior/teacher disagreement from visit-policy exploration.
  - 3.27 seconds/search and exactly 32 simulations/search on average.

Search behavior used the same parent checkpoint for both sides. Search-generated actions were not treated as PPO on-policy actions.

## Distillation

- 903 discovered examples; 902 passed the quality filter.
- 100 actor/value updates, side-balanced minibatches.
- Loss: visit-policy CE + terminal-value MSE + expected-search-Q MSE + parent-policy anchor.
- Weights: policy `1.0`, terminal value `0.1`, search Q `0.05`, anchor `0.2`.
- Final minibatch teacher accuracy: 90.6%.
- Final minibatch US/USSR sample-weight split: 50% / 50%.

Only true root examples receive final-outcome value targets. Hierarchical counterfactual labels remain policy/Q supervision and do not inherit a false trajectory return.

## Paired evaluation

### Raw NN1 versus raw NN0

- 4 matched seed pairs, 8 side-swapped games.
- NN1 score: 6/8 = 75%.
- As US: 3/4 = 75%.
- As USSR: 3/4 = 75%.
- Elo point estimate: +190.85.
- Paired bootstrap 95% score interval: [0.25, 1.00].
- Average wall time: 10.0 seconds/game.

### NN1 + D2/32 versus NN0 + D2/32

- 2 matched seed pairs, 4 side-swapped games.
- NN1 score: 3/4 = 75%.
- As US: 1/2 = 50%.
- As USSR: 2/2 = 100%.
- Elo point estimate: +190.85.
- Paired bootstrap 95% score interval: [0.50, 1.00].
- Average searched fraction: 29.2% of candidate decisions.
- Average searches/game: 126.75.
- Average simulations/game: 4,056.
- Search flip rate: 15.4%.
- Average search time/game: 574.9 seconds.
- Average wall time/game: 664.6 seconds.

The point estimates satisfy both desired inequalities (`NN1 > NN0` and `NN1+D2/32 > NN0+D2/32`), but the confidence intervals do not establish a statistically supported gain.

## Throughput findings

Exact chance expansion dominates several roots. One Che coup-target root took about 39 seconds, and one broad Iran coup root reached the 512-node cap in about 37 seconds. Search failed closed when the cap was hit.

The planner now limits allocation-triggered searches to one per side/action round. This prevents repeated full searches on each country click after a macro-allocation search fails closed. The completed tournament predates that cap and therefore measures the slower implementation.

## Promotion decision

`NN1` is **not promoted** yet.

Reasons:

- Only four raw seed pairs and two search seed pairs were evaluated.
- Confidence intervals remain wide.
- The full frozen tactical diagnostic suite was not rerun for this student.
- D2/32 whole-game cost is still too high for an economical statistically powered tournament.

Next gate: rerun at least 10 matched seed pairs per configuration after branch-width/chance batching optimization, then run the frozen diagnostic suite before promotion.

## Verification

- Repository tests: 155 passed, 2 skipped.
- Complete-game replay transactions reject truncated games.
- Partial/cutoff search labels are rejected without discarding valid labels from the same complete game.
- Sampled hidden information is never inserted into actor observations.
- Exact small chance outcomes and player-perspective backup remain covered by regression tests.
