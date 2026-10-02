# PUCT Joint Cycle 1 Report

## Data generation

- Parent: `cycle0-student-balanced2games-100updates-gpu4-20260812/checkpoints/distilled`
- D2/32 complete self-play: 80 games, 87,722 replay examples before D4/D8 augmentation.
- Main 64-game collection: 71,170 examples, 10,203 searched roots, 64/64 complete games, 52.8 games/hour, 74.8 simulations/second.
- D4/128: 16 balanced roots, 16/16 complete, 15 raw-action flips.
- D8/2048: one complete label; one root was rejected fail-closed after a search cutoff.
- Hidden hands were not serialized into actor observations.

## First joint student

The conservative update-200 student achieved:

- Raw vs parent: 55% overall, US 50%, USSR 60% over 20 games.
- D2/32 vs parent D2/32: 35% overall, US 40%, USSR 30% over 20 games.

It was not promoted because cheap-search strength regressed.

## Actor/critic isolation

Mixed evaluators used one frozen checkpoint for priors and another for leaf values. Results combine two disjoint four-pair screens per configuration:

| Candidate components | Games | Overall | US | USSR |
|---|---:|---:|---:|---:|
| update-200 policy + parent value | 16 | 56.25% | 50.0% | 62.5% |
| parent policy + update-200 value | 16 | 31.25% | 12.5% | 50.0% |

The update-200 actor transferred useful search behavior. Its critic harmed D2 search ordering even though its on-distribution complete-root value audit improved:

| Critic | Calibrated MSE | Sign accuracy | Saturated abs(raw)>2 |
|---|---:|---:|---:|
| parent | 0.9903 | 60.49% | 0.43% |
| update-200 | 0.8869 | 63.17% | 0.00% |

Terminal MSE therefore does not predict PUCT usefulness by itself. Local action-value ranking must remain a promotion gate.

## Corrective experiment

Joint actor/value training now supports a frozen-parent value anchor:

`L = L_policy + lambda_anchor L_policy_anchor + lambda_v L_terminal + lambda_q L_searchQ + lambda_va L_parent_value`

The value-anchor sweep used terminal-value weight 0.1, search-Q weight 0.02,
and parent-value anchor weights 1.0 and 3.0. Both remained joint
policy/value updates; the critic was not frozen.

Raw screens selected the 1.0-anchor student over the 3.0-anchor student:

| Student | Raw overall | Raw US | Raw USSR |
|---|---:|---:|---:|
| value anchor 1.0 | 55% | 40% | 70% |
| value anchor 3.0 | 45% | 50% | 40% |

The selected student then failed the complete 20-pair D2/32 promotion gate:

- 17/40, or 42.5% overall
- US 40%, USSR 45%
- Elo delta -52.5, paired-bootstrap score CI95 27.5%-57.5%
- 157.7 searched decisions and 5,047 simulations per game on average
- 19.0% raw/search action-flip rate

The two disjoint 10-pair shards scored 60% and 25%, so the larger matched
sample was necessary. The checkpoint was not promoted.

## Frozen-encoder joint update

The failed full-model student changed the shared state representation even
while its scalar outputs were anchored. A new `actor_value_heads` scope now
updates only:

- action encoder
- US and USSR policy queries and action biases
- value head

It freezes the country, region, card, history, global, graph, and combined
state encoders. Saved-checkpoint comparison against the parent confirmed zero
delta in every excluded tensor for both students.

Two 200-update students isolated search-Q supervision:

| Student | Raw overall | Raw US | Raw USSR | Final value-anchor MSE |
|---|---:|---:|---:|---:|
| head-only, search-Q 0.02 | 55% | 70% | 40% | 0.00141 |
| head-only, no search-Q | 40% | 60% | 20% | 0.00139 |

The Q-weighted head-only student advanced to the D2/32 promotion gate. The
no-Q student did not, because its USSR side failed the minimum side gate.

The Q-weighted `actor_value_heads` student then scored 47.5% over 40 D2/32
games (US 55%, USSR 40%, Elo delta -17.4). This was a substantial recovery
from the full-model student's 42.5%, but still failed promotion.

An auditable composed checkpoint retained the distilled actor/action encoder
and restored the parent's four value-head tensors exactly. Its 20-game D2
screen scored 45% (US 60%, USSR 30%). Parent-critic restoration therefore did
not remove the side regression; actor/action changes also mattered.

## Strict policy/value heads

A stricter `policy_value_heads` scope freezes both the shared state encoder and
shared action encoder. It updates only:

- US and USSR policy queries
- US and USSR action biases
- value head

Saved-checkpoint comparison confirmed exactly 12 changed tensors and zero
delta outside these prefixes. Two raw screens produced:

| Student | Raw overall | Raw US | Raw USSR |
|---|---:|---:|---:|
| equal side weight | 65% | 50% | 80% |
| USSR policy weight 1.5 | 40% | 40% | 40% |

The side-weighted run was rejected. The equal strict-head student advanced to
the D2/32 screen. Its first matched 10-pair gate scored 60% overall (US 70%,
USSR 50%, Elo delta +70.4), but the paired-bootstrap score interval remained
wide at 40%-80%. A disjoint 20-pair extension scored exactly 50% (US 55%,
USSR 45%). Combined over 30 pairs, the candidate scored 53.3% (US 60%, USSR
46.7%, Elo delta +23.2) with paired-bootstrap score CI95 41.7%-65.0%.

This is not statistically supported improvement. The candidate is retained as
an inconclusive diagnostic student and is not promoted. A verified composition
that restored the parent's four value-head tensors then scored only 40% over
20 D2/32 games (US 20%, USSR 60%, Elo delta -70.4). Parent-critic restoration
therefore did not repair the candidate; its distilled policy heads are the
remaining source of the D2 regression.

## Targeted correction replay

Replay metadata exposed why broad distillation was unstable. Across the
87,804 generated examples, search changed only 3.15% of US labels and 2.68%
of USSR labels. Mean Q advantage over the raw action was 0.0052 for US and
0.0042 for USSR. Training all labels consequently spent most policy gradient
on near-identity targets instead of PUCT corrections.

The distiller now supports separate supervision pools:

- actor policy batches may require actual search flips;
- terminal-value batches independently sample all complete root states with
  equal US/USSR counts;
- the broad safe replay remains the raw-policy anchor pool.

This preserves complete-game value learning while concentrating actor updates
on high-regret search disagreements. The replacement strict-head student uses
128 flip labels, 64 independent terminal roots, and 256 anchor states per
update. This student is not promotable until raw, D2, and frozen diagnostics
all pass.

The first targeted update-100 student scored 45% raw against the parent (US
40%, USSR 50%), but D2/32 scored 55% over 60 games (US 56.7%, USSR 53.3%, Elo
+34.9). Its paired-bootstrap score CI95 was 41.7%-68.3%, so the point gain was
reproducible across two shards but not statistically supported. It was not
promoted because raw play regressed and the D2 interval still crossed 50%.

Filtering actor labels to Q advantage at least 0.02 and teacher visit share at
least 20% left 1,293 labels (634 US, 659 USSR). A 100-update strict-head
student still scored 45% raw (US 40%, USSR 50%). Blending each policy target
50/50 between PUCT visits and the replayed parent policy also scored 45% raw:
update 50 was US 30% / USSR 60%, while update 100 was US 40% / USSR 50%.
Neither advanced to the searched gate.

These results reject broad policy drift and label-quality filtering as
sufficient fixes. The next screen interpolates the already useful targeted
student's policy/value-head delta toward the parent, with exact tensor-hash
manifests. This tests whether a smaller parameter-space trust region can retain
the D2 prior improvement while restoring raw behavior.

The interpolation raw screen produced a non-monotonic but reproducible
threshold effect on the common 10 seed pairs:

| Student-head delta | Raw overall | Raw US | Raw USSR |
|---:|---:|---:|---:|
| 10% | 50% | 40% | 60% |
| 25% | 45% | 30% | 60% |
| 50% | 45% | 30% | 60% |
| 75% | 50% | 40% | 60% |
| 100% | 45% | 40% | 50% |

This is consistent with deterministic argmax trajectories crossing different
near-tie boundaries; strength is not expected to vary monotonically under a
small logit interpolation.

On the 1,293 confident flipped replay labels, teacher top-1 accuracy moved
from 0% in the parent to only 0.85% in the full student. Mean teacher
probability moved from 0.276646 to 0.276922. Even this small prior movement was
enough for the full student's D2/32 point estimate to reach 55%, confirming
that PUCT outcomes are sensitive to small prior changes. The 10% and 75%
interpolations therefore advanced to D2 screens.

Replay evaluation was optimized to apply metadata filters before copying
observation tensors and to batch model inference. Per-checkpoint resident
memory fell from roughly 24 GB to roughly 0.8-2.4 GB on this replay set.

PPO-integrated replay training now has the same joint semantics as standalone
distillation: configurable local policy targets, policy/value-head scopes,
terminal-value and search-Q losses, and an independent broad complete-root
value sampler. PUCT-generated behavior remains outside PPO importance ratios.

The first D2/32 interpolation gate then separated the candidates:

| Student-head delta | D2 overall | D2 US | D2 USSR | Elo delta |
|---:|---:|---:|---:|---:|
| 10% | 55% | 50% | 60% | +34.9 |
| 75% | 40% | 30% | 50% | -70.4 |

The 10% checkpoint preserved the 50% raw score and both D2 side gates. Its
20-game paired-bootstrap CI95 remained 30%-80%, so it is a promising but not
yet promotable candidate. A disjoint 20-pair extension was launched. The 75%
checkpoint was rejected. The D2 evaluator averaged 1,164-1,383 seconds per
game; one alpha-10 final-scoring straggler took 3,762 seconds, showing that
resumable per-game scheduling and tail control are the next tournament
throughput priorities.

The disjoint 20-pair alpha-10 extension scored 45% (US 50%, USSR 40%, Elo
-34.9). Combined with the first shard, alpha-10 scored 48.3% over 30 pairs / 60
games (US 50%, USSR 46.7%, Elo -11.6) with paired-bootstrap CI95
35.0%-61.7%. It is therefore rejected: parameter interpolation recovered raw
play, but did not preserve a statistically supported D2 strength gain.

Across the combined alpha-10 evaluation, search averaged 179.7 decisions and
5,749.9 simulations per game with an 18.5% action-flip rate. Terminal reasons
were 33 final scoring, 23 VP threshold, three thermonuclear war, and one
Wargames. Nuclear endings were not the cause of the rejection.

The evaluator now writes an exact checkpoint/search/seed identity into partial
progress and can resume only matching, unique seed/orientation records. It
refuses old identity-free or mismatched progress rather than risking mixed
agent evidence. This protects future long gates from losing completed games.

The iteration driver now forwards side-specific policy-loss weights and uses
the parallel dual-GPU evaluator for raw, D2/32, and D4/128 promotion gates.
Parallel evaluation also writes an atomic `progress.json` after every complete
game. That file is explicitly marked partial and never substitutes for final
`results.json` evidence. Evaluation defaults to one inference server per
distinct checkpoint so sparse paired requests are not fragmented across
redundant queues; self-play generation retains multi-replica support.
