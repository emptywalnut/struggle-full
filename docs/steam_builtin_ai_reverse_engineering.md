# Steam built-in AI 1.4.11: behavioral reconstruction

> **Fidelity correction (2026-08-31):** this document records static evidence.
> The runnable code is a **Steam-reconstructed proxy**, not a faithful numerical
> evaluator or behaviorally validated Steam AI. See
> `docs/steam_builtin_ai_fidelity_audit.md`. Earlier language implying compatibility
> or parity is superseded.

## Scope and method

This is a clean-room behavioral reconstruction of the normal AI in the legally installed macOS Steam build 1.4.11/build 167. It is not recovered source code. Static evidence came from the arm64 slice of `TwilightStruggleLib-OSX`, the matching IL2CPP artifacts, and all six supplied Lua files. Addresses below are unslid arm64 virtual addresses. No proprietary binary or card text is redistributed.

Confidence labels are **confirmed** (direct symbol/data flow), **strong** (multiple static observations with one missing type/enum label), and **hypothesis** (useful approximation not established by the binary).

## Policy and call structure

**Confirmed.** Normal play constructs `twilight_ai::CMediumAI`; tutorial difficulty value 3 constructs `CTutorialAI`. `CMediumAI::Evaluate` is a pruned forward search over the native rules state machine:

1. `PrepareEvaluation` derives side-relative scoring fields and estimates the hidden opposing hand.
2. `BuildEvaluateOptionList` partitions legal choices into category buffers and calls category filters.
3. `Evaluate` copies the world, clones/updates the state machine, selects an option, evaluates descendants, and rolls back.
4. Dice and draw states dispatch to dedicated chance evaluators.
5. Current/action-round/turn/game boundaries dispatch to distinct scoring functions.

Key evidence:

| Address | Symbol / observation | Confidence |
|---:|---|---|
| `0x16574` | `PrepareEvaluation`; establishes player/opponent indexes and a `+1/-1` side sign at `0x16624-0x16644` | confirmed |
| `0x16f2c` | `ApplyEstimateOpponentHand` | confirmed |
| `0x18f14` | `BuildEvaluateOptionList` | confirmed |
| `0x1a8f0` | `Evaluate`; depth check at `0x1a9a8`, `CopyWorld` at `0x1ab70`, option build at `0x1ada0`, state-machine update at `0x1b114`, rollback at `0x1b0cc/0x1bb7c`, cloning at `0x1b92c/0x1b980` | confirmed |
| `0x1bf3c` / `0x1c8d0` | random dice / draw evaluation; separate depth-40 checks at `0x1bf94` and `0x1c91c` | confirmed |
| `0x1d23c` / `0x1d104` | background start/check wrappers; UI-thread scheduling layer, not a second policy | confirmed |

The guard is 40 state-machine levels, not 40 complete turns or necessarily 40 player plies.

The complete thread wrapper family is `IsActive 0x1cf60`, `Init 0x1cf74`,
`Shutdown 0x1cfec`, `StopEvaluation 0x1d0f0`, `CheckEvaluationResult 0x1d104`,
`StartEvaluation 0x1d23c`, and `ThreadMain 0x1d418`. Init allocates one global
handler; start locks, records a pending job and wakes the thread (`0x1d2e8-0x1d32c`);
check copies the completed option/result words and clears the pending flag
(`0x1d1c8-0x1d218`); stop calls the evaluator's cancellation vtable entry; shutdown
sets the exit byte and wakes/joins the worker. `ThreadMain` waits when idle and invokes
the evaluator virtual method when work is pending. No scoring or difficulty branch is
introduced by this layer.

## Options, pruning, and ordering

The local static buffers named in the symbol table prove these categories: operations influence; place influence; relocate-from; relocate-to; remove influence; realignment; coup; war; own/neutral/opponent/scoring headlines; own/opponent/revealed/scoring event play; own/opponent discard; statecraft; and other.

Every named filter was inspected:

| Function | Address | Direct evidence and recovered rule |
|---|---:|---|
| place influence | `0x171dc` | saves rollback position, temporarily calls `SetCountryPlayerInfluence` (twice for compared cases), scores/sorts, rolls back |
| remove influence | `0x1769c` | temporary removal through `SetCountryPlayerInfluence`, score/sort, rollback |
| relocate influence | `0x1796c` | same simulate-and-rollback pattern for source/destination candidates |
| build realignment list | `0x17c60` | queries adjustment, calls exact roll calculator, applies expected influence result temporarily, then rolls back |
| realignment filter | `0x17f10` | builds two country lists and sorts the merged candidates |
| coup filter | `0x1806c` | resolves player/country, sends a rules query, calls `CalculateCoupRollResults`, applies expected influence changes, rolls back, sorts |
| war filter | `0x18444` | reads current country influence and prunes/ranks the compact candidate records |
| own/neutral event | `0x1855c` | checks realignment, coup, queried ops, Space Race legality, and event `CanUseEffect`; reads per-card flags |
| opponent event | `0x18af8` | checks realignment/coup/ops and per-card opponent-event ordering/DEFCON flags |
| statecraft | `0x18cd0` | ranks statecraft records and uses `GetRandomInt`, so ties/selection can be stochastic |

**Strong inference.** Influence/coup/realignment filters rank by their resulting heuristic delta rather than fixed country names: they actually mutate the rollback world before sorting. Exact source-level comparators and every packed flag name remain unavailable. The reconstructed proxy consequently implements category-aware stable ordering, Lua timing/DEFCON gates, battleground preference, and a candidate cap, but does not claim bit-for-bit native pruning.

**Confirmed quota.** Place influence uses eight native region buckets, sorts each
8-byte country candidate record, and copies `min(count, 4)` candidates from each
bucket (`0x17464-0x17584`). The proxy implements this four-per-region limit. Other
static buffer sizes are capacities and are not treated as recovered logical quotas.

## Evaluation and the fourteen weights

Normal construction copies 14 floats from `0x80564`; the preceding word at `0x80560` is 14. The vector is:

`[0.741206, 0, 0.229500, 0.375208, 0, 0.2, 0.384150, 0.395302, 0.051567, 0, 0.013033, 0.126330, 0.2, 0.15]`

If the constructor receives no external array it copies the fallback vector from `0x80528`:

`[0.5, 0.1, 1.0, 0.6, 0.2, 0.3, 0.1, 0.2, 0.2, 0.2, 0.2, 0.4, 0.2, 0.15]`

Constructor evidence is `0x1646c-0x1653c`: external `count * 4` bytes are copied when count and pointer are present; otherwise the constant block is copied.

The original field names are stripped. The most defensible slot semantics are
therefore reported as data flow. Unresolved slots are not mapped into runnable
observation features:

| Slot / object offset | Native use in `PrepareEvaluation` / scoring | Runnable use | Confidence |
|---|---|---|---|
| 0 / `+76` | multiplies twice the world VP-like integer at world `+5528` in `ScoreBaseState 0x1a4a8` | only in exact prepared-accumulator API | confirmed use; strong label |
| 1 / `+80` | early-war (`world +5584 <= 3`) baseline/cutoff adjustment at `0x16678-0x1669c` | not mapped | confirmed use, unresolved label |
| 2 / `+84` | copied into one of eight region/card accumulators for held scoring cards at `0x166c0-0x16700` | not mapped | confirmed use, strong scoring-card association |
| 3 / `+88` | copied into region/card accumulators for usable opponent effects at `0x16790-0x168fc` | not mapped | confirmed use, strong event-potential association |
| 4 / `+92` | first term of a per-region paired influence/control difference at `0x16974-0x169f0` | not mapped | confirmed regional term |
| 5 / `+96` | second term of that paired regional difference | not mapped | confirmed regional term |
| 6 / `+100` | first coefficient of another per-region paired difference | not mapped | confirmed regional term |
| 7 / `+104` | second coefficient of that paired regional difference | not mapped | confirmed regional term |
| 8 / `+108` | multiplied by `-5` and a world timing/track integer at `0x16a08-0x16a24` | not mapped | confirmed use, unresolved label |
| 9 / `+112` | multiplied by `+5`, side-indexed, then by a player track at `0x16a28-0x16a80` | not mapped | confirmed use, unresolved label |
| 10 / `+116` | multiplied by signed `5` and a difference of world/player counters at `0x16a84-0x16aac` | not mapped | confirmed use, unresolved label |
| 11 / `+120` | `-10` times count-derived values stored in player evaluation accumulators at `0x16718-0x1673c` | not mapped | confirmed use, unresolved label |
| 12 / `+124` | `-10`, stored in another player evaluation accumulator at `0x16740-0x1674c` | not mapped | confirmed use, unresolved label |
| 13 / `+128` | `+10`, conditioned on player card/effect vector presence at `0x16750-0x16834` | not mapped | confirmed use, unresolved label |

`ScoreBaseState` (`0x1a4a8`) adds seven region accumulators at world offsets `+6168` through `+6984`, two player accumulators at `+7000/+7032`, and writes `[score, -score]`. `ScoreCurrentState` (`0x1a51c`) adds the positive-part difference of two counters and three player accumulators (`player +228/+236/+244`). `ScoreEndActionRoundState` (`0x1a5b0`) has the same visible additions. `ScoreEndTurnState` (`0x1a644`) retains only player `+244` after base scoring.

The end-game utility table at `0x73f10` is `[0, 55000, 60000, 50000, 70000, 100000, 80000, 60000]`. `ScoreEndGameState` (`0x1a684`) indexes it by the native reason at world `+68`, applies `-100 * world[+5588]` at `0x1a6cc-0x1a6dc`, and adds/subtracts the result according to winner. Native enum labels for the eight reasons are unresolved.

## Hidden information

**Confirmed.** `PrepareEvaluation` distinguishes `IsKnownCardInHand` (`0x16b08`) from unknown opponent cards. Unknown non-special cards are removed from the simulated hand and returned to the deck (`0x16c14/0x16c20`). It builds estimates, uses `GetRandomInt` at `0x16ed0`, then `ApplyEstimateOpponentHand` removes estimated cards from the deck and places them in the opponent hand (`0x16f58-0x16f9c`). The estimator can therefore vary by RNG seed. The Python module preserves known cards and uniformly samples remaining slots; native bias from event usability/value is only partially recovered and is not claimed exact.

## Chance nodes

**Confirmed.** `EvaluateRandomDiceResult` and `EvaluateRandomDrawResult` save rollback positions, clone/copy branch state, recursively evaluate, normalize probability mass (`fdiv` sequences including `0x1c130`, `0x1c1ec`, `0x1cac0`), and aggregate. Realignment dispatches through `QueryRealignmentAttemptAdjustment` and `CalculateRealignmentRollResults` (`0x1c62c-0x1c644`). Coup paths use `CalculateCoupRollResults` (`0x17154`). Two-die enumeration yields 36 equiprobable ordered outcomes; card draw branches normalize by eligible draw count. The implementation exposes exact generic die distributions and normalized aggregation, plus explicitly named proxy coup/realignment rules helpers; those proxy helpers do not claim calculator parity.

## Lua policy data

All six Lua files were treated as code/data. `twilight_load.lua` defines load order; `twilight_map.lua` defines country indexes, regions, stability, adjacency and initial influence; `twilight_cards.lua`, `twilight_promos.lua`, and `twilight_turnzero.lua` define cards and AI blocks; `load_database.lua` maps fields to native setters.

The reproducible extractor found 112 annotated cards: 105 headline values, 51 `resolve_opponent_event_first`, 33 `resolve_opponent_event_last`, 12 play-before-AR, 8 play-after-AR, 7 do-not-predict-headline, 6 DEFCON-ops, 3 DEFCON-drop, 2 DEFCON-draw, 2 DEFCON-event, and 1 DEFCON-discard annotations. Common fields are inherited by USA/USSR side blocks.

At `load_database.lua:263-298`, fields map directly to headline quality, opponent-event ordering, five DEFCON contexts, headline prediction, and AR timing setters. One confirmed loader anomaly is preserved as uncertainty: `resolve_opponent_event_last` also calls `SetAI_ResolveOpponentEventFirst` (`load_database.lua:270-272`) with the raw value. It may encode false/last rather than call a separately named setter.

## Unity setup bridge and difficulty

The managed wrapper contains `AppPlayerData.aiDifficultyLevel`, `playerType`, `UI_OfflineCreateGame`, Easy/Hard icon fields, and the `StartGame(IntPtr, Int32, AppPlayerData[], UInt32)` bridge. Matching metadata strings occur at `global-metadata.dat` offsets `0x121537` (`AppPlayerData`), `0x121568` (`aiDifficultyLevel`), `0x128ed5` (`UI_OfflineCreateGame`), `0x129331/0x129343` (Easy/Hard). Wrapper identifiers occur at `Assembly-CSharp.dll` offsets `0xdeacb`, `0x114dcf`, `0x131dc4`, and `0x1082fc`. This verifies the managed setup fields against the matching artifact set; generated wrapper IL contains native pointer registrations rather than original managed bodies.

The authoritative construction branch is native `_StartGame` (`0xa84c`). It reads the per-player difficulty byte at record `+45`, compares it with 3 at `0xaacc-0xaad4`, and calls `CTutorialAI` at `0xaafc` only on equality. Every other value calls `CMediumAI` at `0xab2c`, passing count 14 and the same tuned vector at `0x80564`. `_StartTutorial` repeats the same branch at `0xb040-0xb0a0`; `_ResumeGame` follows the same construction family. Thus Easy and Hard UI choices do not select different normal evaluator classes or weights in this build. Tutorial behavior is a separate scripted `TutorialAISelection` evaluator.

## Uncertainty and implementation correspondence

| Item | Status | Baseline treatment |
|---|---|---|
| class/search/chance/depth architecture | confirmed native; proxy differs | bounded recursive Saito max/min, default depth 2; native evidence limit retained as 40 |
| tuned and fallback numeric weights | confirmed | exact constants |
| all 14 original human-readable names | unresolved | not mapped into runnable features; native data-flow table above |
| exact packed comparator fields and option quotas | partially recovered | stable category heuristics and configurable cap |
| hidden-hand sampling biases | partially recovered | known-card preservation plus seeded uniform sampling |
| terminal reason enum names | unresolved | exact numeric table/timing arithmetic only when the native numeric fields are supplied |
| Lua annotations | confirmed | generated 112-card structured table |
| Easy versus Hard policy distinction | confirmed absent in construction path | one observed native normal configuration; no runnable parity claim |
| exact parity with proprietary rules engine | unresolved by static analysis | Saito bridge remains authoritative for legal actions/transitions |

The runnable implementation is suitable only as a reproducible, evidence-informed
proxy. It is not a genuine Steam-AI baseline, and no binary-identical or aggregate
behavioral agreement has been demonstrated.
