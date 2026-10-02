# Steam built-in AI fidelity audit

Date: 2026-08-31  
Reference: macOS Steam Twilight Struggle 1.4.11/build 167  
Runnable subject: **Steam-reconstructed proxy**

## Verdict

Genuine Steam-AI behavioral parity is **not demonstrated**. Genuine Steam-AI
Elo is **not currently identifiable** from this repository. It is verified that
a headless macOS game containing player type 2 can initialize and run. No
`CMediumAI` option or selected action has been observed. The next blockers are
AI-action capture with verified actor provenance, lossless native state/action
mapping, and transition parity. The runnable Python policy remains only a proxy.

Labels in this audit mean:

- **CONFIRMED**: direct symbol, assembly, constant, or Lua-loader data flow.
- **APPROXIMATION**: independently written behavior inspired by evidence, without
  demonstrated numerical or action parity.
- **UNRESOLVED**: current static inputs do not establish the needed behavior.

## Claim-by-claim audit

| Claim | Label | Evidence / correction |
|---|---|---|
| Normal native AI is `CMediumAI` | CONFIRMED | constructor `0x1646c`; `_StartGame 0xab2c` |
| A headless game containing player type 2 can initialize/run without Unity | CONFIRMED | checksum-pinned macOS ctypes harness initialized using the exact `StreamingAssets/Lua` path, started one local player plus player type 2, and shut down successfully; bounded `--max-iterations 1000 --max-prompts 6` audit observed 104 events |
| The smoke captured `CMediumAI` options or selected actions | CONFIRMED false | all six callbacks were `player_id=0` human USSR setup prompts; zero `CMediumAI` options and zero AI selections were observed |
| The default-run count of 10,083 events is AI-decision evidence | CONFIRMED false | it was the same six human prompts followed by a spun/stalled loop; the accumulated 10,083 count has no evidentiary value |
| `GetGamePlayerAIState` supplies a selected action | CONFIRMED false | it reports thinking/liveness and is not accepted as selection provenance |
| `StartGame` player layout | CONFIRMED | `AppPlayerData` stride is `0x34`; `playerType` is byte `+0xc`, `aiDifficulty` byte `+0xd`; player type 2 constructs `CMediumAI` |
| Player type 2 is the native AI-evaluator construction path | CONFIRMED | both slices: arm64 `0xaa6c-0xaa78`; x86_64 `0x8359-0x835f`; other types pass no evaluator |
| Raw difficulty 2 uniquely denotes a Medium policy | CONFIRMED false | callsites distinguish only difficulty 3 (`CTutorialAI`) from every other value (`CMediumAI`); arm64 `0xaacc-0xab2c`, x86_64 `0x83c6-0x842e` |
| `GameParameters` ABI size | CONFIRMED | 16-byte ctypes structure in the verified harness |
| `UpdateGame` is a `uint64[]` ABI | CONFIRMED false | argument 1 is byte capacity; output is a heterogeneous tagged byte stream and return is logical record count (`0xa3a0`; x86_64 `0x7be7`) |
| An `UpdateGame` event proves a normal AI selection | UNRESOLVED/insufficient | no producer is confirmed to emit actor/index/selection-ID/hint together; type 36 lacks the option tuple and type 48 is tutorial-only without actor/index |
| `OfflineOptionSelectFunc` is the CMediumAI choice boundary | CONFIRMED false | it receives the exported host/network selection and stages a move; CMediumAI returns directly through `CGameStateOptions::UpdateState` to `CGameStateOptions::SelectOption` |
| Native CMediumAI selection boundary | CONFIRMED | `CGameStateOptions::SelectOption` arm64 `0x11d98`, x86_64 `0x1090e`; its caller bounds-checks the evaluator result at arm64 `0x11d2c-0x11d4c`, x86_64 `0x108e9-0x108ff` |
| Native AI option-list ABI | CONFIRMED | `CMediumAI::Evaluate 0x1a8f0` (x86_64 `0x19a0c`); actor at evaluator `+8`; count/storage at `CGameStateOptions +88/+96`; 32-byte entries |
| Game-option callback ABI | CONFIRMED | packed 64-byte record: `int32 optionIndex@0`, `uint16 selectionID@4`, `uint16 selectionHint@6`, `uint8 isHidden@8`, inline UTF-8 `optionText@9..63` |
| A game containing an AI proves behavioral parity | CONFIRMED false | host-game feasibility is not AI-action capture; genuine comparison requires verified AI selection provenance, lossless state/action mapping, and transition parity |
| Tutorial difficulty 3 selects `CTutorialAI` | CONFIRMED | byte compare `0xaacc-0xaad4`, constructor call `0xaafc` |
| Easy and Hard select distinct normal policies | CONFIRMED false | every non-3 value follows the same `CMediumAI` call with count 14 and pointer `0x80564` |
| Tuned/fallback vectors contain 14 constants | CONFIRMED | tuned count/value block `0x80560/0x80564`; fallback copied at constructor `0x1651c-0x16538` from `0x80528` |
| All 14 constants map directly to raw VP/region/space/milops/hand features | CONFIRMED false | `PrepareEvaluation 0x16574` uses them to construct region and player accumulators before `ScoreBaseState`; the prior Python dot product was invented and has been removed |
| Human-readable semantics and exact raw-state formula for all slots are known | UNRESOLVED | several field types/enums and weight-to-accumulator paths remain unnamed |
| Exact prepared `ScoreBaseState` formula is known | CONFIRMED | `0x1a4a8-0x1a518`: weighted doubled VP-like counter + seven prepared region accumulators, perspective sign, two player accumulators, symmetric score pair |
| Runnable Saito leaf evaluator is numerically native | APPROXIMATION | it is now a separately named five-feature proxy with independent coefficients; native constants are not used |
| Terminal utility table is known | CONFIRMED | eight floats at `0x73f10`: `[0,55000,60000,50000,70000,100000,80000,60000]` |
| Terminal timing adjustment is known | CONFIRMED | `ScoreEndGameState 0x1a6cc-0x1a6dc` applies `-100 * world[+5588]`; enum label and exact public observation mapping remain unresolved |
| Search copies/clones state and rolls back | CONFIRMED | `CopyWorld 0x1ab70`, state clone `0x1b92c/0x1b980`, rollback `0x1b0cc/0x1bb7c` |
| Native cutoff is 40 state-machine levels | CONFIRMED | checks at `0x1a9a8`, `0x1bf94`, `0x1c91c` |
| Runnable games execute recovered depth-40 search | CONFIRMED false | live runner is connected but defaults to bounded depth 2 for practical Saito execution; leaf values and filters are proxy behavior |
| Live recursive runner is connected to actual rules transitions | CONFIRMED for proxy | `BridgeProxySearchRunner` recursively calls Saito snapshot/restore/step with max/min actor handling; tested with a branching bridge and live smoke |
| Native chance handling branches dice/draw results | CONFIRMED | `EvaluateRandomDiceResult 0x1bf3c`, `EvaluateRandomDrawResult 0x1c8d0`, normalization divisions, coup/realignment calculators |
| Live runner reproduces native chance behavior | APPROXIMATION | it follows one restored Saito RNG sequence; it neither enumerates every native chance branch nor matches native RNG calls |
| Option categories are recovered | CONFIRMED | named static buffers for influence, relocation, coup, realignment, war, headline classes, event classes, discard, statecraft, other |
| Influence/coup/realignment filters simulate score deltas | CONFIRMED | temporary influence writes, sorting, and rollback in `0x171dc-0x18444` |
| Place influence retains at most four per region bucket | CONFIRMED | eight-bucket loop and `min(count,4)` at `0x17464-0x17584`; now implemented |
| All comparator fields, quotas, fallback and tie rules are known | UNRESOLVED | packed records and multiple branch conditions remain unnamed; buffer capacities do not prove every logical quota |
| Static category capacities | CONFIRMED | 24-byte category record indexing in `BuildEvaluateOptionList`; 0x900 buffers imply 96 entries, 0xf0 card/headline buffers imply 10, statecraft 0xc0 implies 8 |
| Lua policy annotation extraction | CONFIRMED | `load_database.lua:263-298`; generated table contains 112 annotated cards and reproduces exactly |
| Uniform hidden-hand sampling matches Steam | APPROXIMATION | known-card preservation is evidence-backed; uniform sampling is not |
| Hidden-hand architecture | CONFIRMED partial | known-card check `0x16b08`, unknown return to deck `0x16c14/0x16c20`, `GetRandomInt 0x16ed0`, apply estimate `0x16f2c-0x16fac` |
| Hidden-hand bias and RNG call sequence | UNRESOLVED | requires runtime capture or fuller decompilation of selection loops and RNG state |
| Saito `optional_us_plus_2` has Steam rules/state-machine parity | UNRESOLVED | no transition-by-transition differential suite or complete option mapping exists |
| Unit tests establish Steam behavioral agreement | CONFIRMED false | tests establish implementation arithmetic/control flow and evidence gates only; the macOS smoke proves invocation, not a losslessly mapped differential decision corpus |
| Prior 8-step smoke demonstrated expectimax fidelity | CONFIRMED false | it was one-ply Saito lookahead and has been superseded by a qualified recursive proxy smoke |
| The six-prompt stall was caused solely by no AI thread | CONFIRMED false | the primary confirmed host defect is an uncommitted temporary move buffer; mode 1 also skipped `AIHandlerThread_Init`, but `CMediumAI` has a synchronous fallback |
| The corrected mode-2 host advances beyond setup | CONFIRMED | local bounded macOS run committed six staged human setup choices and reached `Select a Card to Headline`; 123 logical events/23 nonempty updates; zero AI selections captured |
| LLDB action capture has run locally | CONFIRMED false | disabled macOS Developer Mode prevents even an LLDB `/usr/bin/true` child launch; security setting unchanged; accepted actions remain zero |

## Corrected implementation boundary

`NativePreparedAccumulators` and `score_native_base_pair` implement only the
confirmed post-preparation base-score data flow. They require already derived
native values and therefore cannot score ordinary Saito observations.

`evaluate_proxy_leaf`, Lua-based ordering, uniform hidden-hand estimation, and
the bounded Saito runner are **APPROXIMATION** components. The native 14 constants
remain available as research evidence/configuration data but are not used as
coefficients for invented observation features.

The runnable name is consistently **Steam-reconstructed proxy**. The source
directory retains `steam_builtin` only as provenance for the reverse-engineering
task, not as a parity claim.

## Fidelity gates

The machine-readable gate state is
`artifacts/steam_ai_fidelity_validation.json`.

| Gate | Current state | Pass condition |
|---|---|---|
| native host-game invocation | HOST FEASIBILITY ONLY | checksum-pinned smoke proves a game containing type 2 can run; it does not capture an AI option or selection |
| evaluator mapping | UNRESOLVED | complete raw native state → prepared accumulator formula for all 14 constants, verified against native accumulator/value dumps |
| candidate filters/quotas | PARTIAL | recover all comparator fields, retained counts, fallbacks, exclusions and tie/random rules; verify ordered candidate lists |
| hidden hand/RNG | UNRESOLVED | reproduce estimated hand and random call stream from captured complete hidden state/RNG |
| recursive runner | PASS_FOR_PROXY_ONLY | genuine pass additionally requires native-equivalent depth/cutoff/chance behavior through a parity rules engine |
| rules parity | UNRESOLVED | transition and legal-option equivalence across a stratified panel, including nested event prompts |
| differential actions | NOT RUN | genuine native-oracle panels with lossless state/action mappings, transition parity and repeated native choices |

## Differential trace harness

Schema: `artifacts/steam_ai_trace_schema.json`  
Harness: `scripts/compare_steam_ai_traces.py`

Native oracle smoke harness: `tools/native_oracle_smoke_macos.py`  
Harness SHA-256: `5e257fedca3b392d981fc46bd604341fcc8fc3f3f8f2f40241940628e3ae6371`  
Machine-readable ABI/gate specification: `artifacts/steam_native_oracle_spec.json`

Experimental action host: `tools/native_oracle_capture_macos.py`  
Experimental LLDB observer: `tools/native_oracle_capture_lldb.py`  
Capture validator: `tools/validate_native_oracle_capture.py`  
Capture schema: `artifacts/steam_native_action_capture_schema.json`  
Static evidence and local commands: `docs/native_cmediumai_action_capture.md`

The harness is macOS-only and was deliberately not executed on Linux. It uses
`Path.home()` and contains no personal username. Its default library path is the
installed app bundle and `Initialize` receives the exact
`Resources/Data/StreamingAssets/Lua` directory. Export availability includes
`StartGame`, `UpdateGame`, `SetGameOptionsListener`, `SelectGameOption`, and
`GetGamePlayerAIState`.

The bounded audit used player 0 local and player 1 type 2 `CMediumAI`, with
`--max-iterations 1000 --max-prompts 6`. It observed exactly six callbacks, all
`player_id=0` human USSR setup prompts, and 104 engine events. `SelectGameOption`
advanced those six human setup prompts. No `CMediumAI` option or selected action
was observed. The default 10,000/8 run produced the same six prompts and 10,083
events only because the loop spun/stalled; that count has no evidentiary value.

Static analysis now identifies the host boundary. Normal human choices are
staged at AppWorldData `+380`; the smoke never called exported
`CommitTemporaryMoveBuffer`, so its six setup choices never crossed the main
move-buffer boundary. It also used initialization mode 1, while the AI handler
is constructed only for mode 2 or above. Because `CMediumAI` has a synchronous
fallback, the missing thread alone is not claimed as the permanent-stall cause.
The experimental bounded host fixes both issues and accounts for
`CDelayAIPlayer` wall-clock delay. That host fix is now locally verified on
macOS: mode 2, one human plus one AI, six automatic human setup choices, and
`CommitTemporaryMoveBuffer` advanced to the human `Select a Card to Headline`
callback. The report recorded `temporary_move_buffer_committed=true`, 123
logical events, 23 nonempty updates, and a 10-second wall-bound exit. This is
host-fix evidence only; zero AI selections were captured.

The revised LLDB observer snapshots actor and ordered options at genuine
`CMediumAI::Evaluate`, then observes the selected index at
`CGameStateOptions::SelectOption`. It accepts a selection only when the exact
thread, actor, state-machine pointer, options pointer, prompt and ordered-option
fingerprint agree. It rejects selections made while any `CMediumAI::Evaluate`
frame is still active, preventing simulated search transitions from satisfying
the live-action gate, and consumes each matched evaluation once. Capture-session
identity, contiguous record sequence, observer/host checksums and a matching host
report are also mandatory. Both universal slices have exact symbol-address and
full-binary checksum gates. This is target-memory observation only; it
does not patch code or replace results. There are currently zero locally
accepted native actions, so this implementation advances the capture route but
does not yet advance the behavioral-parity gate.

The native-to-Saito field audit is machine-readable at
`artifacts/steam_native_saito_mapping.json`. Ordinary static analysis has now
located raw `CWorld` fields for VP (`+5528`), DEFCON (`+5532`), the two milops
slots (`+5544/+5548`), the two space slots (`+5552/+5556`), turn (`+5584`) and
action round (`+5588`) in both architecture slices. The observer records these
as raw native scalars. They are not yet called Saito-equivalent: player-index to
faction, VP sign, turn/AR base and transition timing still need a mapped native
panel. Static exported getter ABIs also cover world counters, both private
hands, ordered draw/discard/removed lists, country state, headlines and cards in
play. Those exports have not been dynamically validated by the revised host;
native card/country ID crosswalks, per-card knownness, state-machine stack, RNG,
fixed-draw cursor and effect auxiliary payloads remain unresolved.

The smallest concrete evidence gap is therefore one validator-accepted
`ai_option_selection` in a single capture session, with installation, AI-owned
prompt, matched evaluator entry, direct post-evaluator `SelectOption`, and its
matching host report. It is smaller than a full state mapping and is currently
blocked only at execution time by the unchanged macOS debugger policy. After
that record exists, the next increment is a panel-local exhaustive native
option↔Saito legal-action bijection plus full state extraction; complete-game
traces remain mandatory before behavioral-parity or genuine-Elo claims.

The LLDB observer is currently blocked by the Mac's unchanged security state:
`DevToolsSecurity` reports Developer Mode disabled, and even
`lldb --batch -o run -- /usr/bin/true` fails with `Not allowed to attach to
process`. No security setting was changed. The exact one-time prerequisite is
`sudo /usr/sbin/DevToolsSecurity -enable`, but that is a macOS security-policy
change and must be explicitly authorized and performed by the Mac owner or
administrator. It has **not** been run. After authorization, first verify with
`/usr/sbin/DevToolsSecurity -status` and the `/usr/bin/true` LLDB command before
running the capture harness.

`GetGamePlayerAIState` is useful only for thinking/liveness. It is explicitly not
accepted as selected-action provenance. A valid differential oracle panel must
instead identify the AI actor through the instrumented evaluator-to-
`CGameStateOptions::SelectOption` path. No `UpdateGame` event currently qualifies.

The comparator makes a genuinely captured `CMediumAI` selected action the
preferred gate. Mapped
repository participant outputs are compared against the native canonical
selection only after all state, option and transition mapping flags are true.
The old Python reconstruction comparison is disabled by default and is available
only with `--include-proxy-diagnostic`; its result is explicitly excluded from
the parity gate.

Genuine native action comparability does not require `proxy_observation`. That
field is consulted only when `--include-proxy-diagnostic` is requested.

The schema intentionally stores hashes and private local blob references rather
than embedding proprietary native serializations. A panel is not action-comparable
until every native option has a canonical key and verified Saito action mapping.

### Exact macOS observations still required

Native loading and host-game construction are no longer blockers. AI option and
selection capture remain unresolved. For each decision panel, the macOS oracle
must now capture:

1. SHA-256/build identity and exact variant/setup parameters.
2. Serialized `CWorld` and `CStateMachine` immediately before evaluation, retained
   privately, plus hashes in the portable panel.
3. Both private hands, per-card known/unknown flags, deck order, discard and removed
   piles, headline cards, effects and all pending state-machine prompts.
4. World RNG state, die-roll and draw buffers, and the RNG call sequence from
   `PrepareEvaluation` through final selection.
5. The estimated opponent hand actually installed by `ApplyEstimateOpponentHand`.
6. Complete native `CGameStateOptions` in original order: option type, instance ID,
   option ID, packed option bytes, filter category, retained ordering and score where
   observable.
7. Prepared world region/player accumulators consumed by every `Score*State`, plus
   candidate leaf values and terminal reason/timing fields.
8. Selected native option index and stable canonical action key.
9. Repeated evaluations of the identical serialized state/RNG configuration, and
   controlled runs with changed RNG state, to distinguish deterministic tie behavior
   from sampling.
10. A manually verified Steam↔Saito state/action translation for the panel.

Screenshots, visible board state, hand-only logs, or the selected card/country alone
are insufficient because native choice depends on hidden-hand estimation, option
ordering, nested state-machine state and RNG.

## Current repository policy/checkpoint inventory

These are distinct participants; search configuration is part of participant identity:

| ID | Path | Status / intended identity |
|---|---|---|
| `legacy_mixed_side_best` | `runs/ppo/selected_policies/best_us57005_ussr66001` | legacy side-specific PPO bundle; US episode 57005 and USSR episode 66001; stable Elo anchor candidate |
| `transformer_riskshape10` | `runs/ppo/selected_policies/transformer_history_riskshape10_best_20260626-042440` | selected transformer-history risk-shaping bundle |
| `transformer_defconfilter5` | `runs/ppo/selected_policies/transformer_history_defconfilter5_side_selected_20260627-1110` | side-selected transformer; checkpoint-100 US/checkpoint-200 USSR, based on small 20/30-game ratings |
| `phase3_u75_parent` | `runs/search_distill/root-mechanics-actor-correction-actionhead-u100-gpu4-20260819/checkpoints/updates-000075` | retained current general parent despite historical directory name containing `u100` |
| `p30u25_value` | `runs/search_distill/search-scaling-phase3u75-valuetower-weighted-p30-u25-gpu2-20260827/checkpoints/updates-000025` | retained critic-specific value candidate; actor intended unchanged |
| `phase3_actor_p30u25_composed` | `runs/search_distill/search-scaling-phase3u75-parentactor-p30u25-composed-20260828/checkpoints/composed` | current composed search backbone; fixed-root smoke previously reported cutoff-free |

The canonical current raw parent is `phase3_u75_parent`. The canonical current
search participant is the composed checkpoint with a fully specified search mode,
for example `D2/32`; “the checkpoint” without search settings is not a unique player.

## Canonical paired `optional_us_plus_2` Elo protocol

This protocol is specified only; no matches were launched.

### Environment and identities

- Rules: Saito `deck=optional`, `usbonus=2`, exact repository optional-card list,
  no Turn Zero/promos unless a separately named pool is created.
- No scripted setup, persistent random override, heuristic override or reward shaping.
- Record the code revision, policy weight hashes, feature schema, DEFCON filter mode,
  maximum episode steps, and complete search configuration.
- Treat raw, D2/32, D4/128, and any proxy runner configuration as different players.
- For genuine Steam comparison, use one common authoritative rules engine. Do not
  mix Steam-produced outcomes and Saito-produced outcomes in one Elo likelihood
  until rules/action parity is established.

### Pairing and counts

- One seed pair consists of two games with identical seed/setup stream: candidate
  as US then candidate as USSR.
- Wiring smoke: 2 pairs/4 games, excluded from ratings.
- Screening edge: 25 pairs/50 games, explicitly preliminary.
- Canonical rating edge: 200 pairs/400 games.
- Promotion/close result: extend the same preregistered disjoint schedule to 500
  pairs/1000 games when the paired-bootstrap interval overlaps the promotion margin.
- Seed ranges must be disjoint across tuning, screening and canonical evaluation.

### Anchored multiplayer Elo

Fit all participants jointly with a Bradley–Terry logistic model rather than
chaining pairwise deltas. For game `g` between players `i` and `j`:

`logit(P(i scores)) = ln(10)/400 * (R_i - R_j + beta_US * I[i is US] - beta_US * I[j is US])`.

Use score 1/0.5/0 for win/draw/loss and fit ratings plus one global US-side
intercept. Fix `legacy_mixed_side_best raw` to 1500 to identify the scale. Ensure
the evaluation graph is connected: every new participant plays the anchor and at
least one current parent; the current raw and search participants play each other.
Report joint MLE ratings, direct paired score, side-specific scores, terminal reasons,
timeouts/cutoffs and decision latency. A genuine Steam participant must be labeled by
build, difficulty path and capture/harness version.

For uncertainty, resample whole seed pairs with replacement (never individual games),
refit the entire multiplayer model for 20,000 bootstrap replicates, and report 2.5/97.5
percentile intervals for ratings and pairwise differences. Predeclare draws/timeouts
and never turn incomplete games into losses silently.

## External-prior exclusion

The Chinese human-strategy note `量化TS基础` was **NOT USED**. No value, formula,
weight, feature, test expectation or result in this implementation comes from
that note. `UNRESOLVED` does not mean that a native value is unrecoverable.

Future use is permitted only if normal static analysis, dynamic native
instrumentation, native accumulator/candidate dumps and the headless oracle have
first shown a specific required proxy value map to be genuinely hidden or not
recoverable. Any later use must be a separate, disableable `EXTERNAL_PRIOR`
component and must report an on/off ablation. Until that specific trigger is
documented claim by claim, the note remains outside the code and evidence base.

## Why genuine Steam-AI Elo is not identifiable now

An Elo belongs to a game-playing policy in a defined rules environment. A
headless game containing `CMediumAI` is feasible, but selected AI actions have
not yet been captured. The repository also lacks a lossless common Steam/Saito
state/action boundary and complete mapped native game traces. An Elo measured for
the current runnable Python code would be the Elo of the **Steam-reconstructed
proxy**, not Steam AI.

The exact minimum evidence needed before a genuine Steam-AI Elo point estimate is:

1. extend the now-working legal macOS host from feasibility smoke to capture
   genuine `CMediumAI` selected actions with verified AI-actor provenance and
   complete native state for every decision
   of complete games;
2. implement and verify a lossless native-state/native-option to Saito mapping,
   with current policies callable on the mapped state and transition parity;
3. capture complete hidden state and RNG data sufficient to reproduce each native choice;
4. fix one `optional_us_plus_2` rules/card manifest shared by every participant;
5. collect at least the canonical 200 paired seeds (400 complete games) per connected rating
   edge, with side swaps and durable raw results;
6. anchored joint Bradley–Terry fitting and pair-cluster bootstrap intervals as above.

Until items 1–4 exist, even running more proxy games cannot identify genuine
Steam-AI Elo. Items 5–6 then determine the canonical estimate and uncertainty.

## Focused validation performed

No Elo, tournament, complete paired match, training job or historical run artifact
was started.

```text
pytest -q tests/test_steam_builtin_ai.py tests/test_policies.py tests/test_env.py
41 passed in 5.87s

pytest -q tests/test_native_oracle_capture.py tests/test_steam_builtin_ai.py
14 passed in 0.09s

fidelity/native-oracle/mapping artifacts: valid JSON; both trace schemas valid Draft 2020-12
capture/validator/comparator modules: Python bytecode compilation passed
native-oracle harness: checksum inspected only; not executed or imported on Linux
npm run smoke: ok=true, backend=saito, optional_us_plus_2

recursive proxy smoke:
seed=7 steps=4 search_depth=2 max_candidates=4 expanded_nodes=80
terminal=false turn=1 vp=0

paired evaluator:
dry_run_no_games_started, 2 seed pairs / 4 scheduled games

differential harness:
incomplete synthetic panel rejected; native comparability tested without proxy_observation
proxy_observation required only for explicitly requested Python diagnostic
```
