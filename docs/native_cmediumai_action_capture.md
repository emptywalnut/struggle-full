# Experimental native CMediumAI selected-action capture

Date: 2026-08-31  
Native subject: macOS Twilight Struggle 1.4.11/build 167  
Binary SHA-256: `b45dc29df33f8c11312c2fe64d8b22764d153afb874a8e3f6867465ede0a1fa7`  
Status: **HOST_FIX_LOCALLY_VERIFIED_BUT_ACTION_CAPTURE_BLOCKED_BY_DISABLED_DEVELOPER_MODE**

This is the preferred route to the first genuine native `CMediumAI` selected
action. It does not demonstrate Steam behavioral parity or make genuine
Steam-AI Elo identifiable.

## Player construction semantics

These facts come from both native architecture slices, not from the smoke:

| Fact | arm64 evidence | x86_64 evidence | Status |
|---|---:|---:|---|
| `StartGame` copies `min(count,4) * 0x34` bytes | `0xa8cc-0xa8e8` | `0x8176-0x8193` | CONFIRMED |
| `player_type` is record byte `+0xc` | load/compare AppWorldData `+44` at `0xaa6c-0xaa78` | `cmpb $2` at `0x8359`, with 52-byte loop stride | CONFIRMED |
| only `player_type == 2` constructs an AI evaluator | branch above | branch above | CONFIRMED |
| `ai_difficulty` is byte `+0xd`, consulted only on the type-2 path | load AppWorldData `+45` at `0xaacc` | `cmpb $3` at `0x83c6` | CONFIRMED |
| difficulty `3` constructs `CTutorialAI` | call `0xaafc` | call `0x83f7` | CONFIRMED |
| every non-3 difficulty constructs `CMediumAI` | call `0xab2c` | call `0x842e` | CONFIRMED |
| wrapper/evaluator actor ID is stored at `this+8` | wrapper `0xab50`; constructor `0x1647c` | wrapper `0x8471`; constructor body `0x14d1c` | CONFIRMED |

The exact UI/network labels for non-AI types 0 and 1 remain partly unresolved.
The instrumentation uses only the confirmed fact that type 2 is the AI-evaluator
construction path. Raw difficulty 2 is not claimed to be a separate policy:
every value except 3 reaches the same `CMediumAI` constructor.

`GameParameters` is copied as one 16-byte vector at arm64 `0xa8b4-0xa8b8` and
x86_64 `0x8153-0x8158`.

## `UpdateGame` event ABI

`UpdateGame` is arm64 `0xa3a0` and x86_64 `0x7be7`.

- Argument 0 is a byte buffer; argument 1 is byte capacity, not record count.
  The prior smoke allocated 8192 bytes but passed `1024`.
- The return value is the number of logical records, not bytes written.
- The stream is packed and tagged by a `uint32 event_type` at record offset 0.
  It is not an array of fixed 8-byte or `uint64` records.
- Each producer advances by its own record size. The capacity check is strict:
  `write + record_size < base + byte_capacity`.

Representative confirmed records are:

| Type | Bytes | Payload after type | Evidence | Selected-action value? |
|---:|---:|---|---:|---|
| 35 | 8 | 4-byte load/update progress | `0x6f84-0x6fcc`, `0xa184-0xa1b8` | no |
| 36 | 12 | temporary-move count and function argument | `0x692c-0x698c` | no option index/ID/hint |
| 48 | 8 | two tutorial `uint16` fields | `0x6994-0x6a78` | no actor or index; tutorial only |
| 99 | 8 | internal game-log update count | `0xa424-0xa440` | no |

`OnGameStateChange 0x78e8` emits many other variable records (confirmed sizes
include 8, 12, 16, 20, 24, 28, 32, 36 and 60 bytes). No output-buffer producer
copies the normal 16-byte move entry as a unit. No event is confirmed to contain
the complete actor plus option index plus selection ID/hint tuple. Consequently
the new host retains events only as raw diagnostics; they are insufficient
selected-action provenance.

## The native action boundary

The normal move entry is 16 bytes (`StoreMoveEntry 0x66b8`, retrieve `0x6824`):

| Offset | Field |
|---:|---|
| 0 | `uint32 actor_player_id` |
| 4 | `int16 option_index` |
| 6 | `uint16 selection_id` |
| 8 | `uint16 selection_hint` |
| 12 | `uint32 optional_data` |

`OfflineOptionSelectFunc` is **not** the CMediumAI selection point. It is the
exported host/network staging path:

- arm64 file address `0x6b80`; x86_64 `0x4171`;
- signature `(CStateMachine*, void*, uint32 actor, int option_index, uint16
  selection_id, uint16 selection_hint, uint32* optional_data)`;
- host choices are staged at AppWorldData `+380 + 16*n`; special hints 40992
  and 41138 go directly to the main move buffer (`0x6bc0-0x6c50`).

The actual evaluator-return boundary is `CGameStateOptions::SelectOption`
(arm64 `0x11d98`, x86_64 `0x1090e`). Static control flow proves that
`CGameStateOptions::UpdateState` calls the configured evaluator, bounds-checks
its returned index, and then invokes `SelectOption` at arm64 `0x11d40-0x11d4c`
or x86_64 `0x108d9-0x108ff`. At `SelectOption` entry the ABI is options `this`,
state machine, selected index, and optional-data pointer. The selected option's
ID and hint are still readable from the live option array. This corrects the
earlier instrumentation plan, which would have observed human/network staging
but not the direct CMediumAI return path.

The option list is observed on entry to
`twilight_ai::CMediumAI::Evaluate` (arm64 `0x1a8f0`, x86_64 `0x19a0c`):

- evaluator `this+8` is the actor ID;
- `CGameStateOptions+88` is signed option count and `+96` the array pointer;
- each `CUserOption` is 32 bytes: ID `+0`, hint `+2`, display-string pointer
  `+8`, select callback `+16`, callback data `+24`;
- getter evidence is arm64 `0x11f04/0x11f0c` and x86_64
  `0x10a30/0x10a3a`.

The prompt and its owning player are observed earlier at
`CGameStateOptions::DisplayOptionList` (arm64 `0x120a8`, x86_64 `0x10bb0`).
At entry, the prompt is argument 3 and the player ID is `player+24`. This entry
point is reached even when the function then suppresses public display for an
AI evaluator, so the AI prompt does not depend on the human callback.

For the first setup-country action only, static semantics are narrower and
stronger: `CStateProcessPlaceInfluence::BuildInfluenceOptionList` emits hint
41010 with `selection_id == callback_data == native country index`, and its
callback is `SelectInfluenceCountry` (arm64 `0x493f8`, x86_64 `0x48782`). The
observer records the ASLR-normalized callback file address and labels this
native action family `country_click` only when every condition matches. It does
not invent a Saito country ID: a bijective native-country-index↔Saito-country-ID
crosswalk is still required, and every other hint remains hard-unmapped.

The LLDB observer correlates a selection using the exact thread, actor,
state-machine pointer and options pointer. It requires matching prompt and
option fingerprints, rejects any `SelectOption` reached while a
`CMediumAI::Evaluate` frame is still active (search/simulation), and consumes a
matched evaluation after one accepted selection. Every record also carries one
explicit capture-session ID and the observer-script checksum; the validator
rejects mixed sessions, stale records, sequence gaps, script drift, or a missing
host report. It
auto-continues and reads target memory only: no code patch, target call, changed
register or replaced return value. DYLD interposition was not selected because
direct intra-image calls need not traverse an interposable stub.
`GetGamePlayerAIState` was not selected because it reports liveness/progress,
not a choice.

## Why the six-prompt smoke stopped

Two independent host defects are confirmed:

1. Ordinary human choices are staged by `OfflineOptionSelectFunc` at
   `0x6c98-0x6cdc`. The old smoke selected six setup countries but never called
   exported `CommitTemporaryMoveBuffer`. That function copies staged 16-byte
   entries to the main buffer and clears the temporary count
   (`0x6a94-0x6b68`). `HasTemporaryMoveBuffer 0x7294` is the defensive check.
2. The old smoke called `Initialize(lua_path, 1)`. `_Initialize` at
   `0x4c88-0x4c90` (x86_64 `0x231d-0x2322`) starts the AI handler only when the
   second argument is at least 2. `CMediumAI` has a synchronous fallback, so
   this alone is not claimed as the permanent-stall cause, but mode 2 is the
   correct bounded AI-host route.

The zero-sleep loop also interacts with `CDelayAIPlayer`'s real-time delay
(`0x7134-0x7270`). The earlier 10,083 logical-event total remains a stalled-loop
observation with no AI-decision value.

The new host uses mode 2, passes the true event-buffer byte size, sleeps 10 ms,
commits only after the six independently observed USSR setup selections, and has
iteration and wall-clock bounds.

This host fix is now locally verified on macOS. With `human-ussr-ai-us`, six
automatic human setup selections and `CommitTemporaryMoveBuffer`, the game
advanced to the human `Select a Card to Headline` callback. The bounded report
contained `temporary_move_buffer_committed=true`, 123 logical events, 23
nonempty updates, and a 10-second wall-bound exit. This verifies the stall
diagnosis and host correction. It captured zero AI selections because the LLDB
observer could not run.

The local Mac reports Developer Mode disabled through `DevToolsSecurity`; even
`lldb --batch -o run -- /usr/bin/true` fails with `Not allowed to attach to
process`. No security setting was changed. The one-time prerequisite is a
security-policy change that must be explicitly authorized and run by the Mac
owner or administrator:

```bash
sudo /usr/sbin/DevToolsSecurity -enable
```

That command was **not run**. After explicit authorization, verify the state and
LLDB child launch before attempting the native capture:

```bash
/usr/sbin/DevToolsSecurity -status
lldb --batch -o run -- /usr/bin/true
```

If either check remains blocked, stop and use the Mac's administrator or device
management policy rather than weakening any additional protections.

## Layout safety

- `human-ussr-ai-us` is the first run: one confirmed non-AI local player and one
  type-2 AI, then the six staged USSR setup choices are committed.
- `ai-ussr-human-us` is the side-swapped follow-up. It retains exactly one
  non-AI local player and lets the USSR AI reach the first setup decision.
- `both-ai` is statically plausible because `StartGame` defaults the local ID to
  zero and `SetLocalPlayer(0)` can resolve player 0, but it is not locally
  verified. It requires an explicit flag and is not the first-run route.

## Exact local commands

After the explicitly authorized Developer Mode prerequisite succeeds, run from
the repository root on macOS with a Python architecture matching one slice of
the universal library.

```bash
python3 tools/native_oracle_capture_macos.py --self-check

TRACE="$PWD/artifacts/native_cmediumai_first_action.jsonl"
REPORT="$PWD/artifacts/native_cmediumai_first_action_host.json"
VALIDATION="$PWD/artifacts/native_cmediumai_first_action_validation.json"
SESSION_ID="$(python3 -c 'import uuid; print(uuid.uuid4().hex)')"
rm -f "$TRACE" "$REPORT" "$VALIDATION"

TS_NATIVE_CAPTURE_PATH="$TRACE" \
TS_NATIVE_CAPTURE_SESSION_ID="$SESSION_ID" \
TS_NATIVE_AI_PLAYER_IDS=1 \
TS_NATIVE_START_SEED=0x5A17C0DE \
TS_NATIVE_CAPTURE_RNG=1 \
lldb --batch \
  -o "command script import $PWD/tools/native_oracle_capture_lldb.py" \
  -o run -- \
  python3 "$PWD/tools/native_oracle_capture_macos.py" \
    --layout human-ussr-ai-us --seed 0x5A17C0DE \
    --trace-jsonl "$TRACE" --report "$REPORT" \
    --max-wall-seconds 60 --stop-after-ai-selections 1

python3 tools/validate_native_oracle_capture.py "$TRACE" \
  --host-report "$REPORT" --output "$VALIDATION"
```

If that reaches no verified AI selection, use the side swap without a human selection:

```bash
TRACE="$PWD/artifacts/native_cmediumai_side_swap.jsonl"
REPORT="$PWD/artifacts/native_cmediumai_side_swap_host.json"
SESSION_ID="$(python3 -c 'import uuid; print(uuid.uuid4().hex)')"
rm -f "$TRACE" "$REPORT"

TS_NATIVE_CAPTURE_PATH="$TRACE" \
TS_NATIVE_CAPTURE_SESSION_ID="$SESSION_ID" \
TS_NATIVE_AI_PLAYER_IDS=0 \
TS_NATIVE_START_SEED=0x5A17C0DE \
TS_NATIVE_CAPTURE_RNG=1 \
lldb --batch \
  -o "command script import $PWD/tools/native_oracle_capture_lldb.py" \
  -o run -- \
  python3 "$PWD/tools/native_oracle_capture_macos.py" \
    --layout ai-ussr-human-us \
    --max-auto-human-selections 0 --commit-after-auto-selections 0 \
    --seed 0x5A17C0DE --trace-jsonl "$TRACE" --report "$REPORT" \
    --max-wall-seconds 60 --stop-after-ai-selections 1
```

Do not use `both-ai` until a single-AI layout has a clean capture. The host
rejects it without `--allow-both-ai-experimental`.

## Required evidence fields

The JSONL schema is `artifacts/steam_native_action_capture_schema.json`. The
milestone passes only when the validator reports
`first_genuine_cmedium_selected_action_with_verified_ai_actor_provenance=true`
and the accepted record contains:

1. checksum, architecture, symbol and exact file address;
2. configured type-2 actor and `CMediumAI::Evaluate this+8` actor;
3. native prompt and owning player plus start seed, pre-evaluation 7040-byte
   world hash and state-machine prefix hash;
4. complete ordered options with index, ID, hint, text, ASLR-normalized callback
   file address and callback data. The raw 32-byte entry is retained only as a
   local diagnostic because it contains process pointers;
5. selected index, ID, hint and optional-data pointer at the direct
   `CGameStateOptions::SelectOption` boundary;
6. matching thread, actor, state-machine pointer, options pointer, evaluation
   record and option fingerprint, with no active CMediumAI frame above the
   selection;
7. `selection_snapshot_matches_evaluation=true`,
   `prompt_context_matches_selection=true` and
   `ai_actor_provenance_gate=true`;
8. one capture session with contiguous records and matching observer checksum;
9. matching host player layout, seed, session and checksum.

Optional RNG instrumentation records call entry arguments and receiver-state
hashes. It explicitly records `return_value_captured=false`; the native RNG
sequence therefore remains unresolved. A successful action capture advances the
oracle provenance gate, but evaluator parity, hidden-hand/RNG parity,
native-to-Saito mapping, rules parity and differential action parity remain
unresolved.

The Chinese note `量化TS基础` was not read into, referenced by, or used in this
instrumentation, tests, or any value map.
