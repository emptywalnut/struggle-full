# Steam-reconstructed proxy: usage and limits

This runnable policy is an evidence-informed **Steam-reconstructed proxy**. It is
not genuine Steam AI, not numerically compatible with `CMediumAI`, and has not
passed differential action parity. The authoritative status is in
`docs/steam_builtin_ai_fidelity_audit.md` and
`artifacts/steam_ai_fidelity_validation.json`.

## Implemented

- Exact archival native constants and terminal utility table.
- Exact `ScoreBaseState` calculation only when callers supply already prepared
  native accumulators via `NativePreparedAccumulators`.
- A separately named five-feature Saito leaf heuristic that does not reuse or
  mislabel the 14 native constants.
- The 112-card derived Lua annotation table.
- Proxy action ordering plus the confirmed place-influence maximum of four
  candidates per native region bucket.
- Seeded uniform hidden-hand approximation, clearly separate from unresolved
  native sampling bias and RNG sequencing.
- A generic deterministic/chance callback search API.
- A live bounded recursive Saito runner using snapshot/restore, max/min actor
  handling, configurable depth and candidate bounds.

The live runner follows one restored Saito RNG sequence. It does not reproduce
native chance enumeration, native depth-40 throughput, or Steam/Saito rules parity.

## Commands

The default CLI uses live recursive bridge transitions:

```bash
python -m struggle_ai.baselines.steam_builtin.cli \
  --seed 7 --steps 4 --search-depth 2 --max-candidates 8
```

The shallow mode must be requested explicitly:

```bash
python -m struggle_ai.baselines.steam_builtin.cli --ordering-only --steps 8
```

Interactive play uses the qualified name and recursive runner:

```bash
python -m struggle_ai.play --human us --bot steam_proxy --seed 7
```

The paired evaluator is non-interactive and dry-runs by default. This command
does not play games:

```bash
python scripts/eval_steam_reconstructed_proxy_pair.py \
  --seed-pairs 10 --opponent heuristic
```

Actual games require the explicit `--execute` flag. No aggregate evaluation was
launched during this task.

## Differential panels

The preferred parity authority is a genuinely captured macOS `CMediumAI`
selected action. The checksum-pinned `tools/native_oracle_smoke_macos.py` proves
only that a headless game containing player type 2 can initialize and run. Its
six callbacks were all player-0 human setup prompts; it captured no AI option or
selection. The bounded audit produced 104 events. The earlier 10,083 count came
from a spun/stalled loop and has no evidentiary value. The harness was not run on
Linux. See `artifacts/steam_native_oracle_spec.json`.

Genuine macOS captures must conform to
`artifacts/steam_ai_trace_schema.json`. Compare available JSONL panels with:

```bash
python scripts/compare_steam_ai_traces.py private_panels.jsonl
```

The harness refuses to call a panel action-comparable unless it includes genuine
AI selected-action provenance, minimum native evidence, complete canonical action
mapping, lossless mapping verification, and transition parity. Repository
participant outputs are then compared directly with the captured `CMediumAI`
selection. `GetGamePlayerAIState` liveness is insufficient. Genuine comparison
does not require `proxy_observation`; that input is required only when the shallow
diagnostic is explicitly enabled with `--include-proxy-diagnostic`, and the
diagnostic cannot pass the parity gate.

## Current validation

Focused tests cover exact prepared-accumulator arithmetic, separation of native
and proxy weights, terminal utility, card annotations, confirmed placement quota,
chance helpers, hidden-hand approximation, recursive max/min, real bridge-runner
control flow, paired schedules, and differential evidence gates. Validation
results are recorded in `docs/steam_builtin_ai_fidelity_audit.md` after each run.
