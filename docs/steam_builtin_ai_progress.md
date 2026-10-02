# Steam AI reverse-engineering and reconstructed-proxy progress

Updated: 2026-08-31 (Asia/Shanghai)

- Read the supplied task, input README, and prior policy analysis.
- Confirmed the workspace root is not a Git worktree; no Git status is available.
- Inventory complete: native universal Mach-O, six Lua policy/rules files, matching IL2CPP artifacts, and the existing Saito-backed Python environment are present.
- Completed native symbol/disassembly evidence extraction for construction, preparation,
  every named filter, deterministic evaluation, dice/draw chance handling, scoring,
  hidden-hand application, and background evaluation-thread entry points.
- Verified the matching managed setup names/fields from `Assembly-CSharp.dll` and
  `global-metadata.dat`, then confirmed the actual Easy/Hard/tutorial construction
  branch in native `_StartGame` and `_StartTutorial`.
- Extracted 112 Lua-annotated cards into a generated structured policy table.
- Initial implementation completed, then independently audited and corrected:
  the runnable policy is now consistently named **Steam-reconstructed proxy**.
- Removed the unsupported dot product between the 14 native constants and invented
  raw features. Added an exact `ScoreBaseState` API that requires already prepared
  native accumulators, plus a separately named five-feature Saito proxy leaf value.
- Implemented card policy, partial filters/order (including confirmed four-per-region
  placement quota), hidden-hand approximation, chance helpers, bounded recursive
  bridge minimax, RLlib-shaped proxy adapter, qualified CLI, and
  `play.py --bot steam_proxy` integration.
- Added a default-dry-run non-interactive paired evaluator; no aggregate games run.
- Added genuine macOS differential trace schema/harness and explicit machine-readable
  fidelity gates. A headless game containing player type 2 is verified to
  initialize/run, but the six observed callbacks were human player-0 setup prompts.
  No AI option/selection or losslessly mapped action-comparable panel exists yet.
- Added `docs/steam_builtin_ai_fidelity_audit.md` with claim labels, checkpoint
  inventory, canonical paired Elo protocol, and exact missing evidence.
- Superseded initial test counts are retained only as historical progress; current
  corrected validation is recorded in the fidelity audit.
- Connected recursive proxy smoke complete: 4 steps, depth 2, candidate cap 4,
  80 expanded nodes, seed 7, turn 1, VP 0.
- Paired evaluator dry-run produced a four-game paired schedule and started no games.
- Differential harness correctly rejected an incomplete panel as non-comparable.
- Corrected final validation: `41 passed in 5.87s`; the fidelity and native-oracle
  artifacts are valid JSON, the trace schema is valid Draft 2020-12, and
  `npm run smoke` passed. The macOS-only oracle harness was checksum-inspected but
  deliberately not executed or imported on Linux.
- Existing `npm run smoke` bridge self-test complete (`ok=true`, Saito backend,
  `optional_us_plus_2`).
- Final reverse-engineering, corrected usage, and fidelity-audit reports complete.

Safety: reference binaries remain local and are not copied into the Python package or reports.
