# Steam trace differential audit — final corrected stack

The 19 supplied Steam traces contain 14,318 lines (320,566 bytes). The parser recognized all 9,724 nonblank lines: 9,210 semantic lines and 514 known narrative lines. The final current-stack result is 7,569 deterministic checks: **7,444 exact matches, 125 parser limitations, and 0 hard mismatches**. Both `bridge_differentials` and `materialized_differentials` are empty.

## Confirmed defects and fixes

| Defect | Exact Steam evidence | Root cause and corrected behavior | Layer/training impact |
| --- | --- | --- | --- |
| Repeat selection disappeared after click one | Socialist Governments `3347414.txt:20-22` (Italy −2); Liberation Theology `3347843.txt:295-297` (Panama +2); The Reformer `3352341.txt:802-805` (three +2 placements); Warsaw Pact add mode `3352375.txt:79-83` (East Germany and Poland +2) | `HeadlessDecisionAdapter.trigger` unconditionally removed a class-selected country. `detectBoundedCountryPrompt` now recognizes all four prompts and retains the target until Saito's true max-two-per-country cap. Click one and two are legal; click three is absent/rejected. | Materialized Saito was correct. Bridge, browser catalog, `TwilightStruggleEnv`, encoder, and masks inherited the defect. In 157 saved self-play games/66,688 actions, affected prompts occurred 221 SG, 97 Liberation, 158 Reformer, and 380 Warsaw-add times. |
| Regional bonus replayed the printed Ops command | Vietnam: `3347401.txt:196-202`, `3347401.txt:242-244`, and coup `3346438.txt:134-138`; China: `3346480.txt:86-95` (+5 with the China Card) | `SaitoTwilightKernel.stepAsync` force-drained Saito's pending one-Op prompt and re-entered the original Ops queue item. It now returns the pending restricted target decision and consumes it once. Bonus realignment `click` targets are also region-filtered instead of being discarded by the placement-only `mouseup` filter. | Materialized Saito was correct. Bridge/browser/env/masks exposed the replay. Saved data contains 180 Vietnam Revolts and 239 China Card selections; the saved format cannot determine the exact number that reached an eligible bonus boundary, so these are exposure opportunities, not claimed defect counts. |

Verified semantics include: Socialist Governments removes three total, max two per country; Liberation places three, max two; Reformer and Warsaw add mode retain Saito's event totals with max two; Vietnam gives exactly one extra Op restricted to Southeast Asia; China gives exactly one extra Op restricted to Asia/Southeast Asia. Regressions cover repeated placement, opponent-control cost, coup strength, realignment, combined China+Vietnam (+2 total), and the one-Op Red Purge/Vietnam exception.

No materialized Saito gameplay file or trained checkpoint was changed or retrained. The browser UI continues to project the authoritative bridge catalog; the environment and action encoder still mask that same catalog without a second rules implementation.

## Non-defects and limitations

The four OAS Founded asterisk observations are parser limitations, not engine defects: `3341939.txt:322`, `3346510.txt:475`, `3348222.txt:514`, and `3352375.txt:725`. Steam omits its removal marker in these triggered-event contexts, while its bundled Lua metadata and materialized Saito both define OAS as remove-if-used-as-event. They are included in the 125 parser limitations.

Steam logs omit full hands, deck/shuffle state, options, and some random/hidden state. The audit never promotes an underdetermined transition to a hard mismatch. The remaining parser limitations are explicit in each file's `checks` entries.

## Artifacts and reproduction

- Machine-readable final result: `artifacts/steam_trace_differential_20260831/audit_results.json`
- Preserved pre-fix result: `artifacts/steam_trace_differential_20260831/audit_results_pre_fix_confirmed.json`
- Parser/replay harness: `artifacts/steam_trace_differential_20260831/audit_steam_traces.py`
- Saito oracle and minimized probes: `saito_oracle.js`, `bridge_diagnostics.js`
- Env/mask projection: `env_mask_diagnostics.py`, `env_mask_diagnostics.json`
- Training exposure scan: `training_exposure_scan.py`, `training_exposure.json`
- Per-trace inventory: the `files` array in `audit_results.json`
- Final consistency gate: `self_check.py`

Run from the repository root:

```bash
/mnt2/users/kaile/miniconda3/envs/struggle-ppo/bin/python artifacts/steam_trace_differential_20260831/audit_steam_traces.py
/mnt2/users/kaile/miniconda3/envs/struggle-ppo/bin/python artifacts/steam_trace_differential_20260831/self_check.py
```
