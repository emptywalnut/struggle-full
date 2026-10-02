# Browser Frontend

## What it is

The browser frontend is a thin, private interface over the repository's
existing headless Saito integration. It does not implement a second set of
Twilight Struggle rules.

- One `bridge/saito_bridge.js` child process owns each browser game.
- Browser choices are accepted only as indices into the bridge's current
  `legal_actions`. Stale or out-of-range choices receive an explicit 409/422
  error and are never forwarded as invented Saito actions.
- `struggle_ai/browser_ai.py` invokes either the existing
  `struggle_ai.policies.choose_heuristic_action` baseline or a validated
  repository checkpoint through the canonical Python encoder/masked model.
  JavaScript never implements model inference or accepts a checkpoint path.
- The web client visualizes every legal choice in the Actions panel. Country
  choices are also highlighted on the map, and card-list choices are also
  highlighted in the appropriate hand.

## Modes and information boundaries

### Human versus AI

Choose either US or USSR. The human sees only that side's hand plus public
hand counts, piles, board state, logs, and statuses. When the AI owns the
current forced response, the human's legal action list is empty and the
server advances only through the configured AI policy.

### AI versus AI

Both sides independently select a baseline or registered trained model. A new
match starts paused and supports resume, pause, one-decision step, and speed
controls. Both hands are hidden by default; a local observer toggle can reveal
them without changing either AI observation or action API.

## Trained model registry

`GET /api/models` reports the safe server-side registry. It exposes opaque IDs,
repository-relative provenance, architecture/schema compatibility, and
per-side readiness/device state. `POST /api/models/<id>/load` accepts only a
registered ID plus `us` or `ussr`; arbitrary filesystem paths never cross the
HTTP boundary.

The registry validates serialized observation and action spaces, constructs
`TorchMaskedActions`, and restores weights before exposing a checkpoint. Older
selected PPO policies use `copy_compatible_or_widened_weights`, the same
zero-column widening used by training restores. Inference uses
`features.encode_observation`, the checkpoint's recorded DEFCON filtering mode,
the padded legal-action mask, and deterministic masked logits. It maps a
filtered model index back to the unchanged authoritative Saito legal-action
index. There is no heuristic/random fallback after a model error.

Transformer history and card-history records are maintained with the canonical
environment helpers for every human and AI transition. They are isolated per
browser session; side is encoded in each record as it was during training. The
models are stateless between calls beyond those explicit observation histories,
and cached model instances are keyed by registry ID and side.

The server prefers an available CUDA device with the most free memory. Set
`STRUGGLE_AI_DEVICE=cpu` for the tested CPU path or `cuda:N` to pin a device. A
CUDA load failure retries the same model on CPU and reports that device fallback
in registry readiness; it never changes policy identity.

## Engine state surfaced

The interface shows:

- setup/headline/action phase, active side, turn, and action round;
- VP, DEFCON, Military Operations, Space Race, and China Card ownership;
- live influence, control, battleground status, stability, regions, and the
  Saito board's printed scoring aids;
- the viewer-appropriate hand, hand counts, draw/discard/removed piles,
  headline cards, and active event flags;
- every current legal action, rejected illegal/stale choices, browser action
  history, dice consumed by choices, and the raw Saito rules log;
- terminal winner and the server's AI progress.

Event-versus-operations order, influence, coup, realignment, Space Race,
scoring, forced scoring-card play, round transitions, DEFCON/Mil Ops, China
Card, and card-specific prompts all continue through the bridge's structured
legal-action callbacks. The UI is intentionally generic enough to display
list, option, back, card, and country decisions exposed by those callbacks.

## Board and coordinates

The board is the supplied Saito `TS Map-11.jpg` at its native 5100 x 3300
resolution. Each country hit target is positioned from the `top` and `left`
values on the live Saito country object; stability, battleground, region,
ordinary adjacency, influence, and control are also read from that object.
No Natural Earth layer and no Steam artwork are used.

The country data remains authoritative even when a card temporarily changes a
battleground flag or placement legality. Superpower adjacency is already
printed on the Saito board and is resolved by Saito's special realignment and
scoring branches.

### Native overlay geometry

Saito's country `top` is the top of the printed 100px-high colored influence
placement area, not the top of the complete printed country box. The
country-name/stability strip begins 32px above that coordinate. The upstream
renderer places adjacent `.us` and `.ussr` panes at `(0, 0)` and `(100, 0)`,
each exactly 100 x 100 native pixels. This frontend uses the same geometry;
the surrounding 202 x 132 interactive target does not shift those panes.

The pinned upstream `twilight-base.css` used to verify this behavior is:

`https://raw.githubusercontent.com/SaitoTech/saito-lite-rust/8781f1d71964487fbc507fac4b423351f4aaa0b6/mods/twilight/web/css/twilight-base.css`

It was fetched narrowly for verification and has SHA-256
`f07f5e41f6ac2b6a7403f9437761102593f9f99dcdefe576874ceda0a763b2a0`.
It is not copied into the browser build.

The live track markers use the coordinate tables in the materialized Saito
implementation (`state.defcon_ps`, `state.milops_ps`, `state.vp_ps`, and
`state.space_race_ps`):

- DEFCON 5 through 1: top `2592`, left `1526`, `1682`, `1838`, `1994`, or
  `2150`; the tile is 120 x 120.
- Military Operations 0 through 5: top `2940`, left `1520`, `1675`, `1830`,
  `1985`, `2150`, or `2305`; tiles are 100 x 100. As in Saito, the USSR tile
  is offset `(+10, +40)` inside the represented cell so both tiles remain
  visible when the values coincide.
- Victory Points use all 41 entries of Saito's `vp_ps`, indexed by
  `clamp(vp, -20, 20) + 20`. In this engine negative VP are USSR-positive and
  positive VP are US-positive. The endpoints are USSR 20 at `(3108, 2466)`,
  neutral 0 at `(3572, 2748)`, and US 20 at `(4042, 3030)`; tiles are
  120 x 120.
- Space Race 0 through 8: top `510`, left `3465`, `3638`, `3810`, `3980`,
  `4150`, `4320`, `4490`, `4660`, or `4830`; tiles are 100 x 100. Saito's
  USSR tile offset `(+10, +40)` is preserved so coincident markers remain
  visible.

All coordinates are native board pixels and are transformed only by the
existing `.game-board` zoom, including Fit. Marker state is rebuilt from the
bridge observation after every authoritative state version.

## Private start

Build and start the production-static variant:

```bash
npm run build
npm run browser -- --host 0.0.0.0 --port 41783 --production
```

Health check:

```bash
curl -fsS http://127.0.0.1:41783/api/health
```

Forward from a local workstation (replace only the SSH destination if the
machine is reached through an alias):

```bash
ssh -N -L 41783:127.0.0.1:41783 kaile@10.10.41.30
```

Open `http://127.0.0.1:41783` locally.

The server has no public deployment configuration and should not be exposed
as a public website. Sessions are in-memory and disappear when the server
stops.

## Full-game trained-model acceptance gate

Run the bounded validator against the final production server. It creates an
AI-vs-AI session through `/api/sessions`, sets the supported 80 ms pacing, and
advances every decision through the observer `step` control. It does not call
the bridge or policy service directly.

```bash
npm run validate:browser-full-game -- \
  --base http://127.0.0.1:41783 \
  --us phase3-u75-parent \
  --ussr phase3-composed-p30u25 \
  --seed 20260831 \
  --max-decisions 2500 \
  --output artifacts/browser-full-game-acceptance-20260831.json
```

The validator fails closed on a non-model policy, non-legal recorded action,
wrong or baseline history attribution, skipped or duplicate history number,
unhandled prompt, server error, request timeout, polling failure, missing
terminal result, or unhealthy post-game model cache. It retains the completed
session by default. A retained session can be restored in another browser with
`/?session=<opaque-session-id>`; pass `--delete-session` when retention is not
wanted.

## Asset conditions

The board is owned by GMT Games and/or the original artists and is available
through Saito's open-source module only under GMT's electronically assisted
game-module terms: personal, non-commercial use, with at least one participant
owning a copy of Twilight Struggle. The browser start form requires this
acknowledgement. Full source and verification details are in
`docs/browser_assets.md` and `reference/map-assets/saito/LICENSE.txt`.
