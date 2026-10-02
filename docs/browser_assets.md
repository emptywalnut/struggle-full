# Browser Asset Sources and Licenses

## Included board

| Field | Value |
| --- | --- |
| Local file | `reference/map-assets/saito/TS Map-11.jpg` |
| Upstream | `https://raw.githubusercontent.com/SaitoTech/saito-lite-rust/master/mods/twilight/web/img/TS%20Map-11.jpg` |
| Pinned Saito revision used by this engine | `8781f1d71964487fbc507fac4b423351f4aaa0b6` |
| Resolution | 5100 x 3300 JPEG |
| Size | 1,619,682 bytes |
| SHA-256 | `8f21635165f354214fa0b7329b3f3e9f2c2769be76ba2dafd192252591f53217` |
| Owner | GMT Games and/or the original game artists |
| Allowed use here | Personal, non-commercial electronically assisted play, with at least one participant owning the game |

The browser build verifies the hash before copying the board into its private
build output. The runtime serves the supplied source file at
`/assets/saito-board.jpg`.

## Notices checked

- GMT Games electronically assisted game module terms:
  <https://www.gmtgames.com/t-GMTVASSAL.aspx>
- Saito's Twilight module license notice at the pinned revision:
  <https://raw.githubusercontent.com/SaitoTech/saito-lite-rust/8781f1d71964487fbc507fac4b423351f4aaa0b6/mods/twilight/license/LICENSE.txt>
- Saito Twilight module README:
  <https://github.com/SaitoTech/saito-lite-rust/tree/master/mods/twilight>
- Official GMT rules reference used to check map concepts and terminology:
  <https://www.gmtgames.com/living_rules/TS_Rules_Deluxe.pdf>

The module's JavaScript rule code is MIT licensed. That MIT grant does **not**
cover GMT's board or other game art.

## Excluded material

- No Steam/Playdek board, card art, UI art, sounds, or other proprietary Steam
  assets are copied.
- No Saito card faces or event artwork are copied into the browser frontend.
  Cards are original CSS text components made from the engine's existing card
  identifiers, names, sides, operations values, era, scoring flag, and
  recurring flag.
- Natural Earth was researched as a public-domain fallback but is not included
  or required after the high-resolution Saito board was supplied.

This frontend is intentionally documented and served as a private
SSH-forwarded development service, not publicly deployed.
