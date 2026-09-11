# GRM-D1 — A+ alias probes on the CPU fake session

`GRM_ALIAS_FOLD_MERGE=1`, `GRM_ADMISSION_RULE=margin_first` (both pinned and read back, R1 idiom), arena width 96, 164 turns replayed.

Evidence class: CPU fake session (lineage/routing/admission only; NOT a language-model quality measurement)

Alias fold jobs pending before the sweep: 1; merges executed: 10; pending after: 0.

| probe | d | alias -> base | LT1 GPU arm A | fold job fired? | digest node | digest identifiers (FIX-8) | admitted? | mounted ids | served value (CPU double) |
|---|---|---|---|---|---|---|---|---|---|
| `recall_6_10` | 10 | the Hauler -> Kestrel | correct | **yes** | [164] | `37`, `a`, `alias`, `allowance`, `an`, `archive`, `be`, `call`, `cargo`, `crates`, …(31 total) | **yes** | [164] | `37 crates……………………………………………………………………………` |
| `recall_7_10` | 10 | the Beacon -> Lantern | **WRONG** | **yes** | [165] | `18`, `2196`, `a`, `alias`, `an`, `archive`, `be`, `beacon`, `call`, `crew`, …(32 total) | **yes** | [165] | `18 October 2196………………………………………………………………………` |
| `recall_6_25` | 25 | the Hauler -> Kestrel | correct | **yes** | [164] | `37`, `a`, `alias`, `allowance`, `an`, `archive`, `be`, `call`, `cargo`, `crates`, …(31 total) | **yes** | [164] | `37 crates……………………………………………………………………………` |
| `recall_7_25` | 25 | the Beacon -> Lantern | **WRONG** | **yes** | [165] | `18`, `2196`, `a`, `alias`, `an`, `archive`, `be`, `beacon`, `call`, `crew`, …(32 total) | **yes** | [165] | `18 October 2196………………………………………………………………………` |
| `recall_6_50` | 50 | the Hauler -> Kestrel | correct | **yes** | [164] | `37`, `a`, `alias`, `allowance`, `an`, `archive`, `be`, `call`, `cargo`, `crates`, …(31 total) | **yes** | [164] | `37 crates……………………………………………………………………………` |
| `recall_7_50` | 50 | the Beacon -> Lantern | **WRONG** | **yes** | [165] | `18`, `2196`, `a`, `alias`, `an`, `archive`, `be`, `beacon`, `call`, `crew`, …(32 total) | **yes** | [165] | `18 October 2196………………………………………………………………………` |
| `recall_6_100` | 100 | the Hauler -> Kestrel | correct | **yes** | [164] | `37`, `a`, `alias`, `allowance`, `an`, `archive`, `be`, `call`, `cargo`, `crates`, …(31 total) | **yes** | [164] | `37 crates……………………………………………………………………………` |
| `recall_7_100` | 100 | the Beacon -> Lantern | **WRONG** | **yes** | [165] | `18`, `2196`, `a`, `alias`, `an`, `archive`, `be`, `beacon`, `call`, `crew`, …(32 total) | **yes** | [165] | `18 October 2196………………………………………………………………………` |
| `recall_6_150` | 150 | the Hauler -> Kestrel | correct | **yes** | [164] | `37`, `a`, `alias`, `allowance`, `an`, `archive`, `be`, `call`, `cargo`, `crates`, …(31 total) | **yes** | [164] | `37 crates……………………………………………………………………………` |
| `recall_7_150` | 150 | the Beacon -> Lantern | **WRONG** | **yes** | [165] | `18`, `2196`, `a`, `alias`, `an`, `archive`, `be`, `beacon`, `call`, `crew`, …(32 total) | **yes** | [165] | `18 October 2196………………………………………………………………………` |

## Arm A (flag OFF) contrast

| probe | fold job fired? | digest node | admitted? | mounted ids | served value |
|---|---|---|---|---|---|
| `recall_6_10` | no | — | no | [4] | `unknown…………………………………………………………………………………` |
| `recall_7_10` | no | — | no | [5] | `unknown…………………………………………………………………………………` |
| `recall_6_25` | no | — | no | [4] | `unknown…………………………………………………………………………………` |
| `recall_7_25` | no | — | no | [5] | `unknown…………………………………………………………………………………` |
| `recall_6_50` | no | — | no | [4] | `unknown…………………………………………………………………………………` |
| `recall_7_50` | no | — | no | [5] | `unknown…………………………………………………………………………………` |
| `recall_6_100` | no | — | no | [4] | `unknown…………………………………………………………………………………` |
| `recall_7_100` | no | — | no | [5] | `unknown…………………………………………………………………………………` |
| `recall_6_150` | no | — | no | [4] | `unknown…………………………………………………………………………………` |
| `recall_7_150` | no | — | no | [5] | `unknown…………………………………………………………………………………` |

## Counts

- A+ fold jobs fired: 10/10 probes
- A+ digests naming BOTH names: 10/10
- A+ digests ADMITTED by routing: 10/10
- A+ digests MOUNTED: 10/10
- A  fold jobs fired: 0/10 (flag OFF: must be 0)

