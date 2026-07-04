# Qwen3.5 2B-to-9B Graft Translation PoC Final Write-Up

**Date:** 2026-07-04

## Claim

The surviving claim is **binding transfer**, qualified.

The translated 2B-to-9B grafts passed the frozen G3 binding threshold:
`25 / 32` translated probes had positive gold-minus-best-decoy margins, above
the pre-registered `>= 14 / 32` pass line.

This is not a claim of full Qwen3.5 hybrid-state portability. The 9B side is
still a prefix-restore target for this PoC, and live G0 logit identity failed
the strict max-abs-delta floor even though top-1 stayed stable.

## Gate Results

### G0 Identity

- Capture identity passed structurally on all held-out target shards.
- Live G0 logit identity smoke failed the frozen `2e-3` max-abs-delta
  threshold:
  - max abs delta: `0.1875`
  - mean abs delta: `0.019217236981522855`
  - top-1 flip rate: `0.0`

Interpretation: downstream geometry and binding results are meaningful, but
the live reinjection noise floor is not clean enough to call this a full
runtime-equivalence proof.

### G1 Key Fidelity

- Held-out shards: `1006`
- Held-out tokens per layer-pair: `254,556`
- Key recall@16 range: `0.5634406672489058` to `0.6904513192234587`
- Average key recall@16: `0.637847516976028`
- Shuffled key recall@16 range: `0.047558302250840366` to
  `0.05274420215654368`
- Minimum key/shuffled ratio: `10.688533519012191`

G1 passed the frozen gate: average recall is above `0.60`, and every layer band
is at least `3x` shuffled. The first band is individually below `0.60`, but the
registered gate was average recall plus per-band shuffled separation.

### G2 Value Fidelity

- Value-output cosine range: `0.9094160406409361` to
  `0.9932101022103578`
- Translated/wrong-layer MSE ratio range: `0.01193908268665701` to
  `0.1447668773166899`

G2 passed the frozen gate: every layer band has cosine `>= 0.90` and translated
MSE is `<= 25%` of the wrong-layer baseline.

### G3 Binding

| Mode | Positive margins | Mean margin | Min margin | Frozen pass |
| --- | ---: | ---: | ---: | --- |
| amnesia | `20 / 32` | `0.7479729879753355` | `-2.897013226382356` | `true` |
| source-native | `32 / 32` | `19.93564477957448` | `15.442655127356694` | `true` |
| target-native | `32 / 32` | `18.62211288181452` | `15.489111058558944` | `true` |
| translated | `25 / 32` | `1.3904626497223576` | `-2.2494257231745074` | `true` |

G3 passed the frozen `>= 14 / 32` binding threshold for the translated mode.
The native ceilings are clean at `32 / 32` on both source and target.

The amnesia floor is high at `20 / 32`, so this probe set is not a clean
zero-memory floor. The translated mode still clears the frozen binding
threshold and improves over amnesia, but the final interpretation should be
binding transfer signal under this harness, not an isolated proof that all
binding signal comes only from translated graft state.

## Primary Artifacts

- Eval metrics:
  `/mnt/ForgeRealm/qwen35_graft_translation_poc/translator/eval_metrics.json`
  - sha256:
    `c42847747374bb28b5b033d2a203d91dd6e14cef03f572eca5a0ff54541bfa9a`
- Binding metrics:
  `/mnt/ForgeRealm/qwen35_graft_translation_poc/gates/binding_eval_metrics.json`
  - sha256:
    `0d72858222abb8a2a23a0079fec087e0e6c53f8f29c009dfc80a820f47ae954f`
- Final pipeline status:
  `/mnt/ForgeRealm/qwen35_graft_translation_poc/pipeline_status.json`
  - sha256:
    `f4d0cbfe03fced8aec3c634a81a670be20e15edb36d0d31c8aa4709b8c7931af`
  - status: `complete`
  - stage: `complete`

## Bottom Line

The PoC completed the frozen gate ladder. The defensible result is:

**Qwen3.5 2B-to-9B attention-plane graft translation shows attention geometry
transfer and a qualified binding-transfer signal on the registered probe
harness.**

The result should not be described as full live GRM portability until the live
G0 logit delta is brought under the registered noise floor and the binding
probe floor is tightened.
