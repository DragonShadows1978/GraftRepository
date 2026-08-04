# DOCS1 — GraftRepository Public Documentation Rework

YOUR WRITABLE TARGET is `/mnt/ForgeRealm/GraftRepository` — edits under
`README.md` and `docs/` AUTHORIZED. This is a documentation-only order:
no code, no adapters, no GPU runs, no model workloads.

## Boundaries

- WRITE: `README.md`, `docs/*.md` (new files allowed in `docs/`).
- READ-ONLY: everything else in the repo (core/, scripts/, tests/,
  ledgers, plans — read them for evidence, never modify). Ledgers and
  plan documents are immutable under house rules: do NOT edit them; move
  content OUT of the README, never out of a ledger.
- NO git — the lead commits. NO subagents. No network.
- RED honesty: if an acceptance check cannot be met, say so with receipts;
  a documented failure is a valid result.
- No monitor-idling: work to completion, then stop.

## Mission

Execute the GraftRepository portion of
`orders/REFERENCE_PUBLIC_DOC_RECOMMENDATIONS.md` (read it in full first —
it is the spec; its "GraftRepository README", "Specific edits →
GraftRepository", "Required public status vocabulary", and "Recommended
visual hierarchy" sections are binding). Deliverables:

1. **README.md rework** — target 120–180 lines, in the reference doc's
   recommended order: plain-language product identity → six-step "how it
   works" flow (the ASCII diagram in the reference doc's "Recommended
   visual hierarchy" section is pre-approved; use it) → user-visible
   outcome and constraints → GRM certification matrix → minimal runnable
   example → current limitations → links. The suggested opening
   blockquote is pre-approved; use it. Only the strongest six-to-eight
   receipts above the fold: lossless mounting, bounded residency,
   persistence/restart, one MLA result, one GQA result, GPT-OSS E2E, and
   one honest negative. "Effectively unbounded" may survive ONLY
   immediately paired with the qualification: bounded active residency
   does not mean zero host/disk growth, universal recall, or unlimited
   addressable storage.
2. **`docs/RESULTS_INDEX.md`** — move the complete gated-result table /
   results dump out of the README into this index. Change location, not
   honesty: every receipt, failure, and negative that leaves the README
   must land here with a link to its source ledger. Do not delete or
   soften anything.
3. **Certification matrix before the quickstart** — using the reference
   doc's status vocabulary verbatim (reproduce the label table or link to
   it). Distinguish "GRM adapter" (dialect code exists) from "GRM
   certified" (deposit, route, mount, recall, persistence/restart, and
   relevant controls all passed). Label each model row strictly from
   receipts you find in-repo (the ledgers in `docs/` are the receipt
   pool); anything you cannot back with an in-repo document is
   `unconfirmed` — do NOT invent receipts, do NOT upgrade an adapter to
   certified.
4. **Quickstart relabel** — the current MiniCPM-only quickstart becomes an
   explicitly-labeled "MiniCPM3 developer example," with links to other
   adapters. State clearly that the runtime depends on Project-Tensor
   (`/mnt/ForgeRealm/Project-Tensor`, GitHub sibling repo) and is NOT a
   drop-in layer for arbitrary Hugging Face / llama.cpp / MLX / vLLM
   models.
5. **Cross-links** — the README links to Project-Tensor's `docs/APA.md`
   and `docs/SUPPORT_MATRIX.md` (being created in a parallel order; link
   by repo-relative GitHub path, check `git remote -v` for the org/repo
   names and use those; if no usable remote URL exists for Project-Tensor,
   name the repo in prose instead of a dead link and flag it in ## Done).
6. **Repo-local consistency sweep** (reference doc Phase 4): grep
   README.md and every doc you touched (plus any doc a README link
   reaches) for `any transformer`, `architecture-agnostic`,
   `supported models`, `four architectures`, `unbounded`,
   `infinite context`, `no PyTorch`, `CuPy`, `MQA`, `Hunyuan`, `HY3D`,
   `DiT` — fix each occurrence to the qualified vocabulary. List every
   hit and its disposition in ## Done. Immutable ledgers/plans are exempt
   from edits (but not from the listing).

## Gemma-4 / MQA framing (lead-supplied, binding — for any mention)

Gemma-4 12B: Engine port complete and parity-gated; APA verdict NEGATIVE
with law (closed 2026-07-04) — APA is a multi-KV-head mechanism; MQA's
single shared KV head and ≤1024-key sliding windows fail both the
statistics and economics axes; the operative APA-engaged measurement
showed net cost. GRM status for Gemma-4: label strictly per in-repo
receipts (a demonstrated mount is not lifecycle certification). Do not
present the earlier near-tie readings as current; they were a
non-engaged-scoring protocol artifact. Detail lives in Project-Tensor's
`docs/GEMMA4_MQA_ADJUDICATION.md` (parallel order) — link there.

## Acceptance checks (verify before ## Done)

- An average technical reader can explain GRM correctly from the first
  two screens: frozen model, attention state captured once, stored as
  grafts, routed per request, mounted into a bounded arena; weights
  unchanged; old history never re-prefills; active VRAM bounded.
- The certification matrix appears before the quickstart, and no row
  presents adapter-presence as certification.
- "Effectively unbounded" (and every `unbounded`/`infinite` phrasing)
  carries the operational qualification inline.
- Every receipt that left the README is reachable via
  `docs/RESULTS_INDEX.md`; nothing was deleted or softened.
- README is 120–180 lines and the full results dump is gone from it.

## Done

Your final message MUST contain, verbatim:

1. `wc -l` for README.md and every doc you created or modified.
2. The full list of files created/modified.
3. The Phase-4 sweep table: every phrase hit, file:line, disposition.
4. The certification matrix as shipped, with the receipt file each row
   cites (or `unconfirmed`).
5. Any acceptance check you could NOT satisfy, stated RED with the
   reason.
