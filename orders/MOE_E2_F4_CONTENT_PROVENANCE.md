# ORDER MOE-E2-F4 — provenance must hash array content, not npz bytes

WRITABLE TARGET `/mnt/ForgeRealm/GraftRepository`; same rules as prior F rounds.

Defect (lead-diagnosed): `prepare` re-ran during F3's CPU eval_abi
self-run and regenerated `prepared_windows.npz` — arrays identical,
but np.savez zip bytes differ (member timestamps) → manifest
prepared_windows sha changed (c4f61cdf… → adfaefa8…) → ALL r4 capture/
pair receipts now fail `verified_key_captures` (r5 log). The r4 work
(key captures, 64 pairs, trained adapter, G2/G3 GREEN) is content-valid
and must be salvaged, not regenerated.

Work:
1. Provenance hashing moves to CONTENT: per-array dtype-normalized
   sha256 (and a combined content digest) instead of npz file bytes,
   everywhere prepared_windows provenance is checked or recorded.
2. `prepare` becomes strictly idempotent: if the manifest exists and
   array CONTENT digests match, do not rewrite npz/manifest.
3. Migration: re-validate ALL existing receipts (key captures, pairs,
   train, eval) against content digests; where a receipt stores only
   the old byte-sha, accept iff its recorded arrays match content
   (record a migration note in each touched receipt-side ledger — do
   not mutate original receipts; a sidecar validation file is fine).
4. Self-run: your contracts + a CPU proof that r4's narrative_00 and
   at least one pair capture validate under the new scheme.
5. Regenerate chain scripts only if invocation shapes changed.

Done: mechanism confirmation, diff summary, salvage proof (which r4
artifacts validate), receipts, anything not done.
