# GRM-WC1 receipt path relocation index (lead, 2026-09-09)

`artifacts/grm_wc1_opus/grm_wc1_results.json` and its nested receipts
carry 31 literal `artifacts/grm_wc1/...` paths from the worktree the
campaign ran in. The bytes live under `artifacts/grm_wc1_opus/` in the
canonical checkout. GRM-C6 verified every one of the 31 candidates by
SHA-256 against the hash recorded in the aggregate: **31/31 match**
(`docs/grm_c6_review/wc1_path_resolution.json`, entries
`original` → `candidate`, `sha256` = `expected`, `matches: true`).

Rule: resolve `artifacts/grm_wc1/<rest>` as
`artifacts/grm_wc1_opus/<rest>` when reading WC1 receipts. The old
receipt bytes are unchanged; this index is the relocation record, not a
rewrite. Any operational resolver (code that follows these paths) is a
seat order, not a docs edit (C6 rank 1).
