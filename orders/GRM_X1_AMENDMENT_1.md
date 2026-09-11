# GRM-X1 amendment 1 (lead, 2026-09-08) — first oracle cell RED on the card: `payload_digest` on a node without a materialized `h`

Your r1 is committed (ad5741b). Lead-run: `oracle_m1_s0` RED after 34.6 s
(receipt `artifacts/grm_x1/receipts/gpu_oracle_m1_s0_0c1bf6…json`);
the registered stop then blocked every successor:

```
scripts/grm_x1_gpu.py:169 run_cell -> hashes = {i: payload_digest(nodes[i]) ...}
scripts/grm_x1_gpu.py:121 payload_digest -> for layer in node["h"]:
TypeError: 'NoneType' object is not iterable
```

On the real EB1 repository, nodes can carry `h = None` (cold / paged /
manifest-only nodes; see `core/graft_repository.py` reload path and the
capture-provenance finding in `docs/GRM_SCOUT_2026-09-08.md` Part B #3).
Your digest must not assume a resident tensor. Fix in your scripts only:
hash the node by its durable identity (manifest sha / page ids) or
materialize through the repository's own accessor, whichever the
production read path uses; add a CPU test with a `h=None` node; re-run
your CPU gates; refresh `lead_commands.txt`. Same rules (no git, no
subagents, no background waits, never kill, additive flag-OFF only).
Reasoning effort for this and every later amendment: **high**.

## Done (verbatim)
1. The fix (file/lines) and the new test name.
2. CPU gate results; refreshed lead commands.
3. RED; process safety; model id and effort.
