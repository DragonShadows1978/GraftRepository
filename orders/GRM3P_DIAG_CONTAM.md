# ORDER GRM3P-DIAG-CONTAM — convict the contaminator-wash mechanism (Opus 4.8 seat)

WRITABLE TARGET /home/vader/GraftRepository-three-pass. Production +
merge-train READ-ONLY. No git. No subagents. GPU under
`flock -w 7200 /tmp/forge-gpu.lock`. Read
docs/GRM3P_DIAG_UNBOUNDED_REPORT.md fully; append, never edit.

## Frozen theory + prediction (registered before any run)

THEORY (lead): the miss mechanism is contaminator-side — the turn-3
cypher deposit (node 3), recency-mounted and perpetually LRU-hot at
budgets >=16MB, has never been pack/unpack-canonicalized; its device
payload hijacks the read (attractor class). PREDICTIONS:
P1: per-probe, pass <=> the Vortex-carrying node in the mounted set
    was pack->evict->rehydrate washed before that turn (correlation
    keyed on the CONTAMINATOR, not the source).
P2: at 8MB, node 3 IS evicted+rehydrated between turn 3 and turn 5;
    at 16MB it is NOT (telemetry, not reconstruction).
P3: node 3's mount-time device payload differs between the 8MB and
    16MB legs (key statistics / bytes); the packed host copies do not.
Any receipt violating P1-P3 is falsifying and reported as such.

## Work items

1. **Per-node paging telemetry (additive, default-off).** Add an
   opt-in env/flag that logs every payload page-in/eviction/upload
   with node id + turn + step to a session JSONL. MUST NOT alter the
   default path: after wiring, rerun default smoke — transcript sha
   must equal 68da84b8824eb79a14f65a692a602215bc753... STOP: use the
   exact registered sha 68da84b8824eb79a14f65a692a602215bc53d84232cd041002410f51657bba4f.
2. **Dual mount-time snapshot (opt-in, same rail).** When enabled,
   dump the dequantized device payload (or exact device bytes) of
   every RECENCY-mounted node at probe turns.
3. **Runs**: with telemetry+snapshot enabled — 8MB, 16MB, unbounded,
   `--mode full --turn-pipeline single`, env GRM_GQA_CUDA_ROUTE=1
   GRM_GRAFT_STORAGE_BITS=8, all else default. Scorecards must
   reproduce the ladder (9/9, 7/9, 7/9) — if telemetry changes ANY
   scorecard, that is a RED result to report, not to debug around.
4. **Analysis**: P1 correlation table over all probes/legs; P2 node-3
   lifecycle timeline per leg; P3 payload diff (max/mean per-layer
   dequant delta + key-norm stats) node 3 at t5, 8MB vs 16MB vs
   unbounded, plus packed-host-copy control. Verdict per frozen
   predictions.

## Honesty
No fixes. RED/falsifying rows verbatim. No prediction edits after
results. Lead verifies against disk.

## Done
Verbatim: default-smoke byte-identity line, three scorecard lines,
P1 table, P2 timelines, P3 diff numbers, verdict, session dirs,
residuals.
