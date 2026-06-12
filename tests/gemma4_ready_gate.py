"""READY-TO-WORK gate for the Gemma 4 APA-GRM stack — the only valid
done signal: the corpus driver's OWN shard loop (run_shard, its
generate(), its validate(), its repair path) authoring a real
fact-shard through the shim. PASS = the wave could run on this stack
today.

Assumes the shim is up:  python3 scripts/gemma4_server.py 11436

  python3 tests/gemma4_ready_gate.py
"""
import sys
import time

sys.path.insert(0, "/mnt/ForgeRealm/GRAPA-Native-LLM")
from corpus.templates import local_wave as lw               # noqa: E402

lw.OLLAMA = "http://127.0.0.1:11436/api/generate"

# one real fact-shard call, built exactly as the driver's plan builds it
REL = "STORED_IN"
calls = [(
    10, {"kind": "fact", "relation": REL, "tags": []},
    "ONE complete declarative sentence asserting the relation. "
    f"Mix tenses. {lw.FACT_REGISTERS['a']}\n\n{lw.rel_block(REL)}",
    "{A} exactly once and {B} exactly once",
    lw.FACT_EX[REL],
)]

t0 = time.perf_counter()
records, asked = lw.run_shard("ready_gate_test", calls, False,
                              "gemma4:12b-apa-grm", rounds=3,
                              existing_keys=set())
dt = time.perf_counter() - t0

print(f"\naccepted {len(records)}/{asked} validated templates "
      f"in {dt:.0f}s", flush=True)
for r in records[:5]:
    print("  ", r["text"], flush=True)
ok = len(records) >= 8                       # driver-grade yield
print(f"READY GATE: {'PASS' if ok else 'FAIL'}", flush=True)
print("DONE", flush=True)
