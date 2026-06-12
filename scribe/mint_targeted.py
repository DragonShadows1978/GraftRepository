"""Error-directed mint round 1 (protocol §3 minting policy): the student
fails on SHORT FACT-DENSE docs (G4 OOD collapse, measured 2026-06-11).
Ask the teacher exactly that question: ~400 synthetic short documents,
varied templates/registers, identifier-dense, domain="factual".

Deterministic (seeded); ~12 held out. T_train trajectory: this round
adds ~30K tokens to the ~813K organic seed.

  python3 scribe/mint_targeted.py /mnt/ForgeRealm/scribe_mint_v1
"""
import hashlib
import os
import sys

import numpy as np

sys.path.insert(0, "/mnt/ForgeRealm/Project-Tensor/tensor_cuda")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SYL = ["vor", "mek", "tal", "rin", "sub", "kez", "pla", "dro", "fen", "gul",
       "mar", "tos", "bel", "kra", "nim", "osh", "pid", "rud", "sev", "wex"]
OBJECTS = ["irrigation pump", "dome motor", "crane gearbox", "film scanner",
           "proofing cabinet", "brake assembly", "monitoring well",
           "pantograph arm", "frost fan", "ladle preheater", "projector",
           "rotation bearing", "seam welder", "turf heater", "vent fan",
           "bow thruster", "mash tun", "conveyor belt", "water chiller",
           "yaw drive", "relay cabinet", "valve manifold", "sensor mast",
           "filter bank", "loading ramp"]
PLACES = ["bay", "dock", "cellar", "loft", "annex", "shed", "vault", "yard",
          "gallery", "platform"]
TEMPLATES = [
    "SERVICE RECORD. The {o} (unit {c}) received its overhaul; technician "
    "logged {n} hours runtime and signed the sheet.",
    "WORK ORDER {c}. Replace the worn bushing on the {o} in {p} {n}; "
    "parts staged; close ticket on completion.",
    "INVENTORY NOTE. Asset {c} — one {o} — moved to {p} {n}; seal intact; "
    "manifest countersigned at the gate.",
    "INCIDENT LOG. The {o} tripped its breaker at shift change; fault code "
    "{c}; reset after {n} minutes; monitoring continues.",
    "CALIBRATION CERT. Instrument {c} ({o}) passed at {n} points; next "
    "due in twelve weeks; sticker affixed.",
    "SHIPPING SLIP. Crate {c} containing the {o} departs {p} {n} at dawn; "
    "tare verified; driver acknowledged.",
    "MAINTENANCE BULLETIN. Torque spec for the {o} is {n} Nm; lot {c} "
    "fasteners only; older lots quarantined.",
    "AUDIT FINDING. The {o} under tag {c} lacks a guard rail at {p} {n}; "
    "remediation due Friday; supervisor notified.",
]


def main(root):
    import tensor_cuda as tc                              # noqa: F401
    from core.minicpm3_tc import MiniCPM3_TC, _snap
    from scribe.mint import Minter
    from tokenizers import Tokenizer as HFTok

    tok = HFTok.from_file(os.path.join(_snap(), "tokenizer.json"))
    m, info = MiniCPM3_TC.from_pretrained()
    m.extend_rope(4096)
    print(f"loaded: {info}", flush=True)
    minter = Minter(m, lambda t: tok.encode(t).ids, root)
    done = {r["sha"] for r in minter.rows()}
    rng = np.random.default_rng(4242)
    n, toks = 0, 0
    for i in range(400):
        code = ("".join(rng.choice(SYL, 2)).upper()
                + f"-{rng.integers(100, 9900)}")
        doc = TEMPLATES[int(rng.integers(len(TEMPLATES)))].format(
            o=OBJECTS[int(rng.integers(len(OBJECTS)))], c=code,
            p=PLACES[int(rng.integers(len(PLACES)))],
            n=int(rng.integers(2, 96)))
        if hashlib.sha256(doc.encode()).hexdigest()[:16] in done:
            continue
        split = "heldout" if i % 33 == 32 else "train"
        minter.mint(doc, domain="factual", split=split)
        n += 1
        toks += len(tok.encode(doc).ids)
        if n % 100 == 0:
            print(f"  {n} docs / {toks} tokens", flush=True)
    print(f"TARGETED MINT: {n} docs, {toks} tokens", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
