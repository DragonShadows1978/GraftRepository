# SCRIBE — results ledger (house format: every gate, dated, raw numbers)

| Gate | Date | Verdict | Measurement |
|---|---|---|---|
| G0 instrument | 2026-06-11 | **PASS** | Minter (scribe/mint.py) pipeline Δ = 0.00000 vs direct harvest (3 docs, 3 domains); graft-vs-in-context floor 0.3125 max\|Δlogit\|; flips only at ≤1-ULP tie margins. Harness: tests/test_scribe_g0.py |
| Fit probe | 2026-06-11 | **co-resident OK** | target+student+Adam peak 4,223 MB / 8,192 (S=224); functional loss needs NO swap mode. ARM-L 45.7M / ARM-S 31.5M params |
| Throughput (§5 ratio) | 2026-06-11 | measured | **C_student/C_target: ARM-L 0.008, ARM-S 0.022** (student 0.03/0.10 ms/tok vs teacher prefill 4.2/4.6 ms/tok) — minting at 1-2%% of teacher cost; cost is architecture-set, training buys fidelity |
| Resume-from-kill | 2026-06-11 | **PASS** | warm trainer: ckpt -> hard-kill -> fresh-process resume, steps 4-7 BIT-IDENTICAL to unbroken run; cursor + per-layer error EMA continuous |

**PHASE 0 EXIT: 2026-06-11 — all criteria met** (G0 green; fit mode
chosen+verified co-resident; thresholds registered; resume-from-kill
green). Phase 1 (ARM-L floor) is next: needs the minted seed set.
| G1 (ARM-L) | 2026-06-11 | recorded | per-layer error: L0 0.002 (solved), BULGE peak L36 ~0.20, decline to ~0.17 at L61 — **H3's monotone form falsified**: contextualization hump, not growth. Full trajectory in arm_l ckpts |
| G2 (ARM-L floor) | 2026-06-11 | FAIL (floor recorded) | top-1 51-53%%, KL 0.85-1.02 vs thresholds 90%%/0.5 |
| G3 (ARM-L floor) | 2026-06-11 | FAIL (floor recorded) | predicted 0/10 vs exact 10/10 in-context 10/10 — **H2 strong form at the floor: content collapses, logits half-right** |
| G4 (ARM-L floor) | 2026-06-11 | INSTRUMENT FLAW | same-template sibling docs broke centroid routing for EXACT too (5/20@1) — corpus-100 regime reproduced; predicted trailed by only 3/20. Gate re-instrumented with diverse-topic docs BEFORE any ARM-S run |

**Phase 1 verdict: floor recorded. The problem is NOT easier than
believed — linear maps solve routing-adjacent layers, cannot write
readable content.** ARM-S (Phase 2) is the real test.
| G1 (ARM-S) | 2026-06-11 | recorded | bulge CRACKED vs floor: L31 0.168 vs floor plateau ~0.19; L0 0.042 (worse than floor's 0.002 — shared trunk vs 62 free linears); window 512 (OOM at 1024, measured) |
| G2 (ARM-S warm) | 2026-06-11 | FAIL, closing | top-1 75-83%%, KL 0.48-0.62 (floor: ~52%%/0.9); held-out MATH is the BEST domain (83.3%%/0.483) — no dist-shift penalty |
| G3 (ARM-S warm) | 2026-06-11 | FAIL 0/10 | H2 strong form persists through warm phase: content unreadable while logits 80%% right. The registered answer: L-func |
| G4 (ARM-S warm, fixed instrument) | 2026-06-11 | FAIL | exact 14/20@1 18/20@3 (instrument healthy); predicted 2/20@1 4/20@3 — H1 failing at warm phase; suspect systematic latent-space bias (affine calibration diagnostic next) |

**Phase 2 (warm) verdict: attention buys fidelity (G2 +30pts), not yet
content or addressability. Next: bias diagnostic -> functional loss.**
| Variance-collapse diagnostic | 2026-06-11 | root cause found | warm Huber regresses to the mean: pred-pred centroid cos 0.989 (exact 0.863), match-mismatch margin 0.009. CENTERING exposes the buried doc signal: margin 0.263, recall 1/12 -> 7/12 on held-out organic chunks |
| G4 centered (ARM-S warm) | 2026-06-11 | FAIL + finding | centered router: EXACT 20/20@1 (perfect — centering improves the production router, transferable to GraftRepository); predicted 4/20@1 7/20@3 on SYNTHETIC short docs vs 7/12 organic — predicted addressability is distribution-sensitive (§7 door in routing). Error-directed minting of short fact-dense docs is the registered answer |

**Standing next steps: (1) L-func build (content/G3 — the registered
"real loss"); (2) error-directed mint round: short fact-dense docs;
(3) port centered routing to the production router (exact-graft win).**
| Targeted mint round 1 | 2026-06-11 | done | 400 synthetic short fact-dense docs (8 templates, seed 4242), domain=factual, 16,184 tokens, ~1/33 heldout; warm top-up epoch over expanded set completed (4,928 steps total) |
| L-func training curve | 2026-06-11 | recorded | 1200 steps, lr 1e-4, pool = 404 short rows (~96%% factual): KL ema 4.52 -> 4.22, PLATEAU 4.15-4.26 from step ~250 (epoch-2 revisits no better than first visit); Huber anchor monotone 0.27 -> 0.20. KL found little traction where Huber kept finding it |
| G2 (ARM-S L-func) | 2026-06-11 | FAIL + split finding | **factual 88.9%%/KL 0.116 — best per-domain KL ever measured, 1.1pt from top-1 threshold** vs ALL organic domains REGRESSED vs warm: math 68.1/0.994 (was 83.3/0.483), narrative 69.4/0.973 (75.0/0.618), research 77.8/1.026 (83.3/0.520), technical 59.7/1.192 (75.0/0.580). L-func on a factual-skewed pool traded general fidelity for in-distribution fidelity — catastrophic-forgetting-shaped; Huber anchor on the SAME skewed rows does not anchor organic domains |
| G3 (ARM-S L-func) | 2026-06-11 | FAIL 0/10 | the main bet did not pay: training on what the reader reads did NOT unlock fresh-doc needle readback (in-context 10/10, exact 10/10, predicted 0/10 — unchanged through floor, warm, L-func) |
| G4 (ARM-S L-func) | 2026-06-11 | FAIL | exact 20/20@1 (centered router stays perfect); predicted 4/20@1 (= warm), @3 5/20 (down from 7/20) — L-func bought zero addressability |

**L-func round-1 verdict: the loss works mechanically (engine VJP
verified; factual KL 0.116 proves the reader CAN be satisfied by
predicted latents in-distribution) but content generation (G3) and
addressability (G4) did not move, and the factual-skewed pool cost
every organic domain. Open levers, in evidence order: (1) balanced
L-func pool (cap factual fraction / include organic ≤512-tok rows);
(2) training-KL decomposition — ema 4.2 vs heldout-factual 0.116
says the readback probe's ANSWER positions carry ~all the loss, i.e.
the student may need capacity/architecture (ARM-T) not more steps;
(3) lr/step sweep before any architecture call.**
| L-func first run (engine unlock) | 2026-06-11 | RUNNING | first execution surfaced TWO engine gaps, both fixed in Project-Tensor: (1) int4_linear was inference-only — graph cut at every INT4 projection, caught by one-time zero-grad tripwire (|g|₁=0); VJP added: dx = g @ W_kn^T, weight re-dequantized transiently in backward, nothing weight-sized retained — verified vs analytic to 1.5e-07 rel. (2) backward kept ALL intermediate grads alive until graph death — 62-layer reader backward OOM'd 8 GB at step 2 (after AdamW moments allocated); fix: free each op node's grad after its grad_fn consumes it. Peak now 6,637 MiB (1.5 GB headroom). Student injection cast to bf16 (reader compute dtype; astype is differentiable). Tripwire |g|₁ = 9.929e+03 bit-identical across relaunches. KL ema 4.52 / anchor 0.224 at step 50 |
