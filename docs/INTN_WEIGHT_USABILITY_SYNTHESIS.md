# INT3 Weight Usability Synthesis

The Qwen3.5-9B INT3 question is no longer only a memory question. The model
loads, and the small PPL smoke suggested that INT3 may be damaged but not
collapsed. The open question is whether the damage is acceptable for work.

The first usability gate therefore asks a narrow operational question: can INT3
still extract planted facts under greedy decode in cases where INT4 should pass?
This is intentionally simpler than a broad benchmark. If the model cannot
recover exact facts from a short document, the PPL savings are not worth using.

The gate is ready but not yet run. Results will be appended after the broad 9B
PPL sweep finishes.

The first usability gate is now in. INT4 passed the planted-fact exact
extraction test in both standard attention and APA r0.15: 5/5 hits in each
mode. INT3 also passed: 5/5 hits in standard attention and 5/5 hits in APA
r0.15. The generated answers were exact or capitalization-only variants of the
expected values, with no degeneration.

This changes the interpretation of INT3 on Qwen3.5-9B. The broad PPL gate says
INT3 has a real quality cost of about +23% PPL versus INT4. The usability gate
says that cost does not automatically make the model useless for simple exact
fact extraction. So the answer is not "INT3 is free" and not "INT3 is dead."
The current evidence says INT3 is a plausible memory-saving mode for constrained
retrieval/extraction-style work, but it needs a harder battery before being
trusted for general generation or production use.

The next gate should increase difficulty rather than repeat the same test:
longer source text, more distractors, multiple simultaneous facts, questions
that require selecting among similar values, and open-ended answers that expose
reasoning or fluency degradation.
