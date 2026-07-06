# INT2/INT3 Weight Model PPL Synthesis

The prior low-bit work proved that the INT2 and INT3 packed kernels exist and
can match their reference path. That is not enough. The actual question is
whether a real model can be loaded with INT3 or INT2 weights and retain usable
perplexity under APA at practical refine levels.

The first real target is Qwen3.5-2B. It is local, unquantized, and already
served by the same `Qwen35_TC` adapter family used for the 9B work. That makes
it the right bring-up target: if the wrapper, memory reporting, and PPL loop do
not work on 2B, scaling to 9B would only add noise.

The important implementation gap is narrow and concrete: `QuantLinearTC` is a
real model weight wrapper, but it is INT4-specific. The native Project-Tensor
INT2/INT3 kernels exist; they need to be routed into the same adapter path so
the model, not a synthetic matrix, is being tested.

Failure remains a valid outcome. If INT3 or INT2 collapse in PPL, if a specific
layer class fails, or if the model cannot load/run within memory, that is the
result to record.

The first real-model gate is now in. Qwen3.5-2B loaded successfully at INT4,
INT3, and INT2. There was no OOM layer in this 2B load-only pass. The memory
ordering was as expected: INT4 loaded at about 1433 MiB, INT3 at 1209 MiB, and
INT2 at 953 MiB. INT3 took longer to load because INT3 packing is the slowest
low-bit layout.

The first PPL gate scored one real 512-token WikiText window, with the last 255
tokens contributing to NLL. This is a small gate, but it is a real model run.
INT4 landed around 9.1 PPL across standard and APA settings. INT3 landed around
18 PPL, roughly doubling the loss surface compared with INT4. INT2 collapsed:
PPL was in the 34k-37k range. APA refine levels at 0.15, 0.10, and 0.05 did
not rescue the low-bit weight damage.

So the first answer is not an OOM story. It is a quality story. INT3 and INT2
fit in memory on Qwen3.5-2B, but INT3 already has a major PPL penalty and INT2
is functionally unusable on this gate. The next step is to broaden the gate:
more windows, then the 9B target if the 2B results remain stable.

The broader 2B run confirmed the direction with 6138 scored tokens per setting.
INT4 sat at about 12.41 PPL across standard and APA refine levels. INT3 sat at
about 28.2-28.3 PPL, so its quality loss is not a single-window artifact. INT2
remained collapsed, around 61k-63k PPL. The APA refine sweep again did not
repair the damage from low-bit weights.

This closes the first real 2B answer: the tested low-bit weights do not hit an
OOM wall on Qwen3.5-2B. INT3 and INT2 are memory-feasible, but INT3 is already
too damaging for a serious operating point on this affine quantization path,
and INT2 is unusable. The next unresolved axis is whether Qwen3.5-9B behaves
similarly or exposes a different load/OOM boundary.
