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
