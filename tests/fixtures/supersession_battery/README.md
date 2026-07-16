# Supersession battery fixtures

These four static JSON sessions implement `G0-SUP` from
`docs/GRM_SUPERSESSION_PLAN.md`: short-correction/long-competitor,
multi-hop A to B to C, correction then restatement, and fresh-fact controls.

`supersedes` uses the repository's existing M5 direction: the newer node
lists older node IDs. The GPU harness converts fixture IDs to graft indices
and writes the same edge under `graft["metadata"]["supersedes"]`, plus the
existing reverse `superseded_by` edge. It deliberately does not retire the
older scripted node: L2 is a post-route mount-resolution experiment, so every
fixture revision must remain visible in route-rank diagnostics.

Run the stdlib/schema checks and CPU math/resolution tests with pytest:

```text
python3 -m pytest tests/test_grm_supersession_battery.py -q
```

The real-model battery is guarded by `--run-gpu`; invoking the file without
that flag validates fixtures and exits without loading a model.
