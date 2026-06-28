# GRM C++ Runtime Scaffold

This is the C++ host-runtime target for the Python RAM-first semantics in
`core/graft_repository.py`.

Implemented here:

- `DialectDescriptor`
- `HostGraftStore`
- `RouterIndex`
- `DirtyQueue`
- `DurabilityWriter`
- `DeviceArena` swap/evict planner and host tensor swap/evict references
- dependency-free C ABI in `grm_runtime_c.h`

The CUDA arena is intentionally not implemented in this scaffold yet. The rule
is the same as the Python runtime: host RAM payloads are authoritative; device
copies are disposable mounts. The host tensor swap and evict operations are the
byte-level oracles for later CUDA kernels.
