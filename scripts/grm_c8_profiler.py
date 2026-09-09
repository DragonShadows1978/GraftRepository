"""Opt-in EB1 inclusive-turn / exclusive-stage profiler, CPU-safe import.

Prior art: Python trace API (Python/Guido van Rossum, 1991 onward), nested
exclusive profiling (gprof, Graham/Kessler/McKusick, 1982); NVIDIA CUDA events
(2007 onward) and stream-ordered pool high water marks (CUDA 11.2, 2020).
External dates/attribution unverified — lead to check: 'gprof 1982', 'Python
sys.settrace', 'CUDA event elapsed time memory pool UsedMemHigh'. Taken: call
stack attribution, monotonic wall, synced events and pool high water counters.
Ours: EB1 source-bound stage map. No new profiling algorithm claimed.
"""
from __future__ import annotations
import ast
import ctypes as C
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
STAGES = ('lexical_scan', 'route', 'admission', 'cold_fetch', 'seat_mount',
          'prefill', 'decode', 'deposit_harvest', 'fold', 'other')
# Local prior art: EB1/C2/RS3 (GRM contributors, 2026). Reuse exact code
# boundaries; fold and deposit forwards stay in their owning operation.
METHODS = {
    'core/graft_arena.py': {
        'lexical_scan': ('_rare_tokens', '_query_content_tokens', '_query_lex_tokens',
                         '_node_text_tokens', '_query_lex_needs_rescore', '_lex_bonus'),
        'route': ('route', '_probe_key'),
        'admission': ('_resolve_revision_mounts', '_rs3_seat_plan',
                      '_split_unseatable', '_identifier_bearing_children'),
        'cold_fetch': ('_ensure_h',),
        'seat_mount': ('swap', '_set_injection_host', '_rs3_rotate_injection',
                       '_commit_native_mount', 'eb1_begin_turn'),
        'prefill': ('_forward',),
        'deposit_harvest': ('deposit', 'deposit_from_cache', '_harvest',
                            'deposit_deferred_turn'),
        'fold': ('consolidate',),
        'other': ('_finish_attempt',),
    },
    'core/graft_repository.py': {
        'cold_fetch': ('_load_node', '_read_payload_file', '_ensure_host_payload'),
        'fold': ('_librarian', '_fold_once', 'fold_pending'),
        'deposit_harvest': ('_guard_deposit_range', 'apply_memory_command'),
    },
    'core/grm_admission.py': {'admission': ('decisive_admission_profile',
        'plan_priority_fit', 'identifier_unbound_abstention'),
        'lexical_scan': ('is_identifier_binding', 'normalized_words')},
    'scripts/grm_e2e_session.py': {'admission': ('_budget_fit_mounts',)},
}


def source_map(root=ROOT):
    """Read source with AST; no import, CUDA initialization or product edits."""
    result = {}
    for relative, groups in METHODS.items():
        path = (root / relative).resolve()
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            label = next((s for s, names in groups.items() if node.name in names), None)
            phases = None
            if relative == 'core/graft_arena.py' and node.name == '_attempt':
                lines = path.read_text().splitlines()
                start = next(i + 1 for i in range(node.lineno - 1, node.end_lineno)
                             if lines[i].strip().startswith('prompt_ids ='))
                decode = next(i + 1 for i in range(start, node.end_lineno)
                              if lines[i].strip().startswith('out = [int(row.argmax())]'))
                label = 'seat_mount'
                phases = (start, decode)
            elif relative == 'core/graft_arena.py' and node.name == '_resume_attempt':
                label = 'decode'
            if label:
                # co_firstlineno includes the first decorator, when present.
                first = min([node.lineno] + [x.lineno for x in node.decorator_list])
                result[(str(path), first, node.name)] = (label, phases)
    return result


class CpuMeter:
    gpu = False
    def begin(self):
        return None
    def end(self):
        return {'gpu_ms': None, 'peak_pool_used_bytes': None,
                'device_used_boundary_peak_bytes': None}
    def close(self):
        pass


class CudaMeter:
    """CUDA event elapsed interval, NOT kernel active time; includes idle gaps.

    Memory peak is exact for the current device's DEFAULT ASYNC POOL only.
    Legacy allocations/other pools are not covered. Device used memory at
    boundaries is a sampled lower bound, never labeled a whole-device peak.
    Counter resets alter measurement counters only, not allocation policy.
    """
    gpu = True
    def __init__(self, library='/usr/local/cuda/lib64/libcudart.so'):
        self.lib = C.CDLL(library)
        self.path = str(Path(library).resolve())
        self.start, self.stop, self.pool = C.c_void_p(), C.c_void_p(), C.c_void_p()
        self._call('cudaEventCreate', C.byref(self.start))
        self._call('cudaEventCreate', C.byref(self.stop))
        device = C.c_int()
        self._call('cudaGetDevice', C.byref(device))
        self._call('cudaDeviceGetDefaultMemPool', C.byref(self.pool), device)
    def _call(self, name, *args):
        status = getattr(self.lib, name)(*args)
        if status:
            raise RuntimeError(f'{name} failed: CUDA status {status}')
    def used(self):
        free, total = C.c_size_t(), C.c_size_t()
        self._call('cudaMemGetInfo', C.byref(free), C.byref(total))
        return total.value - free.value
    def begin(self):
        self._call('cudaDeviceSynchronize')
        self.initial_used = self.used()
        zero = C.c_uint64(0)
        self._call('cudaMemPoolSetAttribute', self.pool, C.c_int(8), C.byref(zero))
        self._call('cudaEventRecord', self.start, C.c_void_p())
    def end(self):
        # Device sync before stop captures work on nondefault streams, too.
        self._call('cudaDeviceSynchronize')
        self._call('cudaEventRecord', self.stop, C.c_void_p())
        self._call('cudaEventSynchronize', self.stop)
        elapsed, high = C.c_float(), C.c_uint64()
        self._call('cudaEventElapsedTime', C.byref(elapsed), self.start, self.stop)
        self._call('cudaMemPoolGetAttribute', self.pool, C.c_int(8), C.byref(high))
        return {'gpu_ms': float(elapsed.value), 'peak_pool_used_bytes': high.value,
                'device_used_boundary_peak_bytes': max(self.initial_used, self.used())}
    def close(self):
        self._call('cudaEventDestroy', self.start)
        self._call('cudaEventDestroy', self.stop)


class TurnProfiler:
    """Flag OFF by default. run() returns the original value unchanged.

    Single-thread synchronous serving only. An existing Python tracer is
    refused instead of replaced. Instrumentation overhead is retained in
    total wall and reported separately; nothing is subtracted to make a win.
    """
    def __init__(self, *, enabled=False, meter=None, mapping=None):
        self.enabled = enabled
        self.meter = meter if meter is not None else CpuMeter()
        self.mapping = source_map() if enabled and mapping is None else (mapping or {})
        self.receipt = None

    def _transition(self, stage):
        if stage == self.stage:
            return
        boundary = time.perf_counter_ns()
        values = self.meter.end()
        end = time.perf_counter_ns()
        row = self.rows[self.stage]
        row['wall_ms'] += (end - self.started) / 1e6
        row['segments'] += 1
        for key, value in values.items():
            if value is not None:
                row[key] = (row[key] or 0) + value if key == 'gpu_ms' else max(row[key] or 0, value)
        if stage is not None:
            self.meter.begin()
        self.overhead_ns += time.perf_counter_ns() - boundary
        self.started = time.perf_counter_ns()
        self.stage = stage

    def _effective(self):
        stages = [entry[1] for entry in self.stack]
        # A fold owns all its nested work; a deposit owns route-key harvest.
        for owner in ('fold', 'deposit_harvest'):
            if owner in stages:
                return owner
        # _forward inherits decode; standalone _forward is prefill.
        if stages and stages[-1] == 'prefill' and 'decode' in stages:
            return 'decode'
        return stages[-1] if stages else 'other'

    def _trace(self, frame, event, arg):
        if event == 'call':
            code = frame.f_code
            spec = self.mapping.get((code.co_filename, code.co_firstlineno, code.co_name))
            if spec is None:
                return None
            label, phases = spec
            self.stack.append([frame, label, phases])
            self.rows[label]['calls'] += 1
            self._transition(self._effective())
            frame.f_trace_lines = phases is not None
            return self._trace
        if event == 'line':
            entry = self.stack[-1]
            if entry[0] is frame and entry[2]:
                prefill, decode = entry[2]
                entry[1] = ('decode' if frame.f_lineno >= decode else
                            'prefill' if frame.f_lineno >= prefill else 'seat_mount')
                self._transition(self._effective())
        elif event == 'return':
            if not self.stack or self.stack[-1][0] is not frame:
                raise RuntimeError('profiler stack mismatch')
            self.stack.pop()
            self._transition(self._effective())
        return self._trace

    def run(self, function, *args, **kwargs):
        if not self.enabled:
            return function(*args, **kwargs)
        if sys.gettrace() is not None:
            raise RuntimeError('existing trace: profiler refuses to replace it')
        self.rows = {s: dict(wall_ms=0.0, gpu_ms=0.0 if self.meter.gpu else None,
                            peak_pool_used_bytes=0 if self.meter.gpu else None,
                            device_used_boundary_peak_bytes=0 if self.meter.gpu else None,
                            calls=0, segments=0) for s in STAGES}
        self.stack, self.stage, self.overhead_ns = [], 'other', 0
        outer = time.perf_counter_ns()
        self.meter.begin()
        self.started = time.perf_counter_ns()
        success = False
        try:
            sys.settrace(self._trace)
            value = function(*args, **kwargs)
            success = True
            return value
        finally:
            sys.settrace(None)
            self._transition(None)
            total = (time.perf_counter_ns() - outer) / 1e6
            assigned = sum(x['wall_ms'] for x in self.rows.values())
            # Gaps in boundary instrumentation belong to OTHER; no lost wall.
            self.rows['other']['wall_ms'] += total - assigned
            self.receipt = {'status': 'COMPLETE' if success else 'RED_EXCEPTION',
                'evidence_class': 'end-to-end gate' if self.meter.gpu else 'unit test',
                'turn_wall_ms': total, 'stages': self.rows,
                'measured_boundary_overhead_ms': self.overhead_ns / 1e6,
                'trace_dispatch_overhead': 'included, not independently measured',
                'gpu_time_scope': 'synced CUDA event intervals, not kernel active time',
                'memory_scope': 'default async pool high water; boundary device samples',
                'full_device_peak_bytes': None}


def stats(values):
    # Prior art: nearest-rank empirical quantiles, established descriptive
    # statistics (no specific origin known to me); no interpolation invented.
    ordered = sorted(values)
    if not ordered:
        return {'mean': None, 'p50': None, 'p95': None}
    return {'mean': sum(ordered) / len(ordered),
            'p50': ordered[math.ceil(.5 * len(ordered)) - 1],
            'p95': ordered[math.ceil(.95 * len(ordered)) - 1]}


def summarize(rows, *, minimum=30):
    if len(rows) < minimum or any(r['status'] != 'COMPLETE' for r in rows):
        return {'status': 'NOT_MEASURED_OR_INCOMPLETE', 'turns': len(rows),
                'decode_fraction': None, 'nondecode_fraction': None, 'decision': None}
    for row in rows:
        vals = [row['turn_wall_ms']] + [row['stages'][s]['wall_ms'] for s in STAGES]
        if any(not math.isfinite(v) or v < 0 for v in vals) or vals[0] <= 0:
            raise ValueError('invalid timing')
        if not math.isclose(sum(vals[1:]), vals[0], abs_tol=1e-5):
            raise ValueError('stage wall does not conserve turn wall')
    wall = sum(r['turn_wall_ms'] for r in rows)
    fraction = sum(r['stages']['decode']['wall_ms'] for r in rows) / wall
    # Prior art: Amdahl (1967), fixed-component end-to-end bound. The 50%
    # allocation rule is David/GRM Scout (2026), not a fitted threshold.
    stages = {}
    for stage in STAGES:
        stages[stage] = {}
        for metric in ('wall_ms', 'gpu_ms', 'peak_pool_used_bytes',
                       'device_used_boundary_peak_bytes'):
            values = [r['stages'][stage][metric] for r in rows]
            stages[stage][metric] = stats(values) if all(v is not None for v in values) else stats([])
    return {'status': 'COMPLETE', 'turns': len(rows), 'stages': stages,
            'turn_wall_ms': stats([r['turn_wall_ms'] for r in rows]),
            'decode_fraction': fraction, 'nondecode_fraction': 1 - fraction,
            'per_turn_decode_fraction': stats([r['stages']['decode']['wall_ms'] / r['turn_wall_ms'] for r in rows]),
            'decision': 'session routing/admission wins' if fraction < .5 else
                        'APA decode integration competitive; lead reports both',
            'decision_scope': 'instrumented turn wall, including tracing/sync overhead',
            'apa_sp_counterfactual_turn_saving_fraction': fraction * (82.6 - 84.2) / 82.6,
            'apa_counterfactual_scope': 'reasoning only: SP5 S=2048 ratio transferred; NOT measured at W96'}
