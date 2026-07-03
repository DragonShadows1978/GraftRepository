#!/usr/bin/env python3
"""Build and run a ThreadSanitizer stress gate for native GRM router snapshots."""

from __future__ import annotations

import argparse
import platform
import shutil
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


HARNESS = r'''
#include "grm_runtime_c.h"

#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr std::uint64_t kKvHeads = 2;
constexpr std::uint64_t kHeadDim = 3;
constexpr std::uint64_t kKeyTokens = 2;
constexpr std::uint64_t kQueryHeads = 4;
constexpr std::uint64_t kQueryTokens = 4;

bool check_rc(grm_store_handle* handle, int rc, const char* label) {
  if (rc == 0) {
    return true;
  }
  const char* error = grm_store_last_error(handle);
  std::cerr << label << " failed: " << (error == nullptr ? "<null>" : error)
            << "\n";
  return false;
}

std::vector<float> route_key(std::uint64_t node_id, std::uint64_t step) {
  std::vector<float> key(kKvHeads * kKeyTokens * kHeadDim, 0.0F);
  for (std::uint64_t kv = 0; kv < kKvHeads; ++kv) {
    for (std::uint64_t tok = 0; tok < kKeyTokens; ++tok) {
      for (std::uint64_t d = 0; d < kHeadDim; ++d) {
        const auto idx = ((kv * kKeyTokens) + tok) * kHeadDim + d;
        const auto v = static_cast<float>(
            ((node_id + 3 * step + 5 * kv + 7 * tok + 11 * d) % 17) - 8);
        key[static_cast<std::size_t>(idx)] = v / 8.0F;
      }
    }
  }
  return key;
}

std::vector<float> query() {
  std::vector<float> q(kQueryHeads * kQueryTokens * kHeadDim, 0.0F);
  for (std::uint64_t h = 0; h < kQueryHeads; ++h) {
    for (std::uint64_t tok = 0; tok < kQueryTokens; ++tok) {
      for (std::uint64_t d = 0; d < kHeadDim; ++d) {
        const auto idx = ((h * kQueryTokens) + tok) * kHeadDim + d;
        const auto v = static_cast<float>(((13 * h + 5 * tok + d) % 19) - 9);
        q[static_cast<std::size_t>(idx)] = v / 9.0F;
      }
    }
  }
  return q;
}

}  // namespace

int main(int argc, char** argv) {
  const std::uint64_t nodes = argc > 1 ? std::strtoull(argv[1], nullptr, 10) : 48;
  const std::uint64_t iterations =
      argc > 2 ? std::strtoull(argv[2], nullptr, 10) : 1200;
  const std::uint64_t readers =
      argc > 3 ? std::strtoull(argv[3], nullptr, 10) : 4;

  grm_store_handle* handle = grm_store_create_gqa(
      "Qwen3.5_TC", 24, 2048, 1024, 3,
      static_cast<int>(kKvHeads), static_cast<int>(kHeadDim));
  if (handle == nullptr) {
    std::cerr << "failed to create GQA store\n";
    return 2;
  }

  std::vector<std::uint64_t> ids;
  ids.reserve(static_cast<std::size_t>(nodes));
  for (std::uint64_t i = 0; i < nodes; ++i) {
    std::uint64_t node_id = 0;
    const std::string text = "router tsan node " + std::to_string(i);
    if (!check_rc(handle,
                  grm_store_add_node(
                      handle, text.c_str(), 1, nullptr, 0, &node_id),
                  "add_node")) {
      grm_store_destroy(handle);
      return 3;
    }
    ids.push_back(node_id);
    auto key = route_key(i, 0);
    const std::string lexical = "node-" + std::to_string(i);
    if (!check_rc(handle,
                  grm_store_set_route_multi(
                      handle, node_id, key.data(), 1,
                      static_cast<std::uint64_t>(key.size()),
                      lexical.c_str()),
                  "set_route_multi")) {
      grm_store_destroy(handle);
      return 4;
    }
  }

  const auto q = query();
  std::uint64_t out[16] = {};
  std::uint64_t out_count = 0;
  if (!check_rc(handle,
                grm_store_route_gqa(
                    handle, q.data(), kQueryHeads, kQueryTokens, kHeadDim,
                    "", "", "", "", "", 8, out, 16, &out_count),
                "warm route_gqa")) {
    grm_store_destroy(handle);
    return 5;
  }

  std::atomic<bool> start{false};
  std::atomic<bool> stop{false};
  std::atomic<int> errors{0};

  auto reader = [&]() {
    while (!start.load(std::memory_order_acquire)) {
      std::this_thread::yield();
    }
    while (!stop.load(std::memory_order_acquire)) {
      std::uint64_t local_out[16] = {};
      std::uint64_t local_count = 0;
      const int rc = grm_store_route_gqa(
          handle, q.data(), kQueryHeads, kQueryTokens, kHeadDim,
          "", "", "", "", "", 8, local_out, 16, &local_count);
      if (rc != 0 || local_count > 16) {
        errors.fetch_add(1, std::memory_order_relaxed);
        continue;
      }
      for (std::uint64_t i = 0; i < local_count; ++i) {
        if (local_out[i] >= nodes) {
          errors.fetch_add(1, std::memory_order_relaxed);
          break;
        }
      }
    }
  };

  std::vector<std::thread> reader_threads;
  reader_threads.reserve(static_cast<std::size_t>(readers));
  for (std::uint64_t i = 0; i < readers; ++i) {
    reader_threads.emplace_back(reader);
  }

  std::thread writer([&]() {
    while (!start.load(std::memory_order_acquire)) {
      std::this_thread::yield();
    }
    for (std::uint64_t step = 0; step < iterations; ++step) {
      const auto idx = step % nodes;
      const auto node_id = ids[static_cast<std::size_t>(idx)];
      bool ok = true;
      if (step % 5 == 0) {
        auto key = route_key(idx, step + 1);
        ok = check_rc(handle,
                      grm_store_set_route_multi(
                          handle, node_id, key.data(), 1,
                          static_cast<std::uint64_t>(key.size()), ""),
                      "writer set_route_multi");
      } else if (step % 5 == 1) {
        ok = check_rc(handle, grm_store_set_active(handle, node_id, 0),
                      "writer set_active false") &&
             check_rc(handle, grm_store_set_active(handle, node_id, 1),
                      "writer set_active true");
      } else if (step % 5 == 2) {
        ok = check_rc(handle,
                      grm_store_set_route_metadata(
                          handle, node_id, "doc", "project", "session",
                          "mutable"),
                      "writer set_route_metadata");
      } else if (step % 5 == 3) {
        const auto replacement = ids[static_cast<std::size_t>((idx + 1) % nodes)];
        ok = check_rc(handle,
                      grm_store_apply_revision(handle, replacement, &node_id, 1),
                      "writer apply_revision");
      } else {
        ok = check_rc(handle, grm_store_apply_expire(handle, &node_id, 1),
                      "writer apply_expire") &&
             check_rc(handle, grm_store_set_active(handle, node_id, 1),
                      "writer reactivate");
      }
      if (!ok) {
        errors.fetch_add(1, std::memory_order_relaxed);
      }
    }
    stop.store(true, std::memory_order_release);
  });

  start.store(true, std::memory_order_release);
  writer.join();
  for (auto& thread : reader_threads) {
    thread.join();
  }

  if (errors.load(std::memory_order_relaxed) != 0) {
    std::cerr << "router TSAN stress observed "
              << errors.load(std::memory_order_relaxed) << " errors\n";
    grm_store_destroy(handle);
    return 6;
  }

  out_count = 0;
  if (!check_rc(handle,
                grm_store_route_gqa(
                    handle, q.data(), kQueryHeads, kQueryTokens, kHeadDim,
                    "", "", "", "", "", 8, out, 16, &out_count),
                "final route_gqa")) {
    grm_store_destroy(handle);
    return 7;
  }

  grm_store_destroy(handle);
  std::cout << "router TSAN stress passed: nodes=" << nodes
            << " iterations=" << iterations
            << " readers=" << readers << "\n";
  return 0;
}
'''


def _replay(completed: subprocess.CompletedProcess[str]) -> None:
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="", file=sys.stderr)


def _run_tsan_gate(run_cmd: list[str], *, retry_setarch: bool) -> None:
    completed = subprocess.run(run_cmd, text=True, capture_output=True)
    _replay(completed)
    if completed.returncode == 0:
        return

    combined = (completed.stdout or "") + (completed.stderr or "")
    setarch = shutil.which("setarch")
    if (
        retry_setarch
        and setarch is not None
        and "ThreadSanitizer: unexpected memory mapping" in combined
    ):
        retry_cmd = [setarch, platform.machine() or "x86_64", "-R", *run_cmd]
        print("retry:", " ".join(retry_cmd), flush=True)
        retry = subprocess.run(retry_cmd, text=True, capture_output=True)
        _replay(retry)
        retry.check_returncode()
        return

    completed.check_returncode()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cxx", default="g++")
    parser.add_argument("--nodes", type=int, default=48)
    parser.add_argument("--iterations", type=int, default=1200)
    parser.add_argument("--readers", type=int, default=4)
    parser.add_argument("--build-dir", type=Path)
    parser.add_argument(
        "--no-setarch-retry",
        action="store_true",
        help="Do not retry under setarch -R when GCC TSAN hits host ASLR mapping issues.",
    )
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="grm-router-tsan-") as td:
        build_dir = args.build_dir or Path(td)
        build_dir.mkdir(parents=True, exist_ok=True)
        source = build_dir / "grm_router_tsan_gate.cpp"
        exe = build_dir / "grm_router_tsan_gate"
        source.write_text(textwrap.dedent(HARNESS).lstrip(), encoding="utf-8")
        cmd = [
            args.cxx,
            "-std=c++17",
            "-O1",
            "-g",
            "-fsanitize=thread",
            "-fno-omit-frame-pointer",
            "-pthread",
            "-I",
            str(ROOT / "cpp"),
            str(ROOT / "cpp" / "grm_runtime.cpp"),
            str(source),
            "-o",
            str(exe),
        ]
        print("compile:", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)
        run_cmd = [
            str(exe),
            str(args.nodes),
            str(args.iterations),
            str(args.readers),
        ]
        print("run:", " ".join(run_cmd), flush=True)
        _run_tsan_gate(run_cmd, retry_setarch=not args.no_setarch_retry)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
