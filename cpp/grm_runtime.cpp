#include "grm_runtime.hpp"
#include "grm_runtime_c.h"

#include <algorithm>
#include <cstring>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <sstream>

namespace grm {

std::string DialectDescriptor::dialect_id() const {
  if (payload_kind == PayloadKind::MLA) {
    return model_type + ":" + std::to_string(num_layers) + "x" +
           std::to_string(hidden_dim) + ":r" + std::to_string(latent_rank);
  }
  return model_type + ":" + std::to_string(num_layers) + "x" +
         std::to_string(hidden_dim) + ":g" + std::to_string(num_kv_heads) +
         "x" + std::to_string(head_dim);
}

std::uint64_t HostPayload::bytes() const {
  std::uint64_t n = 0;
  for (const auto& t : tensors) {
    n += static_cast<std::uint64_t>(t.bytes.size());
  }
  return n;
}

std::uint64_t HostPayload::tensor_count() const {
  return static_cast<std::uint64_t>(tensors.size());
}

void DirtyQueue::mark(std::uint64_t node_id, bool payload, bool metadata) {
  auto& s = dirty_[node_id];
  s.payload = s.payload || payload;
  s.metadata = s.metadata || metadata;
}

void DirtyQueue::clear(std::uint64_t node_id) { dirty_.erase(node_id); }

void DirtyQueue::clear_all() { dirty_.clear(); }

bool DirtyQueue::empty() const { return dirty_.empty(); }

std::vector<std::uint64_t> DirtyQueue::node_ids() const {
  std::vector<std::uint64_t> ids;
  ids.reserve(dirty_.size());
  for (const auto& kv : dirty_) {
    ids.push_back(kv.first);
  }
  std::sort(ids.begin(), ids.end());
  return ids;
}

namespace {

constexpr char kCheckpointMagicV1[] = "GRMSTORE1";
constexpr char kCheckpointMagicV2[] = "GRMSTORE2";
constexpr char kCheckpointMagicV3[] = "GRMSTORE3";
constexpr char kCheckpointMagic[] = "GRMSTORE3";

void write_u64(std::ostream& out, std::uint64_t v) {
  out.write(reinterpret_cast<const char*>(&v), sizeof(v));
}

std::uint64_t read_u64(std::istream& in) {
  std::uint64_t v = 0;
  in.read(reinterpret_cast<char*>(&v), sizeof(v));
  if (!in) {
    throw std::runtime_error("truncated GRM checkpoint");
  }
  return v;
}

void write_bool(std::ostream& out, bool v) {
  const char c = v ? 1 : 0;
  out.write(&c, 1);
}

bool read_bool(std::istream& in) {
  char c = 0;
  in.read(&c, 1);
  if (!in) {
    throw std::runtime_error("truncated GRM checkpoint");
  }
  return c != 0;
}

void write_string(std::ostream& out, const std::string& s) {
  write_u64(out, static_cast<std::uint64_t>(s.size()));
  if (!s.empty()) {
    out.write(s.data(), static_cast<std::streamsize>(s.size()));
  }
}

std::string read_string(std::istream& in) {
  const auto n = read_u64(in);
  std::string s(static_cast<std::size_t>(n), '\0');
  if (n > 0) {
    in.read(s.data(), static_cast<std::streamsize>(n));
    if (!in) {
      throw std::runtime_error("truncated GRM checkpoint string");
    }
  }
  return s;
}

std::uint64_t checked_mul(std::uint64_t a,
                          std::uint64_t b,
                          const char* label) {
  if (a != 0 && b > std::numeric_limits<std::uint64_t>::max() / a) {
    throw std::overflow_error(std::string(label) + " size overflow");
  }
  return a * b;
}

std::uint64_t checked_product(const std::vector<std::uint64_t>& dims,
                              std::size_t first,
                              std::size_t last,
                              const char* label) {
  std::uint64_t out = 1;
  for (std::size_t i = first; i < last; ++i) {
    out = checked_mul(out, dims[i], label);
  }
  return out;
}

void require_byte_count(std::uint64_t actual,
                        std::uint64_t expected,
                        const char* label) {
  if (actual != expected) {
    std::ostringstream ss;
    ss << label << " byte count mismatch: expected " << expected
       << ", got " << actual;
    throw std::runtime_error(ss.str());
  }
}

}  // namespace

HostGraftStore::HostGraftStore(DialectDescriptor dialect)
    : dialect_(std::move(dialect)) {}

std::uint64_t HostGraftStore::add_node(HostGraftNode node) {
  const std::uint64_t id = next_id_++;
  node.node_id = id;
  node.lifecycle.host_present = !node.payload.tensors.empty();
  node.lifecycle.device_present = false;
  node.lifecycle.dirty = true;
  node.lifecycle.durable = false;
  node.lifecycle.cold_only = false;
  nodes_[id] = std::move(node);
  dirty_.mark(id, true, true);
  return id;
}

HostGraftNode* HostGraftStore::get(std::uint64_t node_id) {
  auto it = nodes_.find(node_id);
  return it == nodes_.end() ? nullptr : &it->second;
}

const HostGraftNode* HostGraftStore::get(std::uint64_t node_id) const {
  auto it = nodes_.find(node_id);
  return it == nodes_.end() ? nullptr : &it->second;
}

void HostGraftStore::set_tensor(std::uint64_t node_id, HostTensor tensor) {
  auto* n = get(node_id);
  if (n == nullptr) {
    throw std::out_of_range("unknown GRM node id");
  }
  for (auto& existing : n->payload.tensors) {
    if (existing.name == tensor.name) {
      existing = std::move(tensor);
      n->lifecycle.host_present = true;
      mark_dirty(node_id, true, true);
      return;
    }
  }
  n->payload.tensors.push_back(std::move(tensor));
  n->lifecycle.host_present = true;
  mark_dirty(node_id, true, true);
}

PayloadStats HostGraftStore::payload_stats(std::uint64_t node_id) const {
  const auto* n = get(node_id);
  if (n == nullptr) {
    throw std::out_of_range("unknown GRM node id");
  }
  PayloadStats s;
  s.tensor_count = n->payload.tensor_count();
  s.payload_bytes = n->payload.bytes();
  return s;
}

void HostGraftStore::set_metadata_json(std::uint64_t node_id,
                                       std::string metadata_json) {
  auto* n = get(node_id);
  if (n == nullptr) {
    throw std::out_of_range("unknown GRM node id");
  }
  n->metadata.json = std::move(metadata_json);
  mark_dirty(node_id, false, true);
}

const std::string& HostGraftStore::metadata_json(std::uint64_t node_id) const {
  const auto* n = get(node_id);
  if (n == nullptr) {
    throw std::out_of_range("unknown GRM node id");
  }
  return n->metadata.json;
}

void HostGraftStore::set_active(std::uint64_t node_id, bool active) {
  auto* n = get(node_id);
  if (n == nullptr) {
    throw std::out_of_range("unknown GRM node id");
  }
  if (n->metadata.active == active) {
    return;
  }
  n->metadata.active = active;
  mark_dirty(node_id, false, true);
}

bool HostGraftStore::is_active(std::uint64_t node_id) const {
  const auto* n = get(node_id);
  if (n == nullptr) {
    throw std::out_of_range("unknown GRM node id");
  }
  return n->metadata.active;
}

void HostGraftStore::set_route_metadata(std::uint64_t node_id,
                                        std::string kind,
                                        std::string scope,
                                        std::string durability,
                                        std::string mutability) {
  auto* n = get(node_id);
  if (n == nullptr) {
    throw std::out_of_range("unknown GRM node id");
  }
  bool changed = false;
  if (!kind.empty() && n->metadata.kind != kind) {
    n->metadata.kind = std::move(kind);
    changed = true;
  }
  if (!scope.empty() && n->metadata.scope != scope) {
    n->metadata.scope = std::move(scope);
    changed = true;
  }
  if (!durability.empty() && n->metadata.durability != durability) {
    n->metadata.durability = std::move(durability);
    changed = true;
  }
  if (!mutability.empty() && n->metadata.mutability != mutability) {
    n->metadata.mutability = std::move(mutability);
    changed = true;
  }
  if (changed) {
    mark_dirty(node_id, false, true);
  }
}

void HostGraftStore::mark_dirty(std::uint64_t node_id, bool payload,
                                bool metadata) {
  auto* n = get(node_id);
  if (n == nullptr) {
    throw std::out_of_range("unknown GRM node id");
  }
  n->lifecycle.dirty = true;
  if (payload) {
    n->lifecycle.durable = false;
  }
  dirty_.mark(node_id, payload, metadata);
}

void HostGraftStore::mark_durable(std::uint64_t node_id) {
  auto* n = get(node_id);
  if (n == nullptr) {
    throw std::out_of_range("unknown GRM node id");
  }
  n->lifecycle.dirty = false;
  n->lifecycle.durable = true;
  n->lifecycle.cold_only = !n->lifecycle.host_present;
  dirty_.clear(node_id);
}

void HostGraftStore::evict_device_copy(std::uint64_t node_id) {
  auto* n = get(node_id);
  if (n == nullptr) {
    throw std::out_of_range("unknown GRM node id");
  }
  if (!n->lifecycle.host_present) {
    throw std::runtime_error("cannot evict device copy without host payload");
  }
  n->lifecycle.device_present = false;
}

void HostGraftStore::save_checkpoint(const std::string& root) {
  std::filesystem::create_directories(root);
  const auto tmp = root + "/grm_store.bin.tmp";
  const auto dst = root + "/grm_store.bin";
  std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
  if (!out) {
    throw std::runtime_error("failed to open native GRM checkpoint for write");
  }
  out.write(kCheckpointMagic, sizeof(kCheckpointMagic) - 1);
  write_u64(out, static_cast<std::uint64_t>(nodes_.size()));
  std::vector<std::uint64_t> ids;
  ids.reserve(nodes_.size());
  for (const auto& kv : nodes_) {
    ids.push_back(kv.first);
  }
  std::sort(ids.begin(), ids.end());
  for (const auto id : ids) {
    const auto& n = nodes_.at(id);
    write_u64(out, n.node_id);
    write_u64(out, n.ntok);
    write_string(out, n.text);
    write_string(out, n.metadata.json);
    write_bool(out, n.metadata.active);
    write_string(out, n.metadata.kind);
    write_string(out, n.metadata.scope);
    write_string(out, n.metadata.durability);
    write_string(out, n.metadata.mutability);
    write_bool(out, n.lifecycle.host_present);
    write_bool(out, n.lifecycle.device_present);
    write_bool(out, false);
    write_bool(out, true);
    write_bool(out, n.lifecycle.cold_only);
    write_u64(out, n.payload.tensor_count());
    for (const auto& t : n.payload.tensors) {
      write_string(out, t.name);
      write_string(out, t.dtype);
      write_u64(out, static_cast<std::uint64_t>(t.shape.size()));
      for (const auto d : t.shape) {
        write_u64(out, d);
      }
      write_u64(out, static_cast<std::uint64_t>(t.bytes.size()));
      if (!t.bytes.empty()) {
        out.write(reinterpret_cast<const char*>(t.bytes.data()),
                  static_cast<std::streamsize>(t.bytes.size()));
      }
    }
  }
  if (!out) {
    throw std::runtime_error("failed while writing native GRM checkpoint");
  }
  out.close();
  std::filesystem::rename(tmp, dst);
  for (auto& kv : nodes_) {
    auto& n = kv.second;
    n.lifecycle.dirty = false;
    n.lifecycle.durable = true;
    n.lifecycle.cold_only = n.lifecycle.durable && !n.lifecycle.host_present;
  }
  dirty_.clear_all();
}

void HostGraftStore::load_checkpoint(const std::string& root) {
  const auto src = root + "/grm_store.bin";
  std::ifstream in(src, std::ios::binary);
  if (!in) {
    throw std::runtime_error("failed to open native GRM checkpoint for read");
  }
  char magic[sizeof(kCheckpointMagic) - 1] = {};
  in.read(magic, sizeof(magic));
  const std::string magic_s(magic, sizeof(magic));
  const bool checkpoint_v1 =
      magic_s == std::string(kCheckpointMagicV1, sizeof(kCheckpointMagicV1) - 1);
  const bool checkpoint_v2 =
      magic_s == std::string(kCheckpointMagicV2, sizeof(kCheckpointMagicV2) - 1);
  const bool checkpoint_v3 =
      magic_s == std::string(kCheckpointMagicV3, sizeof(kCheckpointMagicV3) - 1);
  if (!in || (!checkpoint_v1 && !checkpoint_v2 && !checkpoint_v3)) {
    throw std::runtime_error("invalid native GRM checkpoint magic");
  }
  std::unordered_map<std::uint64_t, HostGraftNode> loaded;
  const auto count = read_u64(in);
  std::uint64_t max_id = 0;
  for (std::uint64_t ni = 0; ni < count; ++ni) {
    HostGraftNode n;
    n.node_id = read_u64(in);
    n.ntok = read_u64(in);
    n.text = read_string(in);
    n.metadata.json = read_string(in);
    n.metadata.active = (checkpoint_v2 || checkpoint_v3) ? read_bool(in) : true;
    if (checkpoint_v3) {
      n.metadata.kind = read_string(in);
      n.metadata.scope = read_string(in);
      n.metadata.durability = read_string(in);
      n.metadata.mutability = read_string(in);
    }
    n.lifecycle.host_present = read_bool(in);
    n.lifecycle.device_present = read_bool(in);
    n.lifecycle.dirty = read_bool(in);
    n.lifecycle.durable = read_bool(in);
    n.lifecycle.cold_only = read_bool(in);
    const auto tensor_count = read_u64(in);
    for (std::uint64_t ti = 0; ti < tensor_count; ++ti) {
      HostTensor t;
      t.name = read_string(in);
      t.dtype = read_string(in);
      const auto rank = read_u64(in);
      t.shape.reserve(static_cast<std::size_t>(rank));
      for (std::uint64_t di = 0; di < rank; ++di) {
        t.shape.push_back(read_u64(in));
      }
      const auto bytes = read_u64(in);
      t.bytes.resize(static_cast<std::size_t>(bytes));
      if (bytes > 0) {
        in.read(reinterpret_cast<char*>(t.bytes.data()),
                static_cast<std::streamsize>(bytes));
        if (!in) {
          throw std::runtime_error("truncated GRM checkpoint tensor");
        }
      }
      n.payload.tensors.push_back(std::move(t));
    }
    n.lifecycle.host_present = n.payload.tensor_count() > 0;
    n.lifecycle.device_present = false;
    n.lifecycle.dirty = false;
    n.lifecycle.durable = true;
    n.lifecycle.cold_only = false;
    const auto node_id = n.node_id;
    loaded[node_id] = std::move(n);
    max_id = std::max(max_id, node_id);
  }
  nodes_ = std::move(loaded);
  next_id_ = count == 0 ? 0 : max_id + 1;
  dirty_.clear_all();
}

StoreStats HostGraftStore::stats() const {
  StoreStats s;
  s.nodes = static_cast<std::uint64_t>(nodes_.size());
  for (const auto& kv : nodes_) {
    const auto& n = kv.second;
    s.host_payload_bytes += n.payload.bytes();
    s.host_payload_tensors += n.payload.tensor_count();
    if (n.lifecycle.dirty) {
      ++s.dirty_nodes;
    }
    if (n.lifecycle.durable) {
      ++s.durable_nodes;
    }
  }
  return s;
}

void RouterIndex::upsert(std::uint64_t node_id, std::vector<float> route_key,
                         std::vector<std::string> lexical_keys) {
  std::vector<std::vector<float>> keys;
  keys.push_back(std::move(route_key));
  upsert_multi(node_id, std::move(keys), std::move(lexical_keys));
}

void RouterIndex::upsert_multi(std::uint64_t node_id,
                               std::vector<std::vector<float>> route_keys,
                               std::vector<std::string> lexical_keys) {
  for (auto& e : entries_) {
    if (e.node_id == node_id) {
      e.route_keys = std::move(route_keys);
      e.lexical_keys = std::move(lexical_keys);
      return;
    }
  }
  entries_.push_back({node_id, std::move(route_keys), std::move(lexical_keys)});
}

void RouterIndex::set_active(std::uint64_t node_id, bool active) {
  for (auto& e : entries_) {
    if (e.node_id == node_id) {
      e.active = active;
    }
  }
}

void RouterIndex::set_route_metadata(std::uint64_t node_id,
                                     std::string kind,
                                     std::string scope,
                                     std::string durability,
                                     std::string mutability) {
  for (auto& e : entries_) {
    if (e.node_id == node_id) {
      if (!kind.empty()) {
        e.kind = kind;
      }
      if (!scope.empty()) {
        e.scope = scope;
      }
      if (!durability.empty()) {
        e.durability = durability;
      }
      if (!mutability.empty()) {
        e.mutability = mutability;
      }
    }
  }
}

static float cosine(const std::vector<float>& a, const std::vector<float>& b) {
  if (a.empty() || a.size() != b.size()) {
    return 0.0F;
  }
  double dot = 0.0;
  double na = 0.0;
  double nb = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    dot += static_cast<double>(a[i]) * static_cast<double>(b[i]);
    na += static_cast<double>(a[i]) * static_cast<double>(a[i]);
    nb += static_cast<double>(b[i]) * static_cast<double>(b[i]);
  }
  return static_cast<float>(dot / (std::sqrt(na) * std::sqrt(nb) + 1.0e-8));
}

static float max_cosine(const std::vector<float>& query,
                        const std::vector<std::vector<float>>& keys) {
  float best = 0.0F;
  bool have = false;
  for (const auto& key : keys) {
    const float score = cosine(query, key);
    if (!have || score > best) {
      best = score;
      have = true;
    }
  }
  return have ? best : 0.0F;
}

static bool filter_allows(const std::vector<std::string>& filters,
                          const std::string& value) {
  return filters.empty() ||
         std::find(filters.begin(), filters.end(), value) != filters.end();
}

std::vector<std::uint64_t> RouterIndex::route(
    const std::vector<float>& query, const std::vector<std::string>& lexical,
    std::size_t topk, const std::vector<std::string>& kinds,
    const std::vector<std::string>& scopes,
    const std::vector<std::string>& durabilities,
    const std::vector<std::string>& mutabilities) const {
  std::vector<std::pair<float, std::uint64_t>> scored;
  scored.reserve(entries_.size());
  for (const auto& e : entries_) {
    if (!e.active) {
      continue;
    }
    if (!filter_allows(kinds, e.kind) || !filter_allows(scopes, e.scope) ||
        !filter_allows(durabilities, e.durability) ||
        !filter_allows(mutabilities, e.mutability)) {
      continue;
    }
    float score = max_cosine(query, e.route_keys);
    std::size_t lexical_hits = 0;
    for (const auto& q : lexical) {
      if (std::find(e.lexical_keys.begin(), e.lexical_keys.end(), q) !=
          e.lexical_keys.end()) {
        ++lexical_hits;
      }
    }
    if (!lexical.empty()) {
      score += static_cast<float>(lexical_hits) /
               static_cast<float>(lexical.size());
    }
    scored.push_back({score, e.node_id});
  }
  std::sort(scored.begin(), scored.end(),
            [](const auto& a, const auto& b) { return a.first > b.first; });
  std::vector<std::uint64_t> out;
  for (std::size_t i = 0; i < std::min(topk, scored.size()); ++i) {
    out.push_back(scored[i].second);
  }
  return out;
}

DurabilityWriter::DurabilityWriter(std::string root) : root_(std::move(root)) {}

void DurabilityWriter::write_checkpoint(const HostGraftStore& store) {
  std::filesystem::create_directories(root_);
  std::ofstream out(root_ + "/checkpoint.txt", std::ios::trunc);
  const auto s = store.stats();
  out << "dialect " << store.dialect().dialect_id() << "\n";
  out << "nodes " << s.nodes << "\n";
  out << "dirty_nodes " << s.dirty_nodes << "\n";
  out << "durable_nodes " << s.durable_nodes << "\n";
  out << "host_payload_bytes " << s.host_payload_bytes << "\n";
}

void DeviceArena::configure(std::uint64_t sink_tokens,
                            std::uint64_t arena_width) {
  sink_tokens_ = sink_tokens;
  arena_width_ = arena_width;
  mount_tokens_ = 0;
  mounted_.clear();
}

ArenaSwapPlan DeviceArena::plan_swap(
    std::uint64_t new_mount_tokens,
    std::uint64_t input_cache_tokens) const {
  ArenaSwapPlan p;
  p.sink_tokens = sink_tokens_;
  p.arena_width = arena_width_;
  p.old_mount_tokens = mount_tokens_;
  p.new_mount_tokens = new_mount_tokens;
  p.old_mount_end = sink_tokens_ + mount_tokens_;
  p.input_cache_tokens = input_cache_tokens;
  p.live_tail_start = std::min(p.old_mount_end, input_cache_tokens);
  p.live_tail_tokens = input_cache_tokens - p.live_tail_start;
  p.output_cache_tokens = sink_tokens_ + new_mount_tokens + p.live_tail_tokens;
  p.overflow = new_mount_tokens > arena_width_;
  return p;
}

ArenaEvictPlan DeviceArena::plan_evict(
    std::uint64_t drop_tokens,
    std::uint64_t input_cache_tokens) const {
  ArenaEvictPlan p;
  p.sink_tokens = sink_tokens_;
  p.arena_width = arena_width_;
  p.mount_tokens = mount_tokens_;
  p.head_tokens = sink_tokens_ + mount_tokens_;
  p.drop_tokens = drop_tokens;
  p.input_cache_tokens = input_cache_tokens;
  p.underflow = input_cache_tokens < p.head_tokens ||
                drop_tokens > input_cache_tokens - p.head_tokens;
  p.output_cache_tokens = p.underflow ? 0 : input_cache_tokens - drop_tokens;
  return p;
}

TensorSwapResult DeviceArena::apply_swap_tensor(
    const std::vector<std::uint64_t>& old_shape,
    std::uint64_t seq_dim,
    std::uint64_t elem_size,
    const std::uint8_t* old_payload,
    std::uint64_t old_payload_len,
    const std::vector<std::uint64_t>& mount_shape,
    const std::uint8_t* mount_payload,
    std::uint64_t mount_payload_len,
    std::uint64_t new_mount_tokens,
    std::uint64_t input_cache_tokens) const {
  if (old_shape.empty()) {
    throw std::runtime_error("swap tensor rank must be nonzero");
  }
  if (seq_dim >= old_shape.size()) {
    throw std::runtime_error("swap seq_dim out of range");
  }
  if (elem_size == 0) {
    throw std::runtime_error("swap elem_size must be nonzero");
  }
  if (mount_shape.size() != old_shape.size()) {
    throw std::runtime_error("swap mount rank must match old rank");
  }
  if (old_payload == nullptr && old_payload_len > 0) {
    throw std::runtime_error("null old payload with nonzero length");
  }
  if (mount_payload == nullptr && mount_payload_len > 0) {
    throw std::runtime_error("null mount payload with nonzero length");
  }

  const auto plan = plan_swap(new_mount_tokens, input_cache_tokens);
  if (plan.overflow) {
    throw std::runtime_error("arena mount exceeds configured width");
  }
  if (old_shape[static_cast<std::size_t>(seq_dim)] != input_cache_tokens) {
    throw std::runtime_error("input_cache_tokens must match old seq length");
  }
  if (mount_shape[static_cast<std::size_t>(seq_dim)] != new_mount_tokens) {
    throw std::runtime_error("new_mount_tokens must match mount seq length");
  }
  if (input_cache_tokens < sink_tokens_) {
    throw std::runtime_error("input cache shorter than arena sink");
  }
  for (std::size_t i = 0; i < old_shape.size(); ++i) {
    if (i != static_cast<std::size_t>(seq_dim) &&
        old_shape[i] != mount_shape[i]) {
      throw std::runtime_error("mount tensor shape mismatch outside seq_dim");
    }
  }

  const auto outer_elems = checked_product(
      old_shape, 0, static_cast<std::size_t>(seq_dim), "swap outer");
  const auto inner_elems = checked_product(
      old_shape, static_cast<std::size_t>(seq_dim) + 1, old_shape.size(),
      "swap inner");
  const auto row_bytes = checked_mul(inner_elems, elem_size, "swap row");
  const auto old_rows = checked_mul(outer_elems, input_cache_tokens,
                                    "swap old payload");
  const auto mount_rows = checked_mul(outer_elems, new_mount_tokens,
                                      "swap mount payload");
  const auto out_rows = checked_mul(outer_elems, plan.output_cache_tokens,
                                    "swap output payload");
  const auto old_expected = checked_mul(old_rows, row_bytes,
                                        "swap old payload");
  const auto mount_expected = checked_mul(mount_rows, row_bytes,
                                          "swap mount payload");
  const auto out_expected = checked_mul(out_rows, row_bytes,
                                        "swap output payload");
  require_byte_count(old_payload_len, old_expected, "old payload");
  require_byte_count(mount_payload_len, mount_expected, "mount payload");
  if (out_expected > static_cast<std::uint64_t>(
                         std::numeric_limits<std::size_t>::max())) {
    throw std::overflow_error("swap output exceeds addressable host size");
  }

  TensorSwapResult result;
  result.plan = plan;
  result.shape = old_shape;
  result.shape[static_cast<std::size_t>(seq_dim)] = plan.output_cache_tokens;
  result.bytes.resize(static_cast<std::size_t>(out_expected));

  const auto sink_bytes = checked_mul(sink_tokens_, row_bytes,
                                      "swap sink bytes");
  const auto mount_bytes = checked_mul(new_mount_tokens, row_bytes,
                                       "swap mount bytes");
  const auto tail_bytes = checked_mul(plan.live_tail_tokens, row_bytes,
                                      "swap tail bytes");
  const auto old_stride = checked_mul(input_cache_tokens, row_bytes,
                                      "swap old stride");
  const auto mount_stride = checked_mul(new_mount_tokens, row_bytes,
                                        "swap mount stride");
  const auto out_stride = checked_mul(plan.output_cache_tokens, row_bytes,
                                      "swap output stride");
  for (std::uint64_t outer = 0; outer < outer_elems; ++outer) {
    const auto* old_base = old_payload == nullptr
                               ? nullptr
                               : old_payload + static_cast<std::size_t>(
                                                 outer * old_stride);
    const auto* mount_base = mount_payload == nullptr
                                 ? nullptr
                                 : mount_payload + static_cast<std::size_t>(
                                                     outer * mount_stride);
    auto* out_base =
        result.bytes.data() + static_cast<std::size_t>(outer * out_stride);
    std::uint64_t cursor = 0;
    if (sink_bytes > 0) {
      std::memcpy(out_base, old_base, static_cast<std::size_t>(sink_bytes));
      cursor += sink_bytes;
    }
    if (mount_bytes > 0) {
      std::memcpy(out_base + static_cast<std::size_t>(cursor), mount_base,
                  static_cast<std::size_t>(mount_bytes));
      cursor += mount_bytes;
    }
    if (tail_bytes > 0) {
      const auto tail_offset = checked_mul(plan.live_tail_start, row_bytes,
                                           "swap tail offset");
      std::memcpy(out_base + static_cast<std::size_t>(cursor),
                  old_base + static_cast<std::size_t>(tail_offset),
                  static_cast<std::size_t>(tail_bytes));
    }
  }
  return result;
}

TensorEvictResult DeviceArena::apply_evict_tensor(
    const std::vector<std::uint64_t>& old_shape,
    std::uint64_t seq_dim,
    std::uint64_t elem_size,
    const std::uint8_t* old_payload,
    std::uint64_t old_payload_len,
    std::uint64_t drop_tokens,
    std::uint64_t input_cache_tokens) const {
  if (old_shape.empty()) {
    throw std::runtime_error("evict tensor rank must be nonzero");
  }
  if (seq_dim >= old_shape.size()) {
    throw std::runtime_error("evict seq_dim out of range");
  }
  if (elem_size == 0) {
    throw std::runtime_error("evict elem_size must be nonzero");
  }
  if (old_payload == nullptr && old_payload_len > 0) {
    throw std::runtime_error("null old payload with nonzero length");
  }

  const auto plan = plan_evict(drop_tokens, input_cache_tokens);
  if (plan.underflow) {
    throw std::runtime_error("evict drop exceeds live tail");
  }
  if (old_shape[static_cast<std::size_t>(seq_dim)] != input_cache_tokens) {
    throw std::runtime_error("input_cache_tokens must match old seq length");
  }

  const auto outer_elems = checked_product(
      old_shape, 0, static_cast<std::size_t>(seq_dim), "evict outer");
  const auto inner_elems = checked_product(
      old_shape, static_cast<std::size_t>(seq_dim) + 1, old_shape.size(),
      "evict inner");
  const auto row_bytes = checked_mul(inner_elems, elem_size, "evict row");
  const auto old_rows = checked_mul(outer_elems, input_cache_tokens,
                                    "evict old payload");
  const auto out_rows = checked_mul(outer_elems, plan.output_cache_tokens,
                                    "evict output payload");
  const auto old_expected = checked_mul(old_rows, row_bytes,
                                        "evict old payload");
  const auto out_expected = checked_mul(out_rows, row_bytes,
                                        "evict output payload");
  require_byte_count(old_payload_len, old_expected, "old payload");
  if (out_expected > static_cast<std::uint64_t>(
                         std::numeric_limits<std::size_t>::max())) {
    throw std::overflow_error("evict output exceeds addressable host size");
  }

  TensorEvictResult result;
  result.plan = plan;
  result.shape = old_shape;
  result.shape[static_cast<std::size_t>(seq_dim)] = plan.output_cache_tokens;
  result.bytes.resize(static_cast<std::size_t>(out_expected));

  const auto head_bytes = checked_mul(plan.head_tokens, row_bytes,
                                      "evict head bytes");
  const auto tail_tokens = input_cache_tokens - plan.head_tokens - drop_tokens;
  const auto tail_bytes = checked_mul(tail_tokens, row_bytes,
                                      "evict tail bytes");
  const auto old_stride = checked_mul(input_cache_tokens, row_bytes,
                                      "evict old stride");
  const auto out_stride = checked_mul(plan.output_cache_tokens, row_bytes,
                                      "evict output stride");
  const auto tail_offset = checked_mul(plan.head_tokens + drop_tokens,
                                       row_bytes, "evict tail offset");
  for (std::uint64_t outer = 0; outer < outer_elems; ++outer) {
    const auto* old_base = old_payload == nullptr
                               ? nullptr
                               : old_payload + static_cast<std::size_t>(
                                                 outer * old_stride);
    auto* out_base =
        result.bytes.data() + static_cast<std::size_t>(outer * out_stride);
    std::uint64_t cursor = 0;
    if (head_bytes > 0) {
      std::memcpy(out_base, old_base, static_cast<std::size_t>(head_bytes));
      cursor += head_bytes;
    }
    if (tail_bytes > 0) {
      std::memcpy(out_base + static_cast<std::size_t>(cursor),
                  old_base + static_cast<std::size_t>(tail_offset),
                  static_cast<std::size_t>(tail_bytes));
    }
  }
  return result;
}

void DeviceArena::commit_mount(std::vector<std::uint64_t> node_ids,
                               std::uint64_t mount_tokens) {
  mounted_ = std::move(node_ids);
  mount_tokens_ = mount_tokens;
}

}  // namespace grm

struct grm_store_handle {
  std::unique_ptr<grm::HostGraftStore> store;
  grm::RouterIndex router;
  grm::DeviceArena arena;
  std::string last_error;
};

namespace {

int grm_fail(grm_store_handle* handle, const std::exception& exc) {
  if (handle != nullptr) {
    handle->last_error = exc.what();
  }
  return -1;
}

int grm_fail_msg(grm_store_handle* handle, const char* msg) {
  if (handle != nullptr) {
    handle->last_error = msg;
  }
  return -1;
}

std::vector<std::string> split_lexical_keys(const char* lexical_keys) {
  std::vector<std::string> out;
  if (lexical_keys == nullptr) {
    return out;
  }
  std::istringstream in(lexical_keys);
  std::string item;
  while (std::getline(in, item)) {
    if (!item.empty()) {
      out.push_back(item);
    }
  }
  return out;
}

void sync_router_node_state(grm_store_handle* handle, std::uint64_t node_id) {
  const auto* node = handle->store->get(node_id);
  if (node == nullptr) {
    throw std::out_of_range("unknown GRM node id");
  }
  handle->router.set_active(node_id, node->metadata.active);
  handle->router.set_route_metadata(node_id, node->metadata.kind,
                                    node->metadata.scope,
                                    node->metadata.durability,
                                    node->metadata.mutability);
}

}  // namespace

extern "C" {

grm_store_handle* grm_store_create_mla(const char* model_type,
                                       int num_layers,
                                       int hidden_dim,
                                       int vals_per_tok_layer,
                                       int route_layer,
                                       int latent_rank,
                                       int rope_dim) {
  try {
    auto* handle = new grm_store_handle();
    grm::DialectDescriptor d;
    d.model_type = model_type == nullptr ? "" : model_type;
    d.num_layers = num_layers;
    d.hidden_dim = hidden_dim;
    d.payload_kind = grm::PayloadKind::MLA;
    d.vals_per_tok_layer = vals_per_tok_layer;
    d.route_layer = route_layer;
    d.latent_rank = latent_rank;
    d.rope_dim = rope_dim;
    handle->store = std::make_unique<grm::HostGraftStore>(std::move(d));
    return handle;
  } catch (...) {
    return nullptr;
  }
}

void grm_store_destroy(grm_store_handle* handle) { delete handle; }

int grm_store_dialect_id(grm_store_handle* handle, char* out, size_t out_cap) {
  try {
    if (handle == nullptr || handle->store == nullptr || out == nullptr ||
        out_cap == 0) {
      return grm_fail_msg(handle, "invalid dialect_id arguments");
    }
    const auto s = handle->store->dialect().dialect_id();
    const size_t n = std::min(out_cap - 1, s.size());
    std::memcpy(out, s.data(), n);
    out[n] = '\0';
    return 0;
  } catch (const std::exception& exc) {
    return grm_fail(handle, exc);
  }
}

int grm_store_add_node(grm_store_handle* handle,
                       const char* text,
                       uint64_t ntok,
                       const uint8_t* payload,
                       uint64_t payload_len,
                       uint64_t* out_node_id) {
  try {
    if (handle == nullptr || handle->store == nullptr || out_node_id == nullptr) {
      return grm_fail_msg(handle, "invalid add_node arguments");
    }
    grm::HostGraftNode node;
    node.text = text == nullptr ? "" : text;
    node.ntok = ntok;
    if (payload != nullptr && payload_len > 0) {
      node.payload.tensors.push_back({"payload", "uint8", {payload_len}, {}});
      auto& bytes = node.payload.tensors.back().bytes;
      bytes.assign(payload, payload + payload_len);
    }
    *out_node_id = handle->store->add_node(std::move(node));
    return 0;
  } catch (const std::exception& exc) {
    return grm_fail(handle, exc);
  }
}

int grm_store_set_tensor(grm_store_handle* handle,
                         uint64_t node_id,
                         const char* name,
                         const char* dtype,
                         const uint64_t* shape,
                         uint64_t rank,
                         const uint8_t* payload,
                         uint64_t payload_len) {
  try {
    if (handle == nullptr || handle->store == nullptr || name == nullptr) {
      return grm_fail_msg(handle, "invalid set_tensor arguments");
    }
    if (shape == nullptr && rank > 0) {
      return grm_fail_msg(handle, "null shape with nonzero rank");
    }
    if (payload == nullptr && payload_len > 0) {
      return grm_fail_msg(handle, "null payload with nonzero length");
    }
    grm::HostTensor tensor;
    tensor.name = name;
    tensor.dtype = dtype == nullptr ? "uint8" : dtype;
    tensor.shape.reserve(static_cast<std::size_t>(rank));
    for (uint64_t i = 0; i < rank; ++i) {
      tensor.shape.push_back(shape[i]);
    }
    if (payload != nullptr && payload_len > 0) {
      tensor.bytes.assign(payload, payload + payload_len);
    }
    handle->store->set_tensor(node_id, std::move(tensor));
    return 0;
  } catch (const std::exception& exc) {
    return grm_fail(handle, exc);
  }
}

int grm_store_payload_stats(grm_store_handle* handle,
                            uint64_t node_id,
                            grm_payload_stats_c* out) {
  try {
    if (handle == nullptr || handle->store == nullptr || out == nullptr) {
      return grm_fail_msg(handle, "invalid payload_stats arguments");
    }
    const auto s = handle->store->payload_stats(node_id);
    out->tensor_count = s.tensor_count;
    out->payload_bytes = s.payload_bytes;
    return 0;
  } catch (const std::exception& exc) {
    return grm_fail(handle, exc);
  }
}

int grm_store_tensor_info(grm_store_handle* handle,
                          uint64_t node_id,
                          const char* name,
                          uint64_t* out_shape,
                          uint64_t shape_cap,
                          char* out_dtype,
                          size_t dtype_cap,
                          grm_tensor_info_c* out) {
  try {
    if (handle == nullptr || handle->store == nullptr || name == nullptr ||
        out == nullptr) {
      return grm_fail_msg(handle, "invalid tensor_info arguments");
    }
    if (out_shape == nullptr && shape_cap > 0) {
      return grm_fail_msg(handle, "null shape buffer with nonzero capacity");
    }
    const auto* node = handle->store->get(node_id);
    if (node == nullptr) {
      return grm_fail_msg(handle, "unknown GRM node id");
    }
    for (const auto& t : node->payload.tensors) {
      if (t.name == name) {
        out->rank = static_cast<uint64_t>(t.shape.size());
        out->payload_bytes = static_cast<uint64_t>(t.bytes.size());
        const uint64_t n = std::min<uint64_t>(out->rank, shape_cap);
        for (uint64_t i = 0; i < n; ++i) {
          out_shape[i] = t.shape[static_cast<std::size_t>(i)];
        }
        if (out_dtype != nullptr && dtype_cap > 0) {
          const size_t dn = std::min(dtype_cap - 1, t.dtype.size());
          std::memcpy(out_dtype, t.dtype.data(), dn);
          out_dtype[dn] = '\0';
        }
        return 0;
      }
    }
    return grm_fail_msg(handle, "unknown tensor name");
  } catch (const std::exception& exc) {
    return grm_fail(handle, exc);
  }
}

int grm_store_read_tensor(grm_store_handle* handle,
                          uint64_t node_id,
                          const char* name,
                          uint8_t* out_payload,
                          uint64_t payload_cap,
                          uint64_t* out_count) {
  try {
    if (handle == nullptr || handle->store == nullptr || name == nullptr ||
        out_count == nullptr) {
      return grm_fail_msg(handle, "invalid read_tensor arguments");
    }
    if (out_payload == nullptr && payload_cap > 0) {
      return grm_fail_msg(handle, "null payload buffer with nonzero capacity");
    }
    const auto* node = handle->store->get(node_id);
    if (node == nullptr) {
      return grm_fail_msg(handle, "unknown GRM node id");
    }
    for (const auto& t : node->payload.tensors) {
      if (t.name == name) {
        const uint64_t n = std::min<uint64_t>(
            static_cast<uint64_t>(t.bytes.size()), payload_cap);
        if (n > 0) {
          std::memcpy(out_payload, t.bytes.data(), static_cast<size_t>(n));
        }
        *out_count = n;
        return 0;
      }
    }
    return grm_fail_msg(handle, "unknown tensor name");
  } catch (const std::exception& exc) {
    return grm_fail(handle, exc);
  }
}

int grm_store_set_metadata_json(grm_store_handle* handle,
                                uint64_t node_id,
                                const char* metadata_json) {
  try {
    if (handle == nullptr || handle->store == nullptr ||
        metadata_json == nullptr) {
      return grm_fail_msg(handle, "invalid set_metadata_json arguments");
    }
    handle->store->set_metadata_json(node_id, metadata_json);
    return 0;
  } catch (const std::exception& exc) {
    return grm_fail(handle, exc);
  }
}

int grm_store_set_active(grm_store_handle* handle,
                         uint64_t node_id,
                         int active) {
  try {
    if (handle == nullptr || handle->store == nullptr) {
      return grm_fail_msg(handle, "invalid set_active arguments");
    }
    const bool is_active = active != 0;
    handle->store->set_active(node_id, is_active);
    handle->router.set_active(node_id, is_active);
    return 0;
  } catch (const std::exception& exc) {
    return grm_fail(handle, exc);
  }
}

int grm_store_set_route_metadata(grm_store_handle* handle,
                                 uint64_t node_id,
                                 const char* kind,
                                 const char* scope,
                                 const char* durability,
                                 const char* mutability) {
  try {
    if (handle == nullptr || handle->store == nullptr) {
      return grm_fail_msg(handle, "invalid set_route_metadata arguments");
    }
    handle->store->set_route_metadata(node_id, kind == nullptr ? "" : kind,
                                      scope == nullptr ? "" : scope,
                                      durability == nullptr ? "" : durability,
                                      mutability == nullptr ? "" : mutability);
    handle->router.set_route_metadata(node_id, kind == nullptr ? "" : kind,
                                      scope == nullptr ? "" : scope,
                                      durability == nullptr ? "" : durability,
                                      mutability == nullptr ? "" : mutability);
    return 0;
  } catch (const std::exception& exc) {
    return grm_fail(handle, exc);
  }
}

int grm_store_metadata_json(grm_store_handle* handle,
                            uint64_t node_id,
                            char* out_json,
                            size_t out_cap,
                            uint64_t* out_len) {
  try {
    if (handle == nullptr || handle->store == nullptr || out_len == nullptr) {
      return grm_fail_msg(handle, "invalid metadata_json arguments");
    }
    if (out_json == nullptr && out_cap > 0) {
      return grm_fail_msg(handle, "null metadata buffer with nonzero capacity");
    }
    const auto& s = handle->store->metadata_json(node_id);
    *out_len = static_cast<uint64_t>(s.size());
    if (out_json != nullptr && out_cap > 0) {
      const size_t n = std::min(out_cap - 1, s.size());
      std::memcpy(out_json, s.data(), n);
      out_json[n] = '\0';
    }
    return 0;
  } catch (const std::exception& exc) {
    return grm_fail(handle, exc);
  }
}

int grm_store_set_route(grm_store_handle* handle,
                        uint64_t node_id,
                        const float* route_key,
                        uint64_t route_len,
                        const char* lexical_keys) {
  try {
    if (handle == nullptr || handle->store == nullptr) {
      return grm_fail_msg(handle, "invalid set_route arguments");
    }
    if (handle->store->get(node_id) == nullptr) {
      return grm_fail_msg(handle, "unknown GRM node id");
    }
    if (route_key == nullptr && route_len > 0) {
      return grm_fail_msg(handle, "null route key with nonzero length");
    }
    std::vector<float> key;
    key.reserve(static_cast<std::size_t>(route_len));
    for (uint64_t i = 0; i < route_len; ++i) {
      key.push_back(route_key[i]);
    }
    handle->router.upsert(node_id, std::move(key),
                          split_lexical_keys(lexical_keys));
    sync_router_node_state(handle, node_id);
    return 0;
  } catch (const std::exception& exc) {
    return grm_fail(handle, exc);
  }
}

int grm_store_set_route_multi(grm_store_handle* handle,
                              uint64_t node_id,
                              const float* route_keys,
                              uint64_t key_count,
                              uint64_t route_len,
                              const char* lexical_keys) {
  try {
    if (handle == nullptr || handle->store == nullptr) {
      return grm_fail_msg(handle, "invalid set_route_multi arguments");
    }
    if (handle->store->get(node_id) == nullptr) {
      return grm_fail_msg(handle, "unknown GRM node id");
    }
    if (route_keys == nullptr && key_count > 0 && route_len > 0) {
      return grm_fail_msg(handle, "null route keys with nonzero shape");
    }
    std::vector<std::vector<float>> keys;
    keys.reserve(static_cast<std::size_t>(key_count));
    for (uint64_t k = 0; k < key_count; ++k) {
      std::vector<float> key;
      key.reserve(static_cast<std::size_t>(route_len));
      for (uint64_t i = 0; i < route_len; ++i) {
        key.push_back(route_keys[k * route_len + i]);
      }
      keys.push_back(std::move(key));
    }
    handle->router.upsert_multi(node_id, std::move(keys),
                                split_lexical_keys(lexical_keys));
    sync_router_node_state(handle, node_id);
    return 0;
  } catch (const std::exception& exc) {
    return grm_fail(handle, exc);
  }
}

int grm_store_route(grm_store_handle* handle,
                    const float* query,
                    uint64_t query_len,
                    const char* lexical_keys,
                    uint64_t topk,
                    uint64_t* out_node_ids,
                    uint64_t out_cap,
                    uint64_t* out_count) {
  try {
    if (handle == nullptr || handle->store == nullptr || out_count == nullptr) {
      return grm_fail_msg(handle, "invalid route arguments");
    }
    if (query == nullptr && query_len > 0) {
      return grm_fail_msg(handle, "null query with nonzero length");
    }
    if (out_node_ids == nullptr && out_cap > 0) {
      return grm_fail_msg(handle, "null output buffer with nonzero capacity");
    }
    std::vector<float> q;
    q.reserve(static_cast<std::size_t>(query_len));
    for (uint64_t i = 0; i < query_len; ++i) {
      q.push_back(query[i]);
    }
    const auto routed = handle->router.route(
        q, split_lexical_keys(lexical_keys), static_cast<std::size_t>(topk));
    const uint64_t n = std::min<uint64_t>(
        static_cast<uint64_t>(routed.size()), out_cap);
    for (uint64_t i = 0; i < n; ++i) {
      out_node_ids[i] = routed[static_cast<std::size_t>(i)];
    }
    *out_count = n;
    return 0;
  } catch (const std::exception& exc) {
    return grm_fail(handle, exc);
  }
}

int grm_store_route_filtered(grm_store_handle* handle,
                             const float* query,
                             uint64_t query_len,
                             const char* lexical_keys,
                             const char* kind_filters,
                             const char* scope_filters,
                             const char* durability_filters,
                             const char* mutability_filters,
                             uint64_t topk,
                             uint64_t* out_node_ids,
                             uint64_t out_cap,
                             uint64_t* out_count) {
  try {
    if (handle == nullptr || handle->store == nullptr || out_count == nullptr) {
      return grm_fail_msg(handle, "invalid route_filtered arguments");
    }
    if (query == nullptr && query_len > 0) {
      return grm_fail_msg(handle, "null query with nonzero length");
    }
    if (out_node_ids == nullptr && out_cap > 0) {
      return grm_fail_msg(handle, "null output buffer with nonzero capacity");
    }
    std::vector<float> q;
    q.reserve(static_cast<std::size_t>(query_len));
    for (uint64_t i = 0; i < query_len; ++i) {
      q.push_back(query[i]);
    }
    const auto routed = handle->router.route(
        q, split_lexical_keys(lexical_keys), static_cast<std::size_t>(topk),
        split_lexical_keys(kind_filters), split_lexical_keys(scope_filters),
        split_lexical_keys(durability_filters),
        split_lexical_keys(mutability_filters));
    const uint64_t n = std::min<uint64_t>(
        static_cast<uint64_t>(routed.size()), out_cap);
    for (uint64_t i = 0; i < n; ++i) {
      out_node_ids[i] = routed[static_cast<std::size_t>(i)];
    }
    *out_count = n;
    return 0;
  } catch (const std::exception& exc) {
    return grm_fail(handle, exc);
  }
}

int grm_store_configure_arena(grm_store_handle* handle,
                              uint64_t sink_tokens,
                              uint64_t arena_width) {
  try {
    if (handle == nullptr || handle->store == nullptr) {
      return grm_fail_msg(handle, "invalid configure_arena arguments");
    }
    handle->arena.configure(sink_tokens, arena_width);
    return 0;
  } catch (const std::exception& exc) {
    return grm_fail(handle, exc);
  }
}

int grm_store_plan_swap(grm_store_handle* handle,
                        uint64_t new_mount_tokens,
                        uint64_t input_cache_tokens,
                        grm_arena_swap_plan_c* out) {
  try {
    if (handle == nullptr || handle->store == nullptr || out == nullptr) {
      return grm_fail_msg(handle, "invalid plan_swap arguments");
    }
    const auto p = handle->arena.plan_swap(new_mount_tokens, input_cache_tokens);
    out->sink_tokens = p.sink_tokens;
    out->arena_width = p.arena_width;
    out->old_mount_tokens = p.old_mount_tokens;
    out->new_mount_tokens = p.new_mount_tokens;
    out->old_mount_end = p.old_mount_end;
    out->live_tail_start = p.live_tail_start;
    out->live_tail_tokens = p.live_tail_tokens;
    out->input_cache_tokens = p.input_cache_tokens;
    out->output_cache_tokens = p.output_cache_tokens;
    out->overflow = p.overflow ? 1 : 0;
    return 0;
  } catch (const std::exception& exc) {
    return grm_fail(handle, exc);
  }
}

int grm_store_plan_evict(grm_store_handle* handle,
                         uint64_t drop_tokens,
                         uint64_t input_cache_tokens,
                         grm_arena_evict_plan_c* out) {
  try {
    if (handle == nullptr || handle->store == nullptr || out == nullptr) {
      return grm_fail_msg(handle, "invalid plan_evict arguments");
    }
    const auto p = handle->arena.plan_evict(drop_tokens, input_cache_tokens);
    out->sink_tokens = p.sink_tokens;
    out->arena_width = p.arena_width;
    out->mount_tokens = p.mount_tokens;
    out->head_tokens = p.head_tokens;
    out->drop_tokens = p.drop_tokens;
    out->input_cache_tokens = p.input_cache_tokens;
    out->output_cache_tokens = p.output_cache_tokens;
    out->underflow = p.underflow ? 1 : 0;
    return 0;
  } catch (const std::exception& exc) {
    return grm_fail(handle, exc);
  }
}

int grm_store_apply_swap_tensor(grm_store_handle* handle,
                                uint64_t new_mount_tokens,
                                uint64_t input_cache_tokens,
                                const uint64_t* old_shape,
                                uint64_t rank,
                                uint64_t seq_dim,
                                uint64_t elem_size,
                                const uint8_t* old_payload,
                                uint64_t old_payload_len,
                                const uint64_t* mount_shape,
                                uint64_t mount_rank,
                                const uint8_t* mount_payload,
                                uint64_t mount_payload_len,
                                uint64_t* out_shape,
                                uint64_t out_shape_cap,
                                uint8_t* out_payload,
                                uint64_t out_payload_cap,
                                uint64_t* out_payload_len,
                                grm_arena_swap_plan_c* out_plan) {
  try {
    if (handle == nullptr || handle->store == nullptr ||
        out_payload_len == nullptr) {
      return grm_fail_msg(handle, "invalid apply_swap_tensor arguments");
    }
    if (old_shape == nullptr && rank > 0) {
      return grm_fail_msg(handle, "null old_shape with nonzero rank");
    }
    if (mount_shape == nullptr && mount_rank > 0) {
      return grm_fail_msg(handle, "null mount_shape with nonzero rank");
    }
    if (out_shape == nullptr && out_shape_cap > 0) {
      return grm_fail_msg(handle, "null output shape with nonzero capacity");
    }
    if (out_payload == nullptr && out_payload_cap > 0) {
      return grm_fail_msg(handle, "null output payload with nonzero capacity");
    }
    std::vector<std::uint64_t> old_dims;
    old_dims.reserve(static_cast<std::size_t>(rank));
    for (uint64_t i = 0; i < rank; ++i) {
      old_dims.push_back(old_shape[i]);
    }
    std::vector<std::uint64_t> mount_dims;
    mount_dims.reserve(static_cast<std::size_t>(mount_rank));
    for (uint64_t i = 0; i < mount_rank; ++i) {
      mount_dims.push_back(mount_shape[i]);
    }

    const auto result = handle->arena.apply_swap_tensor(
        old_dims, seq_dim, elem_size, old_payload, old_payload_len, mount_dims,
        mount_payload, mount_payload_len, new_mount_tokens, input_cache_tokens);
    *out_payload_len = static_cast<uint64_t>(result.bytes.size());
    if (out_shape != nullptr) {
      if (out_shape_cap < result.shape.size()) {
        return grm_fail_msg(handle, "output shape buffer too small");
      }
      for (std::size_t i = 0; i < result.shape.size(); ++i) {
        out_shape[i] = result.shape[i];
      }
    }
    if (out_payload != nullptr) {
      if (out_payload_cap < result.bytes.size()) {
        return grm_fail_msg(handle, "output payload buffer too small");
      }
      if (!result.bytes.empty()) {
        std::memcpy(out_payload, result.bytes.data(), result.bytes.size());
      }
    }
    if (out_plan != nullptr) {
      out_plan->sink_tokens = result.plan.sink_tokens;
      out_plan->arena_width = result.plan.arena_width;
      out_plan->old_mount_tokens = result.plan.old_mount_tokens;
      out_plan->new_mount_tokens = result.plan.new_mount_tokens;
      out_plan->old_mount_end = result.plan.old_mount_end;
      out_plan->live_tail_start = result.plan.live_tail_start;
      out_plan->live_tail_tokens = result.plan.live_tail_tokens;
      out_plan->input_cache_tokens = result.plan.input_cache_tokens;
      out_plan->output_cache_tokens = result.plan.output_cache_tokens;
      out_plan->overflow = result.plan.overflow ? 1 : 0;
    }
    return 0;
  } catch (const std::exception& exc) {
    return grm_fail(handle, exc);
  }
}

int grm_store_apply_evict_tensor(grm_store_handle* handle,
                                 uint64_t drop_tokens,
                                 uint64_t input_cache_tokens,
                                 const uint64_t* old_shape,
                                 uint64_t rank,
                                 uint64_t seq_dim,
                                 uint64_t elem_size,
                                 const uint8_t* old_payload,
                                 uint64_t old_payload_len,
                                 uint64_t* out_shape,
                                 uint64_t out_shape_cap,
                                 uint8_t* out_payload,
                                 uint64_t out_payload_cap,
                                 uint64_t* out_payload_len,
                                 grm_arena_evict_plan_c* out_plan) {
  try {
    if (handle == nullptr || handle->store == nullptr ||
        out_payload_len == nullptr) {
      return grm_fail_msg(handle, "invalid apply_evict_tensor arguments");
    }
    if (old_shape == nullptr && rank > 0) {
      return grm_fail_msg(handle, "null old_shape with nonzero rank");
    }
    if (out_shape == nullptr && out_shape_cap > 0) {
      return grm_fail_msg(handle, "null output shape with nonzero capacity");
    }
    if (out_payload == nullptr && out_payload_cap > 0) {
      return grm_fail_msg(handle, "null output payload with nonzero capacity");
    }
    std::vector<std::uint64_t> old_dims;
    old_dims.reserve(static_cast<std::size_t>(rank));
    for (uint64_t i = 0; i < rank; ++i) {
      old_dims.push_back(old_shape[i]);
    }

    const auto result = handle->arena.apply_evict_tensor(
        old_dims, seq_dim, elem_size, old_payload, old_payload_len,
        drop_tokens, input_cache_tokens);
    *out_payload_len = static_cast<uint64_t>(result.bytes.size());
    if (out_shape != nullptr) {
      if (out_shape_cap < result.shape.size()) {
        return grm_fail_msg(handle, "output shape buffer too small");
      }
      for (std::size_t i = 0; i < result.shape.size(); ++i) {
        out_shape[i] = result.shape[i];
      }
    }
    if (out_payload != nullptr) {
      if (out_payload_cap < result.bytes.size()) {
        return grm_fail_msg(handle, "output payload buffer too small");
      }
      if (!result.bytes.empty()) {
        std::memcpy(out_payload, result.bytes.data(), result.bytes.size());
      }
    }
    if (out_plan != nullptr) {
      out_plan->sink_tokens = result.plan.sink_tokens;
      out_plan->arena_width = result.plan.arena_width;
      out_plan->mount_tokens = result.plan.mount_tokens;
      out_plan->head_tokens = result.plan.head_tokens;
      out_plan->drop_tokens = result.plan.drop_tokens;
      out_plan->input_cache_tokens = result.plan.input_cache_tokens;
      out_plan->output_cache_tokens = result.plan.output_cache_tokens;
      out_plan->underflow = result.plan.underflow ? 1 : 0;
    }
    return 0;
  } catch (const std::exception& exc) {
    return grm_fail(handle, exc);
  }
}

int grm_store_commit_mount(grm_store_handle* handle,
                           const uint64_t* node_ids,
                           uint64_t node_count,
                           uint64_t mount_tokens) {
  try {
    if (handle == nullptr || handle->store == nullptr) {
      return grm_fail_msg(handle, "invalid commit_mount arguments");
    }
    if (node_ids == nullptr && node_count > 0) {
      return grm_fail_msg(handle, "null node_ids with nonzero count");
    }
    std::vector<std::uint64_t> ids;
    ids.reserve(static_cast<std::size_t>(node_count));
    for (uint64_t i = 0; i < node_count; ++i) {
      ids.push_back(node_ids[i]);
    }
    handle->arena.commit_mount(std::move(ids), mount_tokens);
    return 0;
  } catch (const std::exception& exc) {
    return grm_fail(handle, exc);
  }
}

int grm_store_save_checkpoint(grm_store_handle* handle, const char* root) {
  try {
    if (handle == nullptr || handle->store == nullptr || root == nullptr) {
      return grm_fail_msg(handle, "invalid save_checkpoint arguments");
    }
    handle->store->save_checkpoint(root);
    return 0;
  } catch (const std::exception& exc) {
    return grm_fail(handle, exc);
  }
}

int grm_store_load_checkpoint(grm_store_handle* handle, const char* root) {
  try {
    if (handle == nullptr || handle->store == nullptr || root == nullptr) {
      return grm_fail_msg(handle, "invalid load_checkpoint arguments");
    }
    handle->store->load_checkpoint(root);
    handle->router = grm::RouterIndex();
    return 0;
  } catch (const std::exception& exc) {
    return grm_fail(handle, exc);
  }
}

int grm_store_mark_durable(grm_store_handle* handle, uint64_t node_id) {
  try {
    if (handle == nullptr || handle->store == nullptr) {
      return grm_fail_msg(handle, "invalid mark_durable arguments");
    }
    handle->store->mark_durable(node_id);
    return 0;
  } catch (const std::exception& exc) {
    return grm_fail(handle, exc);
  }
}

int grm_store_evict_device_copy(grm_store_handle* handle, uint64_t node_id) {
  try {
    if (handle == nullptr || handle->store == nullptr) {
      return grm_fail_msg(handle, "invalid evict_device_copy arguments");
    }
    handle->store->evict_device_copy(node_id);
    return 0;
  } catch (const std::exception& exc) {
    return grm_fail(handle, exc);
  }
}

int grm_store_stats(grm_store_handle* handle, grm_store_stats_c* out) {
  try {
    if (handle == nullptr || handle->store == nullptr || out == nullptr) {
      return grm_fail_msg(handle, "invalid stats arguments");
    }
    const auto s = handle->store->stats();
    out->nodes = s.nodes;
    out->dirty_nodes = s.dirty_nodes;
    out->durable_nodes = s.durable_nodes;
    out->host_payload_bytes = s.host_payload_bytes;
    out->host_payload_tensors = s.host_payload_tensors;
    out->route_entries = handle->router.size();
    return 0;
  } catch (const std::exception& exc) {
    return grm_fail(handle, exc);
  }
}

const char* grm_store_last_error(grm_store_handle* handle) {
  if (handle == nullptr) {
    return "invalid GRM store handle";
  }
  return handle->last_error.c_str();
}

}  // extern "C"
