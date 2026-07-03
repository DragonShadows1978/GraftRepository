#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace grm {

enum class PayloadKind {
  MLA,
  GQA,
};

struct DialectDescriptor {
  std::string model_type;
  int num_layers = 0;
  int hidden_dim = 0;
  PayloadKind payload_kind = PayloadKind::MLA;
  int vals_per_tok_layer = 0;
  int route_layer = 0;
  int latent_rank = 0;
  int rope_dim = 0;
  int num_kv_heads = 0;
  int head_dim = 0;
  std::string position_law = "rope_full_kv";
  std::string state_kind = "kv";
  std::string graftability = "seat_remountable";
  bool remountable = true;
  std::string composition = "multi_mount";

  std::string dialect_id() const;
  std::string profile_id() const;
};

struct NodeLifecycle {
  bool host_present = false;
  bool device_present = false;
  bool dirty = false;
  bool durable = false;
  bool cold_only = false;
};

struct NodeMetadata {
  std::string json = "{}";
  std::string kind = "turn";
  std::string durability = "session";
  std::string mutability = "ephemeral";
  std::string scope = "conversation";
  std::string write_intent = "observed";
  double confidence = 1.0;
  bool active = true;
  std::string subject;
  std::string predicate;
  std::string value;
  std::string valid_from;
  std::string expires_at;
  std::vector<std::uint64_t> source_turns;
  std::vector<std::uint64_t> source_grafts;
  std::vector<std::uint64_t> supersedes;
  std::vector<std::uint64_t> superseded_by;
};

struct HostTensor {
  std::string name;
  std::string dtype = "uint8";
  std::vector<std::uint64_t> shape;
  std::vector<std::uint8_t> bytes;
};

struct HostPayload {
  std::vector<HostTensor> tensors;
  std::uint64_t bytes() const;
  std::uint64_t tensor_count() const;
};

struct ProvenanceRecord {
  std::uint64_t segment_id = 0;
  std::uint64_t node_id = 0;
  std::string segment_type;
  double created_at = 0.0;
};

struct HostGraftNode {
  std::uint64_t node_id = 0;
  std::string text;
  std::uint64_t ntok = 0;
  NodeMetadata metadata;
  std::string provenance_json = "[]";
  bool no_fold = false;
  NodeLifecycle lifecycle;
  HostPayload payload;
  std::vector<float> route_key;
  std::vector<std::vector<float>> route_keys;
  std::vector<std::string> lexical_keys;
  std::vector<std::uint64_t> sources;
  std::vector<ProvenanceRecord> provenance;
};

struct StoreStats {
  std::uint64_t nodes = 0;
  std::uint64_t dirty_nodes = 0;
  std::uint64_t durable_nodes = 0;
  std::uint64_t host_payload_bytes = 0;
  std::uint64_t host_payload_tensors = 0;
  std::uint64_t route_entries = 0;
};

struct DirtyNodeInfo {
  std::uint64_t node_id = 0;
  bool payload_dirty = false;
  bool metadata_dirty = false;
  std::uint64_t payload_bytes = 0;
  std::uint64_t durability_priority = 0;
};

struct PayloadStats {
  std::uint64_t tensor_count = 0;
  std::uint64_t payload_bytes = 0;
};

struct GraphEdges {
  std::vector<std::uint64_t> source_turns;
  std::vector<std::uint64_t> source_grafts;
  std::vector<std::uint64_t> supersedes;
  std::vector<std::uint64_t> superseded_by;
};

struct FactIdentity {
  std::string subject;
  std::string predicate;
  std::string value;
  std::string scope = "project";
  std::string valid_from;
  std::string expires_at;
};

struct ArenaSwapPlan {
  std::uint64_t sink_tokens = 0;
  std::uint64_t arena_width = 0;
  std::uint64_t old_mount_tokens = 0;
  std::uint64_t new_mount_tokens = 0;
  std::uint64_t old_mount_end = 0;
  std::uint64_t live_tail_start = 0;
  std::uint64_t live_tail_tokens = 0;
  std::uint64_t input_cache_tokens = 0;
  std::uint64_t output_cache_tokens = 0;
  bool overflow = false;
};

struct ArenaEvictPlan {
  std::uint64_t sink_tokens = 0;
  std::uint64_t arena_width = 0;
  std::uint64_t mount_tokens = 0;
  std::uint64_t head_tokens = 0;
  std::uint64_t drop_tokens = 0;
  std::uint64_t input_cache_tokens = 0;
  std::uint64_t output_cache_tokens = 0;
  bool underflow = false;
};

struct TensorSwapResult {
  ArenaSwapPlan plan;
  std::vector<std::uint64_t> shape;
  std::vector<std::uint8_t> bytes;
};

struct TensorEvictResult {
  ArenaEvictPlan plan;
  std::vector<std::uint64_t> shape;
  std::vector<std::uint8_t> bytes;
};

struct MemoryCommandPlan {
  std::string action;
  std::string command;
  std::string body;
  std::string query;
  std::string replacement;
  std::string durability;
  std::string durability_mode;
  std::string mutability;
  std::string scope;
  std::string kind;
  std::string boundary;
  std::string metadata_key;
  std::string metadata_value;
  std::string reason;
  std::uint64_t node_id = 0;
  std::uint64_t review_id = 0;
  std::uint64_t max_tokens = 0;
  std::uint64_t span_start = 0;
  std::uint64_t span_end = 0;
  bool has_node_id = false;
  bool has_review_id = false;
  bool has_max_tokens = false;
  bool has_span = false;
  bool flush_immediately = false;
};

MemoryCommandPlan parse_memory_command(const std::string& text);
std::string memory_command_plan_json(const MemoryCommandPlan& plan);

struct RememberFlushPlan {
  bool force_flush = false;
  std::string reason;
};

RememberFlushPlan plan_remember_flush(
    const std::string& durability_mode,
    const std::string& durability,
    const std::string& scope,
    bool flush_immediately);
std::string remember_flush_plan_json(const RememberFlushPlan& plan);

struct RuntimeEventPlan {
  bool flush = false;
  bool page = true;
  bool read_only = false;
  std::string reason;
};

RuntimeEventPlan plan_runtime_event(
    const std::string& event,
    const std::string& action,
    bool autosave_enabled,
    bool force_flush,
    bool read_only);
std::string runtime_event_plan_json(const RuntimeEventPlan& plan);

struct DurabilityModePlan {
  std::string durability_mode;
  bool target_wal_enabled = false;
  bool final_wal_enabled = false;
  bool append_config_before = false;
  bool append_config_after = false;
};

DurabilityModePlan plan_durability_mode(
    const std::string& requested_mode,
    const std::string& current_mode,
    bool old_wal_enabled,
    bool wal_enabled_override);
std::string durability_mode_plan_json(const DurabilityModePlan& plan);

struct MetadataUpdatePlan {
  std::string key;
  std::string string_value;
  bool bool_value = false;
  bool value_is_bool = false;
};

MetadataUpdatePlan plan_metadata_update(
    const std::string& command,
    const std::string& metadata_key,
    const std::string& metadata_value);
std::string metadata_update_plan_json(const MetadataUpdatePlan& plan);

struct MemoryMutationPlan {
  std::string action;
  std::string reason;
  std::uint64_t target_count = 0;
  bool apply_expire = false;
  bool apply_revision = false;
  bool write_replacement = false;
  bool update_metadata = false;
};

MemoryMutationPlan plan_memory_mutation(
    const std::string& command,
    bool has_query,
    std::uint64_t target_count,
    bool has_replacement);
std::string memory_mutation_plan_json(const MemoryMutationPlan& plan);

struct LibrarianPlan {
  std::uint64_t pending_jobs = 0;
  std::uint64_t digest_source_count = 0;
  std::uint64_t era_source_count = 0;
  bool deferred_backpressure = false;
  std::string reason;
};

LibrarianPlan plan_librarian(
    std::uint64_t foldable_turn_count,
    std::uint64_t foldable_digest_count,
    std::uint64_t turns_high,
    std::uint64_t turns_fold,
    std::uint64_t digests_high,
    std::uint64_t digests_fold,
    bool era_enabled,
    bool deferred_backpressure);
std::string librarian_plan_json(const LibrarianPlan& plan);

struct ExtractionPolicyPlan {
  std::string action;
  std::string reason;
};

ExtractionPolicyPlan plan_extraction_policy(
    const std::string& action,
    const std::string& write_intent,
    double confidence,
    double write_direct_threshold,
    std::uint64_t conflict_count,
    std::uint64_t requested_supersede_count,
    std::uint64_t requested_id_count,
    std::uint64_t equivalent_count,
    std::uint64_t expire_target_count);
std::string extraction_policy_plan_json(const ExtractionPolicyPlan& plan);

struct ReinforcementPlan {
  std::string write_intent;
  double confidence = 0.0;
  std::uint64_t reinforcement_count = 0;
};

ReinforcementPlan plan_reinforcement(
    const std::string& old_write_intent,
    const std::string& new_write_intent,
    double old_confidence,
    double new_confidence,
    std::uint64_t old_reinforcement_count);
std::string reinforcement_plan_json(const ReinforcementPlan& plan);

struct ReviewTransitionPlan {
  std::string action;
  std::string reason;
};

ReviewTransitionPlan plan_review_transition(
    const std::string& command,
    const std::string& status,
    bool has_approved_node_id);
std::string review_transition_plan_json(const ReviewTransitionPlan& plan);

struct CullSpan {
  std::uint64_t start = 0;
  std::uint64_t end = 0;
};

struct CullSpanPlan {
  std::vector<CullSpan> spans;
  bool retire_parent = true;
};

CullSpanPlan plan_cull_spans(
    std::uint64_t ntok,
    bool has_max_tokens,
    std::uint64_t max_tokens,
    const std::vector<CullSpan>& spans,
    bool retire_parent);
std::string cull_span_plan_json(const CullSpanPlan& plan);

class DirtyQueue {
 public:
  void mark(std::uint64_t node_id, bool payload, bool metadata);
  void clear(std::uint64_t node_id);
  void clear_all();
  bool empty() const;
  std::vector<std::uint64_t> node_ids() const;
  bool payload_dirty(std::uint64_t node_id) const;
  bool metadata_dirty(std::uint64_t node_id) const;

 private:
  struct DirtyState {
    bool payload = false;
    bool metadata = false;
  };
  std::unordered_map<std::uint64_t, DirtyState> dirty_;
};

class HostGraftStore {
 public:
  explicit HostGraftStore(DialectDescriptor dialect);

  std::uint64_t add_node(HostGraftNode node);
  HostGraftNode* get(std::uint64_t node_id);
  const HostGraftNode* get(std::uint64_t node_id) const;

  void set_tensor(std::uint64_t node_id, HostTensor tensor);
  HostTensor slice_tensor(std::uint64_t node_id,
                          const std::string& name,
                          std::uint64_t axis,
                          std::uint64_t start,
                          std::uint64_t length) const;
  PayloadStats payload_stats(std::uint64_t node_id) const;
  void clear_payload(std::uint64_t node_id);
  void set_metadata_json(std::uint64_t node_id, std::string metadata_json);
  const std::string& metadata_json(std::uint64_t node_id) const;
  const std::string& text(std::uint64_t node_id) const;
  std::string node_summary_json(std::uint64_t node_id) const;
  void set_provenance_json(std::uint64_t node_id,
                           std::string provenance_json);
  const std::string& provenance_json(std::uint64_t node_id) const;
  void clear_route(std::uint64_t node_id);
  void set_active(std::uint64_t node_id, bool active);
  bool is_active(std::uint64_t node_id) const;
  void set_no_fold(std::uint64_t node_id, bool no_fold);
  std::vector<std::uint64_t> foldable_nodes(
      const std::string& kind,
      const std::vector<std::uint64_t>& excluded_node_ids = {}) const;
  void set_route_metadata(std::uint64_t node_id,
                          std::string kind,
                          std::string scope,
                          std::string durability,
                          std::string mutability);
  void set_fact_identity(std::uint64_t node_id, FactIdentity identity);
  std::vector<std::uint64_t> fact_matches(const FactIdentity& identity,
                                          std::uint64_t value_mode,
                                          std::uint64_t temporal_mode = 0) const;
  std::vector<std::uint64_t> filter_active_nodes(
      const std::vector<std::uint64_t>& node_ids) const;
  std::vector<std::uint64_t> active_text_matches(
      const std::string& query) const;
  void set_graph_edges(std::uint64_t node_id, GraphEdges edges);
  GraphEdges graph_edges(std::uint64_t node_id) const;
  std::vector<std::uint64_t> source_closure(
      const std::vector<std::uint64_t>& node_ids,
      std::uint64_t max_depth = 3,
      bool include_roots = false) const;
  void apply_revision(std::uint64_t replacement_node_id,
                      std::vector<std::uint64_t> supersedes);
  void apply_expire(std::vector<std::uint64_t> node_ids);
  void mark_dirty(std::uint64_t node_id, bool payload, bool metadata);
  void mark_durable(std::uint64_t node_id);
  void evict_device_copy(std::uint64_t node_id);
  void save_checkpoint(const std::string& root);
  void load_checkpoint(const std::string& root);
  StoreStats stats() const;
  std::vector<std::uint64_t> node_ids() const;
  std::vector<DirtyNodeInfo> dirty_plan() const;

  const DialectDescriptor& dialect() const { return dialect_; }
  DirtyQueue& dirty_queue() { return dirty_; }

 private:
  DialectDescriptor dialect_;
  DirtyQueue dirty_;
  std::uint64_t next_id_ = 0;
  std::unordered_map<std::uint64_t, HostGraftNode> nodes_;
};

class RouterIndex {
 public:
  void upsert(std::uint64_t node_id, std::vector<float> route_key,
              std::vector<std::string> lexical_keys);
  void upsert_multi(std::uint64_t node_id,
                    std::vector<std::vector<float>> route_keys,
                    std::vector<std::string> lexical_keys);
  void erase(std::uint64_t node_id);
  void set_active(std::uint64_t node_id, bool active);
  void set_route_metadata(std::uint64_t node_id,
                          std::string kind,
                          std::string scope,
                          std::string durability,
                          std::string mutability);
  std::vector<std::uint64_t> route(const std::vector<float>& query,
                                   const std::vector<std::string>& lexical,
                                   std::size_t topk,
                                   const std::vector<std::string>& kinds = {},
                                   const std::vector<std::string>& scopes = {},
                                   const std::vector<std::string>& durabilities = {},
                                   const std::vector<std::string>& mutabilities = {}) const;
  std::vector<std::uint64_t> route_gqa_raw(
      const std::vector<float>& query,
      std::uint64_t query_heads,
      std::uint64_t query_tokens,
      std::uint64_t head_dim,
      std::uint64_t kv_heads,
      const std::vector<std::string>& lexical,
      std::size_t topk,
      const std::vector<std::string>& kinds = {},
      const std::vector<std::string>& scopes = {},
      const std::vector<std::string>& durabilities = {},
      const std::vector<std::string>& mutabilities = {}) const;
  std::size_t size() const { return entries_.size(); }

 private:
  struct Entry {
    std::uint64_t node_id = 0;
    std::vector<std::vector<float>> route_keys;
    std::vector<std::string> lexical_keys;
    bool active = true;
    std::string kind = "turn";
    std::string scope = "conversation";
    std::string durability = "session";
    std::string mutability = "ephemeral";
  };
  struct MlaArena {
    bool valid = false;
    bool uniform_dim = true;
    std::size_t dim = 0;
    std::vector<float> rows;
    std::vector<float> norms;
    std::vector<std::size_t> entry_for_row;
  };

  void mark_mla_arena_dirty();
  void rebuild_mla_arena() const;
  bool entry_allowed(
      const Entry& entry,
      const std::vector<std::string>& kinds,
      const std::vector<std::string>& scopes,
      const std::vector<std::string>& durabilities,
      const std::vector<std::string>& mutabilities) const;
  std::vector<std::uint64_t> route_scan(
      const std::vector<float>& query,
      const std::vector<std::string>& lexical,
      std::size_t topk,
      const std::vector<std::string>& kinds,
      const std::vector<std::string>& scopes,
      const std::vector<std::string>& durabilities,
      const std::vector<std::string>& mutabilities) const;
  std::vector<std::uint64_t> route_mla_arena(
      const std::vector<float>& query,
      const std::vector<std::string>& lexical,
      std::size_t topk,
      const std::vector<std::string>& kinds,
      const std::vector<std::string>& scopes,
      const std::vector<std::string>& durabilities,
      const std::vector<std::string>& mutabilities) const;

  std::vector<Entry> entries_;
  mutable MlaArena mla_arena_;
  mutable bool mla_arena_dirty_ = true;
};

class DurabilityWriter {
 public:
  explicit DurabilityWriter(std::string root);
  void write_checkpoint(HostGraftStore& store);
  const std::string& root() const { return root_; }

 private:
  std::string root_;
};

class DeviceArena {
 public:
  void configure(std::uint64_t sink_tokens, std::uint64_t arena_width);
  ArenaSwapPlan plan_swap(std::uint64_t new_mount_tokens,
                          std::uint64_t input_cache_tokens) const;
  ArenaEvictPlan plan_evict(std::uint64_t drop_tokens,
                            std::uint64_t input_cache_tokens) const;
  TensorSwapResult apply_swap_tensor(
      const std::vector<std::uint64_t>& old_shape,
      std::uint64_t seq_dim,
      std::uint64_t elem_size,
      const std::uint8_t* old_payload,
      std::uint64_t old_payload_len,
      const std::vector<std::uint64_t>& mount_shape,
      const std::uint8_t* mount_payload,
      std::uint64_t mount_payload_len,
      std::uint64_t new_mount_tokens,
      std::uint64_t input_cache_tokens) const;
  TensorEvictResult apply_evict_tensor(
      const std::vector<std::uint64_t>& old_shape,
      std::uint64_t seq_dim,
      std::uint64_t elem_size,
      const std::uint8_t* old_payload,
      std::uint64_t old_payload_len,
      std::uint64_t drop_tokens,
      std::uint64_t input_cache_tokens) const;
  void commit_mount(std::vector<std::uint64_t> node_ids,
                    std::uint64_t mount_tokens);
  const std::vector<std::uint64_t>& mounted() const { return mounted_; }
  std::uint64_t sink_tokens() const { return sink_tokens_; }
  std::uint64_t arena_width() const { return arena_width_; }
  std::uint64_t mount_tokens() const { return mount_tokens_; }

 private:
  std::uint64_t sink_tokens_ = 0;
  std::uint64_t arena_width_ = 0;
  std::uint64_t mount_tokens_ = 0;
  std::vector<std::uint64_t> mounted_;
};

}  // namespace grm
