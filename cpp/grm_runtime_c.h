#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct grm_store_handle grm_store_handle;

typedef struct grm_store_stats_c {
  uint64_t nodes;
  uint64_t dirty_nodes;
  uint64_t durable_nodes;
  uint64_t host_payload_bytes;
  uint64_t host_payload_tensors;
  uint64_t route_entries;
} grm_store_stats_c;

typedef struct grm_payload_stats_c {
  uint64_t tensor_count;
  uint64_t payload_bytes;
} grm_payload_stats_c;

typedef struct grm_tensor_info_c {
  uint64_t rank;
  uint64_t payload_bytes;
} grm_tensor_info_c;

typedef struct grm_graph_edges_info_c {
  uint64_t source_turns;
  uint64_t source_grafts;
  uint64_t supersedes;
  uint64_t superseded_by;
} grm_graph_edges_info_c;

typedef struct grm_arena_swap_plan_c {
  uint64_t sink_tokens;
  uint64_t arena_width;
  uint64_t old_mount_tokens;
  uint64_t new_mount_tokens;
  uint64_t old_mount_end;
  uint64_t live_tail_start;
  uint64_t live_tail_tokens;
  uint64_t input_cache_tokens;
  uint64_t output_cache_tokens;
  int overflow;
} grm_arena_swap_plan_c;

typedef struct grm_arena_evict_plan_c {
  uint64_t sink_tokens;
  uint64_t arena_width;
  uint64_t mount_tokens;
  uint64_t head_tokens;
  uint64_t drop_tokens;
  uint64_t input_cache_tokens;
  uint64_t output_cache_tokens;
  int underflow;
} grm_arena_evict_plan_c;

grm_store_handle* grm_store_create_mla(const char* model_type,
                                       int num_layers,
                                       int hidden_dim,
                                       int vals_per_tok_layer,
                                       int route_layer,
                                       int latent_rank,
                                       int rope_dim);
grm_store_handle* grm_store_create_gqa(const char* model_type,
                                       int num_layers,
                                       int hidden_dim,
                                       int vals_per_tok_layer,
                                       int route_layer,
                                       int num_kv_heads,
                                       int head_dim);
void grm_store_destroy(grm_store_handle* handle);

int grm_store_dialect_id(grm_store_handle* handle, char* out, size_t out_cap);
int grm_store_add_node(grm_store_handle* handle,
                       const char* text,
                       uint64_t ntok,
                       const uint8_t* payload,
                       uint64_t payload_len,
                       uint64_t* out_node_id);
int grm_store_set_tensor(grm_store_handle* handle,
                         uint64_t node_id,
                         const char* name,
                         const char* dtype,
                         const uint64_t* shape,
                         uint64_t rank,
                         const uint8_t* payload,
                         uint64_t payload_len);
int grm_store_payload_stats(grm_store_handle* handle,
                            uint64_t node_id,
                            grm_payload_stats_c* out);
int grm_store_clear_payload(grm_store_handle* handle,
                            uint64_t node_id);
int grm_store_tensor_info(grm_store_handle* handle,
                          uint64_t node_id,
                          const char* name,
                          uint64_t* out_shape,
                          uint64_t shape_cap,
                          char* out_dtype,
                          size_t dtype_cap,
                          grm_tensor_info_c* out);
int grm_store_read_tensor(grm_store_handle* handle,
                          uint64_t node_id,
                          const char* name,
                          uint8_t* out_payload,
                          uint64_t payload_cap,
                          uint64_t* out_count);
int grm_store_set_metadata_json(grm_store_handle* handle,
                                uint64_t node_id,
                                const char* metadata_json);
int grm_store_set_active(grm_store_handle* handle,
                         uint64_t node_id,
                         int active);
int grm_store_set_route_metadata(grm_store_handle* handle,
                                 uint64_t node_id,
                                 const char* kind,
                                 const char* scope,
                                 const char* durability,
                                 const char* mutability);
int grm_store_set_graph_edges(grm_store_handle* handle,
                              uint64_t node_id,
                              const uint64_t* source_turns,
                              uint64_t source_turn_count,
                              const uint64_t* source_grafts,
                              uint64_t source_graft_count,
                              const uint64_t* supersedes,
                              uint64_t supersedes_count,
                              const uint64_t* superseded_by,
                              uint64_t superseded_by_count);
int grm_store_graph_edges_info(grm_store_handle* handle,
                               uint64_t node_id,
                               grm_graph_edges_info_c* out);
int grm_store_read_graph_edges(grm_store_handle* handle,
                               uint64_t node_id,
                               uint64_t* out_source_turns,
                               uint64_t source_turn_cap,
                               uint64_t* out_source_grafts,
                               uint64_t source_graft_cap,
                               uint64_t* out_supersedes,
                               uint64_t supersedes_cap,
                               uint64_t* out_superseded_by,
                               uint64_t superseded_by_cap,
                               grm_graph_edges_info_c* out_counts);
int grm_store_apply_revision(grm_store_handle* handle,
                             uint64_t replacement_node_id,
                             const uint64_t* supersedes,
                             uint64_t supersedes_count);
int grm_store_metadata_json(grm_store_handle* handle,
                            uint64_t node_id,
                            char* out_json,
                            size_t out_cap,
                            uint64_t* out_len);
int grm_store_parse_memory_command(grm_store_handle* handle,
                                   const char* text,
                                   char* out_json,
                                   size_t out_cap,
                                   uint64_t* out_len);
int grm_store_set_route(grm_store_handle* handle,
                        uint64_t node_id,
                        const float* route_key,
                        uint64_t route_len,
                        const char* lexical_keys);
int grm_store_set_route_multi(grm_store_handle* handle,
                              uint64_t node_id,
                              const float* route_keys,
                              uint64_t key_count,
                              uint64_t route_len,
                              const char* lexical_keys);
int grm_store_route(grm_store_handle* handle,
                    const float* query,
                    uint64_t query_len,
                    const char* lexical_keys,
                    uint64_t topk,
                    uint64_t* out_node_ids,
                    uint64_t out_cap,
                    uint64_t* out_count);
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
                             uint64_t* out_count);
int grm_store_configure_arena(grm_store_handle* handle,
                              uint64_t sink_tokens,
                              uint64_t arena_width);
int grm_store_plan_swap(grm_store_handle* handle,
                        uint64_t new_mount_tokens,
                        uint64_t input_cache_tokens,
                        grm_arena_swap_plan_c* out);
int grm_store_plan_evict(grm_store_handle* handle,
                         uint64_t drop_tokens,
                         uint64_t input_cache_tokens,
                         grm_arena_evict_plan_c* out);
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
                                grm_arena_swap_plan_c* out_plan);
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
                                 grm_arena_evict_plan_c* out_plan);
int grm_store_commit_mount(grm_store_handle* handle,
                           const uint64_t* node_ids,
                           uint64_t node_count,
                           uint64_t mount_tokens);
int grm_store_save_checkpoint(grm_store_handle* handle, const char* root);
int grm_store_load_checkpoint(grm_store_handle* handle, const char* root);
int grm_store_mark_durable(grm_store_handle* handle, uint64_t node_id);
int grm_store_evict_device_copy(grm_store_handle* handle, uint64_t node_id);
int grm_store_stats(grm_store_handle* handle, grm_store_stats_c* out);
const char* grm_store_last_error(grm_store_handle* handle);

#ifdef __cplusplus
}
#endif
