#include "grm_runtime.hpp"
#include "grm_runtime_c.h"

#include <algorithm>
#include <cerrno>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <limits>
#include <memory>
#include <mutex>
#include <set>
#include <shared_mutex>
#include <stdexcept>
#include <sstream>

#if defined(__unix__) || defined(__APPLE__)
#include <fcntl.h>
#include <unistd.h>
#endif

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

std::string DialectDescriptor::profile_id() const {
  return position_law + "|" + state_kind + "|" + graftability + "|" +
         (remountable ? "1" : "0") + "|" + composition;
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

bool DirtyQueue::payload_dirty(std::uint64_t node_id) const {
  const auto it = dirty_.find(node_id);
  return it != dirty_.end() && it->second.payload;
}

bool DirtyQueue::metadata_dirty(std::uint64_t node_id) const {
  const auto it = dirty_.find(node_id);
  return it != dirty_.end() && it->second.metadata;
}

std::uint64_t durability_priority(const std::string& durability) {
  if (durability == "permanent") {
    return 0;
  }
  if (durability == "project") {
    return 1;
  }
  if (durability == "session") {
    return 2;
  }
  if (durability == "volatile") {
    return 3;
  }
  return 4;
}

namespace {

constexpr char kCheckpointMagicV1[] = "GRMSTORE1";
constexpr char kCheckpointMagicV2[] = "GRMSTORE2";
constexpr char kCheckpointMagicV3[] = "GRMSTORE3";
constexpr char kCheckpointMagicV4[] = "GRMSTORE4";
constexpr char kCheckpointMagicV5[] = "GRMSTORE5";
constexpr char kCheckpointMagicV6[] = "GRMSTORE6";
constexpr char kCheckpointMagicV7[] = "GRMSTORE7";
constexpr char kCheckpointMagicV8[] = "GRMSTORE8";
constexpr char kCheckpointMagicV9[] = "GRMSTORE9";
constexpr char kCheckpointMagicV10[] = "GRMSTOR10";
constexpr char kCheckpointMagic[] = "GRMSTOR10";

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

void write_u64_vector(std::ostream& out, const std::vector<std::uint64_t>& xs) {
  write_u64(out, static_cast<std::uint64_t>(xs.size()));
  for (const auto x : xs) {
    write_u64(out, x);
  }
}

void write_f32_vector(std::ostream& out, const std::vector<float>& xs) {
  write_u64(out, static_cast<std::uint64_t>(xs.size()));
  if (!xs.empty()) {
    out.write(reinterpret_cast<const char*>(xs.data()),
              static_cast<std::streamsize>(xs.size() * sizeof(float)));
  }
}

void write_f32_vectors(std::ostream& out,
                       const std::vector<std::vector<float>>& xs) {
  write_u64(out, static_cast<std::uint64_t>(xs.size()));
  for (const auto& x : xs) {
    write_f32_vector(out, x);
  }
}

void write_string_vector(std::ostream& out,
                         const std::vector<std::string>& xs) {
  write_u64(out, static_cast<std::uint64_t>(xs.size()));
  for (const auto& x : xs) {
    write_string(out, x);
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

std::vector<std::uint64_t> read_u64_vector(std::istream& in) {
  const auto n = read_u64(in);
  std::vector<std::uint64_t> out;
  out.reserve(static_cast<std::size_t>(n));
  for (std::uint64_t i = 0; i < n; ++i) {
    out.push_back(read_u64(in));
  }
  return out;
}

std::vector<float> read_f32_vector(std::istream& in) {
  const auto n = read_u64(in);
  std::vector<float> out(static_cast<std::size_t>(n));
  if (n > 0) {
    in.read(reinterpret_cast<char*>(out.data()),
            static_cast<std::streamsize>(n * sizeof(float)));
    if (!in) {
      throw std::runtime_error("truncated GRM checkpoint float vector");
    }
  }
  return out;
}

std::vector<std::vector<float>> read_f32_vectors(std::istream& in) {
  const auto n = read_u64(in);
  std::vector<std::vector<float>> out;
  out.reserve(static_cast<std::size_t>(n));
  for (std::uint64_t i = 0; i < n; ++i) {
    out.push_back(read_f32_vector(in));
  }
  return out;
}

std::vector<std::string> read_string_vector(std::istream& in) {
  const auto n = read_u64(in);
  std::vector<std::string> out;
  out.reserve(static_cast<std::size_t>(n));
  for (std::uint64_t i = 0; i < n; ++i) {
    out.push_back(read_string(in));
  }
  return out;
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

std::string trim(std::string s) {
  const auto first = std::find_if_not(s.begin(), s.end(), [](unsigned char c) {
    return std::isspace(c) != 0;
  });
  const auto last = std::find_if_not(s.rbegin(), s.rend(),
                                     [](unsigned char c) {
                                       return std::isspace(c) != 0;
                                     }).base();
  if (first >= last) {
    return "";
  }
  return std::string(first, last);
}

std::string ascii_lower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return s;
}

bool starts_with(const std::string& s, const std::string& prefix) {
  return s.size() >= prefix.size() &&
         std::equal(prefix.begin(), prefix.end(), s.begin());
}

std::vector<std::string> command_words(std::string s) {
  for (char& c : s) {
    if (c == ',' || c == ':' || c == '=') {
      c = ' ';
    }
  }
  std::istringstream in(s);
  std::vector<std::string> out;
  std::string word;
  while (in >> word) {
    out.push_back(word);
  }
  return out;
}

bool parse_u64_word(const std::string& word, std::uint64_t* out) {
  if (word.empty()) {
    return false;
  }
  std::uint64_t value = 0;
  for (const unsigned char c : word) {
    if (std::isdigit(c) == 0) {
      return false;
    }
    const auto digit = static_cast<std::uint64_t>(c - '0');
    if (value > (std::numeric_limits<std::uint64_t>::max() - digit) / 10) {
      throw std::overflow_error("integer command field overflows uint64");
    }
    value = value * 10 + digit;
  }
  *out = value;
  return true;
}

bool command_separator(char c) {
  const unsigned char uc = static_cast<unsigned char>(c);
  return std::isspace(uc) != 0 || c == ':' || c == '=' || c == ',';
}

std::string command_suffix_after_keyword(const std::string& original,
                                         const std::string& low,
                                         const std::string& keyword) {
  std::size_t pos = low.find(keyword);
  while (pos != std::string::npos) {
    std::size_t cursor = pos + keyword.size();
    const bool before_boundary = (
        pos == 0 || command_separator(low[pos - 1]));
    const bool after_boundary = (
        cursor >= low.size() || command_separator(low[cursor]));
    if (before_boundary) {
      if (pos > 0 && std::isspace(
              static_cast<unsigned char>(low[pos - 1])) == 0) {
        return "";
      }
      if (!after_boundary) {
        return "";
      }
      while (cursor < original.size()) {
        if (command_separator(original[cursor])) {
          ++cursor;
          continue;
        }
        break;
      }
      return trim(original.substr(cursor));
    }
    pos = low.find(keyword, pos + 1);
  }
  return "";
}

std::string command_suffix_after_label(const std::string& original,
                                       const std::string& low) {
  return command_suffix_after_keyword(original, low, "label");
}

std::string normalize_cull_boundary(const std::string& word) {
  if (word == "section" || word == "sections") {
    return "section";
  }
  if (word == "paragraph" || word == "paragraphs") {
    return "paragraph";
  }
  if (word == "turn" || word == "turns") {
    return "turn";
  }
  if (word == "heading" || word == "headings") {
    return "heading";
  }
  throw std::runtime_error("unknown cull boundary strategy");
}

std::string norm_fact_field(const std::string& value) {
  return ascii_lower(trim(value));
}

std::string norm_fact_scope(const std::string& value) {
  const auto out = norm_fact_field(value);
  return out.empty() ? "project" : out;
}

bool parse_int_span(const std::string& s,
                    std::size_t pos,
                    std::size_t len,
                    int* out) {
  if (out == nullptr || pos + len > s.size()) {
    return false;
  }
  int value = 0;
  for (std::size_t i = 0; i < len; ++i) {
    const unsigned char c = static_cast<unsigned char>(s[pos + i]);
    if (std::isdigit(c) == 0) {
      return false;
    }
    value = value * 10 + static_cast<int>(c - '0');
  }
  *out = value;
  return true;
}

long long days_from_civil(int y, unsigned m, unsigned d) {
  y -= m <= 2;
  const int era = (y >= 0 ? y : y - 399) / 400;
  const unsigned yoe = static_cast<unsigned>(y - era * 400);
  const unsigned mp = static_cast<unsigned>(
      static_cast<int>(m) + (m > 2 ? -3 : 9));
  const unsigned doy = (153 * mp + 2) / 5 + d - 1;
  const unsigned doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
  return static_cast<long long>(era) * 146097 +
         static_cast<long long>(doe) - 719468;
}

bool parse_fact_time_utc_seconds(const std::string& value, long long* out) {
  std::string s = trim(value);
  if (s.empty() || out == nullptr) {
    return false;
  }
  if (s.back() == 'Z' || s.back() == 'z') {
    s.pop_back();
    s += "+00:00";
  }
  int year = 0;
  int month = 0;
  int day = 0;
  if (s.size() < 10 || s[4] != '-' || s[7] != '-' ||
      !parse_int_span(s, 0, 4, &year) ||
      !parse_int_span(s, 5, 2, &month) ||
      !parse_int_span(s, 8, 2, &day)) {
    return false;
  }
  if (month < 1 || month > 12 || day < 1 || day > 31) {
    return false;
  }
  int hour = 0;
  int minute = 0;
  int second = 0;
  std::size_t pos = 10;
  if (pos < s.size() && (s[pos] == 'T' || s[pos] == 't' || s[pos] == ' ')) {
    if (pos + 9 > s.size() ||
        !parse_int_span(s, pos + 1, 2, &hour) ||
        s[pos + 3] != ':' ||
        !parse_int_span(s, pos + 4, 2, &minute) ||
        s[pos + 6] != ':' ||
        !parse_int_span(s, pos + 7, 2, &second)) {
      return false;
    }
    pos += 9;
    while (pos < s.size() && s[pos] != '+' && s[pos] != '-') {
      ++pos;
    }
  }
  if (hour < 0 || hour > 23 || minute < 0 || minute > 59 ||
      second < 0 || second > 60) {
    return false;
  }
  int offset_seconds = 0;
  if (pos < s.size() && (s[pos] == '+' || s[pos] == '-')) {
    const int sign = s[pos] == '-' ? -1 : 1;
    int offset_hour = 0;
    int offset_minute = 0;
    if (pos + 6 > s.size() || s[pos + 3] != ':' ||
        !parse_int_span(s, pos + 1, 2, &offset_hour) ||
        !parse_int_span(s, pos + 4, 2, &offset_minute) ||
        offset_hour > 23 || offset_minute > 59) {
      return false;
    }
    offset_seconds = sign * (offset_hour * 3600 + offset_minute * 60);
  }
  const long long day_seconds =
      days_from_civil(year, static_cast<unsigned>(month),
                      static_cast<unsigned>(day)) *
      86400LL;
  *out = day_seconds + hour * 3600LL + minute * 60LL + second -
         offset_seconds;
  return true;
}

bool fact_time_effective_at(const std::string& valid_from,
                            const std::string& expires_at,
                            long long now_seconds) {
  long long valid = 0;
  if (!trim(valid_from).empty() &&
      parse_fact_time_utc_seconds(valid_from, &valid) &&
      valid > now_seconds) {
    return false;
  }
  long long expires = 0;
  if (!trim(expires_at).empty() &&
      parse_fact_time_utc_seconds(expires_at, &expires) &&
      expires <= now_seconds) {
    return false;
  }
  return true;
}

std::string json_escape(const std::string& s) {
  std::ostringstream out;
  for (const unsigned char c : s) {
    switch (c) {
      case '"': out << "\\\""; break;
      case '\\': out << "\\\\"; break;
      case '\b': out << "\\b"; break;
      case '\f': out << "\\f"; break;
      case '\n': out << "\\n"; break;
      case '\r': out << "\\r"; break;
      case '\t': out << "\\t"; break;
      default:
        if (c < 0x20) {
          const char* hex = "0123456789abcdef";
          out << "\\u00" << hex[(c >> 4) & 0xf] << hex[c & 0xf];
        } else {
          out << static_cast<char>(c);
        }
    }
  }
  return out.str();
}

void append_json_string_field(std::ostringstream& out,
                              bool& first,
                              const char* key,
                              const std::string& value) {
  if (value.empty()) {
    return;
  }
  if (!first) {
    out << ",";
  }
  first = false;
  out << "\"" << key << "\":\"" << json_escape(value) << "\"";
}

void append_json_u64_field(std::ostringstream& out,
                           bool& first,
                           const char* key,
                           std::uint64_t value,
                           bool emit) {
  if (!emit) {
    return;
  }
  if (!first) {
    out << ",";
  }
  first = false;
  out << "\"" << key << "\":" << value;
}

void append_json_double_field(std::ostringstream& out,
                              bool& first,
                              const char* key,
                              double value) {
  if (!first) {
    out << ",";
  }
  first = false;
  out << "\"" << key << "\":" << std::setprecision(17) << value;
}

void append_json_bool_field(std::ostringstream& out,
                            bool& first,
                            const char* key,
                            bool value) {
  if (!first) {
    out << ",";
  }
  first = false;
  out << "\"" << key << "\":" << (value ? "true" : "false");
}

int write_intent_rank(const std::string& intent) {
  if (intent == "imported") {
    return 0;
  }
  if (intent == "inferred") {
    return 1;
  }
  if (intent == "observed") {
    return 2;
  }
  if (intent == "system_asserted") {
    return 3;
  }
  if (intent == "user_asserted") {
    return 4;
  }
  return 0;
}

void fsync_path(const std::filesystem::path& path) {
#if defined(__unix__) || defined(__APPLE__)
  const auto s = path.string();
  const int fd = ::open(s.c_str(), O_RDONLY);
  if (fd == -1) {
    throw std::runtime_error("failed to open for fsync: " + s + ": " +
                             std::strerror(errno));
  }
  if (::fsync(fd) == -1) {
    const int e = errno;
    ::close(fd);
    throw std::runtime_error("failed to fsync: " + s + ": " +
                             std::strerror(e));
  }
  if (::close(fd) == -1) {
    throw std::runtime_error("failed to close after fsync: " + s + ": " +
                             std::strerror(errno));
  }
#else
  (void)path;
#endif
}

void fsync_directory(const std::filesystem::path& path) {
#if defined(__unix__) || defined(__APPLE__)
  const auto s = path.string();
#ifdef O_DIRECTORY
  const int fd = ::open(s.c_str(), O_RDONLY | O_DIRECTORY);
#else
  const int fd = ::open(s.c_str(), O_RDONLY);
#endif
  if (fd == -1) {
    throw std::runtime_error("failed to open directory for fsync: " + s +
                             ": " + std::strerror(errno));
  }
  if (::fsync(fd) == -1) {
    const int e = errno;
    ::close(fd);
    throw std::runtime_error("failed to fsync directory: " + s + ": " +
                             std::strerror(e));
  }
  if (::close(fd) == -1) {
    throw std::runtime_error("failed to close directory after fsync: " + s +
                             ": " + std::strerror(errno));
  }
#else
  (void)path;
#endif
}

void atomic_write_text_file(const std::filesystem::path& dst,
                            const std::string& body) {
  std::filesystem::create_directories(dst.parent_path());
  const auto tmp = dst.string() + ".tmp";
  {
    std::ofstream out(tmp, std::ios::trunc);
    if (!out) {
      throw std::runtime_error("failed to open text checkpoint for write");
    }
    out << body;
    if (!out) {
      throw std::runtime_error("failed while writing text checkpoint");
    }
  }
  fsync_path(tmp);
  std::filesystem::rename(tmp, dst);
  fsync_directory(dst.parent_path());
}

bool contains_text(const std::string& text, const std::string& needle) {
  return text.find(needle) != std::string::npos;
}

bool is_fixed_position_profile(const DialectDescriptor& dialect) {
  const auto law = ascii_lower(dialect.position_law);
  const auto graftability = ascii_lower(dialect.graftability);
  return contains_text(law, "absolute") || contains_text(law, "fixed") ||
         contains_text(graftability, "absolute") ||
         contains_text(graftability, "fixed");
}

bool is_reseatable_position_law(const DialectDescriptor& dialect) {
  const auto law = ascii_lower(dialect.position_law);
  return contains_text(law, "rope") || contains_text(law, "rotary") ||
         contains_text(law, "relative");
}

void validate_dialect_profile(const DialectDescriptor& dialect) {
  if (!dialect.remountable) {
    return;
  }
  if (is_fixed_position_profile(dialect)) {
    throw std::invalid_argument(
        "fixed-position GRM profiles must set remountable=false");
  }
  if (!is_reseatable_position_law(dialect)) {
    throw std::invalid_argument(
        "remountable GRM profiles require a RoPE or relative position law");
  }
}

}  // namespace

MemoryCommandPlan parse_memory_command(const std::string& text) {
  const std::string original = trim(text);
  const std::string low = ascii_lower(original);
  struct RememberPrefix {
    const char* prefix;
    const char* durability;
    const char* scope;
    const char* kind;
    const char* mutability;
    bool flush;
  };
  const RememberPrefix remember_prefixes[] = {
      {"remember permanently:", "permanent", "user", "fact", "", true},
      {"remember this for the project:", "project", "project",
       "task_state", "", false},
      {"remember this for this session:", "session", "session",
       "task_state", "", false},
      {"this is temporary:", "volatile", "session", "task_state",
       "ephemeral", false},
  };

  for (const auto& p : remember_prefixes) {
    if (starts_with(low, p.prefix)) {
      MemoryCommandPlan plan;
      plan.action = "remember";
      plan.body = trim(original.substr(std::strlen(p.prefix)));
      plan.durability = p.durability;
      plan.scope = p.scope;
      plan.kind = p.kind;
      plan.mutability = p.mutability;
      plan.flush_immediately = p.flush;
      return plan;
    }
  }

  const auto words = command_words(low);
  if (words.size() >= 3 && words[1] == "review") {
    MemoryCommandPlan plan;
    if (!parse_u64_word(words[2], &plan.review_id)) {
      throw std::runtime_error("review command requires a numeric review id");
    }
    plan.has_review_id = true;
    if (words[0] == "approve") {
      if (words.size() != 3) {
        throw std::runtime_error("approve review takes only a review id");
      }
      plan.action = "approve_review";
      return plan;
    }
    if (words[0] == "reject") {
      plan.action = "reject_review";
      if (words.size() > 3) {
        if (words[3] != "reason") {
          throw std::runtime_error("unknown reject review option");
        }
        plan.reason = command_suffix_after_keyword(original, low, "reason");
        if (plan.reason.empty()) {
          throw std::runtime_error("reject review reason is missing");
        }
      }
      return plan;
    }
    if (words[0] == "edit") {
      if (words.size() < 5 ||
          (words[3] != "text" && words[3] != "body")) {
        throw std::runtime_error("edit review requires text <replacement>");
      }
      plan.action = "edit_review";
      plan.body = command_suffix_after_keyword(original, low, words[3]);
      if (plan.body.empty()) {
        throw std::runtime_error("edit review text is missing");
      }
      plan.reason = "memory command edit";
      return plan;
    }
    if (words[0] == "change") {
      plan.action = "change_review_scope";
      std::size_t cursor = 3;
      if (cursor >= words.size() || words[cursor] != "scope") {
        throw std::runtime_error("change review requires scope <scope>");
      }
      ++cursor;
      if (cursor >= words.size()) {
        throw std::runtime_error("change review scope is missing");
      }
      plan.scope = words[cursor++];
      while (cursor < words.size()) {
        const auto& word = words[cursor++];
        if (word == "durability") {
          if (cursor >= words.size()) {
            throw std::runtime_error(
                "change review durability is missing");
          }
          plan.durability = words[cursor++];
          continue;
        }
        if (word == "mutability") {
          if (cursor >= words.size()) {
            throw std::runtime_error(
                "change review mutability is missing");
          }
          plan.mutability = words[cursor++];
          continue;
        }
        throw std::runtime_error("unknown change review option");
      }
      return plan;
    }
  }

  if (words.size() >= 6 && words[0] == "select" && words[1] == "graft") {
    MemoryCommandPlan plan;
    plan.action = "select_graft_span";
    if (!parse_u64_word(words[2], &plan.node_id)) {
      throw std::runtime_error("select graft requires a numeric graft id");
    }
    plan.has_node_id = true;
    std::size_t cursor = 3;
    if (words[cursor] != "span" && words[cursor] != "tokens" &&
        words[cursor] != "token") {
      throw std::runtime_error(
          "select graft requires span <start> <end>");
    }
    ++cursor;
    if (cursor + 1 >= words.size()) {
      throw std::runtime_error("select graft span requires start and end");
    }
    if (!parse_u64_word(words[cursor], &plan.span_start) ||
        !parse_u64_word(words[cursor + 1], &plan.span_end)) {
      throw std::runtime_error("select graft span bounds must be numeric");
    }
    if (plan.span_end <= plan.span_start) {
      throw std::runtime_error("select graft span end must be greater than start");
    }
    plan.has_span = true;
    cursor += 2;
    if (cursor < words.size()) {
      if (words[cursor] != "label") {
        throw std::runtime_error("unknown select graft option");
      }
      ++cursor;
      if (cursor >= words.size()) {
        throw std::runtime_error("select graft label is missing");
      }
      plan.body = command_suffix_after_label(original, low);
      if (plan.body.empty()) {
        throw std::runtime_error("select graft label is missing");
      }
    }
    return plan;
  }

  if (words.size() >= 3 &&
      (words[0] == "cull" || words[0] == "split") &&
      words[1] == "graft") {
    MemoryCommandPlan plan;
    plan.action = "cull_graft";
    if (!parse_u64_word(words[2], &plan.node_id)) {
      throw std::runtime_error("cull graft requires a numeric graft id");
    }
    plan.has_node_id = true;
    std::size_t cursor = 3;
    while (cursor < words.size()) {
      const auto& word = words[cursor];
      if (word == "into" || word == "by") {
        ++cursor;
        if (cursor >= words.size()) {
          throw std::runtime_error("cull graft boundary is missing");
        }
        plan.boundary = normalize_cull_boundary(words[cursor]);
        ++cursor;
        continue;
      }
      if (word == "section" || word == "sections" ||
          word == "paragraph" || word == "paragraphs" ||
          word == "turn" || word == "turns" ||
          word == "heading" || word == "headings") {
        plan.boundary = normalize_cull_boundary(word);
        ++cursor;
        continue;
      }
      if (word == "max" || word == "max_tokens" ||
          word == "max-token" || word == "max-tokens") {
        if (word == "max") {
          ++cursor;
          if (cursor < words.size() &&
              (words[cursor] == "token" || words[cursor] == "tokens")) {
            ++cursor;
          }
        } else {
          ++cursor;
        }
        if (cursor >= words.size()) {
          throw std::runtime_error("cull graft max tokens is missing");
        }
        if (!parse_u64_word(words[cursor], &plan.max_tokens)) {
          throw std::runtime_error("cull graft max tokens must be numeric");
        }
        if (plan.max_tokens == 0) {
          throw std::runtime_error("cull graft max tokens must be positive");
        }
        plan.has_max_tokens = true;
        ++cursor;
        continue;
      }
      throw std::runtime_error("unknown cull graft option");
    }
    if (plan.boundary.empty() && !plan.has_max_tokens) {
      throw std::runtime_error(
          "cull graft requires max tokens or a boundary strategy");
    }
    return plan;
  }

  struct MetadataCommandPrefix {
    const char* prefix;
    const char* command;
    const char* key;
    const char* value;
  };
  const MetadataCommandPrefix metadata_prefixes[] = {
      {"pin memory:", "pin_memory", "pinned", "true"},
      {"pin this:", "pin_memory", "pinned", "true"},
      {"unpin memory:", "unpin_memory", "pinned", "false"},
      {"unpin this:", "unpin_memory", "pinned", "false"},
      {"mark memory mutable:", "mark_mutable", "mutability", "mutable"},
      {"mark this as mutable:", "mark_mutable", "mutability", "mutable"},
      {"mark memory stable:", "mark_stable", "mutability", "stable"},
      {"mark this as stable:", "mark_stable", "mutability", "stable"},
      {"mark memory immutable:", "mark_immutable", "mutability", "immutable"},
      {"mark this as immutable:", "mark_immutable", "mutability", "immutable"},
  };
  for (const auto& p : metadata_prefixes) {
    if (starts_with(low, p.prefix)) {
      MemoryCommandPlan plan;
      plan.action = "update_memory_metadata";
      plan.command = p.command;
      plan.query = trim(original.substr(std::strlen(p.prefix)));
      plan.metadata_key = p.key;
      plan.metadata_value = p.value;
      return plan;
    }
  }

  struct ReadCommandPrefix {
    const char* prefix;
    const char* action;
  };
  const ReadCommandPrefix read_prefixes[] = {
      {"show memory about:", "show_memory"},
      {"why do you remember that:", "why_memory"},
      {"why do you remember:", "why_memory"},
  };
  for (const auto& p : read_prefixes) {
    if (starts_with(low, p.prefix)) {
      MemoryCommandPlan plan;
      plan.action = p.action;
      plan.query = trim(original.substr(std::strlen(p.prefix)));
      return plan;
    }
  }

  struct ModeCommandPrefix {
    const char* prefix;
    const char* mode;
  };
  const ModeCommandPrefix mode_prefixes[] = {
      {"switch to volatile mode", "volatile"},
      {"switch to volatile-fast mode", "volatile_fast"},
      {"switch to volatile fast mode", "volatile_fast"},
      {"switch to session-safe mode", "session_safe"},
      {"switch to session safe mode", "session_safe"},
      {"switch to project-safe mode", "project_safe"},
      {"switch to project safe mode", "project_safe"},
      {"switch to durable-strict mode", "durable_strict"},
      {"switch to durable strict mode", "durable_strict"},
  };
  for (const auto& p : mode_prefixes) {
    if (low == p.prefix) {
      MemoryCommandPlan plan;
      plan.action = "set_durability_mode";
      plan.durability_mode = p.mode;
      return plan;
    }
  }

  if (starts_with(low, "forget:")) {
    MemoryCommandPlan plan;
    plan.action = "forget";
    plan.query = trim(original.substr(std::strlen("forget:")));
    return plan;
  }

  if (starts_with(low, "correct memory:") ||
      starts_with(low, "update memory:")) {
    const auto colon = original.find(':');
    const std::string body = colon == std::string::npos
                                 ? ""
                                 : trim(original.substr(colon + 1));
    const auto arrow = body.find("=>");
    if (arrow == std::string::npos) {
      MemoryCommandPlan plan;
      plan.action = "review";
      plan.body = body;
      plan.reason = "correction missing => separator";
      return plan;
    }
    MemoryCommandPlan plan;
    plan.action = "correct";
    plan.query = trim(body.substr(0, arrow));
    plan.replacement = trim(body.substr(arrow + 2));
    return plan;
  }

  if (starts_with(low, "do not remember this")) {
    MemoryCommandPlan plan;
    plan.action = "ignore";
    return plan;
  }

  if (starts_with(low, "flush memory now")) {
    MemoryCommandPlan plan;
    plan.action = "flush";
    return plan;
  }

  throw std::runtime_error("unknown memory command");
}

std::string memory_command_plan_json(const MemoryCommandPlan& plan) {
  std::ostringstream out;
  bool first = true;
  out << "{";
  append_json_string_field(out, first, "action", plan.action);
  append_json_string_field(out, first, "command", plan.command);
  append_json_string_field(out, first, "body", plan.body);
  append_json_string_field(out, first, "query", plan.query);
  append_json_string_field(out, first, "replacement", plan.replacement);
  append_json_string_field(out, first, "durability", plan.durability);
  append_json_string_field(out, first, "durability_mode",
                           plan.durability_mode);
  append_json_string_field(out, first, "mutability", plan.mutability);
  append_json_string_field(out, first, "scope", plan.scope);
  append_json_string_field(out, first, "kind", plan.kind);
  append_json_string_field(out, first, "boundary", plan.boundary);
  append_json_string_field(out, first, "metadata_key", plan.metadata_key);
  append_json_string_field(out, first, "metadata_value", plan.metadata_value);
  append_json_string_field(out, first, "reason", plan.reason);
  append_json_u64_field(out, first, "node_id", plan.node_id, plan.has_node_id);
  append_json_u64_field(out, first, "review_id", plan.review_id,
                        plan.has_review_id);
  append_json_u64_field(out, first, "max_tokens", plan.max_tokens,
                        plan.has_max_tokens);
  append_json_u64_field(out, first, "span_start", plan.span_start,
                        plan.has_span);
  append_json_u64_field(out, first, "span_end", plan.span_end,
                        plan.has_span);
  if (!first) {
    out << ",";
  }
  out << "\"flush_immediately\":"
      << (plan.flush_immediately ? "true" : "false") << "}";
  return out.str();
}

RememberFlushPlan plan_remember_flush(
    const std::string& durability_mode,
    const std::string& durability,
    const std::string& scope,
    bool flush_immediately) {
  RememberFlushPlan plan;
  if (flush_immediately) {
    plan.force_flush = true;
    plan.reason = "command requested immediate flush";
    return plan;
  }
  if (ascii_lower(trim(durability_mode)) != "project_safe") {
    return plan;
  }
  const auto d = ascii_lower(trim(durability));
  const auto s = ascii_lower(trim(scope));
  if (d == "project" || d == "permanent" || s == "project") {
    plan.force_flush = true;
    plan.reason = "project-safe mode requires project memory flush";
  }
  return plan;
}

std::string remember_flush_plan_json(const RememberFlushPlan& plan) {
  std::ostringstream out;
  bool first = true;
  out << "{";
  append_json_bool_field(out, first, "force_flush", plan.force_flush);
  append_json_string_field(out, first, "reason", plan.reason);
  out << "}";
  return out.str();
}

RuntimeEventPlan plan_runtime_event(
    const std::string& event,
    const std::string& action,
    bool autosave_enabled,
    bool force_flush,
    bool read_only) {
  RuntimeEventPlan plan;
  const auto normalized_event = ascii_lower(trim(event));
  const auto normalized_action = ascii_lower(trim(action));
  const bool action_read_only =
      normalized_event == "memory_command" &&
      (normalized_action == "show_memory" ||
       normalized_action == "why_memory");
  if (read_only || action_read_only) {
    plan.read_only = true;
    plan.page = false;
    plan.reason = "read-only runtime event";
    return plan;
  }
  if (force_flush ||
      (normalized_event == "memory_command" && normalized_action == "flush")) {
    plan.flush = true;
    plan.reason = "runtime event requested flush";
    return plan;
  }
  if (autosave_enabled) {
    plan.flush = true;
    plan.reason = "autosave enabled";
  }
  return plan;
}

std::string runtime_event_plan_json(const RuntimeEventPlan& plan) {
  std::ostringstream out;
  bool first = true;
  out << "{";
  append_json_bool_field(out, first, "flush", plan.flush);
  append_json_bool_field(out, first, "page", plan.page);
  append_json_bool_field(out, first, "read_only", plan.read_only);
  append_json_string_field(out, first, "reason", plan.reason);
  out << "}";
  return out.str();
}

std::string normalize_durability_mode(std::string mode) {
  mode = ascii_lower(trim(mode.empty() ? "session_safe" : mode));
  for (char& c : mode) {
    if (c == '-' || c == ' ') {
      c = '_';
    }
  }
  if (mode == "volatile" || mode == "volatile_fast" ||
      mode == "session_safe" || mode == "project_safe" ||
      mode == "durable_strict") {
    return mode;
  }
  throw std::runtime_error("unknown durability mode: " + mode);
}

bool wal_enabled_for_durability_mode(const std::string& mode) {
  const auto normalized = normalize_durability_mode(mode);
  return normalized == "session_safe" || normalized == "project_safe" ||
         normalized == "durable_strict";
}

DurabilityModePlan plan_durability_mode(
    const std::string& requested_mode,
    const std::string& current_mode,
    bool old_wal_enabled,
    bool wal_enabled_override) {
  DurabilityModePlan plan;
  plan.durability_mode = normalize_durability_mode(requested_mode);
  (void)normalize_durability_mode(current_mode);
  plan.target_wal_enabled =
      wal_enabled_for_durability_mode(plan.durability_mode);
  plan.final_wal_enabled =
      wal_enabled_override ? old_wal_enabled : plan.target_wal_enabled;
  plan.append_config_before = old_wal_enabled;
  plan.append_config_after = !old_wal_enabled && plan.final_wal_enabled;
  return plan;
}

std::string durability_mode_plan_json(const DurabilityModePlan& plan) {
  std::ostringstream out;
  bool first = true;
  out << "{";
  append_json_string_field(out, first, "durability_mode",
                           plan.durability_mode);
  append_json_bool_field(out, first, "target_wal_enabled",
                         plan.target_wal_enabled);
  append_json_bool_field(out, first, "final_wal_enabled",
                         plan.final_wal_enabled);
  append_json_bool_field(out, first, "append_config_before",
                         plan.append_config_before);
  append_json_bool_field(out, first, "append_config_after",
                         plan.append_config_after);
  out << "}";
  return out.str();
}

MetadataUpdatePlan plan_metadata_update(
    const std::string& command,
    const std::string& metadata_key,
    const std::string& metadata_value) {
  (void)command;
  MetadataUpdatePlan plan;
  plan.key = trim(metadata_key);
  if (plan.key.empty()) {
    throw std::runtime_error("metadata update requires a key");
  }
  const auto value = trim(metadata_value);
  if (plan.key == "pinned") {
    const auto low = ascii_lower(value);
    if (low != "true" && low != "false") {
      throw std::runtime_error("pinned metadata requires a boolean value");
    }
    plan.value_is_bool = true;
    plan.bool_value = low == "true";
    return plan;
  }
  if (plan.key == "mutability") {
    const auto low = ascii_lower(value);
    if (low != "mutable" && low != "stable" && low != "immutable") {
      throw std::runtime_error("mutability metadata value is invalid");
    }
    plan.string_value = low;
    return plan;
  }
  const auto low = ascii_lower(value);
  if (low == "true" || low == "false") {
    plan.value_is_bool = true;
    plan.bool_value = low == "true";
  } else {
    plan.string_value = value;
  }
  return plan;
}

std::string metadata_update_plan_json(const MetadataUpdatePlan& plan) {
  std::ostringstream out;
  out << "{\"metadata\":{";
  out << "\"" << json_escape(plan.key) << "\":";
  if (plan.value_is_bool) {
    out << (plan.bool_value ? "true" : "false");
  } else {
    out << "\"" << json_escape(plan.string_value) << "\"";
  }
  out << "}}";
  return out.str();
}

MemoryMutationPlan plan_memory_mutation(
    const std::string& command,
    bool has_query,
    std::uint64_t target_count,
    bool has_replacement) {
  const auto cmd = ascii_lower(trim(command));
  MemoryMutationPlan plan;
  plan.target_count = target_count;
  if (cmd == "forget") {
    if (!has_query) {
      plan.action = "no_op";
      plan.reason = "empty query";
      return plan;
    }
    if (target_count == 0) {
      plan.action = "no_op";
      plan.reason = "no active targets";
      return plan;
    }
    plan.action = "expire_targets";
    plan.apply_expire = true;
    return plan;
  }
  if (cmd == "correct") {
    if (!has_replacement) {
      throw std::runtime_error("correction requires replacement");
    }
    plan.write_replacement = true;
    if (!has_query || target_count == 0) {
      plan.action = "write_replacement";
      plan.reason = has_query ? "no active supersedes" : "empty query";
      return plan;
    }
    plan.action = "supersede_targets";
    plan.apply_revision = true;
    return plan;
  }
  if (cmd == "update_metadata" || cmd == "update_memory_metadata") {
    if (!has_query) {
      plan.action = "no_op";
      plan.reason = "empty query";
      return plan;
    }
    if (target_count == 0) {
      plan.action = "no_op";
      plan.reason = "no active targets";
      return plan;
    }
    plan.action = "metadata_update";
    plan.update_metadata = true;
    return plan;
  }
  throw std::runtime_error("unknown memory mutation command: " + command);
}

std::string memory_mutation_plan_json(const MemoryMutationPlan& plan) {
  std::ostringstream out;
  bool first = true;
  out << "{";
  append_json_string_field(out, first, "action", plan.action);
  append_json_string_field(out, first, "reason", plan.reason);
  append_json_u64_field(out, first, "target_count", plan.target_count, true);
  append_json_bool_field(out, first, "apply_expire", plan.apply_expire);
  append_json_bool_field(out, first, "apply_revision", plan.apply_revision);
  append_json_bool_field(out, first, "write_replacement",
                         plan.write_replacement);
  append_json_bool_field(out, first, "update_metadata",
                         plan.update_metadata);
  out << "}";
  return out.str();
}

LibrarianPlan plan_librarian(
    std::uint64_t foldable_turn_count,
    std::uint64_t foldable_digest_count,
    std::uint64_t turns_high,
    std::uint64_t turns_fold,
    std::uint64_t digests_high,
    std::uint64_t digests_fold,
    bool era_enabled,
    bool deferred_backpressure) {
  LibrarianPlan plan;
  plan.deferred_backpressure = deferred_backpressure;
  if (deferred_backpressure) {
    if (turns_high > 0 && turns_fold > 0 &&
        foldable_turn_count >= turns_high * 2) {
      plan.digest_source_count = std::min(turns_fold, foldable_turn_count);
      plan.pending_jobs = 1;
      plan.reason = "deferred turn backpressure";
    } else {
      plan.reason = "below deferred backpressure threshold";
    }
    return plan;
  }
  if (turns_high > 0 && turns_fold > 0 &&
      foldable_turn_count >= turns_high) {
    plan.digest_source_count = std::min(turns_fold, foldable_turn_count);
    plan.pending_jobs += 1;
  }
  if (era_enabled && digests_high > 0 && digests_fold > 0 &&
      foldable_digest_count >= digests_high) {
    plan.era_source_count = std::min(digests_fold, foldable_digest_count);
    plan.pending_jobs += 1;
  }
  if (plan.pending_jobs == 0) {
    plan.reason = "below fold thresholds";
  }
  return plan;
}

std::string librarian_plan_json(const LibrarianPlan& plan) {
  std::ostringstream out;
  bool first = true;
  out << "{";
  append_json_u64_field(out, first, "pending_jobs", plan.pending_jobs, true);
  append_json_u64_field(out, first, "digest_source_count",
                        plan.digest_source_count, true);
  append_json_u64_field(out, first, "era_source_count",
                        plan.era_source_count, true);
  append_json_bool_field(out, first, "deferred_backpressure",
                         plan.deferred_backpressure);
  append_json_string_field(out, first, "reason", plan.reason);
  out << "}";
  return out.str();
}

ExtractionPolicyPlan plan_extraction_policy(
    const std::string& action,
    const std::string& write_intent,
    double confidence,
    double write_direct_threshold,
    std::uint64_t conflict_count,
    std::uint64_t requested_supersede_count,
    std::uint64_t requested_id_count,
    std::uint64_t equivalent_count,
    std::uint64_t expire_target_count) {
  ExtractionPolicyPlan plan;
  const bool authoritative = write_intent == "user_asserted" ||
                             write_intent == "system_asserted";
  if (action == "ignore" || action == "keep_turn_only") {
    plan.action = action;
    return plan;
  }
  if (action == "expire") {
    if (!authoritative) {
      plan.action = "review_candidate";
      plan.reason = "expire action requires authoritative intent";
      return plan;
    }
    if (expire_target_count == 0) {
      plan.action = "review_candidate";
      plan.reason = "expire action found no active target";
      return plan;
    }
    plan.action = "expire";
    return plan;
  }
  if (requested_id_count > 0 && requested_supersede_count == 0) {
    plan.action = "review_candidate";
    plan.reason = "supersede action found no active target";
    return plan;
  }
  if (requested_supersede_count > 0 && !authoritative) {
    plan.action = "review_candidate";
    plan.reason = "supersede action requires authoritative intent";
    return plan;
  }
  if (conflict_count > 0 && !authoritative) {
    plan.action = "review_candidate";
    plan.reason = write_intent == "imported"
                      ? "imported candidate conflicts with active memory"
                      : "conflicts with active memory";
    return plan;
  }
  if (action == "write_direct" && confidence < write_direct_threshold &&
      !authoritative) {
    plan.action = "review_candidate";
    plan.reason = "confidence below direct-write threshold";
    return plan;
  }
  if ((action == "review_candidate" || action == "update_existing" ||
       action == "supersede_existing") &&
      !(authoritative && conflict_count > 0)) {
    plan.action = "review_candidate";
    plan.reason = action + " requires review";
    return plan;
  }
  if (action != "write_direct" && action != "update_existing" &&
      action != "supersede_existing") {
    plan.action = "review_candidate";
    plan.reason = "unsupported extraction action: " + action;
    return plan;
  }
  if (conflict_count > 0 || requested_supersede_count > 0) {
    plan.action = "supersede_existing";
    return plan;
  }
  if (equivalent_count > 0) {
    plan.action = "reinforce_existing";
    return plan;
  }
  plan.action = "write_direct";
  return plan;
}

std::string extraction_policy_plan_json(const ExtractionPolicyPlan& plan) {
  std::ostringstream out;
  bool first = true;
  out << "{";
  append_json_string_field(out, first, "action", plan.action);
  append_json_string_field(out, first, "reason", plan.reason);
  out << "}";
  return out.str();
}

ReinforcementPlan plan_reinforcement(
    const std::string& old_write_intent,
    const std::string& new_write_intent,
    double old_confidence,
    double new_confidence,
    std::uint64_t old_reinforcement_count) {
  ReinforcementPlan plan;
  plan.write_intent = old_write_intent;
  if (write_intent_rank(new_write_intent) >=
      write_intent_rank(old_write_intent)) {
    plan.write_intent = new_write_intent;
  }
  plan.confidence = std::max(old_confidence, new_confidence);
  plan.reinforcement_count = old_reinforcement_count + 1;
  return plan;
}

std::string reinforcement_plan_json(const ReinforcementPlan& plan) {
  std::ostringstream out;
  bool first = true;
  out << "{";
  append_json_string_field(out, first, "write_intent", plan.write_intent);
  append_json_double_field(out, first, "confidence", plan.confidence);
  append_json_u64_field(out, first, "reinforcement_count",
                        plan.reinforcement_count, true);
  out << "}";
  return out.str();
}

ReviewTransitionPlan plan_review_transition(
    const std::string& command,
    const std::string& status,
    bool has_approved_node_id) {
  ReviewTransitionPlan plan;
  const std::string state = status.empty() ? "pending" : status;
  if (command == "edit_review" || command == "change_review_scope") {
    if (state == "approved") {
      plan.action = "error";
      plan.reason = "approved review items cannot be edited";
      return plan;
    }
    if (state == "rejected") {
      plan.action = "error";
      plan.reason = "rejected review items cannot be edited";
      return plan;
    }
    plan.action = command;
    return plan;
  }
  if (command == "reject_review") {
    if (state == "approved") {
      plan.action = "error";
      plan.reason = "approved review items cannot be rejected";
      return plan;
    }
    plan.action = "reject_review";
    return plan;
  }
  if (command == "approve_review") {
    if (state == "rejected") {
      plan.action = "error";
      plan.reason = "rejected review items cannot be approved";
      return plan;
    }
    plan.action = (state == "approved" && has_approved_node_id)
                      ? "return_existing"
                      : "approve_review";
    return plan;
  }
  plan.action = "error";
  plan.reason = "unsupported review transition: " + command;
  return plan;
}

std::string review_transition_plan_json(const ReviewTransitionPlan& plan) {
  std::ostringstream out;
  bool first = true;
  out << "{";
  append_json_string_field(out, first, "action", plan.action);
  append_json_string_field(out, first, "reason", plan.reason);
  out << "}";
  return out.str();
}

CullSpanPlan plan_cull_spans(
    std::uint64_t ntok,
    bool has_max_tokens,
    std::uint64_t max_tokens,
    const std::vector<CullSpan>& spans,
    bool retire_parent) {
  if (ntok == 0) {
    throw std::runtime_error("cannot cull a graft with no token length");
  }
  if (has_max_tokens && max_tokens == 0) {
    throw std::runtime_error("max_tokens must be positive");
  }

  CullSpanPlan plan;
  plan.retire_parent = retire_parent;
  if (spans.empty()) {
    if (!has_max_tokens) {
      throw std::runtime_error("cull_graft requires max_tokens or spans");
    }
    for (std::uint64_t start = 0; start < ntok; start += max_tokens) {
      const auto end = std::min(start + max_tokens, ntok);
      plan.spans.push_back({start, end});
    }
  } else {
    plan.spans = spans;
  }

  for (const auto& span : plan.spans) {
    if (span.start >= span.end || span.end > ntok) {
      throw std::runtime_error("invalid cull span");
    }
  }
  if (plan.spans.empty()) {
    throw std::runtime_error("cull_graft produced no spans");
  }
  if (retire_parent) {
    std::vector<CullSpan> ordered = plan.spans;
    std::sort(ordered.begin(), ordered.end(),
              [](const CullSpan& a, const CullSpan& b) {
                if (a.start != b.start) {
                  return a.start < b.start;
                }
                return a.end < b.end;
              });
    std::uint64_t cursor = 0;
    for (const auto& span : ordered) {
      if (span.start != cursor) {
        throw std::runtime_error(
            "retiring a parent requires cull spans to cover every token "
            "without gaps");
      }
      cursor = span.end;
    }
    if (cursor != ntok) {
      throw std::runtime_error(
          "retiring a parent requires cull spans to cover the full graft");
    }
  }
  return plan;
}

std::string cull_span_plan_json(const CullSpanPlan& plan) {
  std::ostringstream out;
  out << "{\"spans\":[";
  for (std::size_t i = 0; i < plan.spans.size(); ++i) {
    if (i > 0) {
      out << ",";
    }
    out << "[" << plan.spans[i].start << "," << plan.spans[i].end << "]";
  }
  out << "],\"retire_parent\":"
      << (plan.retire_parent ? "true" : "false") << "}";
  return out.str();
}

HostGraftStore::HostGraftStore(DialectDescriptor dialect)
    : dialect_(std::move(dialect)) {
  validate_dialect_profile(dialect_);
}

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

std::vector<std::uint64_t> HostGraftStore::node_ids() const {
  std::vector<std::uint64_t> ids;
  ids.reserve(nodes_.size());
  for (const auto& kv : nodes_) {
    ids.push_back(kv.first);
  }
  std::sort(ids.begin(), ids.end());
  return ids;
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

HostTensor HostGraftStore::slice_tensor(std::uint64_t node_id,
                                        const std::string& name,
                                        std::uint64_t axis,
                                        std::uint64_t start,
                                        std::uint64_t length) const {
  const auto* n = get(node_id);
  if (n == nullptr) {
    throw std::out_of_range("unknown GRM node id");
  }
  const HostTensor* src = nullptr;
  for (const auto& t : n->payload.tensors) {
    if (t.name == name) {
      src = &t;
      break;
    }
  }
  if (src == nullptr) {
    throw std::out_of_range("unknown tensor name");
  }
  if (src->shape.empty()) {
    throw std::runtime_error("slice tensor rank must be nonzero");
  }
  if (axis >= src->shape.size()) {
    throw std::runtime_error("slice axis out of range");
  }
  const auto axis_i = static_cast<std::size_t>(axis);
  if (start > src->shape[axis_i] || length > src->shape[axis_i] - start) {
    throw std::runtime_error("slice range out of bounds");
  }

  const auto elements = checked_product(src->shape, 0, src->shape.size(),
                                        "slice tensor elements");
  if (elements == 0) {
    throw std::runtime_error("slice tensor has zero elements");
  }
  if (elements > static_cast<std::uint64_t>(
                     std::numeric_limits<std::size_t>::max())) {
    throw std::overflow_error("slice tensor exceeds addressable host size");
  }
  if (src->bytes.size() % static_cast<std::size_t>(elements) != 0) {
    throw std::runtime_error("slice tensor byte count is not element-aligned");
  }
  const auto elem_size = static_cast<std::uint64_t>(
      src->bytes.size() / static_cast<std::size_t>(elements));
  if (elem_size == 0) {
    throw std::runtime_error("slice tensor element size must be nonzero");
  }

  const auto outer_elems = checked_product(src->shape, 0, axis_i,
                                           "slice outer");
  const auto inner_elems = checked_product(src->shape, axis_i + 1,
                                           src->shape.size(), "slice inner");
  const auto row_bytes = checked_mul(inner_elems, elem_size, "slice row");
  const auto old_stride = checked_mul(src->shape[axis_i], row_bytes,
                                      "slice old stride");
  const auto out_stride = checked_mul(length, row_bytes, "slice output stride");
  const auto copy_bytes = checked_mul(length, row_bytes, "slice copy");
  const auto out_bytes = checked_mul(outer_elems, out_stride,
                                     "slice output payload");
  if (out_bytes > static_cast<std::uint64_t>(
                      std::numeric_limits<std::size_t>::max())) {
    throw std::overflow_error("slice output exceeds addressable host size");
  }

  HostTensor out;
  out.name = src->name;
  out.dtype = src->dtype;
  out.shape = src->shape;
  out.shape[axis_i] = length;
  out.bytes.resize(static_cast<std::size_t>(out_bytes));
  const auto offset_bytes = checked_mul(start, row_bytes, "slice offset");
  for (std::uint64_t outer = 0; outer < outer_elems; ++outer) {
    const auto* src_base = src->bytes.data() + static_cast<std::size_t>(
        checked_mul(outer, old_stride, "slice source base") + offset_bytes);
    auto* out_base = out.bytes.data() + static_cast<std::size_t>(
        checked_mul(outer, out_stride, "slice output base"));
    if (copy_bytes > 0) {
      std::memcpy(out_base, src_base, static_cast<std::size_t>(copy_bytes));
    }
  }
  return out;
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

void HostGraftStore::clear_payload(std::uint64_t node_id) {
  auto* n = get(node_id);
  if (n == nullptr) {
    throw std::out_of_range("unknown GRM node id");
  }
  if (n->payload.tensors.empty() && !n->lifecycle.host_present &&
      !n->lifecycle.device_present) {
    return;
  }
  n->payload.tensors.clear();
  n->lifecycle.host_present = false;
  n->lifecycle.device_present = false;
  n->lifecycle.cold_only = n->lifecycle.durable;
  mark_dirty(node_id, true, true);
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

const std::string& HostGraftStore::text(std::uint64_t node_id) const {
  const auto* n = get(node_id);
  if (n == nullptr) {
    throw std::out_of_range("unknown GRM node id");
  }
  return n->text;
}

std::string HostGraftStore::node_summary_json(
    std::uint64_t node_id) const {
  const auto* n = get(node_id);
  if (n == nullptr) {
    throw std::out_of_range("unknown GRM node id");
  }
  std::ostringstream out;
  out << "{\"node_id\":" << node_id
      << ",\"text\":\"" << json_escape(n->text) << "\",\"metadata\":";
  if (trim(n->metadata.json).empty()) {
    out << "{}";
  } else {
    out << n->metadata.json;
  }
  out << "}";
  return out.str();
}

void HostGraftStore::set_provenance_json(
    std::uint64_t node_id,
    std::string provenance_json) {
  auto* n = get(node_id);
  if (n == nullptr) {
    throw std::out_of_range("unknown GRM node id");
  }
  if (trim(provenance_json).empty()) {
    provenance_json = "[]";
  }
  if (n->provenance_json == provenance_json) {
    return;
  }
  n->provenance_json = std::move(provenance_json);
  mark_dirty(node_id, false, true);
}

const std::string& HostGraftStore::provenance_json(
    std::uint64_t node_id) const {
  const auto* n = get(node_id);
  if (n == nullptr) {
    throw std::out_of_range("unknown GRM node id");
  }
  return n->provenance_json;
}

void HostGraftStore::clear_route(std::uint64_t node_id) {
  auto* n = get(node_id);
  if (n == nullptr) {
    throw std::out_of_range("unknown GRM node id");
  }
  if (n->route_key.empty() && n->route_keys.empty() &&
      n->lexical_keys.empty()) {
    return;
  }
  n->route_key.clear();
  n->route_keys.clear();
  n->lexical_keys.clear();
  mark_dirty(node_id, false, true);
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

void HostGraftStore::set_no_fold(std::uint64_t node_id, bool no_fold) {
  auto* n = get(node_id);
  if (n == nullptr) {
    throw std::out_of_range("unknown GRM node id");
  }
  if (n->no_fold == no_fold) {
    return;
  }
  n->no_fold = no_fold;
  mark_dirty(node_id, false, true);
}

std::vector<std::uint64_t> HostGraftStore::foldable_nodes(
    const std::string& kind,
    const std::vector<std::uint64_t>& excluded_node_ids) const {
  const auto wanted_kind = trim(kind);
  std::set<std::uint64_t> excluded(
      excluded_node_ids.begin(), excluded_node_ids.end());
  std::vector<std::uint64_t> out;
  for (const auto node_id : node_ids()) {
    if (excluded.find(node_id) != excluded.end()) {
      continue;
    }
    const auto& node = nodes_.at(node_id);
    if (!node.metadata.active || node.no_fold) {
      continue;
    }
    if (!wanted_kind.empty() && node.metadata.kind != wanted_kind) {
      continue;
    }
    out.push_back(node_id);
  }
  return out;
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

void HostGraftStore::set_fact_identity(std::uint64_t node_id,
                                       FactIdentity identity) {
  auto* n = get(node_id);
  if (n == nullptr) {
    throw std::out_of_range("unknown GRM node id");
  }
  bool changed = false;
  if (n->metadata.subject != identity.subject) {
    n->metadata.subject = std::move(identity.subject);
    changed = true;
  }
  if (n->metadata.predicate != identity.predicate) {
    n->metadata.predicate = std::move(identity.predicate);
    changed = true;
  }
  if (n->metadata.value != identity.value) {
    n->metadata.value = std::move(identity.value);
    changed = true;
  }
  if (!identity.scope.empty() && n->metadata.scope != identity.scope) {
    n->metadata.scope = std::move(identity.scope);
    changed = true;
  }
  if (n->metadata.valid_from != identity.valid_from) {
    n->metadata.valid_from = std::move(identity.valid_from);
    changed = true;
  }
  if (n->metadata.expires_at != identity.expires_at) {
    n->metadata.expires_at = std::move(identity.expires_at);
    changed = true;
  }
  if (changed) {
    mark_dirty(node_id, false, true);
  }
}

std::vector<std::uint64_t> HostGraftStore::fact_matches(
    const FactIdentity& identity,
    std::uint64_t value_mode,
    std::uint64_t temporal_mode) const {
  const auto subject = norm_fact_field(identity.subject);
  const auto predicate = norm_fact_field(identity.predicate);
  const auto value = norm_fact_field(identity.value);
  const auto scope = norm_fact_scope(identity.scope);
  std::vector<std::uint64_t> out;
  if (subject.empty() || predicate.empty()) {
    return out;
  }
  const long long now_seconds =
      static_cast<long long>(std::time(nullptr));
  if (temporal_mode == 2 &&
      !fact_time_effective_at(identity.valid_from, identity.expires_at,
                              now_seconds)) {
    return out;
  }
  for (const auto id : node_ids()) {
    const auto& node = nodes_.at(id);
    const auto& meta = node.metadata;
    if (!meta.active) {
      continue;
    }
    if (norm_fact_field(meta.subject) != subject) {
      continue;
    }
    if (norm_fact_field(meta.predicate) != predicate) {
      continue;
    }
    if (norm_fact_scope(meta.scope) != scope) {
      continue;
    }
    const auto existing_value = norm_fact_field(meta.value);
    if (value_mode == 1 && existing_value != value) {
      continue;
    }
    if (value_mode == 2 && (existing_value.empty() || existing_value == value)) {
      continue;
    }
    if (temporal_mode == 1 &&
        (meta.valid_from != identity.valid_from ||
         meta.expires_at != identity.expires_at)) {
      continue;
    }
    if (temporal_mode == 2 &&
        !fact_time_effective_at(meta.valid_from, meta.expires_at,
                                now_seconds)) {
      continue;
    }
    out.push_back(id);
  }
  return out;
}

std::vector<std::uint64_t> HostGraftStore::filter_active_nodes(
    const std::vector<std::uint64_t>& node_ids) const {
  std::vector<std::uint64_t> out;
  std::set<std::uint64_t> seen;
  for (const auto node_id : node_ids) {
    if (!seen.insert(node_id).second) {
      continue;
    }
    const auto* n = get(node_id);
    if (n == nullptr) {
      continue;
    }
    if (!n->metadata.active) {
      continue;
    }
    out.push_back(node_id);
  }
  return out;
}

std::vector<std::uint64_t> HostGraftStore::active_text_matches(
    const std::string& query) const {
  const auto q = ascii_lower(trim(query));
  std::vector<std::uint64_t> out;
  if (q.empty()) {
    return out;
  }
  for (const auto id : node_ids()) {
    const auto& node = nodes_.at(id);
    if (!node.metadata.active) {
      continue;
    }
    if (ascii_lower(node.text).find(q) == std::string::npos) {
      continue;
    }
    out.push_back(id);
  }
  return out;
}

void HostGraftStore::set_graph_edges(std::uint64_t node_id, GraphEdges edges) {
  auto* n = get(node_id);
  if (n == nullptr) {
    throw std::out_of_range("unknown GRM node id");
  }
  const bool changed =
      n->metadata.source_turns != edges.source_turns ||
      n->metadata.source_grafts != edges.source_grafts ||
      n->metadata.supersedes != edges.supersedes ||
      n->metadata.superseded_by != edges.superseded_by;
  if (!changed) {
    return;
  }
  n->metadata.source_turns = std::move(edges.source_turns);
  n->metadata.source_grafts = std::move(edges.source_grafts);
  n->metadata.supersedes = std::move(edges.supersedes);
  n->metadata.superseded_by = std::move(edges.superseded_by);
  mark_dirty(node_id, false, true);
}

GraphEdges HostGraftStore::graph_edges(std::uint64_t node_id) const {
  const auto* n = get(node_id);
  if (n == nullptr) {
    throw std::out_of_range("unknown GRM node id");
  }
  GraphEdges edges;
  edges.source_turns = n->metadata.source_turns;
  edges.source_grafts = n->metadata.source_grafts;
  edges.supersedes = n->metadata.supersedes;
  edges.superseded_by = n->metadata.superseded_by;
  return edges;
}

std::vector<std::uint64_t> HostGraftStore::source_closure(
    const std::vector<std::uint64_t>& node_ids,
    std::uint64_t max_depth,
    bool include_roots) const {
  std::vector<std::uint64_t> out;
  std::set<std::uint64_t> seen;

  const auto emit = [&](std::uint64_t node_id) {
    if (seen.insert(node_id).second) {
      out.push_back(node_id);
    }
  };

  const auto mark_seed = [&](std::uint64_t node_id) {
    if (include_roots) {
      emit(node_id);
    } else {
      seen.insert(node_id);
    }
  };

  std::function<void(std::uint64_t, std::uint64_t)> walk =
      [&](std::uint64_t node_id, std::uint64_t depth) {
        const auto* n = get(node_id);
        if (n == nullptr) {
          throw std::out_of_range("unknown GRM source node id");
        }
        if (depth >= max_depth) {
          return;
        }
        for (const auto child_id : n->metadata.source_grafts) {
          const auto* child = get(child_id);
          if (child == nullptr) {
            throw std::out_of_range("unknown GRM source child id");
          }
          if (seen.insert(child_id).second) {
            out.push_back(child_id);
            walk(child_id, depth + 1);
          }
        }
      };

  for (const auto node_id : node_ids) {
    const auto* n = get(node_id);
    if (n == nullptr) {
      throw std::out_of_range("unknown GRM node id");
    }
    mark_seed(node_id);
  }
  for (const auto node_id : node_ids) {
    walk(node_id, 0);
  }
  return out;
}

void HostGraftStore::apply_revision(std::uint64_t replacement_node_id,
                                    std::vector<std::uint64_t> supersedes) {
  auto* replacement = get(replacement_node_id);
  if (replacement == nullptr) {
    throw std::out_of_range("unknown replacement GRM node id");
  }
  for (const auto old_id : supersedes) {
    if (get(old_id) == nullptr) {
      throw std::out_of_range("unknown superseded GRM node id");
    }
  }

  replacement->metadata.active = true;
  replacement->metadata.supersedes = supersedes;
  mark_dirty(replacement_node_id, false, true);

  for (const auto old_id : supersedes) {
    auto* old = get(old_id);
    old->metadata.active = false;
    if (std::find(old->metadata.superseded_by.begin(),
                  old->metadata.superseded_by.end(),
                  replacement_node_id) == old->metadata.superseded_by.end()) {
      old->metadata.superseded_by.push_back(replacement_node_id);
    }
    mark_dirty(old_id, false, true);
  }
}

void HostGraftStore::apply_expire(std::vector<std::uint64_t> node_ids) {
  for (const auto node_id : node_ids) {
    if (get(node_id) == nullptr) {
      throw std::out_of_range("unknown expired GRM node id");
    }
  }
  std::set<std::uint64_t> seen;
  for (const auto node_id : node_ids) {
    if (!seen.insert(node_id).second) {
      continue;
    }
    auto* node = get(node_id);
    node->metadata.active = false;
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
  {
    std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
    if (!out) {
      throw std::runtime_error("failed to open native GRM checkpoint for write");
    }
    out.write(kCheckpointMagic, sizeof(kCheckpointMagic) - 1);
    write_string(out, dialect_.dialect_id());
    write_string(out, dialect_.position_law);
    write_string(out, dialect_.state_kind);
    write_string(out, dialect_.graftability);
    write_bool(out, dialect_.remountable);
    write_string(out, dialect_.composition);
    write_u64(out, static_cast<std::uint64_t>(nodes_.size()));
    for (const auto id : node_ids()) {
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
      write_string(out, n.metadata.subject);
      write_string(out, n.metadata.predicate);
      write_string(out, n.metadata.value);
      write_string(out, n.metadata.valid_from);
      write_string(out, n.metadata.expires_at);
      write_u64_vector(out, n.metadata.source_turns);
      write_u64_vector(out, n.metadata.source_grafts);
      write_u64_vector(out, n.metadata.supersedes);
      write_u64_vector(out, n.metadata.superseded_by);
      write_string(out, n.provenance_json);
      write_bool(out, n.no_fold);
      write_f32_vectors(out, n.route_keys);
      write_string_vector(out, n.lexical_keys);
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
  }
  fsync_path(tmp);
  std::filesystem::rename(tmp, dst);
  fsync_directory(root);
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
  const bool checkpoint_v4 =
      magic_s == std::string(kCheckpointMagicV4, sizeof(kCheckpointMagicV4) - 1);
  const bool checkpoint_v5 =
      magic_s == std::string(kCheckpointMagicV5, sizeof(kCheckpointMagicV5) - 1);
  const bool checkpoint_v6 =
      magic_s == std::string(kCheckpointMagicV6, sizeof(kCheckpointMagicV6) - 1);
  const bool checkpoint_v7 =
      magic_s == std::string(kCheckpointMagicV7, sizeof(kCheckpointMagicV7) - 1);
  const bool checkpoint_v8 =
      magic_s == std::string(kCheckpointMagicV8, sizeof(kCheckpointMagicV8) - 1);
  const bool checkpoint_v9 =
      magic_s == std::string(kCheckpointMagicV9, sizeof(kCheckpointMagicV9) - 1);
  const bool checkpoint_v10 =
      magic_s == std::string(kCheckpointMagicV10,
                             sizeof(kCheckpointMagicV10) - 1);
  if (!in || (!checkpoint_v1 && !checkpoint_v2 && !checkpoint_v3 &&
              !checkpoint_v4 && !checkpoint_v5 && !checkpoint_v6 &&
              !checkpoint_v7 && !checkpoint_v8 && !checkpoint_v9 &&
              !checkpoint_v10)) {
    throw std::runtime_error("invalid native GRM checkpoint magic");
  }
  const bool has_active = checkpoint_v2 || checkpoint_v3 || checkpoint_v4 ||
                          checkpoint_v5 || checkpoint_v6 || checkpoint_v7 ||
                          checkpoint_v8 || checkpoint_v9 || checkpoint_v10;
  const bool has_route_metadata = checkpoint_v3 || checkpoint_v4 ||
                                  checkpoint_v5 || checkpoint_v6 ||
                                  checkpoint_v7 || checkpoint_v8 ||
                                  checkpoint_v9 || checkpoint_v10;
  const bool has_graph_edges = checkpoint_v4 || checkpoint_v5 ||
                               checkpoint_v6 || checkpoint_v7 ||
                               checkpoint_v8 || checkpoint_v9 ||
                               checkpoint_v10;
  const bool has_dialect_id = checkpoint_v5 || checkpoint_v6 ||
                              checkpoint_v7 || checkpoint_v8 ||
                              checkpoint_v9 || checkpoint_v10;
  const bool has_route_lists = checkpoint_v6 || checkpoint_v7 ||
                               checkpoint_v8 || checkpoint_v9 ||
                               checkpoint_v10;
  const bool has_profile = checkpoint_v7 || checkpoint_v8 || checkpoint_v9 ||
                           checkpoint_v10;
  const bool has_fact_identity = checkpoint_v8 || checkpoint_v9 ||
                                 checkpoint_v10;
  const bool has_provenance = checkpoint_v9 || checkpoint_v10;
  const bool has_no_fold = checkpoint_v10;
  if (has_dialect_id) {
    const auto saved_dialect = read_string(in);
    const auto expected_dialect = dialect_.dialect_id();
    if (saved_dialect != expected_dialect) {
      throw std::runtime_error("native GRM checkpoint dialect mismatch: " +
                               saved_dialect + " vs " + expected_dialect);
    }
  }
  if (has_profile) {
    grm::DialectDescriptor saved = dialect_;
    saved.position_law = read_string(in);
    saved.state_kind = read_string(in);
    saved.graftability = read_string(in);
    saved.remountable = read_bool(in);
    saved.composition = read_string(in);
    if (saved.profile_id() != dialect_.profile_id()) {
      throw std::runtime_error("native GRM checkpoint graft profile mismatch: " +
                               saved.profile_id() + " vs " +
                               dialect_.profile_id());
    }
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
    n.metadata.active = has_active ? read_bool(in) : true;
    if (has_route_metadata) {
      n.metadata.kind = read_string(in);
      n.metadata.scope = read_string(in);
      n.metadata.durability = read_string(in);
      n.metadata.mutability = read_string(in);
    }
    if (has_fact_identity) {
      n.metadata.subject = read_string(in);
      n.metadata.predicate = read_string(in);
      n.metadata.value = read_string(in);
      n.metadata.valid_from = read_string(in);
      n.metadata.expires_at = read_string(in);
    }
    if (has_graph_edges) {
      n.metadata.source_turns = read_u64_vector(in);
      n.metadata.source_grafts = read_u64_vector(in);
      n.metadata.supersedes = read_u64_vector(in);
      n.metadata.superseded_by = read_u64_vector(in);
    }
    if (has_provenance) {
      n.provenance_json = read_string(in);
      if (trim(n.provenance_json).empty()) {
        n.provenance_json = "[]";
      }
    }
    if (has_no_fold) {
      n.no_fold = read_bool(in);
    }
    if (has_route_lists) {
      n.route_keys = read_f32_vectors(in);
      n.lexical_keys = read_string_vector(in);
      if (!n.route_keys.empty()) {
        n.route_key = n.route_keys.front();
      }
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

std::vector<DirtyNodeInfo> HostGraftStore::dirty_plan() const {
  std::vector<DirtyNodeInfo> out;
  for (const auto node_id : dirty_.node_ids()) {
    const auto* n = get(node_id);
    if (n == nullptr || !n->lifecycle.dirty) {
      continue;
    }
    DirtyNodeInfo info;
    info.node_id = node_id;
    info.payload_dirty = dirty_.payload_dirty(node_id);
    info.metadata_dirty = dirty_.metadata_dirty(node_id);
    info.payload_bytes = info.payload_dirty ? n->payload.bytes() : 0;
    info.durability_priority = durability_priority(n->metadata.durability);
    out.push_back(info);
  }
  std::sort(out.begin(), out.end(), [](const auto& a, const auto& b) {
    if (a.durability_priority != b.durability_priority) {
      return a.durability_priority < b.durability_priority;
    }
    if (a.payload_dirty != b.payload_dirty) {
      return a.payload_dirty && !b.payload_dirty;
    }
    return a.node_id < b.node_id;
  });
  return out;
}

constexpr int kKindFilterShift = 0;
constexpr int kScopeFilterShift = 16;
constexpr int kDurabilityFilterShift = 32;
constexpr int kMutabilityFilterShift = 48;

static std::uint64_t route_filter_bit(int shift, int offset) {
  return std::uint64_t{1} << (shift + offset);
}

static std::uint64_t known_route_filter_bit(
    int shift, const std::string& value) {
  if (shift == kKindFilterShift) {
    if (value == "turn") return route_filter_bit(shift, 0);
    if (value == "doc") return route_filter_bit(shift, 1);
    if (value == "digest") return route_filter_bit(shift, 2);
    if (value == "era") return route_filter_bit(shift, 3);
    if (value == "fact") return route_filter_bit(shift, 4);
    if (value == "task_state") return route_filter_bit(shift, 5);
    if (value == "recall") return route_filter_bit(shift, 6);
    if (value == "cull_span") return route_filter_bit(shift, 7);
  } else if (shift == kScopeFilterShift) {
    if (value == "conversation") return route_filter_bit(shift, 0);
    if (value == "project") return route_filter_bit(shift, 1);
    if (value == "user") return route_filter_bit(shift, 2);
    if (value == "task") return route_filter_bit(shift, 3);
    if (value == "global") return route_filter_bit(shift, 4);
    if (value == "session") return route_filter_bit(shift, 5);
  } else if (shift == kDurabilityFilterShift) {
    if (value == "volatile") return route_filter_bit(shift, 0);
    if (value == "session") return route_filter_bit(shift, 1);
    if (value == "project") return route_filter_bit(shift, 2);
    if (value == "permanent") return route_filter_bit(shift, 3);
  } else if (shift == kMutabilityFilterShift) {
    if (value == "ephemeral") return route_filter_bit(shift, 0);
    if (value == "stable") return route_filter_bit(shift, 1);
    if (value == "mutable") return route_filter_bit(shift, 2);
    if (value == "immutable") return route_filter_bit(shift, 3);
  }
  return 0;
}

static std::uint64_t route_filter_bits(
    const std::string& kind,
    const std::string& scope,
    const std::string& durability,
    const std::string& mutability) {
  return known_route_filter_bit(kKindFilterShift, kind) |
         known_route_filter_bit(kScopeFilterShift, scope) |
         known_route_filter_bit(kDurabilityFilterShift, durability) |
         known_route_filter_bit(kMutabilityFilterShift, mutability);
}

static bool env_truthy(const char* name) {
  const char* value = std::getenv(name);
  if (value == nullptr) {
    return false;
  }
  const std::string s = ascii_lower(value);
  return s == "1" || s == "true" || s == "yes" || s == "on";
}

static std::size_t env_size_or(const char* name, std::size_t fallback) {
  const char* value = std::getenv(name);
  if (value == nullptr || *value == '\0') {
    return fallback;
  }
  char* end = nullptr;
  errno = 0;
  const unsigned long long parsed = std::strtoull(value, &end, 10);
  if (errno != 0 || end == value || *end != '\0') {
    return fallback;
  }
  return static_cast<std::size_t>(parsed);
}

static std::vector<std::uint64_t> lexical_hashes(
    const std::vector<std::string>& values);

static std::uint8_t pack_i4(std::int8_t value) {
  return static_cast<std::uint8_t>(value) & 0x0F;
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
  const auto existing = entry_by_node_.find(node_id);
  if (existing != entry_by_node_.end()) {
    auto& e = entries_[existing->second];
    e.route_keys = std::move(route_keys);
    e.lexical_keys = std::move(lexical_keys);
    e.lexical_hashes = lexical_hashes(e.lexical_keys);
    mark_mla_arena_dirty();
    mark_gqa_arena_dirty();
    return;
  }
  Entry entry;
  entry.node_id = node_id;
  entry.route_keys = std::move(route_keys);
  entry.lexical_keys = std::move(lexical_keys);
  entry.lexical_hashes = lexical_hashes(entry.lexical_keys);
  entries_.push_back(std::move(entry));
  auto& e = entries_.back();
  e.filter_bits = route_filter_bits(
      e.kind, e.scope, e.durability, e.mutability);
  entry_by_node_[node_id] = entries_.size() - 1;
  mark_mla_arena_dirty();
  mark_gqa_arena_dirty();
}

void RouterIndex::erase(std::uint64_t node_id) {
  const auto it = entry_by_node_.find(node_id);
  if (it == entry_by_node_.end()) {
    return;
  }
  entries_.erase(entries_.begin() + static_cast<std::ptrdiff_t>(it->second));
  rebuild_entry_map();
  mark_mla_arena_dirty();
  mark_gqa_arena_dirty();
}

void RouterIndex::set_active(std::uint64_t node_id, bool active) {
  const auto it = entry_by_node_.find(node_id);
  if (it != entry_by_node_.end()) {
    entries_[it->second].active = active;
  }
}

void RouterIndex::set_route_metadata(std::uint64_t node_id,
                                     std::string kind,
                                     std::string scope,
                                     std::string durability,
                                     std::string mutability) {
  const auto it = entry_by_node_.find(node_id);
  if (it == entry_by_node_.end()) {
    return;
  }
  auto& e = entries_[it->second];
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
  e.filter_bits = route_filter_bits(
      e.kind, e.scope, e.durability, e.mutability);
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
    if (!std::isfinite(score)) {
      continue;
    }
    if (!have || score > best) {
      best = score;
      have = true;
    }
  }
  return have ? best : std::numeric_limits<float>::quiet_NaN();
}

static float gqa_raw_score(const std::vector<float>& query,
                           std::uint64_t query_heads,
                           std::uint64_t query_tokens,
                           std::uint64_t head_dim,
                           const std::vector<float>& key,
                           std::uint64_t kv_heads) {
  if (query_heads == 0 || query_tokens == 0 || head_dim == 0 ||
      kv_heads == 0 || query_heads % kv_heads != 0) {
    return 0.0F;
  }
  const auto query_expected =
      checked_mul(checked_mul(query_heads, query_tokens, "GQA query"),
                  head_dim, "GQA query");
  const auto key_head_width = checked_mul(kv_heads, head_dim, "GQA key");
  if (query.size() != static_cast<std::size_t>(query_expected) ||
      key_head_width == 0 || key.size() % key_head_width != 0) {
    return 0.0F;
  }
  const auto key_tokens =
      static_cast<std::uint64_t>(key.size()) / key_head_width;
  if (key_tokens == 0) {
    return 0.0F;
  }

  const auto repeat = query_heads / kv_heads;
  const double denom = std::sqrt(static_cast<double>(head_dim));
  double total = 0.0;
  for (std::uint64_t h = 0; h < query_heads; ++h) {
    const auto kh = h / repeat;
    double best = 0.0;
    for (std::uint64_t qi = 0; qi < query_tokens; ++qi) {
      const auto qoff = ((h * query_tokens) + qi) * head_dim;
      for (std::uint64_t ki = 0; ki < key_tokens; ++ki) {
        const auto koff = ((kh * key_tokens) + ki) * head_dim;
        double dot = 0.0;
        for (std::uint64_t d = 0; d < head_dim; ++d) {
          const auto qv = query[static_cast<std::size_t>(qoff + d)];
          const auto kv = key[static_cast<std::size_t>(koff + d)];
          if (!std::isfinite(qv) || !std::isfinite(kv)) {
            return std::numeric_limits<float>::quiet_NaN();
          }
          dot += static_cast<double>(qv) * static_cast<double>(kv);
        }
        best = std::max(best, std::abs(dot) / denom);
      }
    }
    total += best;
  }
  return static_cast<float>(total / static_cast<double>(query_heads));
}

static float max_gqa_raw_score(const std::vector<float>& query,
                               std::uint64_t query_heads,
                               std::uint64_t query_tokens,
                               std::uint64_t head_dim,
                               std::uint64_t kv_heads,
                               const std::vector<std::vector<float>>& keys) {
  float best = 0.0F;
  bool have = false;
  for (const auto& key : keys) {
    const auto score = gqa_raw_score(
        query, query_heads, query_tokens, head_dim, key, kv_heads);
    if (!std::isfinite(score)) {
      continue;
    }
    if (!have || score > best) {
      best = score;
      have = true;
    }
  }
  return have ? best : std::numeric_limits<float>::quiet_NaN();
}

static bool filter_allows(const std::vector<std::string>& filters,
                          const std::string& value) {
  return filters.empty() ||
         std::find(filters.begin(), filters.end(), value) != filters.end();
}

static bool filter_mask_all_known(const std::vector<std::string>& filters,
                                  int shift,
                                  std::uint64_t* out) {
  std::uint64_t mask = 0;
  for (const auto& value : filters) {
    const auto bit = known_route_filter_bit(shift, value);
    if (bit == 0) {
      return false;
    }
    mask |= bit;
  }
  *out = mask;
  return true;
}

static bool filter_allows_bits(const std::vector<std::string>& filters,
                               const std::string& value,
                               std::uint64_t entry_bits,
                               int shift) {
  if (filters.empty()) {
    return true;
  }
  std::uint64_t mask = 0;
  if (filter_mask_all_known(filters, shift, &mask)) {
    return (entry_bits & mask) != 0;
  }
  return filter_allows(filters, value);
}

bool RouterIndex::entry_allowed(
    const Entry& e,
    const std::vector<std::string>& kinds,
    const std::vector<std::string>& scopes,
    const std::vector<std::string>& durabilities,
    const std::vector<std::string>& mutabilities) const {
  return e.active &&
         filter_allows_bits(
             kinds, e.kind, e.filter_bits, kKindFilterShift) &&
         filter_allows_bits(
             scopes, e.scope, e.filter_bits, kScopeFilterShift) &&
         filter_allows_bits(
             durabilities, e.durability, e.filter_bits,
             kDurabilityFilterShift) &&
         filter_allows_bits(
             mutabilities, e.mutability, e.filter_bits,
             kMutabilityFilterShift);
}

static std::size_t lexical_hit_count(
    const std::vector<std::string>& query_lexical,
    const std::vector<std::string>& entry_lexical) {
  std::size_t hits = 0;
  for (const auto& q : query_lexical) {
    if (std::find(entry_lexical.begin(), entry_lexical.end(), q) !=
        entry_lexical.end()) {
      ++hits;
    }
  }
  return hits;
}

static std::uint64_t lexical_hash(const std::string& value) {
  std::uint64_t h = 1469598103934665603ULL;
  for (const unsigned char c : value) {
    h ^= static_cast<std::uint64_t>(c);
    h *= 1099511628211ULL;
  }
  return h;
}

static std::vector<std::uint64_t> lexical_hashes(
    const std::vector<std::string>& values) {
  std::vector<std::uint64_t> out;
  out.reserve(values.size());
  for (const auto& value : values) {
    out.push_back(lexical_hash(value));
  }
  return out;
}

static std::size_t lexical_hit_count(
    const std::vector<std::string>& query_lexical,
    const std::vector<std::uint64_t>& query_hashes,
    const std::vector<std::string>& entry_lexical,
    const std::vector<std::uint64_t>& entry_hashes) {
  if (query_hashes.size() != query_lexical.size() ||
      entry_hashes.size() != entry_lexical.size()) {
    return lexical_hit_count(query_lexical, entry_lexical);
  }
  std::size_t hits = 0;
  for (std::size_t qi = 0; qi < query_lexical.size(); ++qi) {
    const auto qh = query_hashes[qi];
    for (std::size_t ei = 0; ei < entry_lexical.size(); ++ei) {
      if (entry_hashes[ei] == qh && entry_lexical[ei] == query_lexical[qi]) {
        ++hits;
        break;
      }
    }
  }
  return hits;
}

static float lexical_bonus(
    const std::vector<std::string>& query_lexical,
    const std::vector<std::uint64_t>& query_hashes,
    const std::vector<std::string>& entry_lexical,
    const std::vector<std::uint64_t>& entry_hashes) {
  if (query_lexical.empty()) {
    return 0.0F;
  }
  return static_cast<float>(
      lexical_hit_count(
          query_lexical, query_hashes, entry_lexical, entry_hashes)) /
      static_cast<float>(query_lexical.size());
}

void RouterIndex::mark_mla_arena_dirty() {
  mla_arena_dirty_ = true;
  mla_arena_.valid = false;
}

void RouterIndex::mark_gqa_arena_dirty() {
  gqa_arena_dirty_ = true;
  gqa_arena_.valid = false;
}

void RouterIndex::rebuild_entry_map() {
  entry_by_node_.clear();
  entry_by_node_.reserve(entries_.size());
  for (std::size_t i = 0; i < entries_.size(); ++i) {
    entry_by_node_[entries_[i].node_id] = i;
  }
}

void RouterIndex::rebuild_mla_arena() const {
  if (!mla_arena_dirty_) {
    return;
  }
  MlaArena arena;
  std::size_t row_count = 0;
  for (const auto& e : entries_) {
    row_count += e.route_keys.size();
    for (const auto& key : e.route_keys) {
      if (key.empty()) {
        arena.uniform_dim = false;
        continue;
      }
      if (arena.dim == 0) {
        arena.dim = key.size();
      } else if (arena.dim != key.size()) {
        arena.uniform_dim = false;
      }
    }
  }
  if (row_count == 0 || arena.dim == 0 || !arena.uniform_dim) {
    arena.valid = false;
    mla_arena_ = std::move(arena);
    mla_arena_dirty_ = false;
    return;
  }
  arena.single_row_per_entry = row_count == entries_.size();
  arena.rows.reserve(row_count * arena.dim);
  arena.norms.reserve(row_count);
  arena.inv_norms.reserve(row_count);
  arena.entry_row_offsets.reserve(entries_.size() + 1);
  arena.q4_stride = (arena.dim + 1) / 2;
  arena.q4_rows.reserve(row_count * arena.q4_stride);
  arena.q4_values.reserve(row_count * arena.dim);
  arena.q4_scales.reserve(row_count);
  arena.q4_norm_scales.reserve(row_count);
  for (std::size_t entry_idx = 0; entry_idx < entries_.size(); ++entry_idx) {
    arena.entry_row_offsets.push_back(arena.norms.size());
    const auto& e = entries_[entry_idx];
    for (const auto& key : e.route_keys) {
      if (key.size() != arena.dim) {
        arena.uniform_dim = false;
        arena.valid = false;
        mla_arena_ = std::move(arena);
        mla_arena_dirty_ = false;
        return;
      }
      double norm_sq = 0.0;
      float max_abs = 0.0F;
      for (const auto v : key) {
        arena.rows.push_back(v);
        norm_sq += static_cast<double>(v) * static_cast<double>(v);
        max_abs = std::max(max_abs, std::abs(v));
      }
      const float norm = static_cast<float>(std::sqrt(norm_sq));
      const float inv_norm = norm > 0.0F ? 1.0F / norm : 0.0F;
      arena.norms.push_back(norm);
      arena.inv_norms.push_back(inv_norm);
      const float scale = max_abs > 0.0F ? max_abs / 7.0F : 1.0F;
      arena.q4_scales.push_back(scale);
      arena.q4_norm_scales.push_back(scale * inv_norm);
      for (std::size_t d = 0; d < arena.dim; d += 2) {
        const auto q0 = static_cast<std::int8_t>(std::max(
            -7.0F, std::min(7.0F, std::round(key[d] / scale))));
        std::int8_t q1 = 0;
        if (d + 1 < arena.dim) {
          q1 = static_cast<std::int8_t>(std::max(
              -7.0F, std::min(7.0F, std::round(key[d + 1] / scale))));
        }
        arena.q4_values.push_back(q0);
        if (d + 1 < arena.dim) {
          arena.q4_values.push_back(q1);
        }
        arena.q4_rows.push_back(
            static_cast<std::uint8_t>(pack_i4(q0) | (pack_i4(q1) << 4)));
      }
    }
  }
  arena.entry_row_offsets.push_back(arena.norms.size());
  arena.valid = !arena.rows.empty();
  arena.q4_valid = arena.valid &&
                   arena.q4_scales.size() == arena.norms.size() &&
                   arena.q4_norm_scales.size() == arena.norms.size() &&
                   arena.q4_rows.size() ==
                       arena.norms.size() * arena.q4_stride &&
                   arena.q4_values.size() ==
                       arena.norms.size() * arena.dim;
  mla_arena_ = std::move(arena);
  mla_arena_dirty_ = false;
}

void RouterIndex::rebuild_gqa_arena(
    std::uint64_t kv_heads,
    std::uint64_t head_dim) const {
  if (!gqa_arena_dirty_ && gqa_arena_.kv_heads == kv_heads &&
      gqa_arena_.head_dim == head_dim) {
    return;
  }
  GqaArena arena;
  arena.kv_heads = kv_heads;
  arena.head_dim = head_dim;
  if (kv_heads == 0 || head_dim == 0) {
    gqa_arena_ = std::move(arena);
    gqa_arena_dirty_ = false;
    return;
  }
  const auto key_head_width = checked_mul(kv_heads, head_dim, "GQA arena key");
  arena.entry_key_offsets.reserve(entries_.size() + 1);
  for (const auto& e : entries_) {
    arena.entry_key_offsets.push_back(arena.key_tokens.size());
    for (const auto& key : e.route_keys) {
      if (key.size() % key_head_width != 0) {
        arena.valid = false;
        gqa_arena_ = std::move(arena);
        gqa_arena_dirty_ = false;
        return;
      }
      const auto key_tokens =
          static_cast<std::uint64_t>(key.size()) / key_head_width;
      const auto row_count = checked_mul(kv_heads, key_tokens, "GQA arena rows");
      arena.key_row_offsets.push_back(arena.rows.size() /
                                      static_cast<std::size_t>(head_dim));
      arena.key_tokens.push_back(key_tokens);
      bool key_finite = true;
      arena.rows.reserve(arena.rows.size() + key.size());
      for (const auto v : key) {
        key_finite = key_finite && std::isfinite(v);
        arena.rows.push_back(v);
      }
      arena.key_finite.push_back(key_finite ? 1 : 0);
      if (row_count == 0 && !key.empty()) {
        arena.valid = false;
        gqa_arena_ = std::move(arena);
        gqa_arena_dirty_ = false;
        return;
      }
    }
  }
  arena.entry_key_offsets.push_back(arena.key_tokens.size());
  arena.key_row_offsets.push_back(arena.rows.size() /
                                  static_cast<std::size_t>(head_dim));
  arena.valid = !arena.key_tokens.empty() &&
                arena.key_finite.size() == arena.key_tokens.size() &&
                arena.key_row_offsets.size() == arena.key_tokens.size() + 1;
  gqa_arena_ = std::move(arena);
  gqa_arena_dirty_ = false;
}

void RouterIndex::prepare_mla_route() const {
  rebuild_mla_arena();
}

void RouterIndex::prepare_gqa_route(
    std::uint64_t kv_heads,
    std::uint64_t head_dim) const {
  rebuild_gqa_arena(kv_heads, head_dim);
}

std::vector<std::uint64_t> RouterIndex::route_scan(
    const std::vector<float>& query, const std::vector<std::string>& lexical,
    std::size_t topk, const std::vector<std::string>& kinds,
    const std::vector<std::string>& scopes,
    const std::vector<std::string>& durabilities,
    const std::vector<std::string>& mutabilities) const {
  const auto query_lex_hashes = lexical_hashes(lexical);
  std::vector<std::pair<float, std::uint64_t>> scored;
  scored.reserve(entries_.size());
  for (const auto& e : entries_) {
    if (!entry_allowed(e, kinds, scopes, durabilities, mutabilities)) {
      continue;
    }
    float score = max_cosine(query, e.route_keys);
    score += lexical_bonus(
        lexical, query_lex_hashes, e.lexical_keys, e.lexical_hashes);
    if (!std::isfinite(score)) {
      continue;
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

float RouterIndex::exact_mla_entry_score(
    const std::vector<float>& query,
    std::size_t entry_idx,
    double qnorm) const {
  const auto dim = mla_arena_.dim;
  const auto* rows = mla_arena_.rows.data();
  const auto row_start = mla_arena_.entry_row_offsets[entry_idx];
  const auto row_end = mla_arena_.entry_row_offsets[entry_idx + 1];
  float entry_best = 0.0F;
  bool entry_have = false;
  for (std::size_t row = row_start; row < row_end; ++row) {
    const auto* base = rows + (row * dim);
    double dot = 0.0;
    for (std::size_t d = 0; d < dim; ++d) {
      dot += static_cast<double>(query[d]) * static_cast<double>(base[d]);
    }
    const float score = static_cast<float>(
        dot / ((qnorm * static_cast<double>(mla_arena_.norms[row])) + 1.0e-8));
    if (!std::isfinite(score)) {
      continue;
    }
    if (!entry_have || score > entry_best) {
      entry_best = score;
      entry_have = true;
    }
  }
  return entry_have ? entry_best : std::numeric_limits<float>::quiet_NaN();
}

float RouterIndex::int4_mla_entry_score(
    const std::vector<float>& query,
    std::size_t entry_idx,
    double qnorm_inv) const {
  const auto row_start = mla_arena_.entry_row_offsets[entry_idx];
  const auto row_end = mla_arena_.entry_row_offsets[entry_idx + 1];
  float entry_best = 0.0F;
  bool entry_have = false;
  for (std::size_t row = row_start; row < row_end; ++row) {
    const float score = int4_mla_row_score(query, row, qnorm_inv);
    if (!std::isfinite(score)) {
      continue;
    }
    if (!entry_have || score > entry_best) {
      entry_best = score;
      entry_have = true;
    }
  }
  return entry_have ? entry_best : std::numeric_limits<float>::quiet_NaN();
}

float RouterIndex::int4_mla_row_score(
    const std::vector<float>& query,
    std::size_t row,
    double qnorm_inv) const {
  const auto dim = mla_arena_.dim;
  const auto value_base = row * dim;
  const auto* values = mla_arena_.q4_values.data() + value_base;
  const auto* q = query.data();
  float dot_i4 = 0.0F;
#if defined(_OPENMP)
#pragma omp simd reduction(+ : dot_i4)
#endif
  for (std::int64_t d = 0; d < static_cast<std::int64_t>(dim); ++d) {
    const auto idx = static_cast<std::size_t>(d);
    dot_i4 += q[idx] * static_cast<float>(values[idx]);
  }
  return dot_i4 * mla_arena_.q4_norm_scales[row] *
         static_cast<float>(qnorm_inv);
}

float RouterIndex::gqa_arena_key_score(
    const std::vector<float>& query,
    std::uint64_t query_heads,
    std::uint64_t query_tokens,
    std::uint64_t head_dim,
    std::uint64_t kv_heads,
    bool query_finite,
    std::size_t key_idx) const {
  if (query_heads == 0 || query_tokens == 0 || head_dim == 0 ||
      kv_heads == 0 || query_heads % kv_heads != 0 ||
      key_idx >= gqa_arena_.key_tokens.size()) {
    return 0.0F;
  }
  const auto key_tokens = gqa_arena_.key_tokens[key_idx];
  if (key_tokens == 0) {
    return 0.0F;
  }
  if (!query_finite || gqa_arena_.key_finite[key_idx] == 0) {
    return std::numeric_limits<float>::quiet_NaN();
  }
  const auto query_expected =
      checked_mul(checked_mul(query_heads, query_tokens, "GQA arena query"),
                  head_dim, "GQA arena query");
  if (query.size() != static_cast<std::size_t>(query_expected)) {
    return 0.0F;
  }
  const auto repeat = query_heads / kv_heads;
  const auto key_row_start = gqa_arena_.key_row_offsets[key_idx];
  const auto* rows = gqa_arena_.rows.data();
  const float inv_denom =
      1.0F / std::sqrt(static_cast<float>(head_dim));
  float total = 0.0F;
  for (std::uint64_t h = 0; h < query_heads; ++h) {
    const auto kh = h / repeat;
    float best = 0.0F;
    for (std::uint64_t qi = 0; qi < query_tokens; ++qi) {
      const auto qoff = ((h * query_tokens) + qi) * head_dim;
      const auto* q = query.data() + static_cast<std::size_t>(qoff);
      for (std::uint64_t ki = 0; ki < key_tokens; ++ki) {
        const auto row = key_row_start +
                         static_cast<std::size_t>((kh * key_tokens) + ki);
        const auto* k = rows + (row * static_cast<std::size_t>(head_dim));
        float dot = 0.0F;
#if defined(_OPENMP)
#pragma omp simd reduction(+ : dot)
#endif
        for (std::uint64_t d = 0; d < head_dim; ++d) {
          dot += q[static_cast<std::size_t>(d)] *
                 k[static_cast<std::size_t>(d)];
        }
        best = std::max(best, std::abs(dot) * inv_denom);
      }
    }
    total += best;
  }
  return total / static_cast<float>(query_heads);
}

float RouterIndex::gqa_arena_entry_score(
    const std::vector<float>& query,
    std::uint64_t query_heads,
    std::uint64_t query_tokens,
    std::uint64_t head_dim,
    std::uint64_t kv_heads,
    bool query_finite,
    std::size_t entry_idx) const {
  const auto key_start = gqa_arena_.entry_key_offsets[entry_idx];
  const auto key_end = gqa_arena_.entry_key_offsets[entry_idx + 1];
  float entry_best = 0.0F;
  bool entry_have = false;
  for (std::size_t key_idx = key_start; key_idx < key_end; ++key_idx) {
    const auto score = gqa_arena_key_score(
        query, query_heads, query_tokens, head_dim, kv_heads, query_finite,
        key_idx);
    if (!std::isfinite(score)) {
      continue;
    }
    if (!entry_have || score > entry_best) {
      entry_best = score;
      entry_have = true;
    }
  }
  return entry_have ? entry_best : std::numeric_limits<float>::quiet_NaN();
}

std::vector<std::uint64_t> RouterIndex::route_mla_arena(
    const std::vector<float>& query, const std::vector<std::string>& lexical,
    std::size_t topk, const std::vector<std::string>& kinds,
    const std::vector<std::string>& scopes,
    const std::vector<std::string>& durabilities,
    const std::vector<std::string>& mutabilities) const {
  rebuild_mla_arena();
  if (!mla_arena_.valid || mla_arena_.dim == 0 ||
      query.size() != mla_arena_.dim) {
    return route_scan(
        query, lexical, topk, kinds, scopes, durabilities, mutabilities);
  }
  double qnorm_sq = 0.0;
  for (const auto v : query) {
    qnorm_sq += static_cast<double>(v) * static_cast<double>(v);
  }
  const auto qnorm = std::sqrt(qnorm_sq);
  std::vector<float> best(entries_.size(), 0.0F);
  std::vector<unsigned char> have(entries_.size(), 0);
  const auto entry_count = entries_.size();
  constexpr std::size_t kOpenMpRouteThreshold = 32768;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if(entry_count >= kOpenMpRouteThreshold)
#endif
  for (std::int64_t entry_i = 0;
       entry_i < static_cast<std::int64_t>(entry_count);
       ++entry_i) {
    const auto entry_idx = static_cast<std::size_t>(entry_i);
    const auto& e = entries_[entry_idx];
    if (!entry_allowed(e, kinds, scopes, durabilities, mutabilities)) {
      continue;
    }
    const auto score = exact_mla_entry_score(query, entry_idx, qnorm);
    if (std::isfinite(score)) {
      best[entry_idx] = score;
      have[entry_idx] = 1;
    }
  }

  std::vector<std::pair<float, std::uint64_t>> scored;
  const auto query_lex_hashes = lexical_hashes(lexical);
  scored.reserve(entries_.size());
  for (std::size_t entry_idx = 0; entry_idx < entries_.size(); ++entry_idx) {
    if (have[entry_idx] == 0) {
      continue;
    }
    const auto& e = entries_[entry_idx];
    float score = best[entry_idx] + lexical_bonus(
        lexical, query_lex_hashes, e.lexical_keys, e.lexical_hashes);
    if (!std::isfinite(score)) {
      continue;
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

std::vector<std::uint64_t> RouterIndex::route_mla_int4(
    const std::vector<float>& query, const std::vector<std::string>& lexical,
    std::size_t topk, const std::vector<std::string>& kinds,
    const std::vector<std::string>& scopes,
    const std::vector<std::string>& durabilities,
    const std::vector<std::string>& mutabilities) const {
  rebuild_mla_arena();
  if (!mla_arena_.valid || !mla_arena_.q4_valid || mla_arena_.dim == 0 ||
      query.size() != mla_arena_.dim) {
    return route_mla_arena(
        query, lexical, topk, kinds, scopes, durabilities, mutabilities);
  }
  double qnorm_sq = 0.0;
  for (const auto v : query) {
    qnorm_sq += static_cast<double>(v) * static_cast<double>(v);
  }
  const auto qnorm = std::sqrt(qnorm_sq);
  const auto qnorm_inv = 1.0 / (qnorm + 1.0e-8);
  const auto entry_count = entries_.size();
  const auto query_lex_hashes = lexical_hashes(lexical);
  std::vector<float> bulk_scores(entry_count, 0.0F);
  std::vector<unsigned char> bulk_have(entry_count, 0);
  constexpr std::size_t kOpenMpRouteThreshold = 32768;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if(entry_count >= kOpenMpRouteThreshold)
#endif
  for (std::int64_t entry_i = 0;
       entry_i < static_cast<std::int64_t>(entry_count);
       ++entry_i) {
    const auto entry_idx = static_cast<std::size_t>(entry_i);
    const auto& e = entries_[entry_idx];
    if (!entry_allowed(e, kinds, scopes, durabilities, mutabilities)) {
      continue;
    }
    float score = mla_arena_.single_row_per_entry
                      ? int4_mla_row_score(query, entry_idx, qnorm_inv)
                      : int4_mla_entry_score(query, entry_idx, qnorm_inv);
    score += lexical_bonus(
        lexical, query_lex_hashes, e.lexical_keys, e.lexical_hashes);
    if (!std::isfinite(score)) {
      continue;
    }
    bulk_scores[entry_idx] = score;
    bulk_have[entry_idx] = 1;
  }

  const auto requested_m = std::max(
      topk, env_size_or("GRM_ROUTER_INT4_REFINE_M", 4096));
  const auto candidate_better = [](const auto& a, const auto& b) {
    if (a.first != b.first) {
      return a.first > b.first;
    }
    return a.second < b.second;
  };
  const auto worst_candidate_first = [](const auto& a, const auto& b) {
    if (a.first != b.first) {
      return a.first > b.first;
    }
    return a.second < b.second;
  };
  std::vector<std::pair<float, std::size_t>> candidates;
  if (requested_m >= entry_count) {
    candidates.reserve(entry_count);
    for (std::size_t i = 0; i < entry_count; ++i) {
      if (bulk_have[i] != 0) {
        candidates.push_back({bulk_scores[i], i});
      }
    }
  } else {
    candidates.reserve(requested_m);
    for (std::size_t i = 0; i < entry_count; ++i) {
      if (bulk_have[i] == 0) {
        continue;
      }
      const std::pair<float, std::size_t> candidate{bulk_scores[i], i};
      if (candidates.size() < requested_m) {
        candidates.push_back(candidate);
        std::push_heap(
            candidates.begin(), candidates.end(), worst_candidate_first);
        continue;
      }
      if (!candidates.empty() && candidate_better(candidate, candidates.front())) {
        std::pop_heap(
            candidates.begin(), candidates.end(), worst_candidate_first);
        candidates.back() = candidate;
        std::push_heap(
            candidates.begin(), candidates.end(), worst_candidate_first);
      }
    }
  }

  std::vector<std::pair<float, std::uint64_t>> scored;
  scored.reserve(candidates.size());
  for (const auto& cand : candidates) {
    const auto entry_idx = cand.second;
    const auto& e = entries_[entry_idx];
    float score = exact_mla_entry_score(query, entry_idx, qnorm);
    score += lexical_bonus(
        lexical, query_lex_hashes, e.lexical_keys, e.lexical_hashes);
    if (!std::isfinite(score)) {
      continue;
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

std::vector<std::uint64_t> RouterIndex::route(
    const std::vector<float>& query, const std::vector<std::string>& lexical,
    std::size_t topk, const std::vector<std::string>& kinds,
    const std::vector<std::string>& scopes,
    const std::vector<std::string>& durabilities,
    const std::vector<std::string>& mutabilities) const {
  if (env_truthy("GRM_ROUTER_INT4")) {
    return route_mla_int4(
        query, lexical, topk, kinds, scopes, durabilities, mutabilities);
  }
  return route_mla_arena(
      query, lexical, topk, kinds, scopes, durabilities, mutabilities);
}

std::vector<std::uint64_t> RouterIndex::route_gqa_raw(
    const std::vector<float>& query,
    std::uint64_t query_heads,
    std::uint64_t query_tokens,
    std::uint64_t head_dim,
    std::uint64_t kv_heads,
    const std::vector<std::string>& lexical,
    std::size_t topk,
    const std::vector<std::string>& kinds,
    const std::vector<std::string>& scopes,
    const std::vector<std::string>& durabilities,
    const std::vector<std::string>& mutabilities) const {
  struct Scored {
    float raw = 0.0F;
    std::uint64_t node_id = 0;
    std::size_t lexical_hits = 0;
  };
  const auto query_lex_hashes = lexical_hashes(lexical);
  rebuild_gqa_arena(kv_heads, head_dim);
  const bool use_arena = gqa_arena_.valid &&
                         gqa_arena_.kv_heads == kv_heads &&
                         gqa_arena_.head_dim == head_dim &&
                         gqa_arena_.entry_key_offsets.size() ==
                             entries_.size() + 1;
  bool query_finite = true;
  if (use_arena) {
    for (const auto v : query) {
      query_finite = query_finite && std::isfinite(v);
    }
  }
  const auto entry_count = entries_.size();
  std::vector<float> raw_scores(entry_count, 0.0F);
  std::vector<std::size_t> lexical_hits(entry_count, 0);
  std::vector<unsigned char> have(entry_count, 0);
  std::uint64_t max_key_tokens = 1;
  if (use_arena) {
    for (const auto n : gqa_arena_.key_tokens) {
      max_key_tokens = std::max(max_key_tokens, n);
    }
  }
  const double gqa_route_work =
      static_cast<double>(entry_count) *
      static_cast<double>(std::max<std::uint64_t>(1, query_heads)) *
      static_cast<double>(std::max<std::uint64_t>(1, query_tokens)) *
      static_cast<double>(max_key_tokens);
  constexpr double kOpenMpGqaRouteWorkThreshold = 32768.0;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if(gqa_route_work >= kOpenMpGqaRouteWorkThreshold)
#endif
  for (std::int64_t entry_i = 0;
       entry_i < static_cast<std::int64_t>(entry_count);
       ++entry_i) {
    const auto entry_idx = static_cast<std::size_t>(entry_i);
    const auto& e = entries_[entry_idx];
    if (!entry_allowed(e, kinds, scopes, durabilities, mutabilities)) {
      continue;
    }
    const auto raw = use_arena
        ? gqa_arena_entry_score(
              query, query_heads, query_tokens, head_dim, kv_heads,
              query_finite, entry_idx)
        : max_gqa_raw_score(
              query, query_heads, query_tokens, head_dim, kv_heads,
              e.route_keys);
    if (!std::isfinite(raw)) {
      continue;
    }
    raw_scores[entry_idx] = raw;
    lexical_hits[entry_idx] = lexical_hit_count(
        lexical, query_lex_hashes, e.lexical_keys, e.lexical_hashes);
    have[entry_idx] = 1;
  }
  std::vector<Scored> scored;
  scored.reserve(entries_.size());
  float max_abs = 0.0F;
  for (std::size_t entry_idx = 0; entry_idx < entry_count; ++entry_idx) {
    if (have[entry_idx] == 0) {
      continue;
    }
    const auto raw = raw_scores[entry_idx];
    max_abs = std::max(max_abs, std::abs(raw));
    scored.push_back({raw, entries_[entry_idx].node_id, lexical_hits[entry_idx]});
  }
  const float norm = max_abs + 1.0e-8F;
  std::sort(scored.begin(), scored.end(), [&](const auto& a, const auto& b) {
    const float alex = lexical.empty()
        ? 0.0F
        : static_cast<float>(a.lexical_hits) /
              static_cast<float>(lexical.size());
    const float blex = lexical.empty()
        ? 0.0F
        : static_cast<float>(b.lexical_hits) /
              static_cast<float>(lexical.size());
    const float ascore = (a.raw / norm) + alex;
    const float bscore = (b.raw / norm) + blex;
    if (ascore != bscore) {
      return ascore > bscore;
    }
    return a.node_id < b.node_id;
  });
  std::vector<std::uint64_t> out;
  for (std::size_t i = 0; i < std::min(topk, scored.size()); ++i) {
    out.push_back(scored[i].node_id);
  }
  return out;
}

DurabilityWriter::DurabilityWriter(std::string root) : root_(std::move(root)) {}

void DurabilityWriter::write_checkpoint(HostGraftStore& store) {
  std::filesystem::create_directories(root_);
  store.save_checkpoint(root_);
  const auto s = store.stats();
  std::ostringstream summary;
  summary << "dialect " << store.dialect().dialect_id() << "\n";
  summary << "nodes " << s.nodes << "\n";
  summary << "dirty_nodes " << s.dirty_nodes << "\n";
  summary << "durable_nodes " << s.durable_nodes << "\n";
  summary << "host_payload_bytes " << s.host_payload_bytes << "\n";
  summary << "host_payload_tensors " << s.host_payload_tensors << "\n";
  atomic_write_text_file(std::filesystem::path(root_) / "checkpoint.txt",
                         summary.str());
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
  std::shared_mutex router_mutex;
  bool router_needs_prepare = false;
  grm::DeviceArena arena;
  std::mutex error_mutex;
  std::string last_error;
};

namespace {

int grm_fail(grm_store_handle* handle, const std::exception& exc) {
  if (handle != nullptr) {
    std::lock_guard<std::mutex> lock(handle->error_mutex);
    handle->last_error = exc.what();
  }
  return -1;
}

int grm_fail_msg(grm_store_handle* handle, const char* msg) {
  if (handle != nullptr) {
    std::lock_guard<std::mutex> lock(handle->error_mutex);
    handle->last_error = msg;
  }
  return -1;
}

const char* safe_cstr(const char* s, const char* fallback) {
  return s == nullptr ? fallback : s;
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

std::vector<std::uint64_t> read_u64_array(const uint64_t* xs,
                                          uint64_t count,
                                          const char* label) {
  if (xs == nullptr && count > 0) {
    throw std::runtime_error(std::string("null ") + label +
                             " with nonzero count");
  }
  std::vector<std::uint64_t> out;
  out.reserve(static_cast<std::size_t>(count));
  for (uint64_t i = 0; i < count; ++i) {
    out.push_back(xs[i]);
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

void prepare_router_for_store_dialect(grm_store_handle* handle) {
  const auto& dialect = handle->store->dialect();
  if (dialect.payload_kind == grm::PayloadKind::GQA) {
    handle->router.prepare_gqa_route(
        static_cast<std::uint64_t>(dialect.num_kv_heads),
        static_cast<std::uint64_t>(dialect.head_dim));
  } else {
    handle->router.prepare_mla_route();
  }
  handle->router_needs_prepare = false;
}

void rebuild_router_from_store(grm_store_handle* handle) {
  std::unique_lock<std::shared_mutex> lock(handle->router_mutex);
  handle->router = grm::RouterIndex();
  for (const auto node_id : handle->store->node_ids()) {
    const auto* node = handle->store->get(node_id);
    if (node == nullptr) {
      continue;
    }
    if (!node->route_keys.empty()) {
      handle->router.upsert_multi(node_id, node->route_keys,
                                  node->lexical_keys);
    } else if (!node->route_key.empty() || !node->lexical_keys.empty()) {
      handle->router.upsert(node_id, node->route_key, node->lexical_keys);
    } else {
      continue;
    }
    sync_router_node_state(handle, node_id);
  }
  handle->router_needs_prepare = true;
}

}  // namespace

extern "C" {

grm_store_handle* grm_store_create_mla_profile(const char* model_type,
                                               int num_layers,
                                               int hidden_dim,
                                               int vals_per_tok_layer,
                                               int route_layer,
                                               int latent_rank,
                                               int rope_dim,
                                               const char* position_law,
                                               const char* state_kind,
                                               const char* graftability,
                                               int remountable,
                                               const char* composition) {
  try {
    auto handle = std::make_unique<grm_store_handle>();
    grm::DialectDescriptor d;
    d.model_type = model_type == nullptr ? "" : model_type;
    d.num_layers = num_layers;
    d.hidden_dim = hidden_dim;
    d.payload_kind = grm::PayloadKind::MLA;
    d.vals_per_tok_layer = vals_per_tok_layer;
    d.route_layer = route_layer;
    d.latent_rank = latent_rank;
    d.rope_dim = rope_dim;
    d.position_law = safe_cstr(position_law, "rope_partial_mla");
    d.state_kind = safe_cstr(state_kind, "mla_latent_plus_rope");
    d.graftability = safe_cstr(graftability, "seat_remountable");
    d.remountable = remountable != 0;
    d.composition = safe_cstr(composition, "multi_mount");
    handle->store = std::make_unique<grm::HostGraftStore>(std::move(d));
    return handle.release();
  } catch (...) {
    return nullptr;
  }
}

grm_store_handle* grm_store_create_mla(const char* model_type,
                                       int num_layers,
                                       int hidden_dim,
                                       int vals_per_tok_layer,
                                       int route_layer,
                                       int latent_rank,
                                       int rope_dim) {
  return grm_store_create_mla_profile(
      model_type, num_layers, hidden_dim, vals_per_tok_layer, route_layer,
      latent_rank, rope_dim, "rope_partial_mla", "mla_latent_plus_rope",
      "seat_remountable", 1, "multi_mount");
}

grm_store_handle* grm_store_create_gqa_profile(const char* model_type,
                                               int num_layers,
                                               int hidden_dim,
                                               int vals_per_tok_layer,
                                               int route_layer,
                                               int num_kv_heads,
                                               int head_dim,
                                               const char* position_law,
                                               const char* state_kind,
                                               const char* graftability,
                                               int remountable,
                                               const char* composition) {
  try {
    auto handle = std::make_unique<grm_store_handle>();
    grm::DialectDescriptor d;
    d.model_type = model_type == nullptr ? "" : model_type;
    d.num_layers = num_layers;
    d.hidden_dim = hidden_dim;
    d.payload_kind = grm::PayloadKind::GQA;
    d.vals_per_tok_layer = vals_per_tok_layer;
    d.route_layer = route_layer;
    d.num_kv_heads = num_kv_heads;
    d.head_dim = head_dim;
    d.position_law = safe_cstr(position_law, "rope_full_kv");
    d.state_kind = safe_cstr(state_kind, "kv");
    d.graftability = safe_cstr(graftability, "seat_remountable");
    d.remountable = remountable != 0;
    d.composition = safe_cstr(composition, "multi_mount");
    handle->store = std::make_unique<grm::HostGraftStore>(std::move(d));
    return handle.release();
  } catch (...) {
    return nullptr;
  }
}

grm_store_handle* grm_store_create_gqa(const char* model_type,
                                       int num_layers,
                                       int hidden_dim,
                                       int vals_per_tok_layer,
                                       int route_layer,
                                       int num_kv_heads,
                                       int head_dim) {
  return grm_store_create_gqa_profile(
      model_type, num_layers, hidden_dim, vals_per_tok_layer, route_layer,
      num_kv_heads, head_dim, "rope_full_kv", "kv", "seat_remountable", 1,
      "multi_mount");
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

int grm_store_dialect_profile(grm_store_handle* handle,
                              char* out,
                              size_t out_cap) {
  try {
    if (handle == nullptr || handle->store == nullptr || out == nullptr ||
        out_cap == 0) {
      return grm_fail_msg(handle, "invalid dialect_profile arguments");
    }
    const auto s = handle->store->dialect().profile_id();
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

int grm_store_clear_payload(grm_store_handle* handle, uint64_t node_id) {
  try {
    if (handle == nullptr || handle->store == nullptr) {
      return grm_fail_msg(handle, "invalid clear_payload arguments");
    }
    handle->store->clear_payload(node_id);
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

int grm_store_slice_tensor(grm_store_handle* handle,
                           uint64_t node_id,
                           const char* name,
                           uint64_t axis,
                           uint64_t start,
                           uint64_t length,
                           uint64_t* out_shape,
                           uint64_t shape_cap,
                           char* out_dtype,
                           size_t dtype_cap,
                           uint8_t* out_payload,
                           uint64_t payload_cap,
                           grm_tensor_info_c* out) {
  try {
    if (handle == nullptr || handle->store == nullptr || name == nullptr ||
        out == nullptr) {
      return grm_fail_msg(handle, "invalid slice_tensor arguments");
    }
    if (out_shape == nullptr && shape_cap > 0) {
      return grm_fail_msg(handle, "null shape buffer with nonzero capacity");
    }
    if (out_payload == nullptr && payload_cap > 0) {
      return grm_fail_msg(handle, "null payload buffer with nonzero capacity");
    }
    auto t = handle->store->slice_tensor(node_id, name, axis, start, length);
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
    const uint64_t bytes = static_cast<uint64_t>(t.bytes.size());
    if (out_payload == nullptr) {
      return 0;
    }
    if (payload_cap < bytes) {
      return grm_fail_msg(handle, "slice output buffer too small");
    }
    if (bytes > 0) {
      std::memcpy(out_payload, t.bytes.data(), static_cast<std::size_t>(bytes));
    }
    return 0;
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
    std::unique_lock<std::shared_mutex> lock(handle->router_mutex);
    handle->router.set_active(node_id, is_active);
    return 0;
  } catch (const std::exception& exc) {
    return grm_fail(handle, exc);
  }
}

int grm_store_set_no_fold(grm_store_handle* handle,
                          uint64_t node_id,
                          int no_fold) {
  try {
    if (handle == nullptr || handle->store == nullptr) {
      return grm_fail_msg(handle, "invalid set_no_fold arguments");
    }
    handle->store->set_no_fold(node_id, no_fold != 0);
    return 0;
  } catch (const std::exception& exc) {
    return grm_fail(handle, exc);
  }
}

int grm_store_foldable_nodes(grm_store_handle* handle,
                             const char* kind,
                             const uint64_t* excluded_node_ids,
                             uint64_t excluded_count,
                             uint64_t* out_node_ids,
                             uint64_t out_cap,
                             uint64_t* out_count) {
  try {
    if (handle == nullptr || handle->store == nullptr ||
        kind == nullptr || out_count == nullptr) {
      return grm_fail_msg(handle, "invalid foldable_nodes arguments");
    }
    if (excluded_node_ids == nullptr && excluded_count > 0) {
      return grm_fail_msg(handle,
                          "null foldable exclusion list with nonzero count");
    }
    if (out_node_ids == nullptr && out_cap > 0) {
      return grm_fail_msg(handle,
                          "null foldable output buffer with nonzero capacity");
    }
    auto excluded = read_u64_array(
        excluded_node_ids, excluded_count, "foldable excluded");
    const auto nodes = handle->store->foldable_nodes(kind, excluded);
    const auto n = static_cast<uint64_t>(nodes.size());
    *out_count = n;
    const auto copied = std::min<uint64_t>(n, out_cap);
    for (uint64_t i = 0; i < copied; ++i) {
      out_node_ids[i] = nodes[static_cast<std::size_t>(i)];
    }
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
    std::unique_lock<std::shared_mutex> lock(handle->router_mutex);
    handle->router.set_route_metadata(node_id, kind == nullptr ? "" : kind,
                                      scope == nullptr ? "" : scope,
                                      durability == nullptr ? "" : durability,
                                      mutability == nullptr ? "" : mutability);
    return 0;
  } catch (const std::exception& exc) {
    return grm_fail(handle, exc);
  }
}

int grm_store_set_fact_identity(grm_store_handle* handle,
                                uint64_t node_id,
                                const char* subject,
                                const char* predicate,
                                const char* value,
                                const char* scope,
                                const char* valid_from,
                                const char* expires_at) {
  try {
    if (handle == nullptr || handle->store == nullptr) {
      return grm_fail_msg(handle, "invalid set_fact_identity arguments");
    }
    grm::FactIdentity identity;
    identity.subject = safe_cstr(subject, "");
    identity.predicate = safe_cstr(predicate, "");
    identity.value = safe_cstr(value, "");
    identity.scope = safe_cstr(scope, "project");
    identity.valid_from = safe_cstr(valid_from, "");
    identity.expires_at = safe_cstr(expires_at, "");
    handle->store->set_fact_identity(node_id, std::move(identity));
    return 0;
  } catch (const std::exception& exc) {
    return grm_fail(handle, exc);
  }
}

int grm_store_fact_matches(grm_store_handle* handle,
                           const char* subject,
                           const char* predicate,
                           const char* value,
                           const char* scope,
                           uint64_t value_mode,
                           uint64_t* out_node_ids,
                           uint64_t out_cap,
                           uint64_t* out_count) {
  try {
    if (handle == nullptr || handle->store == nullptr || out_count == nullptr) {
      return grm_fail_msg(handle, "invalid fact_matches arguments");
    }
    if (out_node_ids == nullptr && out_cap > 0) {
      return grm_fail_msg(handle, "null fact_matches output buffer");
    }
    grm::FactIdentity identity;
    identity.subject = safe_cstr(subject, "");
    identity.predicate = safe_cstr(predicate, "");
    identity.value = safe_cstr(value, "");
    identity.scope = safe_cstr(scope, "project");
    const auto matches = handle->store->fact_matches(identity, value_mode);
    if (out_node_ids == nullptr || out_cap == 0) {
      *out_count = static_cast<uint64_t>(matches.size());
      return 0;
    }
    const auto n = std::min<uint64_t>(
        static_cast<uint64_t>(matches.size()), out_cap);
    for (uint64_t i = 0; i < n; ++i) {
      out_node_ids[i] = matches[static_cast<std::size_t>(i)];
    }
    *out_count = n;
    return 0;
  } catch (const std::exception& exc) {
    return grm_fail(handle, exc);
  }
}

int grm_store_fact_matches_ex(grm_store_handle* handle,
                              const char* subject,
                              const char* predicate,
                              const char* value,
                              const char* scope,
                              const char* valid_from,
                              const char* expires_at,
                              uint64_t value_mode,
                              uint64_t temporal_mode,
                              uint64_t* out_node_ids,
                              uint64_t out_cap,
                              uint64_t* out_count) {
  try {
    if (handle == nullptr || handle->store == nullptr || out_count == nullptr) {
      return grm_fail_msg(handle, "invalid fact_matches_ex arguments");
    }
    if (out_node_ids == nullptr && out_cap > 0) {
      return grm_fail_msg(handle, "null fact_matches_ex output buffer");
    }
    grm::FactIdentity identity;
    identity.subject = safe_cstr(subject, "");
    identity.predicate = safe_cstr(predicate, "");
    identity.value = safe_cstr(value, "");
    identity.scope = safe_cstr(scope, "project");
    identity.valid_from = safe_cstr(valid_from, "");
    identity.expires_at = safe_cstr(expires_at, "");
    const auto matches = handle->store->fact_matches(
        identity, value_mode, temporal_mode);
    if (out_node_ids == nullptr || out_cap == 0) {
      *out_count = static_cast<uint64_t>(matches.size());
      return 0;
    }
    const auto n = std::min<uint64_t>(
        static_cast<uint64_t>(matches.size()), out_cap);
    for (uint64_t i = 0; i < n; ++i) {
      out_node_ids[i] = matches[static_cast<std::size_t>(i)];
    }
    *out_count = n;
    return 0;
  } catch (const std::exception& exc) {
    return grm_fail(handle, exc);
  }
}

int grm_store_filter_active_nodes(grm_store_handle* handle,
                                  const uint64_t* node_ids,
                                  uint64_t node_count,
                                  uint64_t* out_node_ids,
                                  uint64_t out_cap,
                                  uint64_t* out_count) {
  try {
    if (handle == nullptr || handle->store == nullptr || out_count == nullptr) {
      return grm_fail_msg(handle, "invalid filter_active_nodes arguments");
    }
    if (node_ids == nullptr && node_count > 0) {
      return grm_fail_msg(handle, "null filter_active_nodes input");
    }
    if (out_node_ids == nullptr && out_cap > 0) {
      return grm_fail_msg(handle, "null filter_active_nodes output buffer");
    }
    const auto requested = read_u64_array(
        node_ids, node_count, "filter_active_nodes");
    const auto active = handle->store->filter_active_nodes(requested);
    if (out_node_ids == nullptr || out_cap == 0) {
      *out_count = static_cast<uint64_t>(active.size());
      return 0;
    }
    const auto n = std::min<uint64_t>(
        static_cast<uint64_t>(active.size()), out_cap);
    for (uint64_t i = 0; i < n; ++i) {
      out_node_ids[i] = active[static_cast<std::size_t>(i)];
    }
    *out_count = n;
    return 0;
  } catch (const std::exception& exc) {
    return grm_fail(handle, exc);
  }
}

int grm_store_active_text_matches(grm_store_handle* handle,
                                  const char* query,
                                  uint64_t* out_node_ids,
                                  uint64_t out_cap,
                                  uint64_t* out_count) {
  try {
    if (handle == nullptr || handle->store == nullptr || out_count == nullptr) {
      return grm_fail_msg(handle, "invalid active_text_matches arguments");
    }
    if (out_node_ids == nullptr && out_cap > 0) {
      return grm_fail_msg(handle, "null active_text_matches output buffer");
    }
    const auto matches = handle->store->active_text_matches(
        safe_cstr(query, ""));
    if (out_node_ids == nullptr || out_cap == 0) {
      *out_count = static_cast<uint64_t>(matches.size());
      return 0;
    }
    const auto n = std::min<uint64_t>(
        static_cast<uint64_t>(matches.size()), out_cap);
    for (uint64_t i = 0; i < n; ++i) {
      out_node_ids[i] = matches[static_cast<std::size_t>(i)];
    }
    *out_count = n;
    return 0;
  } catch (const std::exception& exc) {
    return grm_fail(handle, exc);
  }
}

int grm_store_set_graph_edges(grm_store_handle* handle,
                              uint64_t node_id,
                              const uint64_t* source_turns,
                              uint64_t source_turn_count,
                              const uint64_t* source_grafts,
                              uint64_t source_graft_count,
                              const uint64_t* supersedes,
                              uint64_t supersedes_count,
                              const uint64_t* superseded_by,
                              uint64_t superseded_by_count) {
  try {
    if (handle == nullptr || handle->store == nullptr) {
      return grm_fail_msg(handle, "invalid set_graph_edges arguments");
    }
    grm::GraphEdges edges;
    edges.source_turns = read_u64_array(
        source_turns, source_turn_count, "source_turns");
    edges.source_grafts = read_u64_array(
        source_grafts, source_graft_count, "source_grafts");
    edges.supersedes = read_u64_array(
        supersedes, supersedes_count, "supersedes");
    edges.superseded_by = read_u64_array(
        superseded_by, superseded_by_count, "superseded_by");
    handle->store->set_graph_edges(node_id, std::move(edges));
    return 0;
  } catch (const std::exception& exc) {
    return grm_fail(handle, exc);
  }
}

int grm_store_graph_edges_info(grm_store_handle* handle,
                               uint64_t node_id,
                               grm_graph_edges_info_c* out) {
  try {
    if (handle == nullptr || handle->store == nullptr || out == nullptr) {
      return grm_fail_msg(handle, "invalid graph_edges_info arguments");
    }
    const auto edges = handle->store->graph_edges(node_id);
    out->source_turns = static_cast<uint64_t>(edges.source_turns.size());
    out->source_grafts = static_cast<uint64_t>(edges.source_grafts.size());
    out->supersedes = static_cast<uint64_t>(edges.supersedes.size());
    out->superseded_by = static_cast<uint64_t>(edges.superseded_by.size());
    return 0;
  } catch (const std::exception& exc) {
    return grm_fail(handle, exc);
  }
}

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
                               grm_graph_edges_info_c* out_counts) {
  try {
    if (handle == nullptr || handle->store == nullptr || out_counts == nullptr) {
      return grm_fail_msg(handle, "invalid read_graph_edges arguments");
    }
    if ((out_source_turns == nullptr && source_turn_cap > 0) ||
        (out_source_grafts == nullptr && source_graft_cap > 0) ||
        (out_supersedes == nullptr && supersedes_cap > 0) ||
        (out_superseded_by == nullptr && superseded_by_cap > 0)) {
      return grm_fail_msg(handle, "null graph edge buffer with nonzero capacity");
    }
    const auto edges = handle->store->graph_edges(node_id);
    const auto copy_vec = [](const std::vector<std::uint64_t>& src,
                             uint64_t* dst, uint64_t cap) {
      const uint64_t n = std::min<uint64_t>(
          static_cast<uint64_t>(src.size()), cap);
      for (uint64_t i = 0; i < n; ++i) {
        dst[i] = src[static_cast<std::size_t>(i)];
      }
    };
    copy_vec(edges.source_turns, out_source_turns, source_turn_cap);
    copy_vec(edges.source_grafts, out_source_grafts, source_graft_cap);
    copy_vec(edges.supersedes, out_supersedes, supersedes_cap);
    copy_vec(edges.superseded_by, out_superseded_by, superseded_by_cap);
    out_counts->source_turns =
        static_cast<uint64_t>(edges.source_turns.size());
    out_counts->source_grafts =
        static_cast<uint64_t>(edges.source_grafts.size());
    out_counts->supersedes =
        static_cast<uint64_t>(edges.supersedes.size());
    out_counts->superseded_by =
        static_cast<uint64_t>(edges.superseded_by.size());
    return 0;
  } catch (const std::exception& exc) {
    return grm_fail(handle, exc);
  }
}

int grm_store_source_closure(grm_store_handle* handle,
                             const uint64_t* node_ids,
                             uint64_t node_count,
                             uint64_t max_depth,
                             int include_roots,
                             uint64_t* out_node_ids,
                             uint64_t out_cap,
                             uint64_t* out_count) {
  try {
    if (handle == nullptr || handle->store == nullptr || out_count == nullptr) {
      return grm_fail_msg(handle, "invalid source_closure arguments");
    }
    if (node_ids == nullptr && node_count > 0) {
      return grm_fail_msg(handle, "null source_closure seeds");
    }
    if (out_node_ids == nullptr && out_cap > 0) {
      return grm_fail_msg(handle, "null source_closure output buffer");
    }
    const auto seeds = read_u64_array(node_ids, node_count, "source_closure");
    const auto closure = handle->store->source_closure(
        seeds, max_depth, include_roots != 0);
    const uint64_t n = static_cast<uint64_t>(closure.size());
    const uint64_t copied = std::min<uint64_t>(n, out_cap);
    for (uint64_t i = 0; i < copied; ++i) {
      out_node_ids[i] = closure[static_cast<std::size_t>(i)];
    }
    *out_count = n;
    return 0;
  } catch (const std::exception& exc) {
    return grm_fail(handle, exc);
  }
}

int grm_store_apply_revision(grm_store_handle* handle,
                             uint64_t replacement_node_id,
                             const uint64_t* supersedes,
                             uint64_t supersedes_count) {
  try {
    if (handle == nullptr || handle->store == nullptr) {
      return grm_fail_msg(handle, "invalid apply_revision arguments");
    }
    auto superseded = read_u64_array(
        supersedes, supersedes_count, "supersedes");
    handle->store->apply_revision(replacement_node_id, superseded);
    std::unique_lock<std::shared_mutex> lock(handle->router_mutex);
    handle->router.set_active(replacement_node_id, true);
    for (const auto old_id : superseded) {
      handle->router.set_active(old_id, false);
    }
    return 0;
  } catch (const std::exception& exc) {
    return grm_fail(handle, exc);
  }
}

int grm_store_apply_expire(grm_store_handle* handle,
                           const uint64_t* node_ids,
                           uint64_t node_count) {
  try {
    if (handle == nullptr || handle->store == nullptr) {
      return grm_fail_msg(handle, "invalid apply_expire arguments");
    }
    auto expired = read_u64_array(node_ids, node_count, "expired");
    handle->store->apply_expire(expired);
    std::set<std::uint64_t> seen;
    std::unique_lock<std::shared_mutex> lock(handle->router_mutex);
    for (const auto node_id : expired) {
      if (seen.insert(node_id).second) {
        handle->router.set_active(node_id, false);
      }
    }
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

int grm_store_node_text(grm_store_handle* handle,
                        uint64_t node_id,
                        char* out_text,
                        size_t out_cap,
                        uint64_t* out_len) {
  try {
    if (handle == nullptr || handle->store == nullptr || out_len == nullptr) {
      return grm_fail_msg(handle, "invalid node_text arguments");
    }
    if (out_text == nullptr && out_cap > 0) {
      return grm_fail_msg(handle, "null text buffer with nonzero capacity");
    }
    const auto& s = handle->store->text(node_id);
    *out_len = static_cast<uint64_t>(s.size());
    if (out_text != nullptr && out_cap > 0) {
      const size_t n = std::min(out_cap - 1, s.size());
      std::memcpy(out_text, s.data(), n);
      out_text[n] = '\0';
    }
    return 0;
  } catch (const std::exception& exc) {
    return grm_fail(handle, exc);
  }
}

int grm_store_node_summary_json(grm_store_handle* handle,
                                uint64_t node_id,
                                char* out_json,
                                size_t out_cap,
                                uint64_t* out_len) {
  try {
    if (handle == nullptr || handle->store == nullptr || out_len == nullptr) {
      return grm_fail_msg(handle, "invalid node_summary arguments");
    }
    if (out_json == nullptr && out_cap > 0) {
      return grm_fail_msg(handle,
                          "null summary buffer with nonzero capacity");
    }
    const auto json = handle->store->node_summary_json(node_id);
    *out_len = static_cast<uint64_t>(json.size());
    if (out_json != nullptr && out_cap > 0) {
      const size_t n = std::min(out_cap - 1, json.size());
      std::memcpy(out_json, json.data(), n);
      out_json[n] = '\0';
    }
    return 0;
  } catch (const std::exception& exc) {
    return grm_fail(handle, exc);
  }
}

int grm_store_set_provenance_json(grm_store_handle* handle,
                                  uint64_t node_id,
                                  const char* provenance_json) {
  try {
    if (handle == nullptr || handle->store == nullptr ||
        provenance_json == nullptr) {
      return grm_fail_msg(handle, "invalid set_provenance arguments");
    }
    handle->store->set_provenance_json(node_id, provenance_json);
    return 0;
  } catch (const std::exception& exc) {
    return grm_fail(handle, exc);
  }
}

int grm_store_provenance_json(grm_store_handle* handle,
                              uint64_t node_id,
                              char* out_json,
                              size_t out_cap,
                              uint64_t* out_len) {
  try {
    if (handle == nullptr || handle->store == nullptr || out_len == nullptr) {
      return grm_fail_msg(handle, "invalid provenance_json arguments");
    }
    if (out_json == nullptr && out_cap > 0) {
      return grm_fail_msg(handle,
                          "null provenance buffer with nonzero capacity");
    }
    const auto& json = handle->store->provenance_json(node_id);
    *out_len = static_cast<uint64_t>(json.size());
    if (out_json != nullptr && out_cap > 0) {
      const size_t n = std::min(out_cap - 1, json.size());
      std::memcpy(out_json, json.data(), n);
      out_json[n] = '\0';
    }
    return 0;
  } catch (const std::exception& exc) {
    return grm_fail(handle, exc);
  }
}

int grm_store_parse_memory_command(grm_store_handle* handle,
                                   const char* text,
                                   char* out_json,
                                   size_t out_cap,
                                   uint64_t* out_len) {
  try {
    if (handle == nullptr || handle->store == nullptr || text == nullptr ||
        out_len == nullptr) {
      return grm_fail_msg(handle, "invalid parse_memory_command arguments");
    }
    if (out_json == nullptr && out_cap > 0) {
      return grm_fail_msg(handle,
                          "null command-plan buffer with nonzero capacity");
    }
    const auto json = grm::memory_command_plan_json(
        grm::parse_memory_command(text));
    *out_len = static_cast<uint64_t>(json.size());
    if (out_json != nullptr && out_cap > 0) {
      const size_t n = std::min(out_cap - 1, json.size());
      std::memcpy(out_json, json.data(), n);
      out_json[n] = '\0';
    }
    return 0;
  } catch (const std::exception& exc) {
    return grm_fail(handle, exc);
  }
}

int grm_store_plan_remember_flush(grm_store_handle* handle,
                                  const char* durability_mode,
                                  const char* durability,
                                  const char* scope,
                                  int flush_immediately,
                                  char* out_json,
                                  size_t out_cap,
                                  uint64_t* out_len) {
  try {
    if (handle == nullptr || handle->store == nullptr || out_len == nullptr) {
      return grm_fail_msg(handle, "invalid plan_remember_flush arguments");
    }
    if (out_json == nullptr && out_cap > 0) {
      return grm_fail_msg(
          handle, "null remember flush buffer with nonzero capacity");
    }
    const auto json = grm::remember_flush_plan_json(
        grm::plan_remember_flush(
            durability_mode == nullptr ? "" : durability_mode,
            durability == nullptr ? "" : durability,
            scope == nullptr ? "" : scope,
            flush_immediately != 0));
    *out_len = static_cast<uint64_t>(json.size());
    if (out_json != nullptr && out_cap > 0) {
      const size_t n = std::min(out_cap - 1, json.size());
      std::memcpy(out_json, json.data(), n);
      out_json[n] = '\0';
    }
    return 0;
  } catch (const std::exception& exc) {
    return grm_fail(handle, exc);
  }
}

int grm_store_plan_runtime_event(grm_store_handle* handle,
                                 const char* event,
                                 const char* action,
                                 int autosave_enabled,
                                 int force_flush,
                                 int read_only,
                                 char* out_json,
                                 size_t out_cap,
                                 uint64_t* out_len) {
  try {
    if (handle == nullptr || handle->store == nullptr ||
        event == nullptr || action == nullptr || out_len == nullptr) {
      return grm_fail_msg(handle, "invalid plan_runtime_event arguments");
    }
    if (out_json == nullptr && out_cap > 0) {
      return grm_fail_msg(
          handle, "null runtime event buffer with nonzero capacity");
    }
    const auto json = grm::runtime_event_plan_json(
        grm::plan_runtime_event(event, action, autosave_enabled != 0,
                                force_flush != 0, read_only != 0));
    *out_len = static_cast<uint64_t>(json.size());
    if (out_json != nullptr && out_cap > 0) {
      const size_t n = std::min(out_cap - 1, json.size());
      std::memcpy(out_json, json.data(), n);
      out_json[n] = '\0';
    }
    return 0;
  } catch (const std::exception& exc) {
    return grm_fail(handle, exc);
  }
}

int grm_store_plan_durability_mode(grm_store_handle* handle,
                                   const char* requested_mode,
                                   const char* current_mode,
                                   int old_wal_enabled,
                                   int wal_enabled_override,
                                   char* out_json,
                                   size_t out_cap,
                                   uint64_t* out_len) {
  try {
    if (handle == nullptr || handle->store == nullptr ||
        requested_mode == nullptr || out_len == nullptr) {
      return grm_fail_msg(handle, "invalid plan_durability_mode arguments");
    }
    if (out_json == nullptr && out_cap > 0) {
      return grm_fail_msg(
          handle, "null durability mode buffer with nonzero capacity");
    }
    const auto json = grm::durability_mode_plan_json(
        grm::plan_durability_mode(
            requested_mode,
            current_mode == nullptr ? "" : current_mode,
            old_wal_enabled != 0,
            wal_enabled_override != 0));
    *out_len = static_cast<uint64_t>(json.size());
    if (out_json != nullptr && out_cap > 0) {
      const size_t n = std::min(out_cap - 1, json.size());
      std::memcpy(out_json, json.data(), n);
      out_json[n] = '\0';
    }
    return 0;
  } catch (const std::exception& exc) {
    return grm_fail(handle, exc);
  }
}

int grm_store_plan_metadata_update(grm_store_handle* handle,
                                   const char* command,
                                   const char* metadata_key,
                                   const char* metadata_value,
                                   char* out_json,
                                   size_t out_cap,
                                   uint64_t* out_len) {
  try {
    if (handle == nullptr || handle->store == nullptr ||
        metadata_key == nullptr || out_len == nullptr) {
      return grm_fail_msg(handle, "invalid plan_metadata_update arguments");
    }
    if (out_json == nullptr && out_cap > 0) {
      return grm_fail_msg(
          handle, "null metadata update buffer with nonzero capacity");
    }
    const auto json = grm::metadata_update_plan_json(
        grm::plan_metadata_update(
            command == nullptr ? "" : command,
            metadata_key,
            metadata_value == nullptr ? "" : metadata_value));
    *out_len = static_cast<uint64_t>(json.size());
    if (out_json != nullptr && out_cap > 0) {
      const size_t n = std::min(out_cap - 1, json.size());
      std::memcpy(out_json, json.data(), n);
      out_json[n] = '\0';
    }
    return 0;
  } catch (const std::exception& exc) {
    return grm_fail(handle, exc);
  }
}

int grm_store_plan_memory_mutation(grm_store_handle* handle,
                                   const char* command,
                                   int has_query,
                                   uint64_t target_count,
                                   int has_replacement,
                                   char* out_json,
                                   size_t out_cap,
                                   uint64_t* out_len) {
  try {
    if (handle == nullptr || handle->store == nullptr ||
        command == nullptr || out_len == nullptr) {
      return grm_fail_msg(handle, "invalid plan_memory_mutation arguments");
    }
    if (out_json == nullptr && out_cap > 0) {
      return grm_fail_msg(
          handle, "null memory mutation buffer with nonzero capacity");
    }
    const auto json = grm::memory_mutation_plan_json(
        grm::plan_memory_mutation(command, has_query != 0, target_count,
                                  has_replacement != 0));
    *out_len = static_cast<uint64_t>(json.size());
    if (out_json != nullptr && out_cap > 0) {
      const size_t n = std::min(out_cap - 1, json.size());
      std::memcpy(out_json, json.data(), n);
      out_json[n] = '\0';
    }
    return 0;
  } catch (const std::exception& exc) {
    return grm_fail(handle, exc);
  }
}

int grm_store_plan_librarian(grm_store_handle* handle,
                             uint64_t foldable_turn_count,
                             uint64_t foldable_digest_count,
                             uint64_t turns_high,
                             uint64_t turns_fold,
                             uint64_t digests_high,
                             uint64_t digests_fold,
                             int era_enabled,
                             int deferred_backpressure,
                             char* out_json,
                             size_t out_cap,
                             uint64_t* out_len) {
  try {
    if (handle == nullptr || handle->store == nullptr || out_len == nullptr) {
      return grm_fail_msg(handle, "invalid plan_librarian arguments");
    }
    if (out_json == nullptr && out_cap > 0) {
      return grm_fail_msg(
          handle, "null librarian plan buffer with nonzero capacity");
    }
    const auto json = grm::librarian_plan_json(grm::plan_librarian(
        foldable_turn_count,
        foldable_digest_count,
        turns_high,
        turns_fold,
        digests_high,
        digests_fold,
        era_enabled != 0,
        deferred_backpressure != 0));
    *out_len = static_cast<uint64_t>(json.size());
    if (out_json != nullptr && out_cap > 0) {
      const size_t n = std::min(out_cap - 1, json.size());
      std::memcpy(out_json, json.data(), n);
      out_json[n] = '\0';
    }
    return 0;
  } catch (const std::exception& exc) {
    return grm_fail(handle, exc);
  }
}

int grm_store_plan_extraction_policy(grm_store_handle* handle,
                                     const char* action,
                                     const char* write_intent,
                                     double confidence,
                                     double write_direct_threshold,
                                     uint64_t conflict_count,
                                     uint64_t requested_supersede_count,
                                     uint64_t requested_id_count,
                                     uint64_t equivalent_count,
                                     uint64_t expire_target_count,
                                     char* out_json,
                                     size_t out_cap,
                                     uint64_t* out_len) {
  try {
    if (handle == nullptr || handle->store == nullptr || action == nullptr ||
        write_intent == nullptr || out_len == nullptr) {
      return grm_fail_msg(handle, "invalid plan_extraction_policy arguments");
    }
    if (out_json == nullptr && out_cap > 0) {
      return grm_fail_msg(
          handle, "null extraction policy buffer with nonzero capacity");
    }
    const auto json = grm::extraction_policy_plan_json(
        grm::plan_extraction_policy(
            action, write_intent, confidence, write_direct_threshold,
            conflict_count, requested_supersede_count, requested_id_count,
            equivalent_count, expire_target_count));
    *out_len = static_cast<uint64_t>(json.size());
    if (out_json != nullptr && out_cap > 0) {
      const size_t n = std::min(out_cap - 1, json.size());
      std::memcpy(out_json, json.data(), n);
      out_json[n] = '\0';
    }
    return 0;
  } catch (const std::exception& exc) {
    return grm_fail(handle, exc);
  }
}

int grm_store_plan_reinforcement(grm_store_handle* handle,
                                 const char* old_write_intent,
                                 const char* new_write_intent,
                                 double old_confidence,
                                 double new_confidence,
                                 uint64_t old_reinforcement_count,
                                 char* out_json,
                                 size_t out_cap,
                                 uint64_t* out_len) {
  try {
    if (handle == nullptr || handle->store == nullptr ||
        old_write_intent == nullptr || new_write_intent == nullptr ||
        out_len == nullptr) {
      return grm_fail_msg(handle, "invalid plan_reinforcement arguments");
    }
    if (out_json == nullptr && out_cap > 0) {
      return grm_fail_msg(
          handle, "null reinforcement plan buffer with nonzero capacity");
    }
    const auto json = grm::reinforcement_plan_json(
        grm::plan_reinforcement(
            old_write_intent, new_write_intent, old_confidence, new_confidence,
            old_reinforcement_count));
    *out_len = static_cast<uint64_t>(json.size());
    if (out_json != nullptr && out_cap > 0) {
      const size_t n = std::min(out_cap - 1, json.size());
      std::memcpy(out_json, json.data(), n);
      out_json[n] = '\0';
    }
    return 0;
  } catch (const std::exception& exc) {
    return grm_fail(handle, exc);
  }
}

int grm_store_plan_review_transition(grm_store_handle* handle,
                                     const char* command,
                                     const char* status,
                                     int has_approved_node_id,
                                     char* out_json,
                                     size_t out_cap,
                                     uint64_t* out_len) {
  try {
    if (handle == nullptr || handle->store == nullptr || command == nullptr ||
        status == nullptr || out_len == nullptr) {
      return grm_fail_msg(handle, "invalid plan_review_transition arguments");
    }
    if (out_json == nullptr && out_cap > 0) {
      return grm_fail_msg(
          handle, "null review transition plan buffer with nonzero capacity");
    }
    const auto json = grm::review_transition_plan_json(
        grm::plan_review_transition(command, status,
                                    has_approved_node_id != 0));
    *out_len = static_cast<uint64_t>(json.size());
    if (out_json != nullptr && out_cap > 0) {
      const size_t n = std::min(out_cap - 1, json.size());
      std::memcpy(out_json, json.data(), n);
      out_json[n] = '\0';
    }
    return 0;
  } catch (const std::exception& exc) {
    return grm_fail(handle, exc);
  }
}

int grm_store_plan_cull_spans(grm_store_handle* handle,
                              uint64_t ntok,
                              uint64_t max_tokens,
                              int has_max_tokens,
                              const uint64_t* starts,
                              const uint64_t* ends,
                              uint64_t span_count,
                              int retire_parent,
                              char* out_json,
                              size_t out_cap,
                              uint64_t* out_len) {
  try {
    if (handle == nullptr || handle->store == nullptr || out_len == nullptr) {
      return grm_fail_msg(handle, "invalid plan_cull_spans arguments");
    }
    if (out_json == nullptr && out_cap > 0) {
      return grm_fail_msg(handle, "null cull span buffer with nonzero capacity");
    }
    if (span_count > 0 && (starts == nullptr || ends == nullptr)) {
      return grm_fail_msg(handle, "null cull span arrays with nonzero count");
    }
    std::vector<grm::CullSpan> spans;
    spans.reserve(static_cast<std::size_t>(span_count));
    for (uint64_t i = 0; i < span_count; ++i) {
      spans.push_back({starts[i], ends[i]});
    }
    const auto json = grm::cull_span_plan_json(grm::plan_cull_spans(
        ntok, has_max_tokens != 0, max_tokens, spans, retire_parent != 0));
    *out_len = static_cast<uint64_t>(json.size());
    if (out_json != nullptr && out_cap > 0) {
      const size_t n = std::min(out_cap - 1, json.size());
      std::memcpy(out_json, json.data(), n);
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
    if (route_key == nullptr && route_len > 0) {
      return grm_fail_msg(handle, "null route key with nonzero length");
    }
    std::vector<float> key;
    key.reserve(static_cast<std::size_t>(route_len));
    for (uint64_t i = 0; i < route_len; ++i) {
      key.push_back(route_key[i]);
    }
    auto lexical = split_lexical_keys(lexical_keys);
    std::unique_lock<std::shared_mutex> lock(handle->router_mutex);
    auto* node = handle->store->get(node_id);
    if (node == nullptr) {
      return grm_fail_msg(handle, "unknown GRM node id");
    }
    node->route_key = key;
    node->route_keys.clear();
    node->route_keys.push_back(key);
    node->lexical_keys = lexical;
    handle->store->mark_dirty(node_id, false, true);
    handle->router.upsert(node_id, std::move(key), std::move(lexical));
    sync_router_node_state(handle, node_id);
    handle->router_needs_prepare = true;
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
    auto lexical = split_lexical_keys(lexical_keys);
    std::unique_lock<std::shared_mutex> lock(handle->router_mutex);
    auto* node = handle->store->get(node_id);
    if (node == nullptr) {
      return grm_fail_msg(handle, "unknown GRM node id");
    }
    node->route_keys = keys;
    node->route_key = keys.empty() ? std::vector<float>() : keys.front();
    node->lexical_keys = lexical;
    handle->store->mark_dirty(node_id, false, true);
    handle->router.upsert_multi(node_id, std::move(keys), std::move(lexical));
    sync_router_node_state(handle, node_id);
    handle->router_needs_prepare = true;
    return 0;
  } catch (const std::exception& exc) {
    return grm_fail(handle, exc);
  }
}

int grm_store_set_route_list(grm_store_handle* handle,
                             uint64_t node_id,
                             const float* route_values,
                             uint64_t value_count,
                             const uint64_t* route_offsets,
                             uint64_t key_count,
                             const char* lexical_keys) {
  try {
    if (handle == nullptr || handle->store == nullptr) {
      return grm_fail_msg(handle, "invalid set_route_list arguments");
    }
    if (route_values == nullptr && value_count > 0) {
      return grm_fail_msg(handle, "null route values with nonzero length");
    }
    if (route_offsets == nullptr && key_count > 0) {
      return grm_fail_msg(handle, "null route offsets with nonzero key count");
    }
    if (key_count == 0 && value_count != 0) {
      return grm_fail_msg(handle, "route values require route offsets");
    }
    std::vector<std::vector<float>> keys;
    keys.reserve(static_cast<std::size_t>(key_count));
    if (key_count > 0 && route_offsets[0] != 0) {
      return grm_fail_msg(handle, "route offsets must start at zero");
    }
    if (key_count > 0 && route_offsets[key_count] != value_count) {
      return grm_fail_msg(handle, "route offsets must end at value_count");
    }
    for (uint64_t k = 0; k < key_count; ++k) {
      const auto start = route_offsets[k];
      const auto end = route_offsets[k + 1];
      if (end < start || end > value_count) {
        return grm_fail_msg(handle, "invalid route key offsets");
      }
      std::vector<float> key;
      key.reserve(static_cast<std::size_t>(end - start));
      for (uint64_t i = start; i < end; ++i) {
        key.push_back(route_values[i]);
      }
      keys.push_back(std::move(key));
    }
    auto lexical = split_lexical_keys(lexical_keys);
    std::unique_lock<std::shared_mutex> lock(handle->router_mutex);
    auto* node = handle->store->get(node_id);
    if (node == nullptr) {
      return grm_fail_msg(handle, "unknown GRM node id");
    }
    node->route_keys = keys;
    node->route_key = keys.empty() ? std::vector<float>() : keys.front();
    node->lexical_keys = lexical;
    handle->store->mark_dirty(node_id, false, true);
    handle->router.upsert_multi(node_id, std::move(keys), std::move(lexical));
    sync_router_node_state(handle, node_id);
    handle->router_needs_prepare = true;
    return 0;
  } catch (const std::exception& exc) {
    return grm_fail(handle, exc);
  }
}

int grm_store_clear_route(grm_store_handle* handle,
                          uint64_t node_id) {
  try {
    if (handle == nullptr || handle->store == nullptr) {
      return grm_fail_msg(handle, "invalid clear_route arguments");
    }
    std::unique_lock<std::shared_mutex> lock(handle->router_mutex);
    handle->store->clear_route(node_id);
    handle->router.erase(node_id);
    handle->router_needs_prepare = true;
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
    const auto route_once = [&]() {
      const auto routed = handle->router.route(
          q, split_lexical_keys(lexical_keys), static_cast<std::size_t>(topk));
      const uint64_t n = std::min<uint64_t>(
          static_cast<uint64_t>(routed.size()), out_cap);
      for (uint64_t i = 0; i < n; ++i) {
        out_node_ids[i] = routed[static_cast<std::size_t>(i)];
      }
      *out_count = n;
      return 0;
    };
    const bool shared_read_ok =
        handle->store->dialect().payload_kind != grm::PayloadKind::GQA;
    if (shared_read_ok) {
      std::shared_lock<std::shared_mutex> lock(handle->router_mutex);
      if (!handle->router_needs_prepare) {
        return route_once();
      }
    }
    std::unique_lock<std::shared_mutex> lock(handle->router_mutex);
    if (handle->router_needs_prepare) {
      prepare_router_for_store_dialect(handle);
    }
    return route_once();
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
    const auto route_once = [&]() {
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
    };
    const bool shared_read_ok =
        handle->store->dialect().payload_kind != grm::PayloadKind::GQA;
    if (shared_read_ok) {
      std::shared_lock<std::shared_mutex> lock(handle->router_mutex);
      if (!handle->router_needs_prepare) {
        return route_once();
      }
    }
    std::unique_lock<std::shared_mutex> lock(handle->router_mutex);
    if (handle->router_needs_prepare) {
      prepare_router_for_store_dialect(handle);
    }
    return route_once();
  } catch (const std::exception& exc) {
    return grm_fail(handle, exc);
  }
}

int grm_store_route_gqa(grm_store_handle* handle,
                        const float* query,
                        uint64_t query_heads,
                        uint64_t query_tokens,
                        uint64_t head_dim,
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
      return grm_fail_msg(handle, "invalid route_gqa arguments");
    }
    if (handle->store->dialect().payload_kind != grm::PayloadKind::GQA) {
      return grm_fail_msg(handle, "route_gqa requires a GQA dialect store");
    }
    if (head_dim != static_cast<uint64_t>(handle->store->dialect().head_dim)) {
      return grm_fail_msg(handle, "route_gqa head_dim does not match dialect");
    }
    if (query_heads != 0 &&
        query_tokens > std::numeric_limits<uint64_t>::max() / query_heads) {
      return grm_fail_msg(handle, "route_gqa query size overflow");
    }
    const auto head_tokens = query_heads * query_tokens;
    if (head_tokens != 0 &&
        head_dim > std::numeric_limits<uint64_t>::max() / head_tokens) {
      return grm_fail_msg(handle, "route_gqa query size overflow");
    }
    const auto query_len = head_tokens * head_dim;
    if (query == nullptr && query_len > 0) {
      return grm_fail_msg(handle, "null GQA query with nonzero length");
    }
    if (out_node_ids == nullptr && out_cap > 0) {
      return grm_fail_msg(handle, "null output buffer with nonzero capacity");
    }
    std::vector<float> q;
    q.reserve(static_cast<std::size_t>(query_len));
    for (uint64_t i = 0; i < query_len; ++i) {
      q.push_back(query[i]);
    }
    const auto route_once = [&]() {
      const auto routed = handle->router.route_gqa_raw(
          q, query_heads, query_tokens, head_dim,
          static_cast<uint64_t>(handle->store->dialect().num_kv_heads),
          split_lexical_keys(lexical_keys), static_cast<std::size_t>(topk),
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
    };
    {
      std::shared_lock<std::shared_mutex> lock(handle->router_mutex);
      if (!handle->router_needs_prepare) {
        return route_once();
      }
    }
    std::unique_lock<std::shared_mutex> lock(handle->router_mutex);
    if (handle->router_needs_prepare) {
      prepare_router_for_store_dialect(handle);
    }
    return route_once();
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
    rebuild_router_from_store(handle);
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

int grm_store_dirty_nodes(grm_store_handle* handle,
                          uint64_t* out_node_ids,
                          uint64_t out_cap,
                          uint64_t* out_count) {
  try {
    if (handle == nullptr || handle->store == nullptr ||
        out_count == nullptr) {
      return grm_fail_msg(handle, "invalid dirty_nodes arguments");
    }
    if (out_node_ids == nullptr && out_cap > 0) {
      return grm_fail_msg(
          handle, "null dirty_nodes buffer with nonzero capacity");
    }
    const auto ids = handle->store->dirty_queue().node_ids();
    *out_count = static_cast<uint64_t>(ids.size());
    const auto n = std::min<uint64_t>(
        static_cast<uint64_t>(ids.size()), out_cap);
    for (uint64_t i = 0; i < n; ++i) {
      out_node_ids[i] = ids[static_cast<std::size_t>(i)];
    }
    return 0;
  } catch (const std::exception& exc) {
    return grm_fail(handle, exc);
  }
}

int grm_store_dirty_plan(grm_store_handle* handle,
                         grm_dirty_node_c* out_nodes,
                         uint64_t out_cap,
                         uint64_t* out_count) {
  try {
    if (handle == nullptr || handle->store == nullptr ||
        out_count == nullptr) {
      return grm_fail_msg(handle, "invalid dirty_plan arguments");
    }
    if (out_nodes == nullptr && out_cap > 0) {
      return grm_fail_msg(
          handle, "null dirty_plan buffer with nonzero capacity");
    }
    const auto plan = handle->store->dirty_plan();
    *out_count = static_cast<uint64_t>(plan.size());
    const auto n = std::min<uint64_t>(
        static_cast<uint64_t>(plan.size()), out_cap);
    for (uint64_t i = 0; i < n; ++i) {
      const auto& item = plan[static_cast<std::size_t>(i)];
      out_nodes[i].node_id = item.node_id;
      out_nodes[i].payload_dirty = item.payload_dirty ? 1 : 0;
      out_nodes[i].metadata_dirty = item.metadata_dirty ? 1 : 0;
      out_nodes[i].payload_bytes = item.payload_bytes;
      out_nodes[i].durability_priority = item.durability_priority;
    }
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
    std::shared_lock<std::shared_mutex> lock(handle->router_mutex);
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
  thread_local std::string last_error_copy;
  std::lock_guard<std::mutex> lock(handle->error_mutex);
  last_error_copy = handle->last_error;
  return last_error_copy.c_str();
}

}  // extern "C"
