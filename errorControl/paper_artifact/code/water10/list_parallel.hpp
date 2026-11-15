#ifndef TREECODE_LIST_PARALLEL_HPP
#define TREECODE_LIST_PARALLEL_HPP

#include <omp.h>
#include <cerrno>
#include <climits>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace treecode_list {

// Execution policy only: it must not change HNL layers, MAC, or list ordering.
struct Policy {
  int threads = 0;  // 0 inherits the surrounding OpenMP setting.
  omp_sched_t schedule = omp_sched_dynamic;
  const char* schedule_name = "DYNAMIC";
  int chunk = 16;
  bool profile = false;
  bool pooled = true;

  static int integer(const char* name, int fallback, int minimum) {
    const char* value = std::getenv(name);
    if (!value || !*value) return fallback;
    char* end = nullptr;
    errno = 0;
    const long parsed = std::strtol(value, &end, 10);
    if (errno || end == value || *end || parsed < minimum || parsed > INT_MAX)
      throw std::invalid_argument(std::string(name) + " must be an integer >= "
                                  + std::to_string(minimum));
    return static_cast<int>(parsed);
  }

  static Policy environment() {
    Policy policy;
    policy.threads = integer("TREECODE_LIST_THREADS", 0, 0);
    policy.chunk = integer("TREECODE_LIST_CHUNK", 16, 1);
    const int profile = integer("TREECODE_LIST_PROFILE", 0, 0);
    if (profile > 1) throw std::invalid_argument("TREECODE_LIST_PROFILE must be 0 or 1");
    policy.profile = profile != 0;
    if (const char* raw = std::getenv("TREECODE_LIST_STORAGE")) {
      const std::string value(raw);
      if (value == "VECTORS") policy.pooled = false;
      else if (!value.empty() && value != "POOLED")
        throw std::invalid_argument("TREECODE_LIST_STORAGE must be POOLED or VECTORS");
    }
    if (const char* raw = std::getenv("TREECODE_LIST_SCHEDULE")) {
      const std::string value(raw);
      if (value == "STATIC") {
        policy.schedule = omp_sched_static;
        policy.schedule_name = "STATIC";
      } else if (value == "GUIDED") {
        policy.schedule = omp_sched_guided;
        policy.schedule_name = "GUIDED";
      } else if (!value.empty() && value != "DYNAMIC") {
        throw std::invalid_argument("TREECODE_LIST_SCHEDULE must be STATIC, DYNAMIC or GUIDED");
      }
    }
    return policy;
  }

  int team_size() const { return threads ? threads : omp_get_max_threads(); }
};

// Restore the calling task's schedule; do not change evaluation or other loops.
class ScheduleScope {
  omp_sched_t previous_;
  int chunk_;
 public:
  explicit ScheduleScope(const Policy& policy) {
    omp_get_schedule(&previous_, &chunk_);
    omp_set_schedule(policy.schedule, policy.chunk);
  }
  ~ScheduleScope() { omp_set_schedule(previous_, chunk_); }
  ScheduleScope(const ScheduleScope&) = delete;
  ScheduleScope& operator=(const ScheduleScope&) = delete;
};

// No per-target shared writes or atomic timing counters. Profiling is optional
// and its measurements must not be used as production benchmark timings.
struct alignas(64) WorkerProfile {
  double hnl = 0.0, mac = 0.0, store = 0.0, loop = 0.0;
  std::size_t targets = 0;
};
}  // namespace treecode_list
#endif
