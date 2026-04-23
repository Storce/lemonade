#pragma once

#include <string>
#include <atomic>
#include <mutex>
#include <vector>
#include <thread>
#include <nlohmann/json.hpp>
#include "lemon/router.h"

namespace lemon {

class MetricsRegistry {
public:
    MetricsRegistry(Router* router);
    ~MetricsRegistry();

    // Start background poller
    void start();
    
    // Stop background poller
    void stop();

    // Increment/decrement active requests
    void inc_active_requests();
    void dec_active_requests();

    // Record inference values (called right after processing)
    void record_inference(int input_tokens, int output_tokens, int prompt_tokens, double time_to_first_token, double tokens_per_second);

    // Get all metrics in Prometheus text format
    std::string format_prometheus();

private:
    void polling_loop();
    void poll_backend(const std::string& model_name, const std::string& recipe, const std::string& backend_url);
    
    Router* router_;
    std::atomic<bool> running_{false};
    std::thread background_thread_;

    // Native tracking fields
    std::atomic<uint64_t> input_tokens_total_{0};
    std::atomic<uint64_t> output_tokens_total_{0};
    std::atomic<uint64_t> prompt_tokens_total_{0};
    std::atomic<uint64_t> total_sessions_{0};
    std::atomic<int64_t> active_requests_{0};

    // Note: These "last" metrics need to be updated atomically together or protected by a mutex
    std::mutex last_metrics_mutex_;
    int last_input_tokens_{0};
    int last_output_tokens_{0};
    int last_prompt_tokens_{0};
    int last_cached_tokens_{0};
    double last_time_to_first_token_{0.0};
    double last_tokens_per_second_{0.0};

    // Cached backend metrics text
    std::mutex backend_metrics_mutex_;
    std::string cached_backend_metrics_;
};

// RAII helper to automatically decrement active requests when a handler finishes
class ActiveRequestTracker {
public:
    ActiveRequestTracker(MetricsRegistry* registry) : registry_(registry) {
        if (registry_) registry_->inc_active_requests();
    }
    ~ActiveRequestTracker() {
        if (registry_) registry_->dec_active_requests();
    }
private:
    MetricsRegistry* registry_;
};

} // namespace lemon
