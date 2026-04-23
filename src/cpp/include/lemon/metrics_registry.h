#pragma once

#include <string>
#include <atomic>
#include <mutex>
#include <vector>
#include <thread>
#include <nlohmann/json.hpp>
#include "lemon/router.h"

namespace lemon {

struct ModelMetrics {
    std::atomic<uint64_t> input_tokens_total{0};
    std::atomic<uint64_t> output_tokens_total{0};
    std::atomic<uint64_t> prompt_tokens_total{0};
    std::atomic<uint64_t> total_sessions{0};
    
    std::mutex last_metrics_mutex;
    int last_input_tokens{0};
    int last_output_tokens{0};
    int last_prompt_tokens{0};
    int last_cached_tokens{0};
    double last_time_to_first_token{0.0};
    double last_tokens_per_second{0.0};
};

class MetricsRegistry {
public:
    MetricsRegistry(Router* router);
    ~MetricsRegistry();

    // Start background poller and logger
    void start();
    
    // Stop background poller
    void stop();

    // Increment/decrement active requests (global and per-model)
    void inc_active_requests(const std::string& model_name = "");
    void dec_active_requests(const std::string& model_name = "");

    // Record inference values (called right after processing)
    void record_inference(const std::string& model_name, int input_tokens, int output_tokens, int prompt_tokens, double time_to_first_token, double tokens_per_second);

    // Get all metrics in Prometheus text format
    std::string format_prometheus();

private:
    void polling_loop();
    void log_loop();
    std::shared_ptr<ModelMetrics> get_or_create_model_metrics(const std::string& model_name);
    
    Router* router_;
    std::atomic<bool> running_{false};
    std::thread background_thread_;
    std::thread logging_thread_;

    // Native tracking fields
    std::atomic<uint64_t> global_input_tokens_total_{0};
    std::atomic<uint64_t> global_output_tokens_total_{0};
    std::atomic<uint64_t> global_prompt_tokens_total_{0};
    std::atomic<uint64_t> global_total_sessions_{0};
    std::atomic<int64_t> global_active_requests_{0};

    // Per-model metrics
    std::mutex models_mutex_;
    std::unordered_map<std::string, std::shared_ptr<ModelMetrics>> model_metrics_;

    // Per-model active requests
    std::mutex active_req_mutex_;
    std::unordered_map<std::string, std::atomic<int64_t>*> model_active_requests_;

    // Cached backend metrics text
    std::mutex backend_metrics_mutex_;
    std::string cached_backend_metrics_;
};

// RAII helper to automatically decrement active requests when a handler finishes
// Now move-aware to support async/streaming handlers
class ActiveRequestTracker {
public:
    ActiveRequestTracker(MetricsRegistry* registry, const std::string& model_name = "") 
        : registry_(registry), model_name_(model_name) {
        if (registry_) registry_->inc_active_requests(model_name_);
    }
    
    ~ActiveRequestTracker() {
        if (registry_) registry_->dec_active_requests(model_name_);
    }

    // Move-only for safety in lambdas
    ActiveRequestTracker(const ActiveRequestTracker&) = delete;
    ActiveRequestTracker& operator=(const ActiveRequestTracker&) = delete;
    
    ActiveRequestTracker(ActiveRequestTracker&& other) noexcept 
        : registry_(other.registry_), model_name_(std::move(other.model_name_)) {
        other.registry_ = nullptr;
    }
    
    ActiveRequestTracker& operator=(ActiveRequestTracker&& other) noexcept {
        if (this != &other) {
            if (registry_) registry_->dec_active_requests(model_name_);
            registry_ = other.registry_;
            model_name_ = std::move(other.model_name_);
            other.registry_ = nullptr;
        }
        return *this;
    }

private:
    MetricsRegistry* registry_;
    std::string model_name_;
};

} // namespace lemon
