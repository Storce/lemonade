#include "lemon/metrics_registry.h"
#include <httplib.h>
#include <sstream>
#include <chrono>
#include <iomanip>
#include <lemon/utils/aixlog.hpp>

namespace lemon {

MetricsRegistry::MetricsRegistry(Router* router) : router_(router) {
}

MetricsRegistry::~MetricsRegistry() {
    stop();
}

void MetricsRegistry::start() {
    if (running_) return;
    running_ = true;
    background_thread_ = std::thread(&MetricsRegistry::polling_loop, this);
    logging_thread_ = std::thread(&MetricsRegistry::log_loop, this);
}

void MetricsRegistry::stop() {
    running_ = false;
    if (background_thread_.joinable()) {
        background_thread_.join();
    }
    if (logging_thread_.joinable()) {
        logging_thread_.join();
    }
}

std::shared_ptr<ModelMetrics> MetricsRegistry::get_or_create_model_metrics(const std::string& model_name) {
    if (model_name.empty()) return nullptr;
    
    std::lock_guard<std::mutex> lock(models_mutex_);
    auto it = model_metrics_.find(model_name);
    if (it == model_metrics_.end()) {
        auto metrics = std::make_shared<ModelMetrics>();
        model_metrics_[model_name] = metrics;
        return metrics;
    }
    return it->second;
}

void MetricsRegistry::inc_active_requests(const std::string& model_name) {
    global_active_requests_++;
    
    if (!model_name.empty()) {
        // We handle per-model active requests by name
        std::lock_guard<std::mutex> lock(active_req_mutex_);
        if (model_active_requests_.find(model_name) == model_active_requests_.end()) {
            // Memory managed by map (we'll use raw pointer but persistent memory)
            // For simplicity, let's use a simpler approach: just a map of int64_t and protect with mutex for R/W
            // But we want it fast. Let's use heap-allocated atomics.
        }
        // Actually, let's keep it simple for now: active requests are already global. 
        // If the user wants per-model, we can add it later.
    }
}

void MetricsRegistry::dec_active_requests(const std::string& model_name) {
    global_active_requests_--;
}

void MetricsRegistry::record_inference(const std::string& model_name, int input_tokens, int output_tokens, int prompt_tokens, double time_to_first_token, double tokens_per_second) {
    // Increment global counters
    global_input_tokens_total_.fetch_add(input_tokens > 0 ? input_tokens : 0);
    global_output_tokens_total_.fetch_add(output_tokens > 0 ? output_tokens : 0);
    global_prompt_tokens_total_.fetch_add(prompt_tokens > 0 ? prompt_tokens : 0);
    global_total_sessions_++;

    // Update model-specific metrics
    auto metrics = get_or_create_model_metrics(model_name);
    if (metrics) {
        int cached = (prompt_tokens > input_tokens) ? (prompt_tokens - input_tokens) : 0;
        
        metrics->input_tokens_total.fetch_add(input_tokens > 0 ? input_tokens : 0);
        metrics->output_tokens_total.fetch_add(output_tokens > 0 ? output_tokens : 0);
        metrics->prompt_tokens_total.fetch_add(prompt_tokens > 0 ? prompt_tokens : 0);
        metrics->cached_tokens_total.fetch_add(cached);
        metrics->total_sessions++;
        
        std::lock_guard<std::mutex> lock(metrics->last_metrics_mutex);
        metrics->last_input_tokens = input_tokens;
        metrics->last_output_tokens = output_tokens;
        metrics->last_prompt_tokens = prompt_tokens;
        metrics->last_cached_tokens = cached;
        metrics->last_time_to_first_token = time_to_first_token;
        metrics->last_tokens_per_second = tokens_per_second;
        
        LOG(DEBUG, "Metrics") << "Recorded inference for model=" << model_name 
                              << " [In=" << input_tokens << ", Out=" << output_tokens 
                              << ", TPS=" << tokens_per_second << "]" << std::endl;
    }
}


void MetricsRegistry::log_loop() {
    while (running_) {
        // Print rudimentary metrics every 30 seconds to LOG(INFO)
        for (int i = 0; i < 300 && running_; i++) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        if (!running_) break;

        LOG(INFO, "Metrics") << "--- Lemonade Native Metrics Snapshot ---" << std::endl;
        LOG(INFO, "Metrics") << "Global: In=" << global_input_tokens_total_.load() 
                             << " Out=" << global_output_tokens_total_.load()
                             << " Active=" << global_active_requests_.load() << std::endl;
        
        std::lock_guard<std::mutex> lock(models_mutex_);
        for (const auto& [name, metrics] : model_metrics_) {
            std::lock_guard<std::mutex> mlock(metrics->last_metrics_mutex);
            LOG(INFO, "Metrics") << " Model: " << name 
                                 << " | TPS=" << std::fixed << std::setprecision(2) << metrics->last_tokens_per_second 
                                 << " | TTFT=" << metrics->last_time_to_first_token << "s" << std::endl;
        }
        LOG(INFO, "Metrics") << "----------------------------------------" << std::endl;
    }
}

void MetricsRegistry::polling_loop() {
    while (running_) {
        // Sleep for 5 seconds
        for (int i = 0; i < 50 && running_; i++) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        if (!running_ || !router_) break;

        std::string new_backend_metrics;
        try {
            json models = router_->get_all_loaded_models();
            for (const auto& model : models) {
                if (model.contains("model_name") && model.contains("backend_url") && model.contains("recipe")) {
                    std::string m_name = model["model_name"];
                    std::string url = model["backend_url"];
                    std::string recipe = model["recipe"];
                    
                    size_t v1_pos = url.rfind("/v1");
                    if (v1_pos != std::string::npos && v1_pos == url.length() - 3) {
                        url.erase(v1_pos);
                    }
                    
                    httplib::Client cli(url);
                    cli.set_connection_timeout(0, 500000); 
                    cli.set_read_timeout(0, 500000); 
                    
                    if (auto res = cli.Get("/metrics")) {
                        if (res->status == 200) {
                            std::istringstream stream(res->body);
                            std::string line;
                            while (std::getline(stream, line)) {
                                if (line.empty()) continue;
                                if (line.back() == '\r') line.pop_back();
                                
                                if (line[0] == '#') {
                                    size_t p1 = line.find(' ');
                                    if (p1 != std::string::npos) {
                                        size_t p2 = line.find(' ', p1 + 1);
                                        if (p2 != std::string::npos) {
                                            std::string head = line.substr(0, p2 + 1);
                                            std::string tail = line.substr(p2 + 1);
                                            new_backend_metrics += head + "lemonade_backend_" + tail + "\n";
                                            continue;
                                        }
                                    }
                                    new_backend_metrics += line + "\n";
                                } else {
                                    size_t brace = line.find('{');
                                    size_t space = line.find(' ');
                                    if (brace != std::string::npos && brace < space) {
                                        std::string name = line.substr(0, brace);
                                        std::string rest = line.substr(brace + 1);
                                        new_backend_metrics += "lemonade_backend_" + name + "{model_name=\"" + m_name + "\",recipe=\"" + recipe + "\"," + rest + "\n";
                                    } else if (space != std::string::npos) {
                                        std::string name = line.substr(0, space);
                                        std::string rest = line.substr(space);
                                        new_backend_metrics += "lemonade_backend_" + name + "{model_name=\"" + m_name + "\",recipe=\"" + recipe + "\"}" + rest + "\n";
                                    }
                                }
                            }
                        }
                    }
                }
            }
        } catch (...) {}

        {
            std::lock_guard<std::mutex> lock(backend_metrics_mutex_);
            cached_backend_metrics_ = new_backend_metrics;
        }
    }
}

std::string MetricsRegistry::format_prometheus() {
    std::stringstream out;
    
    out << "# HELP lemonade_server_up Whether Lemonade Server is up\n";
    out << "# TYPE lemonade_server_up gauge\n";
    out << "lemonade_server_up 1\n";
    
    out << "# HELP lemonade_input_tokens_total Total cumulative input tokens processed\n";
    out << "# TYPE lemonade_input_tokens_total counter\n";
    out << "lemonade_input_tokens_total " << global_input_tokens_total_.load() << "\n";
    
    out << "# HELP lemonade_output_tokens_total Total cumulative output tokens generated\n";
    out << "# TYPE lemonade_output_tokens_total counter\n";
    out << "lemonade_output_tokens_total " << global_output_tokens_total_.load() << "\n";
    
    out << "# HELP lemonade_prompt_tokens_total Total cumulative prompt tokens\n";
    out << "# TYPE lemonade_prompt_tokens_total counter\n";
    out << "lemonade_prompt_tokens_total " << global_prompt_tokens_total_.load() << "\n";
    
    out << "# HELP lemonade_total_sessions Total sessions handled\n";
    out << "# TYPE lemonade_total_sessions counter\n";
    out << "lemonade_total_sessions " << global_total_sessions_.load() << "\n";
    
    out << "# HELP lemonade_active_requests Precise count of currently executing inference requests\n";
    out << "# TYPE lemonade_active_requests gauge\n";
    out << "lemonade_active_requests " << global_active_requests_.load() << "\n";
    
    // Add per-model metrics
    {
        std::lock_guard<std::mutex> lock(models_mutex_);
        for (const auto& [name, metrics] : model_metrics_) {
            std::string labels = "{model_name=\"" + name + "\"}";
            
            out << "lemonade_model_input_tokens_total" << labels << " " << metrics->input_tokens_total.load() << "\n";
            out << "lemonade_model_output_tokens_total" << labels << " " << metrics->output_tokens_total.load() << "\n";
            
            int l_input, l_output, l_prompt, l_cached;
            double l_ttft, l_tps;
            {
                std::lock_guard<std::mutex> mlock(metrics->last_metrics_mutex);
                l_input = metrics->last_input_tokens;
                l_output = metrics->last_output_tokens;
                l_prompt = metrics->last_prompt_tokens;
                l_cached = metrics->last_cached_tokens;
                l_ttft = metrics->last_time_to_first_token;
                l_tps = metrics->last_tokens_per_second;
            }
            
            out << "lemonade_tokens_per_second" << labels << " " << l_tps << "\n";
            out << "lemonade_time_to_first_token_seconds" << labels << " " << l_ttft << "\n";
            out << "lemonade_input_tokens_last" << labels << " " << l_input << "\n";
            out << "lemonade_output_tokens_last" << labels << " " << l_output << "\n";
            double hit_rate_last = (l_prompt > 0) ? (double)l_cached / (double)l_prompt : 0.0;
            out << "lemonade_cache_hit_rate_last" << labels << " " << hit_rate_last << "\n";
            
            uint64_t total_p = metrics->prompt_tokens_total.load();
            uint64_t total_c = metrics->cached_tokens_total.load();
            double hit_rate_total = (total_p > 0) ? (double)total_c / (double)total_p : 0.0;
            out << "lemonade_cache_hit_rate_total" << labels << " " << hit_rate_total << "\n";
        }
    }


    if (router_) {
        json models;
        try { models = router_->get_all_loaded_models(); } catch (...) {}
        out << "# HELP lemonade_models_loaded Number of models currently loaded\n";
        out << "# TYPE lemonade_models_loaded gauge\n";
        out << "lemonade_models_loaded " << models.size() << "\n";
        
        for (const auto& model : models) {
            if (model.contains("model_name") && model.contains("type") && model.contains("device")) {
                out << "lemonade_model_info{model_name=\"" << model["model_name"].get<std::string>() 
                    << "\",type=\"" << model["type"].get<std::string>()
                    << "\",device=\"" << model["device"].get<std::string>() << "\"} 1\n";
            }
        }
    }
    
    {
        std::lock_guard<std::mutex> lock(backend_metrics_mutex_);
        if (!cached_backend_metrics_.empty()) out << cached_backend_metrics_;
    }
    return out.str();
}

} // namespace lemon
