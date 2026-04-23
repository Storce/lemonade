#include "lemon/metrics_registry.h"
#include <httplib.h>
#include <sstream>
#include <chrono>
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
}

void MetricsRegistry::stop() {
    running_ = false;
    if (background_thread_.joinable()) {
        background_thread_.join();
    }
}

void MetricsRegistry::inc_active_requests() {
    active_requests_++;
}

void MetricsRegistry::dec_active_requests() {
    active_requests_--;
}

void MetricsRegistry::record_inference(int input_tokens, int output_tokens, int prompt_tokens, double time_to_first_token, double tokens_per_second) {
    // Increment counters
    input_tokens_total_.fetch_add(input_tokens > 0 ? input_tokens : 0);
    output_tokens_total_.fetch_add(output_tokens > 0 ? output_tokens : 0);
    prompt_tokens_total_.fetch_add(prompt_tokens > 0 ? prompt_tokens : 0);
    total_sessions_++;

    // Update gauges under lock
    std::lock_guard<std::mutex> lock(last_metrics_mutex_);
    last_input_tokens_ = input_tokens;
    last_output_tokens_ = output_tokens;
    last_prompt_tokens_ = prompt_tokens;
    last_cached_tokens_ = (prompt_tokens > input_tokens) ? (prompt_tokens - input_tokens) : 0;
    last_time_to_first_token_ = time_to_first_token;
    last_tokens_per_second_ = tokens_per_second;
}

void MetricsRegistry::polling_loop() {
    while (running_) {
        // Sleep for 5 seconds, checking running_ periodically
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
                    
                    // Convert http://127.0.0.1:8003/v1 into http://127.0.0.1:8003
                    size_t v1_pos = url.rfind("/v1");
                    if (v1_pos != std::string::npos && v1_pos == url.length() - 3) {
                        url.erase(v1_pos);
                    }
                    
                    // Perform sync scrape
                    httplib::Client cli(url);
                    cli.set_connection_timeout(0, 500000); // 500ms
                    cli.set_read_timeout(0, 500000); // 500ms
                    
                    if (auto res = cli.Get("/metrics")) {
                        if (res->status == 200) {
                            std::istringstream stream(res->body);
                            std::string line;
                            while (std::getline(stream, line)) {
                                if (line.empty()) continue;
                                if (line.back() == '\r') line.pop_back(); // Handle \r\n
                                
                                if (line[0] == '#') {
                                    // It's a comment/help/type. e.g. # HELP llama_prompt_tokens ...
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
                                    // It's a metric. Find first '{' or space.
                                    size_t brace = line.find('{');
                                    size_t space = line.find(' ');
                                    
                                    if (brace != std::string::npos && brace < space) {
                                        // Contains labels
                                        std::string name = line.substr(0, brace);
                                        std::string rest = line.substr(brace + 1);
                                        new_backend_metrics += "lemonade_backend_" + name + "{model_name=\"" + m_name + "\",recipe=\"" + recipe + "\"," + rest + "\n";
                                    } else if (space != std::string::npos) {
                                        // No labels
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
        } catch (...) {
            // Ignore any fetch errors
        }

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
    out << "lemonade_input_tokens_total " << input_tokens_total_.load() << "\n";
    
    out << "# HELP lemonade_output_tokens_total Total cumulative output tokens generated\n";
    out << "# TYPE lemonade_output_tokens_total counter\n";
    out << "lemonade_output_tokens_total " << output_tokens_total_.load() << "\n";
    
    out << "# HELP lemonade_prompt_tokens_total Total cumulative prompt tokens\n";
    out << "# TYPE lemonade_prompt_tokens_total counter\n";
    out << "lemonade_prompt_tokens_total " << prompt_tokens_total_.load() << "\n";
    
    out << "# HELP lemonade_total_sessions Total sessions handled\n";
    out << "# TYPE lemonade_total_sessions counter\n";
    out << "lemonade_total_sessions " << total_sessions_.load() << "\n";
    
    out << "# HELP lemonade_active_requests Precise count of currently executing inference requests\n";
    out << "# TYPE lemonade_active_requests gauge\n";
    out << "lemonade_active_requests " << active_requests_.load() << "\n";
    
    // Copy last_ metrics
    int l_input, l_output, l_prompt, l_cached;
    double l_ttft, l_tps;
    {
        std::lock_guard<std::mutex> lock(last_metrics_mutex_);
        l_input = last_input_tokens_;
        l_output = last_output_tokens_;
        l_prompt = last_prompt_tokens_;
        l_cached = last_cached_tokens_;
        l_ttft = last_time_to_first_token_;
        l_tps = last_tokens_per_second_;
    }

    out << "# HELP lemonade_input_tokens_last Number of input tokens from last request\n";
    out << "# TYPE lemonade_input_tokens_last gauge\n";
    out << "lemonade_input_tokens_last " << l_input << "\n";
    
    out << "# HELP lemonade_output_tokens_last Number of output tokens from last request\n";
    out << "# TYPE lemonade_output_tokens_last gauge\n";
    out << "lemonade_output_tokens_last " << l_output << "\n";
    
    out << "# HELP lemonade_prompt_tokens_last Number of prompt tokens from last request\n";
    out << "# TYPE lemonade_prompt_tokens_last gauge\n";
    out << "lemonade_prompt_tokens_last " << l_prompt << "\n";
    
    out << "# HELP lemonade_cached_tokens_last Cached tokens from last request\n";
    out << "# TYPE lemonade_cached_tokens_last gauge\n";
    out << "lemonade_cached_tokens_last " << l_cached << "\n";
    
    double hit_rate = (l_prompt > 0) ? (double)l_cached / (double)l_prompt : 0.0;
    out << "# HELP lemonade_cache_hit_rate Cache hit rate for last request\n";
    out << "# TYPE lemonade_cache_hit_rate gauge\n";
    out << "lemonade_cache_hit_rate " << hit_rate << "\n";
    
    out << "# HELP lemonade_time_to_first_token_seconds Time to first token in seconds\n";
    out << "# TYPE lemonade_time_to_first_token_seconds gauge\n";
    out << "lemonade_time_to_first_token_seconds " << l_ttft << "\n";
    
    out << "# HELP lemonade_tokens_per_second Tokens generated per second\n";
    out << "# TYPE lemonade_tokens_per_second gauge\n";
    out << "lemonade_tokens_per_second " << l_tps << "\n";
    
    if (router_) {
        json models;
        try {
            models = router_->get_all_loaded_models();
        } catch (...) {}
        
        out << "# HELP lemonade_models_loaded Number of models currently loaded\n";
        out << "# TYPE lemonade_models_loaded gauge\n";
        out << "lemonade_models_loaded " << models.size() << "\n";
        
        if (!models.empty()) {
            out << "# HELP lemonade_model_info Information about loaded models\n";
            out << "# TYPE lemonade_model_info gauge\n";
            for (const auto& model : models) {
                if (model.contains("model_name") && model.contains("type") && model.contains("device")) {
                    out << "lemonade_model_info{model_name=\"" << model["model_name"].get<std::string>() 
                        << "\",type=\"" << model["type"].get<std::string>()
                        << "\",device=\"" << model["device"].get<std::string>() << "\"} 1\n";
                }
            }
        }
    }
    
    std::string backend_out;
    {
        std::lock_guard<std::mutex> lock(backend_metrics_mutex_);
        backend_out = cached_backend_metrics_;
    }
    
    if (!backend_out.empty()) {
        out << backend_out;
    }
    
    return out.str();
}

} // namespace lemon
