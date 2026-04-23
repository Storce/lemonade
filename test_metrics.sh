#!/usr/bin/env bash
# test_metrics.sh — Lemonade /metrics endpoint diagnostic tool
#
# Usage:
#   ./test_metrics.sh [HOST] [PORT] [INTERVAL_SECONDS] [MODEL]
#
# Defaults: HOST=127.0.0.1, PORT=13305, INTERVAL=5, MODEL=(auto-detect)
#
# What it does:
#  1. Checks that /metrics is reachable and prints the full block
#  2. Polls /metrics every INTERVAL seconds, printing delta values
#  3. Fires one non-streaming chat completion and shows before/after counters
#  4. Shows summary of known bugs / likely causes of zero values

HOST="${1:-127.0.0.1}"
PORT="${2:-13305}"
INTERVAL="${3:-5}"
MODEL="${4:-}"
BASE="http://${HOST}:${PORT}"
POLLS=6      # number of timed polls after the chat injection
DIVIDER="────────────────────────────────────────────────────"

color_green() { printf '\033[32m%s\033[0m\n' "$*"; }
color_red()   { printf '\033[31m%s\033[0m\n' "$*"; }
color_cyan()  { printf '\033[36m%s\033[0m\n' "$*"; }
color_yellow(){ printf '\033[33m%s\033[0m\n' "$*"; }

die() { color_red "ERROR: $*"; exit 1; }

# ── helpers ──────────────────────────────────────────────────────────────────

fetch_metrics() {
    curl -sf "${BASE}/metrics" 2>/dev/null
}

fetch_stats() {
    curl -sf "${BASE}/api/v1/stats" 2>/dev/null
}

extract() {
    # extract_metric <metric_name> <prometheus_text>
    local name="$1"; local text="$2"
    echo "$text" | grep -E "^${name}[{ ]" | awk '{print $NF}' | head -1
}

# ── 0. Server reachability ───────────────────────────────────────────────────
echo
color_cyan "=== Lemonade /metrics Diagnostic ==="
echo "Target: ${BASE}"
echo

printf "Checking /health ... "
if ! curl -sf "${BASE}/api/v1/health" >/dev/null; then
    die "Server not reachable at ${BASE}. Is lemond running?"
fi
color_green "OK"

# ── 1. Detect model ──────────────────────────────────────────────────────────
if [[ -z "$MODEL" ]]; then
    printf "Detecting loaded model ... "
    MODEL=$(curl -sf "${BASE}/api/v1/health" | python3 -c "import sys,json; h=json.load(sys.stdin); print(h.get('model_loaded') or '')" 2>/dev/null)
    if [[ -z "$MODEL" ]]; then
        color_yellow "no model loaded (metrics will show 0 for active_requests/tps/tokens)"
        MODEL=""
    else
        color_green "$MODEL"
    fi
fi

# ── 2. Raw /metrics dump ─────────────────────────────────────────────────────
echo
color_cyan "=== Raw /metrics endpoint ==="
RAW=$(fetch_metrics)
if [[ -z "$RAW" ]]; then
    color_red "FAIL — /metrics returned empty/non-200. Check server logs."
    exit 1
fi
echo "$RAW"
echo

# ── 3. Check /stats shows data (baseline) ───────────────────────────────────
echo "$DIVIDER"
color_cyan "=== /api/v1/stats comparison ==="
STATS=$(fetch_stats)
if echo "$STATS" | grep -q '"error"'; then
    color_yellow "/stats returned error (no model loaded): $STATS"
else
    echo "$STATS" | python3 -m json.tool 2>/dev/null || echo "$STATS"
fi
echo

# ── 4. Check counters pre-request ────────────────────────────────────────────
echo "$DIVIDER"
color_cyan "=== Counter baseline (before sending a request) ==="
METRICS_BEFORE=$(fetch_metrics)
for m in lemonade_total_sessions lemonade_input_tokens_total lemonade_output_tokens_total lemonade_active_requests; do
    val=$(extract "$m" "$METRICS_BEFORE")
    printf "  %-40s %s\n" "$m" "${val:-<missing>}"
done
echo

# ── 5. Inject one NON-STREAMING chat completion ───────────────────────────────
if [[ -n "$MODEL" ]]; then
    echo "$DIVIDER"
    color_cyan "=== Firing non-streaming chat completion (model: $MODEL) ==="
    PAYLOAD=$(printf '{"model":"%s","messages":[{"role":"user","content":"Say exactly the word HELLO only."}],"max_tokens":10,"stream":false}' "$MODEL")
    RESP=$(curl -sf -X POST "${BASE}/api/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "$PAYLOAD" 2>&1)
    if echo "$RESP" | grep -q '"choices"'; then
        CONTENT=$(echo "$RESP" | python3 -c "import sys,json; r=json.load(sys.stdin); print(r['choices'][0]['message']['content'])" 2>/dev/null)
        color_green "Response content: $CONTENT"
        # Check if timings or usage present
        HAS_TIMINGS=$(echo "$RESP" | python3 -c "import sys,json; r=json.load(sys.stdin); print('YES' if 'timings' in r else 'NO')" 2>/dev/null)
        HAS_USAGE=$(echo "$RESP" | python3 -c "import sys,json; r=json.load(sys.stdin); print('YES' if 'usage' in r else 'NO')" 2>/dev/null)
        echo "  Response contains 'timings' field: $HAS_TIMINGS"
        echo "  Response contains 'usage' field:   $HAS_USAGE"
        if [[ "$HAS_TIMINGS" == "NO" && "$HAS_USAGE" == "NO" ]]; then
            color_red "  ⚠ NEITHER 'timings' NOR 'usage' found in response!"
            color_red "    → record_inference() will NOT be called → counters stay at 0"
            color_yellow "  Full response:"
            echo "$RESP" | python3 -m json.tool 2>/dev/null || echo "$RESP"
        fi
    else
        color_red "Chat completion failed: $RESP"
    fi
    echo

    # ── 6. Counter snapshot after request ──────────────────────────────────
    color_cyan "=== Counter snapshot after request ==="
    METRICS_AFTER=$(fetch_metrics)
    for m in lemonade_total_sessions lemonade_input_tokens_total lemonade_output_tokens_total lemonade_tokens_per_second lemonade_time_to_first_token_seconds; do
        before=$(extract "$m" "$METRICS_BEFORE")
        after=$(extract "$m" "$METRICS_AFTER")
        delta=""
        if [[ -n "$before" && -n "$after" ]]; then
            delta=$(python3 -c "print(float('$after') - float('$before'))" 2>/dev/null)
        fi
        if [[ "${before:-0}" == "${after:-0}" ]]; then
            color_red "$(printf "  %-44s %s → %s  (delta: %s)" "$m" "${before:-?}" "${after:-?}" "${delta:-0}")"
        else
            color_green "$(printf "  %-44s %s → %s  (delta: %s)" "$m" "${before:-?}" "${after:-?}" "${delta:--}")"
        fi
    done
    echo
else
    color_yellow "No model loaded — skipping chat completion test."
    color_yellow "Load a model with: curl -X POST ${BASE}/api/v1/load -d '{\"model\":\"YOUR_MODEL\"}'"
    echo
fi

# ── 7. Streaming note ────────────────────────────────────────────────────────
echo "$DIVIDER"
color_yellow "⚠ KNOWN ISSUE: Streaming requests do NOT call record_inference()."
color_yellow "  The streaming path calls router_->chat_completion_stream() which"
color_yellow "  returns nothing — there's no response body to parse for timings/usage."
color_yellow "  Only non-streaming (stream:false) requests update the metrics counters."
echo

# ── 8. Periodic polling ───────────────────────────────────────────────────────
echo "$DIVIDER"
color_cyan "=== Polling /metrics every ${INTERVAL}s for $((POLLS * INTERVAL))s ==="
color_cyan "    (Send some chat requests in another terminal to see counters move)"
echo
printf "  %-8s %-16s %-16s %-16s %-16s\n" "Time" "sessions" "input_tok" "output_tok" "tps"
echo "  ────────────────────────────────────────────────────────────────────────"

for i in $(seq 1 $POLLS); do
    sleep "$INTERVAL"
    M=$(fetch_metrics)
    ts=$(date '+%H:%M:%S')
    sessions=$(extract "lemonade_total_sessions" "$M")
    inp=$(extract "lemonade_input_tokens_total" "$M")
    out=$(extract "lemonade_output_tokens_total" "$M")
    tps=$(extract "lemonade_tokens_per_second" "$M")
    printf "  %-8s %-16s %-16s %-16s %-16s\n" "$ts" "${sessions:-0}" "${inp:-0}" "${out:-0}" "${tps:-0}"
done

echo
color_cyan "=== Test complete ==="
echo "Run 'curl ${BASE}/metrics' at any time for raw output."
