diff --git a/start_all.sh b/start_all.sh
--- a/start_all.sh
+++ b/start_all.sh
@@ -0,0 +1,64 @@
+#!/usr/bin/env bash
+set -euo pipefail
+
+PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
+LOG_DIR="$PROJECT_ROOT/data/logs"
+CHAIN_LOG="$LOG_DIR/chain_start.log"
+TRAIN_LOG="$LOG_DIR/train_start.log"
+
+mkdir -p "$LOG_DIR" "$PROJECT_ROOT/data/sponge" "$PROJECT_ROOT/data/checkpoints" "$PROJECT_ROOT/data/chain/ledger"
+
+log() { echo "[RS-AI] $*"; }
+
+have_cmd() { command -v "$1" >/dev/null 2>&1; }
+
+start_metrics() {
+  if have_cmd docker; then
+    if docker compose version >/dev/null 2>&1; then
+      log "Starting Prometheus/Grafana stack (docker compose up -d)"
+      (cd "$PROJECT_ROOT" && docker compose up -d)
+    elif have_cmd docker-compose; then
+      log "Starting Prometheus/Grafana stack (docker-compose up -d)"
+      (cd "$PROJECT_ROOT" && docker-compose up -d)
+    else
+      log "Docker found but compose plugin not available; skipping metrics stack."
+    fi
+  else
+    log "Docker not found; skipping metrics stack."
+  fi
+}
+
+start_chain() {
+  if ! have_cmd python3; then
+    log "python3 not found. Please install Python 3."
+    exit 1
+  fi
+  log "Launching federated chain... (logs: $CHAIN_LOG)"
+  nohup python3 -m tools.cli chain >>"$CHAIN_LOG" 2>&1 & echo $! > "$PROJECT_ROOT/chain.pid"
+}
+
+start_trainer() {
+  log "Launching trainer... (logs: $TRAIN_LOG)"
+  nohup python3 "$PROJECT_ROOT/train.py" >>"$TRAIN_LOG" 2>&1 & echo $! > "$PROJECT_ROOT/trainer.pid"
+}
+
+print_summary() {
+  log "Services launched. Useful endpoints:"
+  echo "- Node metrics: http://127.0.0.1:8080/metrics_prom"
+  echo "- Prometheus (if started): http://localhost:9090"
+  echo "- Grafana (if started):   http://localhost:3000 (admin/admin)"
+  echo ""
+  log "Recent chain log:"; tail -n 5 "$CHAIN_LOG" || true
+  echo ""
+  log "Recent trainer log:"; tail -n 5 "$TRAIN_LOG" || true
+  echo ""
+  log "PIDs: chain=$(cat "$PROJECT_ROOT/chain.pid" 2>/dev/null || echo N/A), trainer=$(cat "$PROJECT_ROOT/trainer.pid" 2>/dev/null || echo N/A)"
+  log "To stop: kill \$(cat chain.pid) \$(cat trainer.pid) (if still running)"
+}
+
+log "Starting RS-AI stack..."
+start_metrics || true
+start_chain
+start_trainer
+sleep 2
+print_summary
