#!/bin/bash
echo "[*] Starting chained Rs-ai runs..."
for i in {1..5}; do
    echo "[*] Cycle $i"
    python3 test_run.py
    sleep 1
done
echo "[✓] All runs complete."
