#!/bin/bash

echo "[⏱] Starting Rs-ai runtime..."
TIMESTAMP=$(date "+%Y-%m-%d %H:%M:%S")
LOG_FILE="rsai_log.txt"
PYTHON_EXEC="python3"

# Check Python environment
if ! command -v $PYTHON_EXEC &> /dev/null; then
    echo "[❌] Python3 not found! Install it before running this script."
    exit 1
fi

# Check test_run.py
if [ ! -f "test_run.py" ]; then
    echo "[❌] test_run.py not found in $(pwd)."
    exit 1
fi

# Log run header
echo "" >> "$LOG_FILE"
echo "==========================" >> "$LOG_FILE"
echo "[$TIMESTAMP] Starting Rs-ai run..." >> "$LOG_FILE"

# Run Rs-ai core
$PYTHON_EXEC test_run.py >> "$LOG_FILE" 2>&1

# Log footer
END_TIMESTAMP=$(date "+%Y-%m-%d %H:%M:%S")
echo "[$END_TIMESTAMP] Run complete." >> "$LOG_FILE"
echo "[✅] Rs-ai run completed and logged to $LOG_FILE."
