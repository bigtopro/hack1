#!/usr/bin/env bash
# Script to run the analysis engine with proper environment setup

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Activate the virtual environment
source "$SCRIPT_DIR/analysis_env/bin/activate"

# Change to the script directory
cd "$SCRIPT_DIR"

# Run the analysis engine
echo "Starting Comment Analysis Engine..."
echo "Monitoring /Users/venuvamsi/Downloads for clustering and sentiment files..."
echo "Press Ctrl+C to stop the monitoring process."
echo ""

python analysis_engine.py