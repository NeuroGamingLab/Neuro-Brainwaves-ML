#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT" || { echo "Cannot cd to project root: $PROJECT_ROOT"; exit 1; }

NB_AI="$PROJECT_ROOT/neuro_brainwave_ai_project"
cd "$NB_AI" || { echo "Cannot cd to $NB_AI"; exit 1; }

# Neuro-Brainwave AI System Dashboard Launcher
# ============================================
# 
# This script launches the interactive dashboard for visualizing
# and analyzing the Neuro-Brainwave AI System results.

echo "Neuro-Brainwave AI System Dashboard"
echo "======================================"
echo ""

# Check if we're in the right directory
if [ ! -f "neuro_brainwave_dashboard.py" ]; then
    echo "Error: neuro_brainwave_dashboard.py not found!"
    echo "Please run this script from the neuro_brainwave_ai_project directory."
    exit 1
fi

# Check for dataset files
echo "Checking for dataset files..."
dataset_files=$(ls *.jsonl 2>/dev/null | wc -l)

if [ $dataset_files -eq 0 ]; then
    echo "Warning: No dataset files (*.jsonl) found!"
    echo "Please run the Neuro-Brainwave AI System first to generate data."
    echo ""
    echo "To generate data, run:"
    echo "python3 neuro_brainwave_ai_system.py --samples 1000"
    echo ""
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo "Found $dataset_files dataset file(s)"
    ls -lh *.jsonl | awk '{print "    " $9 " (" $5 ")"}'
fi

echo ""
echo "Launching Neuro-Brainwave AI Dashboard..."
echo ""
echo "Dashboard Features:"
echo "Dataset Overview & Statistics"
echo "Brain State Analysis"
echo "EEG Signal Visualization"
echo "Frequency Band Analysis"
echo "Behavioral Pattern Analysis"
echo "Interactive Data Explorer"
echo ""
echo "Dashboard will be available at:"
echo "Local:   http://localhost:8502"
echo "Network: http://$(hostname -I | awk '{print $1}'):8502"
echo ""
echo "Press Ctrl+C to stop the dashboard"
echo ""

# Launch Streamlit dashboard
python3 -m streamlit run neuro_brainwave_dashboard.py \
    --server.port 8502 \
    --server.headless false \
    --server.enableCORS false \
    --server.enableXsrfProtection false
