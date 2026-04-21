#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT" || { echo "Cannot cd to project root: $PROJECT_ROOT"; exit 1; }

NB_AI="$PROJECT_ROOT/neuro_brainwave_ai_project"
cd "$NB_AI" || { echo "Cannot cd to $NB_AI"; exit 1; }

# Dynamic Neuro-Brainwave Data Generator Launcher
# ===============================================
# 
# This script launches the interactive dynamic data generator with sliders
# for real-time parameter control and data generation.

echo "Dynamic Neuro-Brainwave Data Generator"
echo "========================================"
echo ""

# Check if we're in the right directory
if [ ! -f "dynamic_data_generator.py" ]; then
    echo "Error: dynamic_data_generator.py not found!"
    echo "Please run this script from the neuro_brainwave_ai_project directory."
    exit 1
fi

# Check for required dependencies
echo "Checking dependencies..."
if ! python3 -c "import streamlit, plotly, pandas, numpy" 2>/dev/null; then
    echo "Warning: Some dependencies may be missing."
    echo "Make sure you have installed: streamlit, plotly, pandas, numpy"
    echo ""
fi

echo "Dynamic Generator Features:"
echo "Interactive Parameter Sliders"
echo "Real-time Data Generation"
echo "Live EEG Signal Visualization"
echo "Frequency Band Analysis"
echo "Behavioral Pattern Analysis"
echo "Dataset Export & Save"
echo ""
echo "Application will be available at:"
echo "Local:   http://localhost:8503"
echo "Network: http://$(hostname -I | awk '{print $1}'):8503"
echo ""
echo "Press Ctrl+C to stop the application"
echo ""

# Launch the dynamic generator
python3 -m streamlit run dynamic_data_generator.py \
    --server.port 8503 \
    --server.headless false \
    --server.enableCORS false \
    --server.enableXsrfProtection false
