#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT" || { echo "Cannot cd to project root: $PROJECT_ROOT"; exit 1; }

NB_AI="$PROJECT_ROOT/neuro_brainwave_ai_project"
cd "$NB_AI" || { echo "Cannot cd to $NB_AI"; exit 1; }

# Enhanced Neuro-Brainwave Data Generator Launcher
# ===============================================
# 
# This script launches the enhanced dynamic data generator that solves
# the issue of static brain state and emotional state distributions.

echo "Enhanced Neuro-Brainwave Data Generator"
echo "=========================================="
echo ""

# Check if we're in the right directory
if [ ! -f "enhanced_data_generator.py" ]; then
    echo "Error: enhanced_data_generator.py not found!"
    echo "Please run this script from the neuro_brainwave_ai_project directory."
    exit 1
fi

echo "Enhanced Features:"
echo "Dynamic State Distributions - Each generation creates different patterns"
echo "Temporal Variation - States change over time within generations"
echo "Custom Distribution Control - Use sliders to control state ratios"
echo "Emotional Variety - More diverse emotional state patterns"
echo "Variation Analysis - Track how distributions change"
echo ""
echo "Application will be available at:"
echo "Local:   http://localhost:8504"
echo "Network: http://$(hostname -I | awk '{print $1}'):8504"
echo ""
echo "Press Ctrl+C to stop the application"
echo ""

# Launch the enhanced generator
python3 -m streamlit run enhanced_data_generator.py \
    --server.port 8504 \
    --server.headless false \
    --server.enableCORS false \
    --server.enableXsrfProtection false
