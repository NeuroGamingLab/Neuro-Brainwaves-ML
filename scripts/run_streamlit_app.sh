#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT" || { echo "Cannot cd to project root: $PROJECT_ROOT"; exit 1; }

# Streamlit Brain Signal App Launcher
# This script sets up and runs the interactive brain signal visualization app

echo "Brain Signal Streamlit App Launcher"
echo "======================================"

# Check if we're in the right directory
if [ ! -f "brain_signal_streamlit_app.py" ]; then
    echo "Error: brain_signal_streamlit_app.py not found!"
    echo "Please run this script from the project root directory."
    exit 1
fi

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed or not in PATH"
    exit 1
fi

# Check if streamlit is installed
if ! python3 -c "import streamlit" 2>/dev/null; then
    echo "Streamlit not found. Installing requirements..."
    
    # Try to install streamlit and plotly
    if command -v pip3 &> /dev/null; then
        pip3 install streamlit plotly
    elif command -v pip &> /dev/null; then
        pip install streamlit plotly
    else
        echo "Error: pip not found. Please install streamlit and plotly manually:"
        echo "pip install streamlit plotly"
        exit 1
    fi
fi

# Check if plotly is installed
if ! python3 -c "import plotly" 2>/dev/null; then
    echo "Plotly not found. Installing..."
    pip3 install plotly || pip install plotly
fi

echo "All dependencies are ready!"
echo ""
echo "Starting Streamlit Brain Signal App..."
echo "The app will open in your default web browser"
echo "If it doesn't open automatically, go to: http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the app"
echo ""

# Run the streamlit app
streamlit run brain_signal_streamlit_app.py --server.port 8501 --server.headless false
