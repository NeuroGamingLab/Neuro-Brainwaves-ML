#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT" || { echo "Cannot cd to project root: $PROJECT_ROOT"; exit 1; }

# Brainwave Generator AI Agent - Environment Setup Script
# This script creates a Python virtual environment and installs all dependencies

set -e  # Exit on any error

echo "Brainwave Generator AI Agent - Environment Setup"
echo "=================================================="

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed. Please install Python 3.7 or higher."
    exit 1
fi

# Get Python version
PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
echo "Found Python $PYTHON_VERSION"

# Create virtual environment (--clear fixes relocated/copied venvs with stale shebangs)
echo "Creating virtual environment..."
python3 -m venv --clear venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Verify installation
echo "Verifying installation..."
python -c "import numpy, scipy, matplotlib, pandas; print(' All core dependencies installed successfully!')"

echo ""
echo "Environment setup complete!"
echo ""
echo "To activate the environment, run:"
echo "source venv/bin/activate"
echo ""
echo "To run the brainwave generator:"
echo "python brainwave_generator.py"
echo ""
echo "To deactivate the environment:"
echo "deactivate"
echo ""
