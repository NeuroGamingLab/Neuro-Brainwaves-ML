#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT" || { echo "Cannot cd to project root: $PROJECT_ROOT"; exit 1; }

echo "AI Brain Signal Producer & Consumer Setup"
echo "=============================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}$1${NC}"
}

print_warning() {
    echo -e "${YELLOW}$1${NC}"
}

print_error() {
    echo -e "${RED}$1${NC}"
}

print_info() {
    echo -e "${BLUE}$1${NC}"
}

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null
then
    print_error "Python 3 is not installed. Please install Python 3 to proceed."
    exit 1
fi

print_status "Found Python $(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")')"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    print_info "Creating virtual environment..."
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        print_error "Failed to create virtual environment."
        exit 1
    fi
    print_status "Virtual environment created successfully!"
else
    print_info "Virtual environment 'venv' already exists."
fi

# Activate virtual environment
print_info "Activating virtual environment..."
source venv/bin/activate
if [ $? -ne 0 ]; then
    print_error "Failed to activate virtual environment."
    exit 1
fi

# Upgrade pip
print_info "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
print_info "Installing dependencies..."
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    print_error "Failed to install dependencies."
    deactivate
    exit 1
fi

# Verify installation
print_info "Verifying installation..."
if python -c "import numpy, matplotlib, pandas, seaborn, scipy" &> /dev/null; then
    print_status "All core dependencies installed successfully!"
else
    print_warning "Some dependencies might not be installed correctly."
fi

echo ""
echo "Setup completed successfully!"
echo ""
echo "Available Commands:"
echo "======================"
echo ""
echo "Quick Start:"
echo "./scripts/run_consumer.sh     - Start the brain signal consumer"
echo "./scripts/run_producer.sh     - Start the brain signal producer"
echo ""
echo "Brain States:"
echo "./scripts/run_producer.sh normal 60 0.3     - Normal state, 60 seconds"
echo "./scripts/run_producer.sh excited 30 0.4    - Excited state, 30 seconds"
echo "./scripts/run_producer.sh relaxed 45 0.2    - Relaxed state, 45 seconds"
echo "./scripts/run_producer.sh focused 120 0.3   - Focused state, 2 minutes"
echo "./scripts/run_producer.sh stressed 90 0.5   - Stressed state, 90 seconds"
echo "./scripts/run_producer.sh sleepy 60 0.3     - Sleepy state, 60 seconds"
echo ""
echo "Utilities:"
echo "bash "$SCRIPT_DIR/cleanup.sh"          - Clean up ports and processes"
echo "./test_connection.sh  - Test producer-consumer connection"
echo ""
echo "Usage Instructions:"
echo "1. Start consumer first: ./scripts/run_consumer.sh"
echo "2. Start producer second: ./scripts/run_producer.sh [brain_state] [duration] [variability]"
echo "3. Press Ctrl+C to stop either agent"
echo ""
echo "For detailed instructions, see README_Brain_Agents.md"
