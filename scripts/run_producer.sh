#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT" || { echo "Cannot cd to project root: $PROJECT_ROOT"; exit 1; }

echo "Starting Brain Signal Producer..."
echo "===================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Virtual environment not found. Please run ./scripts/setup_brain_agents.sh first."
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Check if producer script exists
if [ ! -f "brain_signal_producer.py" ]; then
    echo "Producer script not found. Please make sure brain_signal_producer.py exists."
    exit 1
fi

# Default parameters
BRAIN_STATE=${1:-excited}
DURATION=${2:-60}
VARIABILITY=${3:-0.3}

# Validate brain state
case $BRAIN_STATE in
    normal|excited|relaxed|focused|stressed|sleepy)
        ;;
    *)
        echo "Invalid brain state: $BRAIN_STATE"
        echo "Valid states: normal, excited, relaxed, focused, stressed, sleepy"
        exit 1
        ;;
esac

# Validate duration
if ! [[ "$DURATION" =~ ^[0-9]+\.?[0-9]*$ ]] || (( $(echo "$DURATION <= 0" | bc -l) )); then
    echo "Invalid duration: $DURATION. Must be a positive number."
    exit 1
fi

# Validate variability
if ! [[ "$VARIABILITY" =~ ^[0-9]+\.?[0-9]*$ ]] || (( $(echo "$VARIABILITY < 0.1" | bc -l) )) || (( $(echo "$VARIABILITY > 1.0" | bc -l) )); then
    echo "Invalid variability: $VARIABILITY. Must be between 0.1 and 1.0."
    exit 1
fi

echo "Configuration:"
echo "   • Brain State: $BRAIN_STATE"
echo "   • Duration: $DURATION seconds"
echo "   • Variability: $VARIABILITY"
echo ""
echo "Make sure consumer is running first!"
echo "Press Ctrl+C to stop the producer"
echo ""

# Run the producer
python brain_signal_producer.py "$BRAIN_STATE" "$DURATION" "$VARIABILITY"
