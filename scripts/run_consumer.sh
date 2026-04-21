#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT" || { echo "Cannot cd to project root: $PROJECT_ROOT"; exit 1; }

echo "Starting Brain Signal Consumer..."
echo "===================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Virtual environment not found. Please run ./scripts/setup_brain_agents.sh first."
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Check if consumer script exists
if [ ! -f "simple_brain_consumer.py" ]; then
    echo "Consumer script not found. Please make sure simple_brain_consumer.py exists."
    exit 1
fi

echo "Starting consumer and waiting for producer to connect..."
echo "Press Ctrl+C to stop the consumer"
echo ""

# Run the consumer
python simple_brain_consumer.py
