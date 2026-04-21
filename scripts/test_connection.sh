#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT" || { echo "Cannot cd to project root: $PROJECT_ROOT"; exit 1; }

echo "Testing Brain Signal Producer & Consumer Connection"
echo "====================================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Virtual environment not found. Please run ./scripts/setup_brain_agents.sh first."
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Clean up any existing processes
echo "Cleaning up existing processes..."
bash "$SCRIPT_DIR/cleanup.sh" > /dev/null 2>&1

echo ""
echo "Starting test session..."
echo ""

# Start producer in background
echo "Starting producer (excited, 15 seconds)..."
python brain_signal_producer.py excited 15 0.3 &
PRODUCER_PID=$!

# Wait for producer to start
sleep 3

# Start consumer
echo "Starting consumer..."
python simple_brain_consumer.py &
CONSUMER_PID=$!

# Wait for both to complete
echo "Waiting for test to complete..."
wait $PRODUCER_PID
PRODUCER_EXIT=$?

wait $CONSUMER_PID
CONSUMER_EXIT=$?

echo ""
echo "Test Results:"
echo "================"

if [ $PRODUCER_EXIT -eq 0 ]; then
    echo "Producer completed successfully"
else
    echo "Producer failed with exit code $PRODUCER_EXIT"
fi

if [ $CONSUMER_EXIT -eq 0 ]; then
    echo "Consumer completed successfully"
else
    echo "Consumer failed with exit code $CONSUMER_EXIT"
fi

if [ $PRODUCER_EXIT -eq 0 ] && [ $CONSUMER_EXIT -eq 0 ]; then
    echo ""
    echo "Test passed! Producer and consumer communication is working."
    echo "You can now use ./scripts/run_consumer.sh and ./scripts/run_producer.sh"
else
    echo ""
    echo "Test failed. Please check the error messages above."
    echo "Try running bash "$SCRIPT_DIR/cleanup.sh" and try again"
fi

# Clean up
bash "$SCRIPT_DIR/cleanup.sh" > /dev/null 2>&1
