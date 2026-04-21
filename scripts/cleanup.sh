#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT" || { echo "Cannot cd to project root: $PROJECT_ROOT"; exit 1; }

echo "Cleaning up Brain Signal Agents..."
echo "====================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}$1${NC}"
}

print_warning() {
    echo -e "${YELLOW}$1${NC}"
}

print_error() {
    echo -e "${RED}$1${NC}"
}

# Kill processes using port 12345
echo "Checking for processes using port 12345..."
PIDS=$(lsof -ti:12345 2>/dev/null)

if [ -n "$PIDS" ]; then
    echo "Found processes using port 12345: $PIDS"
    echo "Killing processes..."
    echo "$PIDS" | xargs kill -9 2>/dev/null
    if [ $? -eq 0 ]; then
        print_status "Successfully killed processes on port 12345"
    else
        print_warning "Some processes could not be killed (may require sudo)"
    fi
else
    print_status "No processes found using port 12345"
fi

# Kill any Python processes running brain signal scripts
echo "Checking for running brain signal processes..."
BRAIN_PIDS=$(pgrep -f "brain_signal_producer.py\|simple_brain_consumer.py\|brain_signal_consumer.py" 2>/dev/null)

if [ -n "$BRAIN_PIDS" ]; then
    echo "Found brain signal processes: $BRAIN_PIDS"
    echo "Killing brain signal processes..."
    echo "$BRAIN_PIDS" | xargs kill -9 2>/dev/null
    if [ $? -eq 0 ]; then
        print_status "Successfully killed brain signal processes"
    else
        print_warning "Some processes could not be killed"
    fi
else
    print_status "No brain signal processes found"
fi

# Clean up any temporary files
echo "Cleaning up temporary files..."
rm -f temp_producer*.py 2>/dev/null
rm -f quick_producer.py 2>/dev/null
print_status "Temporary files cleaned up"

echo ""
print_status "Cleanup completed!"
echo "You can now start fresh with ./scripts/run_consumer.sh and ./scripts/run_producer.sh"
