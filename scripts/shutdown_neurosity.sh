#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT" || { echo "Cannot cd to project root: $PROJECT_ROOT"; exit 1; }

# Neurosity-3 Brain Signal Generator & Visualizer - Clean Shutdown Script
# This script safely shuts down the Streamlit application and related processes

echo "Neurosity-3 Brain Signal Generator - Clean Shutdown"
echo "=================================================="

# Function to check if a process is running
check_process() {
    if pgrep -f "$1" > /dev/null; then
        return 0  # Process is running
    else
        return 1  # Process is not running
    fi
}

# Function to kill processes gracefully
kill_processes() {
    local process_name="$1"
    local signal="${2:-TERM}"  # Default to TERM signal
    
    echo "Looking for $process_name processes..."
    
    if check_process "$process_name"; then
        echo "Found running $process_name processes:"
        ps aux | grep "$process_name" | grep -v grep
        
        echo "Sending $signal signal to $process_name processes..."
        pkill -$signal -f "$process_name"
        
        # Wait a moment for graceful shutdown
        sleep 2
        
        # Check if processes are still running
        if check_process "$process_name"; then
            echo "Some processes still running, sending KILL signal..."
            pkill -KILL -f "$process_name"
            sleep 1
        fi
        
        if check_process "$process_name"; then
            echo "Failed to stop some $process_name processes"
            return 1
        else
            echo "Successfully stopped all $process_name processes"
            return 0
        fi
    else
        echo "No $process_name processes found"
        return 0
    fi
}

# Function to check port usage
check_port() {
    local port="$1"
    echo "Checking port $port usage..."
    
    if lsof -i :$port > /dev/null 2>&1; then
        echo "Port $port is in use:"
        lsof -i :$port
        return 0
    else
        echo "Port $port is free"
        return 1
    fi
}

# Main shutdown sequence
echo "Starting clean shutdown sequence..."

# Step 1: Stop Streamlit processes
echo ""
echo "Step 1: Stopping Streamlit processes"
echo "-----------------------------------"
kill_processes "streamlit" "TERM"

# Step 2: Stop any Python processes running the app
echo ""
echo "Step 2: Stopping Python app processes"
echo "------------------------------------"
kill_processes "brain_signal_streamlit_simple.py" "TERM"

# Step 3: Check and free up ports
echo ""
echo "Step 3: Checking port usage"
echo "---------------------------"
check_port 8501
check_port 8502
check_port 8503

# Step 4: Clean up any remaining processes
echo ""
echo "Step 4: Final cleanup"
echo "--------------------"
kill_processes "python.*streamlit" "TERM"

# Step 5: Verification
echo ""
echo "Step 5: Verification"
echo "--------------------"
echo "Checking for any remaining processes..."

remaining_processes=$(ps aux | grep -E "(streamlit|brain_signal)" | grep -v grep | wc -l)

if [ "$remaining_processes" -eq 0 ]; then
    echo "All processes successfully terminated"
else
    echo "Found $remaining_processes remaining processes:"
    ps aux | grep -E "(streamlit|brain_signal)" | grep -v grep
    echo ""
    echo "Attempting final cleanup..."
    pkill -KILL -f "streamlit"
    pkill -KILL -f "brain_signal"
fi

# Final status
echo ""
echo "Final Status"
echo "==============="
echo "Remaining Streamlit processes:"
ps aux | grep streamlit | grep -v grep || echo "None found"

echo ""
echo "Port status:"
for port in 8501 8502 8503; do
    if lsof -i :$port > /dev/null 2>&1; then
        echo "Port $port: IN USE"
    else
        echo "Port $port: FREE"
    fi
done

echo ""
echo "Clean shutdown completed!"
echo ""
echo "Tips:"
echo "- Close any browser tabs with the application"
echo "- Clear browser cache if needed"
echo "- Run this script again if you see any remaining processes"
echo ""
echo "To restart the application:"
echo "cd /Users/admin/NeruoMarketing/neurosity-eval/neurosity-3"
echo "python3 -m streamlit run brain_signal_streamlit_simple.py"
echo ""
