#!/usr/bin/env python3
"""
Test script for Brain Signal Producer and Consumer
Demonstrates real-time communication between the two AI agents
"""

import time
import threading
import subprocess
import sys
import os

def test_producer_consumer():
    """Test the producer-consumer communication"""
    print(" Testing Brain Signal Producer & Consumer")
    print("=" * 60)
    
    # Test configuration
    test_configs = [
        {'brain_state': 'excited', 'duration': 30.0, 'variability': 0.3},
        {'brain_state': 'relaxed', 'duration': 45.0, 'variability': 0.2},
        {'brain_state': 'focused', 'duration': 60.0, 'variability': 0.4},
    ]
    
    for i, config in enumerate(test_configs, 1):
        print(f"\n Test {i}: {config['brain_state']} state for {config['duration']} seconds")
        print("-" * 50)
        
        # Start producer
        producer_script = f"""
import sys
sys.path.append('{os.getcwd()}')
from brain_signal_producer import BrainSignalProducer

producer = BrainSignalProducer()
producer.configure_session(
    brain_state='{config['brain_state']}',
    duration={config['duration']},
    variability={config['variability']}
)

if producer.start_server():
    producer.generate_and_transmit_session()
    producer.stop_producer()
"""
        
        with open(f'temp_producer_{i}.py', 'w') as f:
            f.write(producer_script)
        
        # Start producer process
        producer_process = subprocess.Popen(
            [sys.executable, f'temp_producer_{i}.py'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait for producer to start
        time.sleep(2)
        
        # Start consumer
        consumer_process = subprocess.Popen(
            [sys.executable, 'brain_signal_consumer.py'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        print(f" Started producer and consumer for {config['brain_state']} test")
        
        # Wait for completion
        producer_process.wait()
        consumer_process.terminate()
        consumer_process.wait()
        
        # Cleanup
        os.remove(f'temp_producer_{i}.py')
        
        print(f" Test {i} completed")
        time.sleep(1)
    
    print("\n All tests completed successfully!")
    print(" Check the generated JSON files for brainwave data")

def test_quick_session():
    """Test a quick 10-second session"""
    print(" Quick Test: 10-second excited session")
    print("=" * 50)
    
    # Start producer in background
    producer_script = """
import sys
sys.path.append('{os.getcwd()}')
from brain_signal_producer import BrainSignalProducer

producer = BrainSignalProducer()
producer.configure_session(
    brain_state='excited',
    duration=10.0,
    variability=0.3
)

if producer.start_server():
    producer.generate_and_transmit_session()
    producer.stop_producer()
"""
    
    with open('quick_producer.py', 'w') as f:
        f.write(producer_script)
    
    producer_process = subprocess.Popen(
        [sys.executable, 'quick_producer.py'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    time.sleep(1)
    
    # Start consumer
    consumer_process = subprocess.Popen(
        [sys.executable, 'brain_signal_consumer.py'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    print(" Quick test started - check consumer GUI")
    
    # Wait for completion
    producer_process.wait()
    consumer_process.terminate()
    consumer_process.wait()
    
    # Cleanup
    os.remove('quick_producer.py')
    
    print(" Quick test completed")

def main():
    """Main test function"""
    print(" Brain Signal Producer & Consumer Test Suite")
    print("=" * 60)
    
    print("Choose test mode:")
    print("1. Full test suite (multiple brain states)")
    print("2. Quick test (10-second excited session)")
    print("3. Interactive launcher")
    
    try:
        choice = input("\n Select test (1-3): ").strip()
        
        if choice == '1':
            test_producer_consumer()
        elif choice == '2':
            test_quick_session()
        elif choice == '3':
            print(" Starting interactive launcher...")
            subprocess.run([sys.executable, 'launch_brain_agents.py'])
        else:
            print(" Invalid choice")
            
    except KeyboardInterrupt:
        print("\n Test stopped by user")
    except Exception as e:
        print(f" Test error: {e}")

if __name__ == "__main__":
    main()
