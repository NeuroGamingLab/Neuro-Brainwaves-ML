#!/usr/bin/env python3
"""
Test script to verify producer-consumer connection
"""

import subprocess
import time
import sys
import os

def test_producer_consumer():
    """Test the producer-consumer connection"""
    print(" Testing Producer-Consumer Connection")
    print("=" * 50)
    
    # Start producer in background
    print(" Starting producer...")
    producer_process = subprocess.Popen(
        [sys.executable, 'brain_signal_producer.py', 'excited', '20', '0.3'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Wait for producer to start
    print(" Waiting for producer to start...")
    time.sleep(3)
    
    # Start consumer
    print(" Starting consumer...")
    consumer_process = subprocess.Popen(
        [sys.executable, 'simple_brain_consumer.py'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Wait for both to complete
    print(" Waiting for processes to complete...")
    
    # Wait for producer to finish
    producer_stdout, producer_stderr = producer_process.communicate()
    print(" Producer output:")
    print(producer_stdout)
    if producer_stderr:
        print(" Producer errors:")
        print(producer_stderr)
    
    # Wait for consumer to finish
    consumer_stdout, consumer_stderr = consumer_process.communicate()
    print(" Consumer output:")
    print(consumer_stdout)
    if consumer_stderr:
        print(" Consumer errors:")
        print(consumer_stderr)
    
    print(" Test completed!")

if __name__ == "__main__":
    test_producer_consumer()
