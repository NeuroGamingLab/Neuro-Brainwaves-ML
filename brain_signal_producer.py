#!/usr/bin/env python3
"""
AI Agent - Brain Signal Producer
Generates brainwave signals and transmits them to consumer in real-time
Supports all brain states and flexible durations
"""

import numpy as np
import time
import json
import socket
import threading
from datetime import datetime
import random
from dynamic_brainwave_generator import DynamicBrainwaveGenerator

class BrainSignalProducer:
    def __init__(self, host='localhost', port=12345):
        self.host = host
        self.port = port
        self.socket = None
        self.client_socket = None
        self.is_running = False
        self.generator = DynamicBrainwaveGenerator()
        
        # Producer configuration
        self.current_brain_state = 'normal'
        self.current_duration = 60.0
        self.current_variability = 0.3
        
        print(" AI Brain Signal Producer Initialized")
        print(f" Server: {host}:{port}")
        print("=" * 60)
    
    def _cleanup_port(self):
        """Clean up any existing connections on the port"""
        try:
            import subprocess
            import os
            
            # Kill any processes using the port
            result = subprocess.run(['lsof', '-ti', f':{self.port}'], 
                                  capture_output=True, text=True)
            if result.stdout.strip():
                pids = result.stdout.strip().split('\n')
                for pid in pids:
                    if pid:
                        try:
                            os.kill(int(pid), 9)
                            print(f" Cleaned up process {pid} on port {self.port}")
                        except:
                            pass
        except:
            pass  # Ignore cleanup errors
    
    def start_server(self):
        """Start the producer server to listen for consumer connections"""
        try:
            # Clean up any existing connections on the port
            self._cleanup_port()
            
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.socket.bind((self.host, self.port))
            self.socket.listen(1)
            
            print(f" Producer server started on {self.host}:{self.port}")
            print(" Waiting for consumer to connect...")
            
            # Accept consumer connection
            self.client_socket, address = self.socket.accept()
            print(f" Consumer connected from {address}")
            
            self.is_running = True
            return True
            
        except Exception as e:
            print(f" Error starting server: {e}")
            return False
    
    def configure_session(self, brain_state='normal', duration=60.0, variability=0.3):
        """Configure the brainwave session parameters"""
        self.current_brain_state = brain_state
        self.current_duration = duration
        self.current_variability = variability
        
        print(f"  Session Configuration:")
        print(f"   • Brain State: {brain_state}")
        print(f"   • Duration: {duration} seconds")
        print(f"   • Variability: {variability}")
        print("-" * 40)
    
    def send_control_message(self, message_type, data=None):
        """Send control messages to consumer"""
        message = {
            'type': message_type,
            'timestamp': time.time(),
            'data': data
        }
        
        try:
            message_json = json.dumps(message) + '\n'
            self.client_socket.send(message_json.encode('utf-8'))
            return True
        except Exception as e:
            print(f" Error sending control message: {e}")
            return False
    
    def send_brainwave_sample(self, sample):
        """Send a single brainwave sample to consumer"""
        try:
            # Add producer metadata
            sample['producer_info'] = {
                'producer_id': 'brain_signal_producer_v1.0',
                'transmission_time': time.time(),
                'session_duration': self.current_duration,
                'brain_state': self.current_brain_state
            }
            
            # Wrap sample in message structure
            message = {
                'type': 'brainwave_sample',
                'timestamp': time.time(),
                'data': sample
            }
            
            # Send sample
            message_json = json.dumps(message) + '\n'
            self.client_socket.send(message_json.encode('utf-8'))
            return True
            
        except Exception as e:
            print(f" Error sending brainwave sample: {e}")
            return False
    
    def generate_and_transmit_session(self):
        """Generate and transmit a complete brainwave session"""
        if not self.is_running:
            print(" Producer not running. Start server first.")
            return
        
        print(f" Starting brainwave generation and transmission...")
        print(f" Brain State: {self.current_brain_state}")
        print(f"  Duration: {self.current_duration} seconds")
        print("-" * 60)
        
        # Send session start message
        self.send_control_message('session_start', {
            'brain_state': self.current_brain_state,
            'duration': self.current_duration,
            'variability': self.current_variability,
            'sampling_rate': self.generator.sampling_rate,
            'channels': self.generator.channel_names
        })
        
        # Generate brain state events
        brain_state_events = self.generator.generate_brain_state_events(
            self.current_duration, 
            self.current_brain_state, 
            self.current_variability
        )
        
        # Send events info
        self.send_control_message('events_info', {
            'num_events': len(brain_state_events),
            'events': brain_state_events
        })
        
        # Calculate number of epochs
        num_epochs = int(self.current_duration / self.generator.epoch_duration)
        start_time = random.uniform(0, 65)
        
        print(f" Transmitting {num_epochs} epochs...")
        
        # Generate and transmit each epoch
        for epoch_num in range(num_epochs):
            if not self.is_running:
                break
                
            # Generate brainwave EEG data
            signal = self.generator.generate_brainwave_eeg(
                epoch_num, 
                self.current_duration, 
                start_time, 
                brain_state_events
            )
            
            # Get current brain state
            current_time = start_time + (epoch_num * self.generator.epoch_duration)
            current_brain_state, state_variation = self.generator.get_brain_state_at_time(
                current_time, 
                brain_state_events
            )
            
            # Format as Neurosity data
            neurosity_data = self.generator.format_neurosity_data(
                signal, 
                epoch_num, 
                start_time, 
                current_brain_state, 
                state_variation
            )
            
            # Send to consumer
            success = self.send_brainwave_sample(neurosity_data)
            
            if not success:
                print(" Failed to send sample. Stopping transmission.")
                break
            
            # Progress update every 50 epochs
            if epoch_num % 50 == 0:
                print(f" Epoch {epoch_num+1}/{num_epochs} - "
                      f"Time: {current_time:.2f}s - "
                      f"State: {current_brain_state} - "
                      f"Var: {state_variation:.2f}")
            
            # Simulate real-time transmission (optional delay)
            time.sleep(0.01)  # 10ms delay for real-time feel
        
        # Send session end message
        self.send_control_message('session_end', {
            'total_epochs': num_epochs,
            'duration': self.current_duration,
            'final_time': start_time + self.current_duration
        })
        
        print(f" Session completed! Transmitted {num_epochs} epochs")
        print(" Producer ready for next session...")
    
    def run_continuous_producer(self):
        """Run producer in continuous mode - ready to generate sessions on demand"""
        if not self.start_server():
            return
        
        print(" Producer running in continuous mode...")
        print(" Use configure_session() and generate_and_transmit_session() to start")
        print("=" * 60)
        
        try:
            while self.is_running:
                # Wait for commands or run default session
                time.sleep(1)
                
        except KeyboardInterrupt:
            print("\n Producer stopped by user")
        except Exception as e:
            print(f" Producer error: {e}")
        finally:
            self.stop_producer()
    
    def stop_producer(self):
        """Stop the producer and close connections"""
        self.is_running = False
        
        if self.client_socket:
            self.client_socket.close()
        if self.socket:
            self.socket.close()
        
        print(" Producer connections closed")

def main():
    """
    Main function to run the Brain Signal Producer
    """
    import sys
    
    print(" AI Brain Signal Producer")
    print("=" * 60)
    
    # Get command line arguments
    brain_state = sys.argv[1] if len(sys.argv) > 1 else 'excited'
    duration = float(sys.argv[2]) if len(sys.argv) > 2 else 60.0
    variability = float(sys.argv[3]) if len(sys.argv) > 3 else 0.3
    
    print(f" Configuration:")
    print(f"   • Brain State: {brain_state}")
    print(f"   • Duration: {duration} seconds")
    print(f"   • Variability: {variability}")
    print("=" * 60)
    
    # Initialize producer
    producer = BrainSignalProducer()
    
    # Configure session
    producer.configure_session(
        brain_state=brain_state,
        duration=duration,
        variability=variability
    )
    
    # Start server and wait for consumer
    if producer.start_server():
        # Generate and transmit session
        producer.generate_and_transmit_session()
        
        # Keep producer running for additional sessions
        print("\n Producer ready for next session...")
        print(" Press Ctrl+C to stop")
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n Producer stopped")
        finally:
            producer.stop_producer()
    else:
        print(" Failed to start producer server")

if __name__ == "__main__":
    main()
