#!/usr/bin/env python3
"""
Simple Brain Signal Consumer (Command Line Version)
Receives brainwave signals from producer and displays real-time data
No GUI - command line only for better compatibility
"""

import numpy as np
import json
import socket
import threading
import time
from collections import deque
import matplotlib.pyplot as plt

from paths import DATA_DIR, ensure_data_dir

class SimpleBrainConsumer:
    def __init__(self, host='localhost', port=12345):
        self.host = host
        self.port = port
        self.socket = None
        self.is_running = False
        self.is_connected = False
        
        # Data storage
        self.samples_buffer = deque(maxlen=1000)
        self.time_buffer = deque(maxlen=1000)
        self.state_buffer = deque(maxlen=1000)
        self.variation_buffer = deque(maxlen=1000)
        
        # Session info
        self.session_info = None
        self.events_info = None
        self.current_session = None
        
        # Channel names
        self.channel_names = ['CP3', 'C3', 'F5', 'PO3', 'PO4', 'F6', 'C4', 'CP4']
        
        print(" Simple Brain Signal Consumer Initialized")
        print(f" Connecting to: {host}:{port}")
        print("=" * 60)
    
    def connect_to_producer(self, max_retries=10, retry_delay=2):
        """Connect to the brain signal producer with retries"""
        for attempt in range(max_retries):
            try:
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.socket.connect((self.host, self.port))
                self.is_connected = True
                self.is_running = True
                
                print(f" Connected to producer at {self.host}:{self.port}")
                return True
                
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f" Connection attempt {attempt + 1}/{max_retries} failed: {e}")
                    print(f" Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    print(f" Failed to connect after {max_retries} attempts: {e}")
                    return False
    
    def start_listening(self):
        """Start listening for brainwave signals"""
        if not self.connect_to_producer():
            return False
        
        print(" Started listening for brainwave signals...")
        print(" Real-time data will be displayed below:")
        print("-" * 60)
        
        # Start data reception thread
        self.reception_thread = threading.Thread(target=self.receive_data_loop)
        self.reception_thread.daemon = True
        self.reception_thread.start()
        
        return True
    
    def receive_data_loop(self):
        """Main loop for receiving data from producer"""
        buffer = ""
        sample_count = 0
        
        while self.is_running and self.is_connected:
            try:
                # Receive data
                data = self.socket.recv(4096).decode('utf-8')
                if not data:
                    break
                
                buffer += data
                
                # Process complete messages
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    if line.strip():
                        self.process_message(line.strip())
                        sample_count += 1
                        
                        # Display progress every 50 samples
                        if sample_count % 50 == 0:
                            print(f" Received {sample_count} samples...")
                
            except Exception as e:
                print(f" Error receiving data: {e}")
                break
        
        self.is_connected = False
        print(f" Disconnected from producer. Total samples received: {sample_count}")
    
    def process_message(self, message_json):
        """Process a received message from producer"""
        try:
            message = json.loads(message_json)
            message_type = message.get('type', 'unknown')
            
            if message_type == 'session_start':
                self.handle_session_start(message['data'])
            elif message_type == 'events_info':
                self.handle_events_info(message['data'])
            elif message_type == 'session_end':
                self.handle_session_end(message['data'])
            elif message_type == 'brainwave_sample':
                self.handle_brainwave_sample(message)
            else:
                print(f" Unknown message type: {message_type}")
                
        except Exception as e:
            print(f" Error processing message: {e}")
    
    def handle_session_start(self, data):
        """Handle session start message"""
        self.session_info = data
        self.current_session = {
            'start_time': time.time(),
            'brain_state': data['brain_state'],
            'duration': data['duration'],
            'sampling_rate': data['sampling_rate'],
            'channels': data['channels']
        }
        
        print(f" Session started: {data['brain_state']} for {data['duration']} seconds")
        print(f" Channels: {', '.join(data['channels'])}")
        print(f" Sampling rate: {data['sampling_rate']} Hz")
    
    def handle_events_info(self, data):
        """Handle events info message"""
        self.events_info = data
        print(f" Brain state events: {data['num_events']}")
    
    def handle_session_end(self, data):
        """Handle session end message"""
        print(f" Session ended: {data['total_epochs']} epochs transmitted")
        self.stop_listening()
    
    def handle_brainwave_sample(self, message):
        """Handle brainwave sample data"""
        try:
            # Extract sample from message
            sample = message['data']
            
            # Extract data
            data = np.array(sample['data'])
            timestamp = sample['info']['startTime']
            brain_state = sample['info'].get('brainState', 'unknown')
            state_variation = sample['info'].get('stateVariation', 1.0)
            
            # Calculate time from start
            if self.current_session:
                elapsed_time = time.time() - self.current_session['start_time']
            else:
                elapsed_time = 0.0
            
            # Store in buffers
            self.samples_buffer.append(data)
            self.time_buffer.append(elapsed_time)
            self.state_buffer.append(brain_state)
            self.variation_buffer.append(state_variation)
            
            # Display sample info (every 10th sample)
            if len(self.samples_buffer) % 10 == 0:
                print(f"  {elapsed_time:.1f}s | State: {brain_state} | "
                      f"Var: {state_variation:.2f} | Channels: {data.shape[0]}")
            
        except Exception as e:
            print(f" Error processing brainwave sample: {e}")
    
    def stop_listening(self):
        """Stop listening for brainwave signals"""
        self.is_running = False
        print(" Stopped listening for brainwave signals")
    
    def save_data(self, filename=None):
        """Save received data to file"""
        if len(self.samples_buffer) == 0:
            print(" No data to save")
            return
        
        if filename is None:
            ensure_data_dir()
            filename = str(DATA_DIR / f"consumer_brainwave_data_{int(time.time())}.json")
        
        # Convert to format compatible with other tools
        samples = []
        for i, (data, timestamp, state, variation) in enumerate(zip(
            self.samples_buffer, self.time_buffer, 
            self.state_buffer, self.variation_buffer)):
            
            sample = {
                'label': 'raw',
                'data': data.tolist(),
                'info': {
                    'channelNames': self.channel_names,
                    'samplingRate': 256,
                    'startTime': int(timestamp * 1000),
                    'brainState': state,
                    'stateVariation': variation,
                    'timestamp': timestamp
                }
            }
            samples.append(sample)
        
        # Save to file
        with open(filename, 'w') as f:
            json.dump(samples, f, indent=2)
        
        print(f" Saved {len(samples)} samples to {filename}")
        return filename
    
    def create_visualization(self, filename=None):
        """Create visualization of received data"""
        if len(self.samples_buffer) == 0:
            print(" No data to visualize")
            return
        
        print(" Creating visualization...")
        
        # Convert data for plotting
        all_data = np.array(list(self.samples_buffer))
        all_times = np.array(list(self.time_buffer))
        all_states = list(self.state_buffer)
        all_variations = np.array(list(self.variation_buffer))
        
        # Create plots
        fig, axes = plt.subplots(3, 1, figsize=(12, 8))
        fig.suptitle(' Brain Signal Consumer - Received Data', fontsize=16, fontweight='bold')
        
        # Plot 1: All channels
        ax1 = axes[0]
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
        
        for ch in range(min(8, all_data.shape[1])):
            ax1.plot(all_times, all_data[:, ch], 
                    color=colors[ch], label=self.channel_names[ch], 
                    linewidth=1, alpha=0.8)
        
        ax1.set_title('All Channels - Received Brainwave Data', fontweight='bold')
        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel('Amplitude (μV)')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Brain states
        ax2 = axes[1]
        state_colors = {'normal': 'green', 'excited': 'red', 'relaxed': 'blue', 
                       'focused': 'orange', 'stressed': 'purple', 'sleepy': 'brown'}
        
        current_state = all_states[0]
        start_time = all_times[0]
        
        for i, (time_val, state) in enumerate(zip(all_times, all_states)):
            if state != current_state or i == len(all_times) - 1:
                color = state_colors.get(current_state, 'gray')
                duration = time_val - start_time
                ax2.barh(0, duration, left=start_time, height=0.5, 
                        color=color, alpha=0.7, label=current_state)
                
                current_state = state
                start_time = time_val
        
        ax2.set_title('Brain State Timeline', fontweight='bold')
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel('State')
        ax2.set_ylim(-0.5, 0.5)
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: State variations
        ax3 = axes[2]
        ax3.plot(all_times, all_variations, 'r-', linewidth=2, label='State Variation')
        ax3.set_title('State Variation Over Time', fontweight='bold')
        ax3.set_xlabel('Time (seconds)')
        ax3.set_ylabel('Variation Factor')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        plt.tight_layout()
        
        if filename is None:
            ensure_data_dir()
            filename = str(DATA_DIR / f"consumer_visualization_{int(time.time())}.png")
        
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f" Visualization saved as: {filename}")
        
        return filename

def main():
    """Main function to run the Simple Brain Consumer"""
    print(" Simple Brain Signal Consumer")
    print("=" * 60)
    
    # Initialize consumer
    consumer = SimpleBrainConsumer()
    
    try:
        # Start listening
        if consumer.start_listening():
            print(" Waiting for producer to send data...")
            print(" Press Ctrl+C to stop")
            
            # Keep running until stopped
            while consumer.is_running:
                time.sleep(1)
        else:
            print(" Failed to start consumer")
            
    except KeyboardInterrupt:
        print("\n Consumer stopped by user")
    except Exception as e:
        print(f" Consumer error: {e}")
    finally:
        # Save data and create visualization
        if len(consumer.samples_buffer) > 0:
            print("\n Saving received data...")
            consumer.save_data()
            consumer.create_visualization()
        
        consumer.stop_listening()

if __name__ == "__main__":
    main()
