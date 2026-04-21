#!/usr/bin/env python3
"""
AI Agent - Brain Signal Consumer
Receives brainwave signals from producer and displays real-time graphs and charts
Continuously listens for signals and updates visualizations in real-time
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk
import json
import socket
import threading
import time
from collections import deque
from datetime import datetime
import pandas as pd
import seaborn as sns

class BrainSignalConsumer:
    def __init__(self, host='localhost', port=12345):
        self.host = host
        self.port = port
        self.socket = None
        self.is_running = False
        self.is_connected = False
        
        # Data storage
        self.samples_buffer = deque(maxlen=1000)  # Keep last 1000 samples
        self.time_buffer = deque(maxlen=1000)
        self.state_buffer = deque(maxlen=1000)
        self.variation_buffer = deque(maxlen=1000)
        
        # Session info
        self.session_info = None
        self.events_info = None
        self.current_session = None
        
        # Real-time data
        self.real_time_data = np.zeros((8, 0))  # 8 channels
        self.real_time_time = np.array([])
        self.real_time_states = []
        self.real_time_variations = []
        
        # GUI components
        self.root = None
        self.fig = None
        self.axes = None
        self.canvas = None
        self.ani = None
        
        # Channel names
        self.channel_names = ['CP3', 'C3', 'F5', 'PO3', 'PO4', 'F6', 'C4', 'CP4']
        self.colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
        
        print(" AI Brain Signal Consumer Initialized")
        print(f" Connecting to: {host}:{port}")
        print("=" * 60)
    
    def connect_to_producer(self):
        """Connect to the brain signal producer"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.host, self.port))
            self.is_connected = True
            self.is_running = True
            
            print(f" Connected to producer at {self.host}:{self.port}")
            return True
            
        except Exception as e:
            print(f" Error connecting to producer: {e}")
            return False
    
    def setup_gui(self):
        """Setup the real-time visualization GUI"""
        self.root = tk.Tk()
        self.root.title(" AI Brain Signal Consumer - Real-time Visualization")
        self.root.geometry("1400x900")
        
        # Create main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Status frame
        status_frame = ttk.Frame(main_frame)
        status_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.status_label = ttk.Label(status_frame, text="Connecting to producer...", 
                                    font=('Arial', 12, 'bold'))
        self.status_label.pack(side=tk.LEFT)
        
        self.session_label = ttk.Label(status_frame, text="", font=('Arial', 10))
        self.session_label.pack(side=tk.RIGHT)
        
        # Create matplotlib figure
        self.fig, self.axes = plt.subplots(3, 1, figsize=(12, 8))
        self.fig.suptitle(' Real-time Brainwave Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Real-time brainwave signals
        self.ax1 = self.axes[0]
        self.ax1.set_title('Real-time Brainwave Signals (All Channels)', fontweight='bold')
        self.ax1.set_xlabel('Time (seconds)')
        self.ax1.set_ylabel('Amplitude (μV)')
        self.ax1.grid(True, alpha=0.3)
        
        # Plot 2: Brain state timeline
        self.ax2 = self.axes[1]
        self.ax2.set_title('Brain State Timeline', fontweight='bold')
        self.ax2.set_xlabel('Time (seconds)')
        self.ax2.set_ylabel('State')
        self.ax2.grid(True, alpha=0.3)
        
        # Plot 3: State variation
        self.ax3 = self.axes[2]
        self.ax3.set_title('State Variation Over Time', fontweight='bold')
        self.ax3.set_xlabel('Time (seconds)')
        self.ax3.set_ylabel('Variation Factor')
        self.ax3.grid(True, alpha=0.3)
        
        # Embed matplotlib in tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, main_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Control frame
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=(10, 0))
        
        # Buttons
        self.start_button = ttk.Button(control_frame, text="Start Listening", 
                                     command=self.start_listening)
        self.start_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.stop_button = ttk.Button(control_frame, text="Stop Listening", 
                                    command=self.stop_listening, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.save_button = ttk.Button(control_frame, text="Save Data", 
                                    command=self.save_current_data)
        self.save_button.pack(side=tk.LEFT, padx=(0, 10))
        
        # Info labels
        self.info_label = ttk.Label(control_frame, text="Ready to receive brainwave signals...")
        self.info_label.pack(side=tk.RIGHT)
        
        plt.tight_layout()
    
    def start_listening(self):
        """Start listening for brainwave signals"""
        if not self.is_connected:
            if not self.connect_to_producer():
                return
        
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.status_label.config(text="Listening for brainwave signals...")
        
        # Start data reception thread
        self.reception_thread = threading.Thread(target=self.receive_data_loop)
        self.reception_thread.daemon = True
        self.reception_thread.start()
        
        # Start animation
        self.ani = animation.FuncAnimation(self.fig, self.update_plots, 
                                         interval=100, blit=False)
        
        print(" Started listening for brainwave signals...")
    
    def stop_listening(self):
        """Stop listening for brainwave signals"""
        self.is_running = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.status_label.config(text="Stopped listening")
        
        if self.ani:
            self.ani.event_source.stop()
        
        print(" Stopped listening for brainwave signals")
    
    def receive_data_loop(self):
        """Main loop for receiving data from producer"""
        buffer = ""
        
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
                
            except Exception as e:
                print(f" Error receiving data: {e}")
                break
        
        self.is_connected = False
        self.status_label.config(text="Disconnected from producer")
    
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
        
        # Update GUI
        self.session_label.config(text=f"Session: {data['brain_state']} - {data['duration']}s")
        self.info_label.config(text=f"Receiving {data['brain_state']} brainwave data...")
        
        print(f" Session started: {data['brain_state']} for {data['duration']} seconds")
    
    def handle_events_info(self, data):
        """Handle events info message"""
        self.events_info = data
        print(f" Received {data['num_events']} brain state events")
    
    def handle_session_end(self, data):
        """Handle session end message"""
        self.info_label.config(text=f"Session completed! Received {data['total_epochs']} epochs")
        print(f" Session ended: {data['total_epochs']} epochs transmitted")
    
    def handle_brainwave_sample(self, sample):
        """Handle brainwave sample data"""
        try:
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
            
            # Update real-time data
            if len(self.real_time_data) == 0:
                self.real_time_data = data.T
                self.real_time_time = np.array([elapsed_time] * data.shape[1])
            else:
                self.real_time_data = np.concatenate([self.real_time_data, data.T], axis=1)
                self.real_time_time = np.concatenate([self.real_time_time, 
                                                    np.array([elapsed_time] * data.shape[1])])
            
            self.real_time_states.append(brain_state)
            self.real_time_variations.append(state_variation)
            
            # Keep only last 1000 data points for performance
            if len(self.real_time_time) > 1000:
                self.real_time_data = self.real_time_data[:, -1000:]
                self.real_time_time = self.real_time_time[-1000:]
                self.real_time_states = self.real_time_states[-1000:]
                self.real_time_variations = self.real_time_variations[-1000:]
            
        except Exception as e:
            print(f" Error processing brainwave sample: {e}")
    
    def update_plots(self, frame):
        """Update the real-time plots"""
        if len(self.real_time_time) == 0:
            return
        
        # Clear previous plots
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        
        # Plot 1: Real-time brainwave signals
        self.ax1.set_title('Real-time Brainwave Signals (All Channels)', fontweight='bold')
        self.ax1.set_xlabel('Time (seconds)')
        self.ax1.set_ylabel('Amplitude (μV)')
        
        for ch in range(min(8, self.real_time_data.shape[0])):
            self.ax1.plot(self.real_time_time, self.real_time_data[ch], 
                         color=self.colors[ch], label=self.channel_names[ch], 
                         linewidth=1, alpha=0.8)
        
        self.ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        self.ax1.grid(True, alpha=0.3)
        
        # Plot 2: Brain state timeline
        self.ax2.set_title('Brain State Timeline', fontweight='bold')
        self.ax2.set_xlabel('Time (seconds)')
        self.ax2.set_ylabel('State')
        
        if len(self.real_time_states) > 0:
            # Create state timeline
            state_colors = {'normal': 'green', 'excited': 'red', 'relaxed': 'blue', 
                          'focused': 'orange', 'stressed': 'purple', 'sleepy': 'brown'}
            
            current_state = self.real_time_states[0]
            start_time = self.real_time_time[0]
            
            for i, (time_val, state) in enumerate(zip(self.real_time_time, self.real_time_states)):
                if state != current_state or i == len(self.real_time_time) - 1:
                    # Draw bar for previous state
                    color = state_colors.get(current_state, 'gray')
                    duration = time_val - start_time
                    self.ax2.barh(0, duration, left=start_time, height=0.5, 
                                color=color, alpha=0.7, label=current_state)
                    
                    current_state = state
                    start_time = time_val
        
        self.ax2.set_ylim(-0.5, 0.5)
        self.ax2.grid(True, alpha=0.3)
        
        # Plot 3: State variation over time
        self.ax3.set_title('State Variation Over Time', fontweight='bold')
        self.ax3.set_xlabel('Time (seconds)')
        self.ax3.set_ylabel('Variation Factor')
        
        if len(self.real_time_variations) > 0:
            self.ax3.plot(self.real_time_time, self.real_time_variations, 
                         'r-', linewidth=2, label='State Variation')
        
        self.ax3.grid(True, alpha=0.3)
        self.ax3.legend()
        
        # Update info
        if self.current_session:
            elapsed = time.time() - self.current_session['start_time']
            self.info_label.config(text=f"Elapsed: {elapsed:.1f}s | "
                                       f"States: {len(set(self.real_time_states))} | "
                                       f"Data points: {len(self.real_time_time)}")
    
    def save_current_data(self):
        """Save current received data to file"""
        if len(self.samples_buffer) == 0:
            print(" No data to save")
            return
        
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
        filename = f"consumer_brainwave_data_{int(time.time())}.json"
        with open(filename, 'w') as f:
            json.dump(samples, f, indent=2)
        
        print(f" Saved {len(samples)} samples to {filename}")
        self.info_label.config(text=f"Data saved to {filename}")
    
    def run_consumer(self):
        """Run the consumer GUI"""
        # Setup GUI in main thread
        self.setup_gui()
        
        print("  Consumer GUI started")
        print(" Click 'Start Listening' to begin receiving signals")
        
        # Start GUI main loop
        self.root.mainloop()
    
    def close_consumer(self):
        """Close the consumer and cleanup"""
        self.is_running = False
        self.is_connected = False
        
        if self.socket:
            self.socket.close()
        
        if self.root:
            self.root.quit()
        
        print(" Consumer closed")

def main():
    """
    Main function to run the Brain Signal Consumer
    """
    print(" AI Brain Signal Consumer")
    print("=" * 60)
    
    # Initialize consumer
    consumer = BrainSignalConsumer()
    
    try:
        # Run consumer GUI
        consumer.run_consumer()
    except KeyboardInterrupt:
        print("\n Consumer stopped by user")
    except Exception as e:
        print(f" Consumer error: {e}")
    finally:
        consumer.close_consumer()

if __name__ == "__main__":
    main()
