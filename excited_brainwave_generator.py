#!/usr/bin/env python3
"""
AI Agent for Super Excited Brainwave Generation
Generates 60-second duration brainwave data with random excitement values and duration
Simulates high arousal, excited emotional state with realistic EEG patterns
"""

import numpy as np
import time
import json
from datetime import datetime
import random
import matplotlib.pyplot as plt

class ExcitedBrainwaveGenerator:
    def __init__(self, sampling_rate=256, num_channels=8):
        self.sampling_rate = sampling_rate
        self.num_channels = num_channels
        self.epoch_size = 16  # Samples per epoch
        self.epoch_duration = self.epoch_size / sampling_rate  # ~62.5ms
        
        # Channel names matching Neurosity device
        self.channel_names = ['CP3', 'C3', 'F5', 'PO3', 'PO4', 'F6', 'C4', 'CP4']
        
        # EEG frequency bands with EXCITED state characteristics
        self.frequency_bands = {
            'delta': (0.5, 4),      # Reduced in excited state
            'theta': (4, 8),        # Reduced in excited state
            'alpha': (8, 13),        # Significantly reduced in excited state
            'beta': (13, 30),       # INCREASED in excited state
            'gamma': (30, 45)       # INCREASED in excited state
        }
        
        # Initialize time vector for current epoch
        self.time_vector = np.linspace(0, self.epoch_duration, self.epoch_size)
        
    def generate_excitement_events(self, total_duration=60.0):
        """
        Generate random excitement events throughout the 60-second period
        Returns list of (start_time, duration, excitement_level) tuples
        """
        excitement_events = []
        current_time = 0.0
        
        while current_time < total_duration:
            # Random excitement level (0.3 to 2.0)
            excitement_level = random.uniform(0.3, 2.0)
            
            # Random duration for this excitement event (2 to 8 seconds)
            event_duration = random.uniform(2.0, 8.0)
            
            # Ensure we don't exceed total duration
            if current_time + event_duration > total_duration:
                event_duration = total_duration - current_time
            
            excitement_events.append((current_time, event_duration, excitement_level))
            current_time += event_duration
            
            # Add random calm periods between excitement events
            if current_time < total_duration:
                calm_duration = random.uniform(1.0, 5.0)
                if current_time + calm_duration > total_duration:
                    calm_duration = total_duration - current_time
                excitement_events.append((current_time, calm_duration, 0.3))  # Low excitement
                current_time += calm_duration
        
        return excitement_events
    
    def get_excitement_level_at_time(self, current_time, excitement_events):
        """
        Get the excitement level at a specific time based on events
        """
        for start_time, duration, excitement_level in excitement_events:
            if start_time <= current_time < start_time + duration:
                return excitement_level
        return 0.3  # Default low excitement
    
    def generate_excited_eeg(self, epoch_num, total_duration=60.0, start_time=0.0, excitement_events=None):
        """
        Generate excited state EEG data for one epoch
        Characterized by increased Beta and Gamma activity, reduced Alpha
        """
        # Calculate current time within the excited period
        current_time = start_time + (epoch_num * self.epoch_duration)
        
        # Get excitement level for this time
        if excitement_events is None:
            excitement_level = 1.0  # Default
        else:
            excitement_level = self.get_excitement_level_at_time(current_time, excitement_events)
        
        # Base signal (combination of all frequency bands)
        signal = np.zeros((self.num_channels, self.epoch_size))
        
        for ch in range(self.num_channels):
            # Generate different frequency components with EXCITED characteristics
            for band_name, (low_freq, high_freq) in self.frequency_bands.items():
                # Random frequency within the band
                freq = np.random.uniform(low_freq, high_freq)
                
                # EXCITED STATE amplitude ranges (microvolts) - different from normal
                excited_amplitude_ranges = {
                    'delta': (5, 25),      # Reduced (less deep sleep activity)
                    'theta': (3, 15),      # Reduced (less drowsiness)
                    'alpha': (5, 20),      # Significantly reduced (not relaxed)
                    'beta': (15, 45),      # INCREASED (high cognitive activity)
                    'gamma': (8, 25)       # INCREASED (high-level processing)
                }
                
                # Add excitement modulation based on time and excitement level
                base_amplitude = np.random.uniform(*excited_amplitude_ranges[band_name])
                
                # Scale amplitude by excitement level
                amplitude = base_amplitude * excitement_level
                
                # Add some phase variation
                phase = np.random.uniform(0, 2 * np.pi)
                
                # Generate sinusoidal component
                component = amplitude * np.sin(2 * np.pi * freq * self.time_vector + phase)
                
                # Add some noise (realistic EEG noise)
                noise_level = 2 + excitement_level  # More noise with higher excitement
                noise = np.random.normal(0, noise_level, self.epoch_size)
                
                signal[ch] += component + noise
        
        # Add excited state specific artifacts
        signal = self._add_excited_artifacts(signal, epoch_num, current_time, excitement_level)
        
        return signal
    
    def _add_excited_artifacts(self, signal, epoch_num, current_time, excitement_level):
        """
        Add excited state specific artifacts based on excitement level
        """
        # Artifact probability scales with excitement level
        muscle_prob = 0.05 + (excitement_level * 0.1)  # 5% to 25%
        rem_prob = 0.08 + (excitement_level * 0.04)   # 8% to 16%
        
        # Increased muscle tension (affects all channels more)
        if random.random() < muscle_prob:
            muscle_channels = [0, 1, 6, 7]  # CP3, C3, C4, CP4
            for ch in muscle_channels:
                muscle_amplitude = np.random.uniform(20, 60 + excitement_level * 20)  # Higher with excitement
                muscle_duration = random.randint(3, 6 + int(excitement_level * 2))
                start_idx = random.randint(0, self.epoch_size - muscle_duration)
                signal[ch, start_idx:start_idx+muscle_duration] += muscle_amplitude
        
        # Rapid eye movements (excited state)
        if random.random() < rem_prob:
            rem_channels = [2, 5]  # F5, F6 (frontal channels)
            for ch in rem_channels:
                rem_amplitude = np.random.uniform(30, 70 + excitement_level * 20)
                rem_duration = random.randint(2, 4 + int(excitement_level))
                start_idx = random.randint(0, self.epoch_size - rem_duration)
                signal[ch, start_idx:start_idx+rem_duration] += rem_amplitude
        
        # Increased baseline variability (excited state)
        for ch in range(self.num_channels):
            # More variable baseline in excited state
            drift_amplitude = np.random.uniform(-5 - excitement_level * 3, 5 + excitement_level * 3)
            drift = np.linspace(0, drift_amplitude, self.epoch_size)
            signal[ch] += drift
            
            # Add micro-movements (excited state characteristic)
            micro_movements = np.random.normal(0, 2 + excitement_level * 2, self.epoch_size)
            signal[ch] += micro_movements
        
        return signal
    
    def format_neurosity_data(self, signal, epoch_num, start_time, excitement_level):
        """
        Format the generated signal to match Neurosity data structure
        """
        # Convert to list format (matching Neurosity output)
        data = signal.tolist()
        
        # Create timestamp with proper timing
        timestamp = int(time.time() * 1000) + int(start_time * 1000) + (epoch_num * 62)
        
        # Format as Neurosity data structure
        neurosity_data = {
            "label": "raw",
            "data": data,
            "info": {
                "channelNames": self.channel_names,
                "notchFrequency": "60Hz",
                "samplingRate": self.sampling_rate,
                "startTime": timestamp,
                "excitedState": True,
                "duration": 60.0,
                "startTimeOffset": start_time,
                "excitementLevel": excitement_level
            }
        }
        
        return neurosity_data
    
    def generate_excited_session(self, duration=60.0, start_time_range=(0, 65)):
        """
        Generate a complete excited brainwave session with random excitement events
        """
        # Random start time within the specified range
        start_time = random.uniform(*start_time_range)
        
        # Generate random excitement events
        excitement_events = self.generate_excitement_events(duration)
        
        # Calculate number of epochs needed
        num_epochs = int(duration / self.epoch_duration)
        
        print(f" Generating SUPER EXCITED brainwave session...")
        print(f"  Duration: {duration} seconds")
        print(f" Start time: {start_time:.2f} seconds")
        print(f" Sampling rate: {self.sampling_rate} Hz")
        print(f" Channels: {', '.join(self.channel_names)}")
        print(f" Total epochs: {num_epochs}")
        print(f" Excitement events: {len(excitement_events)}")
        print("-" * 60)
        
        # Print excitement events
        for i, (event_start, event_duration, excitement_level) in enumerate(excitement_events):
            print(f"Event {i+1}: {event_start:.1f}s - {event_start+event_duration:.1f}s (Level: {excitement_level:.2f})")
        
        print("-" * 60)
        
        samples = []
        
        for epoch_num in range(num_epochs):
            # Generate excited EEG data
            signal = self.generate_excited_eeg(epoch_num, duration, start_time, excitement_events)
            
            # Get current excitement level
            current_time = start_time + (epoch_num * self.epoch_duration)
            excitement_level = self.get_excitement_level_at_time(current_time, excitement_events)
            
            # Format as Neurosity data
            neurosity_data = self.format_neurosity_data(signal, epoch_num, start_time, excitement_level)
            
            # Add to samples list
            samples.append(neurosity_data)
            
            # Print progress (every 10 epochs)
            if epoch_num % 10 == 0:
                print(f"Epoch {epoch_num+1:3d}: {len(signal[0])} samples, "
                      f"Time: {current_time:.2f}s, "
                      f"Excitement: {excitement_level:.2f}")
        
        return samples, excitement_events
    
    def save_excited_samples(self, samples, filename="excited_brainwave_samples.json"):
        """
        Save generated excited samples to JSON file
        """
        with open(filename, 'w') as f:
            json.dump(samples, f, indent=2)
        print(f"\n Saved {len(samples)} excited samples to {filename}")
    
    def create_excitement_visualization(self, samples, excitement_events):
        """
        Create visualization of the excited brainwave session
        """
        # Extract data
        data_matrix = np.array([sample['data'] for sample in samples])
        
        # Create time vector
        start_time = samples[0]['info']['startTimeOffset']
        total_duration = samples[0]['info']['duration']
        time_vector = np.linspace(start_time, start_time + total_duration, len(samples) * self.epoch_size)
        
        # Flatten data for visualization
        flat_data = data_matrix.reshape(-1, self.num_channels)
        
        # Create visualization
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        fig.suptitle(' SUPER EXCITED Brainwave Session (60s with Random Events)', fontsize=16, fontweight='bold')
        
        # Plot 1: All channels overlaid
        ax1 = axes[0]
        for ch in range(self.num_channels):
            ax1.plot(time_vector, flat_data[:, ch], 
                    label=self.channel_names[ch], linewidth=1, alpha=0.8)
        ax1.set_title('All Channels - Excited State', fontweight='bold')
        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel('Amplitude (μV)')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Excitement events timeline
        ax2 = axes[1]
        for event_start, event_duration, excitement_level in excitement_events:
            color = 'red' if excitement_level > 1.0 else 'orange' if excitement_level > 0.5 else 'green'
            ax2.barh(0, event_duration, left=event_start, height=0.5, 
                    color=color, alpha=0.7, label=f'Level: {excitement_level:.2f}')
        ax2.set_title('Excitement Events Timeline', fontweight='bold')
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel('Events')
        ax2.set_ylim(-0.5, 0.5)
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Excitement level over time
        ax3 = axes[2]
        excitement_levels = []
        for i in range(len(samples)):
            current_time = start_time + (i * self.epoch_duration)
            excitement = self.get_excitement_level_at_time(current_time, excitement_events)
            excitement_levels.extend([excitement] * self.epoch_size)
        
        ax3.plot(time_vector, excitement_levels, 'r-', linewidth=2, label='Excitement Level')
        ax3.set_title('Excitement Level Over Time', fontweight='bold')
        ax3.set_xlabel('Time (seconds)')
        ax3.set_ylabel('Excitement Factor')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        plt.tight_layout()
        plt.savefig('excited_brainwave_session.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f" Visualization saved as: excited_brainwave_session.png")

def main():
    """
    Main function to generate excited brainwave session
    """
    print(" AI Super Excited Brainwave Generator (Enhanced)")
    print("=" * 70)
    
    # Initialize excited generator
    generator = ExcitedBrainwaveGenerator()
    
    # Generate excited session
    samples, excitement_events = generator.generate_excited_session(duration=60.0, start_time_range=(0, 65))
    
    # Save to file
    generator.save_excited_samples(samples)
    
    # Create visualization
    generator.create_excitement_visualization(samples, excitement_events)
    
    # Print summary
    print(f"\n EXCITED SESSION SUMMARY:")
    print(f"• Total samples: {len(samples)}")
    print(f"• Duration: {samples[0]['info']['duration']} seconds")
    print(f"• Start time: {samples[0]['info']['startTimeOffset']:.2f} seconds")
    print(f"• End time: {samples[0]['info']['startTimeOffset'] + samples[0]['info']['duration']:.2f} seconds")
    print(f"• Excitement events: {len(excitement_events)}")
    print(f"• Average excitement level: {np.mean([event[2] for event in excitement_events]):.2f}")
    print(f"• Max excitement level: {max([event[2] for event in excitement_events]):.2f}")
    
    print(f"\n Generated enhanced super excited brainwave session!")
    print(" Perfect for testing BCI applications with dynamic emotional state detection")

if __name__ == "__main__":
    main()
