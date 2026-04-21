#!/usr/bin/env python3
"""
AI Agent for generating realistic brainwave samples
Simulates Neurosity device data for testing and development
"""

import numpy as np
import time
import json
from datetime import datetime
import random

class BrainwaveGenerator:
    def __init__(self, sampling_rate=256, num_channels=8):
        self.sampling_rate = sampling_rate
        self.num_channels = num_channels
        self.epoch_size = 16  # Samples per epoch
        self.epoch_duration = self.epoch_size / sampling_rate  # ~62.5ms
        
        # Channel names matching Neurosity device
        self.channel_names = ['CP3', 'C3', 'F5', 'PO3', 'PO4', 'F6', 'C4', 'CP4']
        
        # EEG frequency bands (Hz)
        self.frequency_bands = {
            'delta': (0.5, 4),      # Deep sleep
            'theta': (4, 8),        # Drowsiness, meditation
            'alpha': (8, 13),       # Relaxed, eyes closed
            'beta': (13, 30),       # Active thinking, concentration
            'gamma': (30, 45)       # High-level cognitive processing
        }
        
        # Initialize time vector for current epoch
        self.time_vector = np.linspace(0, self.epoch_duration, self.epoch_size)
        
    def generate_realistic_eeg(self, epoch_num):
        """
        Generate realistic EEG data for one epoch
        Combines multiple frequency bands with realistic amplitudes
        """
        # Base signal (combination of all frequency bands)
        signal = np.zeros((self.num_channels, self.epoch_size))
        
        for ch in range(self.num_channels):
            # Generate different frequency components
            for band_name, (low_freq, high_freq) in self.frequency_bands.items():
                # Random frequency within the band
                freq = np.random.uniform(low_freq, high_freq)
                
                # Realistic amplitude ranges for each band (microvolts)
                amplitude_ranges = {
                    'delta': (10, 50),
                    'theta': (5, 30),
                    'alpha': (10, 40),
                    'beta': (2, 20),
                    'gamma': (1, 10)
                }
                
                amplitude = np.random.uniform(*amplitude_ranges[band_name])
                
                # Add some phase variation
                phase = np.random.uniform(0, 2 * np.pi)
                
                # Generate sinusoidal component
                component = amplitude * np.sin(2 * np.pi * freq * self.time_vector + phase)
                
                # Add some noise (realistic EEG noise)
                noise = np.random.normal(0, 2, self.epoch_size)
                
                signal[ch] += component + noise
        
        # Add some realistic artifacts and variations
        signal = self._add_realistic_artifacts(signal, epoch_num)
        
        return signal
    
    def _add_realistic_artifacts(self, signal, epoch_num):
        """
        Add realistic EEG artifacts like eye blinks, muscle activity, etc.
        """
        # Occasional eye blink artifact (affects frontal channels more)
        if random.random() < 0.1:  # 10% chance
            blink_channels = [2, 5]  # F5, F6 (frontal channels)
            for ch in blink_channels:
                blink_amplitude = np.random.uniform(50, 100)
                blink_duration = random.randint(2, 4)
                start_idx = random.randint(0, self.epoch_size - blink_duration)
                signal[ch, start_idx:start_idx+blink_duration] += blink_amplitude
        
        # Occasional muscle artifact (affects temporal channels)
        if random.random() < 0.05:  # 5% chance
            muscle_channels = [0, 1, 6, 7]  # CP3, C3, C4, CP4
            for ch in muscle_channels:
                muscle_amplitude = np.random.uniform(20, 60)
                muscle_duration = random.randint(3, 6)
                start_idx = random.randint(0, self.epoch_size - muscle_duration)
                signal[ch, start_idx:start_idx+muscle_duration] += muscle_amplitude
        
        # Slow baseline drift
        for ch in range(self.num_channels):
            drift_amplitude = np.random.uniform(-5, 5)
            drift = np.linspace(0, drift_amplitude, self.epoch_size)
            signal[ch] += drift
        
        return signal
    
    def format_neurosity_data(self, signal, epoch_num):
        """
        Format the generated signal to match Neurosity data structure
        """
        # Convert to list format (matching Neurosity output)
        data = signal.tolist()
        
        # Create timestamp
        timestamp = int(time.time() * 1000) + (epoch_num * 62)  # Approximate timing
        
        # Format as Neurosity data structure
        neurosity_data = {
            "label": "raw",
            "data": data,
            "info": {
                "channelNames": self.channel_names,
                "notchFrequency": "60Hz",
                "samplingRate": self.sampling_rate,
                "startTime": timestamp
            }
        }
        
        return neurosity_data
    
    def generate_samples(self, num_samples=10, delay_between_samples=0.0625):
        """
        Generate multiple brainwave samples with realistic timing
        """
        print(f" Generating {num_samples} realistic brainwave samples...")
        print(f" Sampling rate: {self.sampling_rate} Hz")
        print(f"  Epoch duration: {self.epoch_duration:.3f}s")
        print(f" Channels: {', '.join(self.channel_names)}")
        print("-" * 60)
        
        samples = []
        
        for i in range(num_samples):
            # Generate realistic EEG data
            signal = self.generate_realistic_eeg(i)
            
            # Format as Neurosity data
            neurosity_data = self.format_neurosity_data(signal, i)
            
            # Add to samples list
            samples.append(neurosity_data)
            
            # Print sample info
            print(f"Sample {i+1:2d}: {len(signal[0])} samples, "
                  f"Channels: {len(signal)}, "
                  f"Timestamp: {neurosity_data['info']['startTime']}")
            
            # Simulate real-time delay
            if i < num_samples - 1:  # Don't delay after last sample
                time.sleep(delay_between_samples)
        
        return samples
    
    def save_samples(self, samples, filename="brainwave_samples.json"):
        """
        Save generated samples to JSON file
        """
        with open(filename, 'w') as f:
            json.dump(samples, f, indent=2)
        print(f"\n Saved {len(samples)} samples to {filename}")
    
    def print_sample_summary(self, samples):
        """
        Print a summary of the generated samples
        """
        print("\n Sample Summary:")
        print("-" * 40)
        
        for i, sample in enumerate(samples):
            data = np.array(sample['data'])
            print(f"Sample {i+1:2d}:")
            print(f"  • Shape: {data.shape}")
            print(f"  • Range: [{data.min():.2f}, {data.max():.2f}] μV")
            print(f"  • Mean: {data.mean():.2f} μV")
            print(f"  • Std: {data.std():.2f} μV")
            print(f"  • Timestamp: {sample['info']['startTime']}")
            print()

def main():
    """
    Main function to generate and display brainwave samples
    """
    print(" AI Brainwave Generator")
    print("=" * 50)
    
    # Initialize generator
    generator = BrainwaveGenerator()
    
    # Generate samples
    samples = generator.generate_samples(num_samples=10)
    
    # Print summary
    generator.print_sample_summary(samples)
    
    # Save to file
    generator.save_samples(samples)
    
    # Display first sample in detail
    print("\n First Sample Details:")
    print("=" * 50)
    first_sample = samples[0]
    print(json.dumps(first_sample, indent=2))
    
    print(f"\n Generated {len(samples)} realistic brainwave samples!")
    print(" Use these samples for testing your BCI applications")

if __name__ == "__main__":
    main()
