#!/usr/bin/env python3
"""
AI Agent for Dynamic Brainwave Generation
Generates brainwave data with different brain characteristics and flexible duration
Simulates various brain states with realistic EEG patterns
"""

import numpy as np
import time
import json
from datetime import datetime
import random
import matplotlib.pyplot as plt

class DynamicBrainwaveGenerator:
    def __init__(self, sampling_rate=256, num_channels=8):
        self.sampling_rate = sampling_rate
        self.num_channels = num_channels
        self.epoch_size = 16  # Samples per epoch
        self.epoch_duration = self.epoch_size / sampling_rate  # ~62.5ms
        
        # Channel names matching Neurosity device
        self.channel_names = ['CP3', 'C3', 'F5', 'PO3', 'PO4', 'F6', 'C4', 'CP4']
        
        # Initialize time vector for current epoch
        self.time_vector = np.linspace(0, self.epoch_duration, self.epoch_size)
        
        # Define different brain state characteristics with updated amplitude ranges
        self.brain_states = {
            'normal': {
                'name': 'Normal Baseline',
                'description': 'Awake, relaxed, eyes closed',
                'frequency_bands': {
                    'delta': (0.5, 4, 5, 20),      # (low_freq, high_freq, min_amp, max_amp)
                    'theta': (4, 8, 8, 25),
                    'alpha': (8, 13, 5, 25),       # Dominant
                    'beta': (13, 30, 25, 60),
                    'gamma': (30, 45, 10, 30)
                },
                'noise_level': (1, 3),
                'artifact_prob': {'muscle': 0.05, 'eye': 0.08, 'drift': 0.1},
                'baseline_drift': (-5, 5)
            },
            'excited': {
                'name': 'Excited State',
                'description': 'High arousal, increased cognitive activity',
                'frequency_bands': {
                    'delta': (0.5, 4, 5, 20),
                    'theta': (4, 8, 8, 25),
                    'alpha': (8, 13, 5, 25),        # Reduced
                    'beta': (13, 30, 25, 60),       # Increased
                    'gamma': (30, 45, 10, 30)       # Increased
                },
                'noise_level': (2, 4),
                'artifact_prob': {'muscle': 0.15, 'eye': 0.12, 'drift': 0.2},
                'baseline_drift': (-8, 8)
            },
            'relaxed': {
                'name': 'Deep Relaxation',
                'description': 'Meditation, deep relaxation, eyes closed',
                'frequency_bands': {
                    'delta': (0.5, 4, 15, 40),      # Increased for deep relaxation
                    'theta': (4, 8, 10, 35),        # Increased
                    'alpha': (8, 13, 30, 80),       # Very dominant
                    'beta': (13, 30, 2, 15),        # Reduced
                    'gamma': (30, 45, 1, 3)         # Reduced
                },
                'noise_level': (0.5, 2),
                'artifact_prob': {'muscle': 0.02, 'eye': 0.03, 'drift': 0.05},
                'baseline_drift': (-3, 3)
            },
            'focused': {
                'name': 'Focused Attention',
                'description': 'Concentrated mental effort, problem-solving',
                'frequency_bands': {
                    'delta': (0.5, 4, 8, 25),       # Moderate
                    'theta': (4, 8, 5, 20),         # Moderate
                    'alpha': (8, 13, 10, 30),       # Reduced
                    'beta': (13, 30, 20, 50),       # Very increased
                    'gamma': (30, 45, 5, 20)        # Increased
                },
                'noise_level': (1.5, 3.5),
                'artifact_prob': {'muscle': 0.08, 'eye': 0.06, 'drift': 0.12},
                'baseline_drift': (-6, 6)
            },
            'stressed': {
                'name': 'Stress/Anxiety',
                'description': 'High stress, anxiety, overthinking',
                'frequency_bands': {
                    'delta': (0.5, 4, 5, 20),
                    'theta': (4, 8, 8, 25),
                    'alpha': (8, 13, 5, 25),        # Reduced
                    'beta': (13, 30, 25, 60),       # Very increased
                    'gamma': (30, 45, 10, 30)       # Very increased
                },
                'noise_level': (3, 6),
                'artifact_prob': {'muscle': 0.25, 'eye': 0.15, 'drift': 0.3},
                'baseline_drift': (-10, 10)
            },
            'sleepy': {
                'name': 'Drowsy/Sleepy',
                'description': 'Drowsiness, microsleeps, fatigue',
                'frequency_bands': {
                    'delta': (0.5, 4, 20, 60),      # Increased
                    'theta': (4, 8, 15, 45),        # Increased
                    'alpha': (8, 13, 10, 40),       # Variable
                    'beta': (13, 30, 3, 15),        # Reduced
                    'gamma': (30, 45, 1, 5)         # Reduced
                },
                'noise_level': (2, 4),
                'artifact_prob': {'muscle': 0.1, 'eye': 0.2, 'drift': 0.15},
                'baseline_drift': (-8, 8)
            }
        }
        
        # Define frequency band colors for visualization
        self.frequency_band_colors = {
            'delta': '#FF6B6B',    # Red
            'theta': '#4ECDC4',    # Teal
            'alpha': '#45B7D1',    # Blue
            'beta': '#96CEB4',     # Green
            'gamma': '#FFEAA7'     # Yellow
        }
        
        # Standard frequency band definitions
        self.standard_frequency_bands = {
            'delta': (0.5, 4, 5, 20),      # Hz, μV
            'theta': (4, 8, 8, 25),        # Hz, μV
            'alpha': (8, 13, 5, 25),       # Hz, μV
            'beta': (13, 30, 25, 60),      # Hz, μV
            'gamma': (30, 45, 10, 30)      # Hz, μV
        }
    
    def analyze_frequency_bands(self, signal_data):
        """
        Analyze the signal data to determine which frequency bands are active
        Returns a dictionary with band names and their activity levels
        """
        from scipy import signal
        
        # Ensure signal_data is 2D (channels x samples)
        if signal_data.ndim == 1:
            signal_data = signal_data.reshape(1, -1)
        
        # Calculate frequency band activities for each channel
        channel_activities = {}
        
        for ch in range(signal_data.shape[0]):
            channel_signal = signal_data[ch, :]
            
            # Apply FFT to get frequency spectrum
            freqs = np.fft.fftfreq(len(channel_signal), 1/self.sampling_rate)
            fft_signal = np.fft.fft(channel_signal)
            power_spectrum = np.abs(fft_signal) ** 2
            
            # Keep only positive frequencies
            positive_freqs = freqs[:len(freqs)//2]
            positive_power = power_spectrum[:len(power_spectrum)//2]
            
            # Analyze each frequency band
            band_activities = {}
            for band_name, (low_freq, high_freq, min_amp, max_amp) in self.standard_frequency_bands.items():
                # Find frequency indices for this band
                band_mask = (positive_freqs >= low_freq) & (positive_freqs <= high_freq)
                
                if np.any(band_mask):
                    # Calculate average power in this band
                    band_power = np.mean(positive_power[band_mask])
                    band_amplitude = np.sqrt(band_power)  # Convert power to amplitude
                    
                    # Normalize to μV range
                    normalized_amplitude = band_amplitude * 1000  # Rough conversion
                    
                    # Determine if this band is active based on amplitude thresholds
                    is_active = min_amp <= normalized_amplitude <= max_amp
                    activity_level = min(1.0, normalized_amplitude / max_amp) if max_amp > 0 else 0
                    
                    band_activities[band_name] = {
                        'amplitude': normalized_amplitude,
                        'is_active': is_active,
                        'activity_level': activity_level,
                        'frequency_range': (low_freq, high_freq)
                    }
                else:
                    band_activities[band_name] = {
                        'amplitude': 0,
                        'is_active': False,
                        'activity_level': 0,
                        'frequency_range': (low_freq, high_freq)
                    }
            
            channel_activities[f'channel_{ch}'] = band_activities
        
        return channel_activities
    
    def get_dominant_frequency_bands(self, signal_data, threshold=0.5, brain_state='normal', activities=None):
        """
        Get the dominant frequency bands for a signal based on brain state
        Returns list of band names that are most active

        Pass ``activities`` from a prior ``analyze_frequency_bands`` call to avoid a duplicate FFT.
        """
        if activities is None:
            activities = self.analyze_frequency_bands(signal_data)
        
        # Get brain state specific characteristics
        if brain_state in self.brain_states:
            state_config = self.brain_states[brain_state]
            state_frequency_bands = state_config['frequency_bands']
        else:
            state_frequency_bands = self.standard_frequency_bands
        
        # Average activity across all channels
        band_totals = {}
        for band_name in self.standard_frequency_bands.keys():
            total_activity = 0
            channel_count = 0
            
            for channel_name, channel_activities in activities.items():
                if band_name in channel_activities:
                    total_activity += channel_activities[band_name]['activity_level']
                    channel_count += 1
            
            if channel_count > 0:
                band_totals[band_name] = total_activity / channel_count
        
        # Adjust threshold based on brain state characteristics
        state_adjusted_threshold = threshold
        if brain_state in self.brain_states:
            # For states with higher amplitude ranges, lower the threshold
            max_amplitude = max([amp_range[3] for amp_range in state_frequency_bands.values()])
            if max_amplitude > 40:  # High activity states
                state_adjusted_threshold = threshold * 0.7  # Lower threshold for high activity
            elif max_amplitude < 20:  # Low activity states
                state_adjusted_threshold = threshold * 1.3  # Higher threshold for low activity
        
        # Return bands above threshold, sorted by activity level
        dominant_bands = [(band, activity) for band, activity in band_totals.items() if activity >= state_adjusted_threshold]
        dominant_bands.sort(key=lambda x: x[1], reverse=True)
        
        # Limit to top 3 most active bands
        return [band for band, _ in dominant_bands[:3]]
        
    def generate_brain_state_events(self, total_duration=60.0, brain_state='normal', event_variability=0.3):
        """
        Generate random brain state events throughout the specified duration
        """
        events = []
        current_time = 0.0
        
        while current_time < total_duration:
            # Random duration for this brain state event (2 to 10 seconds)
            event_duration = random.uniform(2.0, 10.0)
            
            # Ensure we don't exceed total duration
            if current_time + event_duration > total_duration:
                event_duration = total_duration - current_time
            
            # Add some variability to the brain state
            state_variation = random.uniform(1.0 - event_variability, 1.0 + event_variability)
            state_variation = max(0.1, min(2.0, state_variation))  # Clamp between 0.1 and 2.0
            
            events.append((current_time, event_duration, brain_state, state_variation))
            current_time += event_duration
            
            # Add random transitions to other states (optional)
            if current_time < total_duration and random.random() < 0.3:  # 30% chance
                transition_duration = random.uniform(1.0, 3.0)
                if current_time + transition_duration > total_duration:
                    transition_duration = total_duration - current_time
                
                # Random transition state
                transition_states = [s for s in self.brain_states.keys() if s != brain_state]
                transition_state = random.choice(transition_states)
                transition_variation = random.uniform(0.5, 1.5)
                
                events.append((current_time, transition_duration, transition_state, transition_variation))
                current_time += transition_duration
        
        return events
    
    def get_brain_state_at_time(self, current_time, brain_state_events):
        """
        Get the brain state and variation at a specific time
        """
        for start_time, duration, brain_state, variation in brain_state_events:
            if start_time <= current_time < start_time + duration:
                return brain_state, variation
        return 'normal', 1.0  # Default
    
    def generate_brainwave_eeg(self, epoch_num, total_duration=60.0, start_time=0.0, brain_state_events=None):
        """
        Generate brainwave EEG data for one epoch based on brain state
        """
        # Calculate current time
        current_time = start_time + (epoch_num * self.epoch_duration)
        
        # Get brain state for this time
        if brain_state_events is None:
            brain_state, state_variation = 'normal', 1.0
        else:
            brain_state, state_variation = self.get_brain_state_at_time(current_time, brain_state_events)
        
        # Get brain state characteristics
        state_config = self.brain_states[brain_state]
        
        # Base signal (combination of all frequency bands)
        signal = np.zeros((self.num_channels, self.epoch_size))
        
        for ch in range(self.num_channels):
            # Generate different frequency components based on brain state
            for band_name, (low_freq, high_freq, min_amp, max_amp) in state_config['frequency_bands'].items():
                # Random frequency within the band
                freq = np.random.uniform(low_freq, high_freq)
                
                # Scale amplitude by state variation
                base_amplitude = np.random.uniform(min_amp, max_amp)
                amplitude = base_amplitude * state_variation
                
                # Add some phase variation
                phase = np.random.uniform(0, 2 * np.pi)
                
                # Generate sinusoidal component
                component = amplitude * np.sin(2 * np.pi * freq * self.time_vector + phase)
                
                # Add some noise (realistic EEG noise)
                noise_level = np.random.uniform(*state_config['noise_level'])
                noise = np.random.normal(0, noise_level, self.epoch_size)
                
                signal[ch] += component + noise
        
        # Add brain state specific artifacts
        signal = self._add_brain_state_artifacts(signal, epoch_num, current_time, brain_state, state_variation)
        
        return signal
    
    def _add_brain_state_artifacts(self, signal, epoch_num, current_time, brain_state, state_variation):
        """
        Add brain state specific artifacts
        """
        state_config = self.brain_states[brain_state]
        
        # Scale artifact probabilities by state variation
        muscle_prob = state_config['artifact_prob']['muscle'] * state_variation
        eye_prob = state_config['artifact_prob']['eye'] * state_variation
        drift_prob = state_config['artifact_prob']['drift'] * state_variation
        
        # Muscle tension artifacts
        if random.random() < muscle_prob:
            muscle_channels = [0, 1, 6, 7]  # CP3, C3, C4, CP4
            for ch in muscle_channels:
                muscle_amplitude = np.random.uniform(20, 60 + state_variation * 20)
                muscle_duration = random.randint(3, 6 + int(state_variation * 2))
                start_idx = random.randint(0, self.epoch_size - muscle_duration)
                signal[ch, start_idx:start_idx+muscle_duration] += muscle_amplitude
        
        # Eye movement artifacts
        if random.random() < eye_prob:
            eye_channels = [2, 5]  # F5, F6 (frontal channels)
            for ch in eye_channels:
                eye_amplitude = np.random.uniform(30, 70 + state_variation * 20)
                eye_duration = random.randint(2, 4 + int(state_variation))
                start_idx = random.randint(0, self.epoch_size - eye_duration)
                signal[ch, start_idx:start_idx+eye_duration] += eye_amplitude
        
        # Baseline drift and micro-movements
        for ch in range(self.num_channels):
            if random.random() < drift_prob:
                drift_amplitude = np.random.uniform(*state_config['baseline_drift'])
                drift_amplitude *= state_variation
                drift = np.linspace(0, drift_amplitude, self.epoch_size)
                signal[ch] += drift
            
            # Add micro-movements
            micro_movements = np.random.normal(0, 1 + state_variation, self.epoch_size)
            signal[ch] += micro_movements
        
        return signal
    
    def format_neurosity_data(self, signal, epoch_num, start_time, brain_state, state_variation):
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
                "brainState": brain_state,
                "stateVariation": state_variation,
                "startTimeOffset": start_time
            }
        }
        
        return neurosity_data
    
    def generate_brainwave_session(self, duration=60.0, brain_state='normal', start_time_range=(0, 65), event_variability=0.3):
        """
        Generate a complete brainwave session with specified characteristics
        """
        # Random start time within the specified range
        start_time = random.uniform(*start_time_range)
        
        # Generate brain state events
        brain_state_events = self.generate_brain_state_events(duration, brain_state, event_variability)
        
        # Calculate number of epochs needed
        num_epochs = int(duration / self.epoch_duration)
        
        print(f" Generating {self.brain_states[brain_state]['name']} brainwave session...")
        print(f"  Duration: {duration} seconds")
        print(f" Start time: {start_time:.2f} seconds")
        print(f" Sampling rate: {self.sampling_rate} Hz")
        print(f" Channels: {', '.join(self.channel_names)}")
        print(f" Total epochs: {num_epochs}")
        print(f" Brain state events: {len(brain_state_events)}")
        print(f" Description: {self.brain_states[brain_state]['description']}")
        print("-" * 70)
        
        # Print brain state events
        for i, (event_start, event_duration, event_state, event_variation) in enumerate(brain_state_events):
            print(f"Event {i+1}: {event_start:.1f}s - {event_start+event_duration:.1f}s "
                  f"({event_state}, var: {event_variation:.2f})")
        
        print("-" * 70)
        
        samples = []
        
        for epoch_num in range(num_epochs):
            # Generate brainwave EEG data
            signal = self.generate_brainwave_eeg(epoch_num, duration, start_time, brain_state_events)
            
            # Get current brain state
            current_time = start_time + (epoch_num * self.epoch_duration)
            current_brain_state, state_variation = self.get_brain_state_at_time(current_time, brain_state_events)
            
            # Format as Neurosity data
            neurosity_data = self.format_neurosity_data(signal, epoch_num, start_time, current_brain_state, state_variation)
            
            # Add to samples list
            samples.append(neurosity_data)
            
            # Print progress (every 20 epochs)
            if epoch_num % 20 == 0:
                print(f"Epoch {epoch_num+1:3d}: {len(signal[0])} samples, "
                      f"Time: {current_time:.2f}s, "
                      f"State: {current_brain_state}, "
                      f"Var: {state_variation:.2f}")
        
        return samples, brain_state_events
    
    def save_brainwave_samples(self, samples, filename="brainwave_samples.json"):
        """
        Save generated brainwave samples to JSON file
        """
        with open(filename, 'w') as f:
            json.dump(samples, f, indent=2)
        print(f"\n Saved {len(samples)} brainwave samples to {filename}")
    
    def create_brainwave_visualization(self, samples, brain_state_events):
        """
        Create visualization of the brainwave session
        """
        # Extract data
        data_matrix = np.array([sample['data'] for sample in samples])
        
        # Create time vector
        start_time = samples[0]['info']['startTimeOffset']
        total_duration = len(samples) * self.epoch_duration
        time_vector = np.linspace(start_time, start_time + total_duration, len(samples) * self.epoch_size)
        
        # Flatten data for visualization
        flat_data = data_matrix.reshape(-1, self.num_channels)
        
        # Create visualization
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        fig.suptitle(' Dynamic Brainwave Session', fontsize=16, fontweight='bold')
        
        # Plot 1: All channels overlaid
        ax1 = axes[0]
        for ch in range(self.num_channels):
            ax1.plot(time_vector, flat_data[:, ch], 
                    label=self.channel_names[ch], linewidth=1, alpha=0.8)
        ax1.set_title('All Channels - Brainwave Activity', fontweight='bold')
        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel('Amplitude (μV)')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Brain state events timeline
        ax2 = axes[1]
        colors = {'normal': 'green', 'excited': 'red', 'relaxed': 'blue', 
                 'focused': 'orange', 'stressed': 'purple', 'sleepy': 'brown'}
        
        for event_start, event_duration, event_state, event_variation in brain_state_events:
            color = colors.get(event_state, 'gray')
            ax2.barh(0, event_duration, left=event_start, height=0.5, 
                    color=color, alpha=0.7, label=f'{event_state}: {event_variation:.2f}')
        ax2.set_title('Brain State Events Timeline', fontweight='bold')
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel('Events')
        ax2.set_ylim(-0.5, 0.5)
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: State variation over time
        ax3 = axes[2]
        state_variations = []
        for i in range(len(samples)):
            current_time = start_time + (i * self.epoch_duration)
            _, state_variation = self.get_brain_state_at_time(current_time, brain_state_events)
            state_variations.extend([state_variation] * self.epoch_size)
        
        ax3.plot(time_vector, state_variations, 'r-', linewidth=2, label='State Variation')
        ax3.set_title('Brain State Variation Over Time', fontweight='bold')
        ax3.set_xlabel('Time (seconds)')
        ax3.set_ylabel('Variation Factor')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        plt.tight_layout()
        plt.savefig('brainwave_session.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f" Visualization saved as: brainwave_session.png")

def main():
    """
    Main function to generate brainwave session
    """
    print(" AI Dynamic Brainwave Generator")
    print("=" * 70)
    
    # Initialize brainwave generator
    generator = DynamicBrainwaveGenerator()
    
    # Display available brain states
    print(" Available Brain States:")
    for state, config in generator.brain_states.items():
        print(f"  • {state}: {config['name']} - {config['description']}")
    
    print("\n  Available Durations:")
    print("  • 60 seconds (1 minute)")
    print("  • 120 seconds (2 minutes)")
    print("  • 300 seconds (5 minutes)")
    print("  • 600 seconds (10 minutes)")
    print("  • Custom duration")
    
    # Example usage - you can modify these parameters
    brain_state = 'excited'  # Change to: normal, excited, relaxed, focused, stressed, sleepy
    duration = 120.0  # Change to: 60, 120, 300, 600, or any custom value
    
    print(f"\n Selected Configuration:")
    print(f"  • Brain State: {brain_state}")
    print(f"  • Duration: {duration} seconds")
    print("=" * 70)
    
    # Generate brainwave session
    samples, brain_state_events = generator.generate_brainwave_session(
        duration=duration, 
        brain_state=brain_state, 
        start_time_range=(0, 65),
        event_variability=0.3
    )
    
    # Save to file
    generator.save_brainwave_samples(samples)
    
    # Create visualization
    generator.create_brainwave_visualization(samples, brain_state_events)
    
    # Print summary
    print(f"\n BRAINWAVE SESSION SUMMARY:")
    print(f"• Total samples: {len(samples)}")
    print(f"• Duration: {duration} seconds")
    print(f"• Start time: {samples[0]['info']['startTimeOffset']:.2f} seconds")
    print(f"• End time: {samples[0]['info']['startTimeOffset'] + duration:.2f} seconds")
    print(f"• Brain state events: {len(brain_state_events)}")
    print(f"• Average state variation: {np.mean([event[3] for event in brain_state_events]):.2f}")
    print(f"• Max state variation: {max([event[3] for event in brain_state_events]):.2f}")
    
    print(f"\n Generated dynamic brainwave session!")
    print(" Perfect for testing BCI applications with various brain states")

if __name__ == "__main__":
    main()
