#!/usr/bin/env python3
"""
Neuro-Brainwave Data Generator Agent
====================================

An Agentic AI system for generating 100,000 realistic neuro-brainwave data samples
with complex behavioral patterns, temporal dependencies, and multi-modal characteristics.

Author: AI Engineering Team
Purpose: Generate training data for LSTM behavior forecasting model
"""

import numpy as np
import pandas as pd
import json
import time
import random
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from scipy import signal
from scipy.stats import skewnorm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class BrainwaveSample:
    """Individual brainwave data sample with metadata"""
    timestamp: float
    epoch_id: int
    session_id: str
    user_id: str
    brain_state: str
    eeg_channels: Dict[str, List[float]]  # 8 channels: CP3, CP4, C3, C4, F5, F6, PO3, PO4
    frequency_bands: Dict[str, float]  # Delta, Theta, Alpha, Beta, Gamma
    behavioral_markers: Dict[str, float]  # Attention, Stress, Fatigue, Arousal
    environmental_factors: Dict[str, float]  # Time_of_day, Activity_level, Social_context
    cognitive_load: float
    emotional_state: str
    task_complexity: float
    sample_rate: int = 250  # Hz
    duration: float = 4.0  # seconds per sample

class NeuroBrainwaveDataGeneratorAgent:
    """
    Agentic AI for generating realistic neuro-brainwave data samples
    with complex behavioral patterns and temporal dependencies
    """
    
    def __init__(self, target_samples: int = 100000):
        self.target_samples = target_samples
        self.generated_samples = 0
        self.session_counter = 0
        self.user_counter = 0
        
        # EEG Channel configuration (Neurosity format)
        self.eeg_channels = ['CP3', 'CP4', 'C3', 'C4', 'F5', 'F6', 'PO3', 'PO4']
        
        # Brain states with realistic characteristics
        self.brain_states = {
            'focused': {
                'beta_power': (0.7, 1.2),
                'alpha_power': (0.3, 0.6),
                'theta_power': (0.1, 0.4),
                'delta_power': (0.05, 0.3),
                'gamma_power': (0.4, 0.8),
                'attention': (0.8, 1.0),
                'stress': (0.3, 0.7),
                'fatigue': (0.1, 0.4),
                'arousal': (0.6, 0.9)
            },
            'relaxed': {
                'beta_power': (0.2, 0.5),
                'alpha_power': (0.6, 1.0),
                'theta_power': (0.4, 0.8),
                'delta_power': (0.2, 0.6),
                'gamma_power': (0.1, 0.4),
                'attention': (0.3, 0.6),
                'stress': (0.1, 0.3),
                'fatigue': (0.2, 0.5),
                'arousal': (0.3, 0.6)
            },
            'stressed': {
                'beta_power': (0.8, 1.4),
                'alpha_power': (0.2, 0.5),
                'theta_power': (0.3, 0.6),
                'delta_power': (0.1, 0.4),
                'gamma_power': (0.6, 1.2),
                'attention': (0.6, 0.9),
                'stress': (0.7, 1.0),
                'fatigue': (0.4, 0.8),
                'arousal': (0.7, 1.0)
            },
            'sleepy': {
                'beta_power': (0.1, 0.4),
                'alpha_power': (0.3, 0.7),
                'theta_power': (0.5, 1.0),
                'delta_power': (0.6, 1.2),
                'gamma_power': (0.05, 0.3),
                'attention': (0.1, 0.4),
                'stress': (0.2, 0.5),
                'fatigue': (0.7, 1.0),
                'arousal': (0.1, 0.4)
            },
            'excited': {
                'beta_power': (0.6, 1.1),
                'alpha_power': (0.3, 0.6),
                'theta_power': (0.2, 0.5),
                'delta_power': (0.1, 0.3),
                'gamma_power': (0.7, 1.3),
                'attention': (0.7, 1.0),
                'stress': (0.4, 0.8),
                'fatigue': (0.1, 0.3),
                'arousal': (0.8, 1.0)
            }
        }
        
        # Behavioral patterns for temporal modeling
        self.behavioral_patterns = {
            'circadian_rhythm': self._create_circadian_pattern(),
            'attention_cycles': self._create_attention_cycles(),
            'stress_accumulation': self._create_stress_patterns(),
            'fatigue_progression': self._create_fatigue_patterns(),
            'cognitive_load_variation': self._create_cognitive_patterns()
        }
        
        # User profiles for individual differences
        self.user_profiles = self._generate_user_profiles()
        
        logger.info(f"Neuro-Brainwave Data Generator Agent initialized for {target_samples:,} samples")
    
    def _create_circadian_pattern(self) -> np.ndarray:
        """Create 24-hour circadian rhythm pattern"""
        hours = np.linspace(0, 24, 1440)  # 1 minute resolution
        # Base circadian rhythm with morning peak and evening dip
        circadian = 0.5 + 0.3 * np.sin(2 * np.pi * (hours - 6) / 24)
        return circadian
    
    def _create_attention_cycles(self) -> np.ndarray:
        """Create 90-minute ultradian attention cycles"""
        minutes = np.linspace(0, 1440, 1440)  # 24 hours in minutes
        # 90-minute attention cycles with variability
        attention = 0.5 + 0.3 * np.sin(2 * np.pi * minutes / 90) + 0.1 * np.sin(2 * np.pi * minutes / 20)
        return np.clip(attention, 0, 1)
    
    def _create_stress_patterns(self) -> np.ndarray:
        """Create stress accumulation and recovery patterns"""
        minutes = np.linspace(0, 1440, 1440)
        # Stress builds up during day, peaks in afternoon, recovers in evening
        stress = 0.2 + 0.4 * np.sin(2 * np.pi * (minutes - 360) / 720) + 0.1 * np.random.normal(0, 1, len(minutes))
        return np.clip(stress, 0, 1)
    
    def _create_fatigue_patterns(self) -> np.ndarray:
        """Create fatigue progression throughout day"""
        minutes = np.linspace(0, 1440, 1440)
        # Fatigue increases linearly during day with evening peak
        fatigue = 0.1 + 0.6 * (minutes / 1440) + 0.2 * np.sin(2 * np.pi * minutes / 480)
        return np.clip(fatigue, 0, 1)
    
    def _create_cognitive_patterns(self) -> np.ndarray:
        """Create cognitive load variation patterns"""
        minutes = np.linspace(0, 1440, 1440)
        # Cognitive load varies with work hours and breaks
        cognitive = 0.3 + 0.4 * np.sin(2 * np.pi * (minutes - 480) / 480) + 0.2 * np.sin(2 * np.pi * minutes / 60)
        return np.clip(cognitive, 0, 1)
    
    def _generate_user_profiles(self) -> List[Dict]:
        """Generate diverse user profiles with individual characteristics"""
        profiles = []
        for i in range(100):  # 100 different user profiles
            profile = {
                'user_id': f'user_{i:03d}',
                'age': random.randint(18, 65),
                'gender': random.choice(['male', 'female', 'non-binary']),
                'baseline_attention': random.uniform(0.4, 0.8),
                'baseline_stress': random.uniform(0.2, 0.6),
                'baseline_fatigue': random.uniform(0.1, 0.5),
                'brain_state_preference': random.choice(list(self.brain_states.keys())),
                'circadian_shift': random.uniform(-2, 2),  # hours
                'stress_sensitivity': random.uniform(0.5, 1.5),
                'attention_span': random.uniform(0.3, 0.9),
                'cognitive_capacity': random.uniform(0.6, 1.2)
            }
            profiles.append(profile)
        return profiles
    
    def _generate_eeg_signal(self, brain_state: str, user_profile: Dict, timestamp: float) -> Dict[str, List[float]]:
        """Generate realistic EEG signal for all channels"""
        state_params = self.brain_states[brain_state]
        sample_rate = 250  # Hz
        duration = 4.0  # seconds
        n_samples = int(sample_rate * duration)
        time = np.linspace(0, duration, n_samples)
        
        eeg_data = {}
        
        for channel in self.eeg_channels:
            # Generate base signal with multiple frequency components
            signal_components = []
            
            # Delta waves (0.5-4 Hz)
            delta_freq = random.uniform(1, 3)
            delta_amp = random.uniform(*state_params['delta_power']) * user_profile['baseline_attention']
            signal_components.append(delta_amp * np.sin(2 * np.pi * delta_freq * time))
            
            # Theta waves (4-8 Hz)
            theta_freq = random.uniform(5, 7)
            theta_amp = random.uniform(*state_params['theta_power']) * user_profile['baseline_attention']
            signal_components.append(theta_amp * np.sin(2 * np.pi * theta_freq * time))
            
            # Alpha waves (8-13 Hz)
            alpha_freq = random.uniform(9, 12)
            alpha_amp = random.uniform(*state_params['alpha_power']) * user_profile['baseline_attention']
            signal_components.append(alpha_amp * np.sin(2 * np.pi * alpha_freq * time))
            
            # Beta waves (13-30 Hz)
            beta_freq = random.uniform(15, 25)
            beta_amp = random.uniform(*state_params['beta_power']) * user_profile['baseline_attention']
            signal_components.append(beta_amp * np.sin(2 * np.pi * beta_freq * time))
            
            # Gamma waves (30-45 Hz)
            gamma_freq = random.uniform(32, 40)
            gamma_amp = random.uniform(*state_params['gamma_power']) * user_profile['baseline_attention']
            signal_components.append(gamma_amp * np.sin(2 * np.pi * gamma_freq * time))
            
            # Combine components
            base_signal = np.sum(signal_components, axis=0)
            
            # Add noise and artifacts
            noise = np.random.normal(0, 0.1, n_samples)
            muscle_artifact = np.random.normal(0, 0.05, n_samples) if random.random() < 0.1 else 0
            eye_blink = np.random.normal(0, 0.3, n_samples) if random.random() < 0.05 else 0
            
            # Apply channel-specific characteristics
            channel_modifiers = {
                'CP3': 1.0, 'CP4': 1.0,  # Motor planning - balanced
                'C3': 1.1, 'C4': 1.1,   # Motor cortex - slightly higher amplitude
                'F5': 0.9, 'F6': 0.9,   # Frontal - more variable
                'PO3': 0.8, 'PO4': 0.8  # Occipital - lower amplitude
            }
            
            final_signal = (base_signal + noise + muscle_artifact + eye_blink) * channel_modifiers[channel]
            
            # Apply bandpass filter (0.5-50 Hz)
            nyquist = sample_rate / 2
            low = 0.5 / nyquist
            high = 50 / nyquist
            b, a = signal.butter(4, [low, high], btype='band')
            filtered_signal = signal.filtfilt(b, a, final_signal)
            
            eeg_data[channel] = filtered_signal.tolist()
        
        return eeg_data
    
    def _calculate_frequency_bands(self, eeg_data: Dict[str, List[float]]) -> Dict[str, float]:
        """Calculate frequency band powers from EEG data"""
        sample_rate = 250
        duration = 4.0
        n_samples = len(list(eeg_data.values())[0])
        
        # Calculate power for each frequency band
        frequency_bands = {'delta': 0, 'theta': 0, 'alpha': 0, 'beta': 0, 'gamma': 0}
        
        for channel, signal_data in eeg_data.items():
            # Apply FFT
            freqs = np.fft.fftfreq(n_samples, 1/sample_rate)
            fft_signal = np.fft.fft(signal_data)
            power_spectrum = np.abs(fft_signal) ** 2
            
            # Calculate power in each band
            for band, (low_freq, high_freq) in [
                ('delta', (0.5, 4)), ('theta', (4, 8)), ('alpha', (8, 13)),
                ('beta', (13, 30)), ('gamma', (30, 50))
            ]:
                band_mask = (freqs >= low_freq) & (freqs <= high_freq)
                band_power = np.mean(power_spectrum[band_mask])
                frequency_bands[band] += band_power
        
        # Average across channels
        for band in frequency_bands:
            frequency_bands[band] /= len(eeg_data)
        
        return frequency_bands
    
    def _generate_behavioral_markers(self, brain_state: str, user_profile: Dict, timestamp: float) -> Dict[str, float]:
        """Generate realistic behavioral markers"""
        state_params = self.brain_states[brain_state]
        
        # Time-based adjustments
        hour = (timestamp / 3600) % 24
        minute = (timestamp / 60) % 60
        
        # Get time-based patterns
        circadian_idx = int((hour * 60 + minute) % 1440)
        
        # Base values from brain state
        attention = random.uniform(*state_params['attention'])
        stress = random.uniform(*state_params['stress'])
        fatigue = random.uniform(*state_params['fatigue'])
        arousal = random.uniform(*state_params['arousal'])
        
        # Apply circadian and user-specific adjustments
        attention *= self.behavioral_patterns['circadian_rhythm'][circadian_idx]
        attention *= user_profile['baseline_attention']
        
        stress *= self.behavioral_patterns['stress_accumulation'][circadian_idx]
        stress *= user_profile['stress_sensitivity']
        
        fatigue *= self.behavioral_patterns['fatigue_progression'][circadian_idx]
        fatigue *= (1 + user_profile['baseline_fatigue'])
        
        arousal *= self.behavioral_patterns['attention_cycles'][circadian_idx]
        
        return {
            'attention': np.clip(attention, 0, 1),
            'stress': np.clip(stress, 0, 1),
            'fatigue': np.clip(fatigue, 0, 1),
            'arousal': np.clip(arousal, 0, 1)
        }
    
    def _generate_environmental_factors(self, timestamp: float) -> Dict[str, float]:
        """Generate environmental context factors"""
        hour = (timestamp / 3600) % 24
        
        # Time of day (0-1, normalized)
        time_of_day = (np.sin(2 * np.pi * (hour - 6) / 24) + 1) / 2
        
        # Activity level (higher during day)
        activity_level = 0.3 + 0.7 * (np.sin(2 * np.pi * (hour - 8) / 16) + 1) / 2
        
        # Social context (work hours vs personal time)
        social_context = 0.8 if 9 <= hour <= 17 else 0.3
        
        return {
            'time_of_day': np.clip(time_of_day, 0, 1),
            'activity_level': np.clip(activity_level, 0, 1),
            'social_context': np.clip(social_context, 0, 1)
        }
    
    def generate_sample(self, timestamp: float) -> BrainwaveSample:
        """Generate a single brainwave data sample"""
        # Select user profile
        user_profile = random.choice(self.user_profiles)
        
        # Determine brain state based on time and user characteristics
        brain_state = self._determine_brain_state(timestamp, user_profile)
        
        # Generate EEG signal
        eeg_channels = self._generate_eeg_signal(brain_state, user_profile, timestamp)
        
        # Calculate frequency bands
        frequency_bands = self._calculate_frequency_bands(eeg_channels)
        
        # Generate behavioral markers
        behavioral_markers = self._generate_behavioral_markers(brain_state, user_profile, timestamp)
        
        # Generate environmental factors
        environmental_factors = self._generate_environmental_factors(timestamp)
        
        # Calculate cognitive load and emotional state
        cognitive_load = self._calculate_cognitive_load(behavioral_markers, environmental_factors)
        emotional_state = self._determine_emotional_state(behavioral_markers, brain_state)
        task_complexity = self._calculate_task_complexity(brain_state, cognitive_load)
        
        # Create session and epoch IDs
        session_id = f"session_{self.session_counter:06d}"
        epoch_id = self.generated_samples
        
        sample = BrainwaveSample(
            timestamp=timestamp,
            epoch_id=epoch_id,
            session_id=session_id,
            user_id=user_profile['user_id'],
            brain_state=brain_state,
            eeg_channels=eeg_channels,
            frequency_bands=frequency_bands,
            behavioral_markers=behavioral_markers,
            environmental_factors=environmental_factors,
            cognitive_load=cognitive_load,
            emotional_state=emotional_state,
            task_complexity=task_complexity
        )
        
        return sample
    
    def _determine_brain_state(self, timestamp: float, user_profile: Dict) -> str:
        """Determine brain state based on time, patterns, and user characteristics"""
        hour = (timestamp / 3600) % 24
        minute = (timestamp / 60) % 60
        
        # Base state probabilities by time of day
        if 22 <= hour or hour <= 6:  # Night/sleep hours
            state_probs = {'sleepy': 0.7, 'relaxed': 0.2, 'stressed': 0.05, 'focused': 0.03, 'excited': 0.02}
        elif 7 <= hour <= 9:  # Morning
            state_probs = {'focused': 0.4, 'excited': 0.3, 'relaxed': 0.2, 'stressed': 0.05, 'sleepy': 0.05}
        elif 10 <= hour <= 17:  # Work hours
            state_probs = {'focused': 0.5, 'stressed': 0.2, 'relaxed': 0.15, 'excited': 0.1, 'sleepy': 0.05}
        elif 18 <= hour <= 21:  # Evening
            state_probs = {'relaxed': 0.4, 'focused': 0.3, 'excited': 0.15, 'stressed': 0.1, 'sleepy': 0.05}
        else:
            state_probs = {'relaxed': 0.3, 'focused': 0.3, 'stressed': 0.2, 'excited': 0.1, 'sleepy': 0.1}
        
        # Apply user preferences
        preferred_state = user_profile['brain_state_preference']
        state_probs[preferred_state] *= 1.5
        
        # Normalize probabilities
        total_prob = sum(state_probs.values())
        state_probs = {k: v/total_prob for k, v in state_probs.items()}
        
        # Sample state
        states = list(state_probs.keys())
        probs = list(state_probs.values())
        return np.random.choice(states, p=probs)
    
    def _calculate_cognitive_load(self, behavioral_markers: Dict, environmental_factors: Dict) -> float:
        """Calculate cognitive load based on behavioral and environmental factors"""
        base_load = (behavioral_markers['attention'] + behavioral_markers['stress']) / 2
        environmental_modifier = environmental_factors['activity_level'] * environmental_factors['social_context']
        cognitive_load = base_load * (0.7 + 0.3 * environmental_modifier)
        return np.clip(cognitive_load, 0, 1)
    
    def _determine_emotional_state(self, behavioral_markers: Dict, brain_state: str) -> str:
        """Determine emotional state based on behavioral markers and brain state"""
        stress = behavioral_markers['stress']
        arousal = behavioral_markers['arousal']
        
        if stress > 0.7 and arousal > 0.6:
            return 'anxious'
        elif stress > 0.6 and arousal < 0.4:
            return 'depressed'
        elif stress < 0.3 and arousal > 0.7:
            return 'excited'
        elif stress < 0.4 and arousal < 0.5:
            return 'calm'
        elif brain_state == 'focused':
            return 'concentrated'
        else:
            return 'neutral'
    
    def _calculate_task_complexity(self, brain_state: str, cognitive_load: float) -> float:
        """Calculate task complexity based on brain state and cognitive load"""
        complexity_map = {
            'focused': 0.8,
            'excited': 0.6,
            'stressed': 0.9,
            'relaxed': 0.3,
            'sleepy': 0.1
        }
        base_complexity = complexity_map.get(brain_state, 0.5)
        return np.clip(base_complexity * cognitive_load, 0, 1)
    
    def generate_dataset(self, output_file: str = "neuro_brainwave_dataset.jsonl") -> str:
        """Generate complete dataset of brainwave samples"""
        logger.info(f"Starting generation of {self.target_samples:,} brainwave samples...")
        
        # Start timestamp (one month of data)
        start_time = time.time() - (30 * 24 * 3600)  # 30 days ago
        current_time = start_time
        
        with open(output_file, 'w') as f:
            for i in range(self.target_samples):
                if i % 10000 == 0:
                    logger.info(f"Generated {i:,}/{self.target_samples:,} samples ({i/self.target_samples*100:.1f}%)")
                
                # Generate sample
                sample = self.generate_sample(current_time)
                
                # Write to file as JSONL
                sample_dict = asdict(sample)
                f.write(json.dumps(sample_dict) + '\n')
                
                # Update counters
                self.generated_samples += 1
                current_time += 4.0  # 4-second intervals
                
                # Create new session every 1000 samples
                if i % 1000 == 0:
                    self.session_counter += 1
        
        logger.info(f"Dataset generation complete! Generated {self.generated_samples:,} samples")
        logger.info(f"Output file: {output_file}")
        
        return output_file

def main():
    """Main function to run the data generator"""
    print(" Neuro-Brainwave Data Generator Agent")
    print("=" * 50)
    
    # Initialize generator
    generator = NeuroBrainwaveDataGeneratorAgent(target_samples=100000)
    
    # Generate dataset
    output_file = generator.generate_dataset()
    
    print(f"\n Dataset generation complete!")
    print(f" Output file: {output_file}")
    print(f" Total samples: {generator.generated_samples:,}")
    print(f"  Time span: 30 days of continuous data")
    print(f" Users: {len(generator.user_profiles)} unique profiles")
    print(f" Brain states: {len(generator.brain_states)} different states")

if __name__ == "__main__":
    main()
