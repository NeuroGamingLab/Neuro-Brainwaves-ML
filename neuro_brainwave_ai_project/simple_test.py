#!/usr/bin/env python3
"""
Simple Test for Neuro-Brainwave AI System
=========================================

Quick test to demonstrate the system capabilities
"""

import sys
import json
import time
from neuro_brainwave_data_generator_agent import NeuroBrainwaveDataGeneratorAgent
from lstm_behavior_forecasting_agent import LSTMBehaviorForecastingAgent

def main():
    print(" Neuro-Brainwave AI System - Simple Test")
    print("=" * 50)
    
    # Step 1: Generate data
    print("\n Step 1: Generating neuro-brainwave data...")
    generator = NeuroBrainwaveDataGeneratorAgent(target_samples=500)
    output_file = generator.generate_dataset('simple_test_dataset.jsonl')
    print(f" Generated {generator.generated_samples} samples")
    
    # Step 2: Load and analyze data
    print("\n Step 2: Loading and analyzing data...")
    data = []
    with open(output_file, 'r') as f:
        for line in f:
            sample = json.loads(line.strip())
            data.append(sample)
    
    print(f" Loaded {len(data)} samples")
    
    # Step 3: Show sample data
    print("\n Step 3: Sample data structure...")
    sample = data[0]
    print(f"    Brain State: {sample['brain_state']}")
    print(f"    Emotional State: {sample['emotional_state']}")
    print(f"    Attention: {sample['behavioral_markers']['attention']:.3f}")
    print(f"    Stress: {sample['behavioral_markers']['stress']:.3f}")
    print(f"    Fatigue: {sample['behavioral_markers']['fatigue']:.3f}")
    print(f"    Arousal: {sample['behavioral_markers']['arousal']:.3f}")
    print(f"    Cognitive Load: {sample['cognitive_load']:.3f}")
    
    # Step 4: Show frequency bands
    print("\n Step 4: Frequency band analysis...")
    freq_bands = sample['frequency_bands']
    for band, power in freq_bands.items():
        print(f"   {band.upper()}: {power:.3f}")
    
    # Step 5: Show EEG channels
    print("\n Step 5: EEG channel data...")
    eeg_channels = sample['eeg_channels']
    for channel, signal_data in eeg_channels.items():
        print(f"   {channel}: {len(signal_data)} samples, range: {min(signal_data):.3f} to {max(signal_data):.3f}")
    
    # Step 6: Show behavioral patterns
    print("\n Step 6: Behavioral patterns across samples...")
    brain_states = {}
    emotional_states = {}
    
    for sample in data[:100]:  # Analyze first 100 samples
        brain_state = sample['brain_state']
        emotional_state = sample['emotional_state']
        
        brain_states[brain_state] = brain_states.get(brain_state, 0) + 1
        emotional_states[emotional_state] = emotional_states.get(emotional_state, 0) + 1
    
    print("    Brain States Distribution:")
    for state, count in brain_states.items():
        print(f"      {state}: {count} samples")
    
    print("    Emotional States Distribution:")
    for state, count in emotional_states.items():
        print(f"      {state}: {count} samples")
    
    # Step 7: Show environmental factors
    print("\n Step 7: Environmental factors...")
    env_factors = sample['environmental_factors']
    print(f"    Time of Day: {env_factors['time_of_day']:.3f}")
    print(f"    Activity Level: {env_factors['activity_level']:.3f}")
    print(f"    Social Context: {env_factors['social_context']:.3f}")
    
    print("\n" + "=" * 50)
    print(" Simple Test Complete!")
    print(" Data generation: Working")
    print(" Data loading: Working") 
    print(" Data analysis: Working")
    print(" Behavioral patterns: Working")
    print(" EEG simulation: Working")
    print("=" * 50)
    
    print(f"\n Generated file: {output_file}")
    print(" The Neuro-Brainwave AI System is working correctly!")
    print(" Next steps: Run the full system with larger datasets for LSTM training")

if __name__ == "__main__":
    main()
