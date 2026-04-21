#!/usr/bin/env python3
"""
Neuro-Brainwave AI Dashboard Demo
================================

Quick demonstration of the dashboard capabilities
"""

import json
import os
from pathlib import Path

def create_demo_data():
    """Create a small demo dataset for testing the dashboard"""
    
    print(" Creating demo dataset for dashboard testing...")
    
    # Create sample data
    demo_data = []
    
    # Sample brain states and their characteristics
    brain_states = {
        'focused': {
            'attention': 0.8, 'stress': 0.3, 'fatigue': 0.2, 'arousal': 0.7,
            'delta': 45.2, 'theta': 67.8, 'alpha': 89.3, 'beta': 156.7, 'gamma': 78.9
        },
        'relaxed': {
            'attention': 0.4, 'stress': 0.1, 'fatigue': 0.3, 'arousal': 0.3,
            'delta': 123.4, 'theta': 145.6, 'alpha': 234.5, 'beta': 67.8, 'gamma': 23.4
        },
        'stressed': {
            'attention': 0.9, 'stress': 0.8, 'fatigue': 0.4, 'arousal': 0.9,
            'delta': 34.5, 'theta': 56.7, 'alpha': 45.6, 'beta': 234.5, 'gamma': 156.7
        },
        'sleepy': {
            'attention': 0.1, 'stress': 0.2, 'fatigue': 0.9, 'arousal': 0.1,
            'delta': 345.6, 'theta': 234.5, 'alpha': 123.4, 'beta': 34.5, 'gamma': 12.3
        },
        'excited': {
            'attention': 0.7, 'stress': 0.6, 'fatigue': 0.1, 'arousal': 0.95,
            'delta': 56.7, 'theta': 78.9, 'alpha': 67.8, 'beta': 178.9, 'gamma': 234.5
        }
    }
    
    emotions = ['calm', 'neutral', 'anxious', 'happy', 'focused']
    
    # Generate 100 demo samples
    for i in range(100):
        # Select random brain state
        state_name = list(brain_states.keys())[i % len(brain_states)]
        state_data = brain_states[state_name]
        
        # Create EEG channel data (simplified)
        eeg_channels = {}
        channel_names = ['CP3', 'CP4', 'C3', 'C4', 'F5', 'F6', 'PO3', 'PO4']
        
        for channel in channel_names:
            # Generate 1000 samples of EEG data
            eeg_data = []
            for j in range(1000):
                # Simple sine wave with noise
                import math
                signal = math.sin(2 * math.pi * 10 * j / 250) * 0.5 + (i % 10 - 5) * 0.1
                eeg_data.append(signal)
            eeg_channels[channel] = eeg_data
        
        # Create sample
        sample = {
            'brain_state': state_name,
            'emotional_state': emotions[i % len(emotions)],
            'eeg_channels': eeg_channels,
            'frequency_bands': {
                'delta': state_data['delta'] + (i % 20 - 10),
                'theta': state_data['theta'] + (i % 20 - 10),
                'alpha': state_data['alpha'] + (i % 20 - 10),
                'beta': state_data['beta'] + (i % 20 - 10),
                'gamma': state_data['gamma'] + (i % 20 - 10)
            },
            'behavioral_markers': {
                'attention': state_data['attention'] + (i % 10 - 5) * 0.05,
                'stress': state_data['stress'] + (i % 10 - 5) * 0.05,
                'fatigue': state_data['fatigue'] + (i % 10 - 5) * 0.05,
                'arousal': state_data['arousal'] + (i % 10 - 5) * 0.05
            },
            'cognitive_load': 0.3 + (i % 10) * 0.05,
            'task_complexity': 0.2 + (i % 8) * 0.1,
            'environmental_factors': {
                'time_of_day': (i % 24) / 24.0,
                'activity_level': 0.3 + (i % 7) * 0.1,
                'social_context': 0.2 + (i % 5) * 0.15
            },
            'sample_rate': 250,
            'duration': 4.0,
            'line_number': i + 1
        }
        
        demo_data.append(sample)
    
    # Save to JSONL file
    output_file = 'demo_dataset.jsonl'
    with open(output_file, 'w') as f:
        for sample in demo_data:
            f.write(json.dumps(sample) + '\n')
    
    print(f" Created demo dataset: {output_file}")
    print(f"    {len(demo_data)} samples")
    print(f"    {len(brain_states)} brain states")
    print(f"    {len(emotions)} emotional states")
    print(f"    {len(channel_names)} EEG channels")
    
    return output_file

def show_dashboard_info():
    """Show information about the dashboard"""
    
    print("\n" + "="*60)
    print(" NEURO-BRAINWAVE AI SYSTEM DASHBOARD")
    print("="*60)
    print()
    print(" Dashboard Features:")
    print("   • Dataset Overview & Statistics")
    print("   • Brain State Analysis with Visualizations")
    print("   • Multi-Channel EEG Signal Display")
    print("   • Frequency Band Analysis")
    print("   • Behavioral Pattern Analysis")
    print("   • Interactive Data Explorer")
    print()
    print(" Launch Commands:")
    print("   ./launch_dashboard.sh")
    print("   python3 -m streamlit run neuro_brainwave_dashboard.py --server.port 8502")
    print()
    print(" Access URLs:")
    print("   Local:   http://localhost:8502")
    print("   Network: http://[your-ip]:8502")
    print()
    print(" Required Files:")
    print("   • neuro_brainwave_dashboard.py")
    print("   • *.jsonl dataset files")
    print("   • launch_dashboard.sh (optional)")
    print()
    print(" Dashboard Tabs:")
    print("   1.  Overview - System metrics and dataset statistics")
    print("   2.  Brain States - State distribution and behavioral markers")
    print("   3.  EEG Signals - Multi-channel signal visualization")
    print("   4.  Frequency Bands - Frequency analysis and power distribution")
    print("   5.  Behavioral Patterns - Correlation and trend analysis")
    print("   6.  Data Explorer - Interactive filtering and data exploration")
    print()

def main():
    """Main demo function"""
    
    print(" Neuro-Brainwave AI Dashboard Demo")
    print("="*40)
    print()
    
    # Check if we're in the right directory
    if not Path("neuro_brainwave_dashboard.py").exists():
        print(" Error: neuro_brainwave_dashboard.py not found!")
        print("Please run this script from the neuro_brainwave_ai_project directory.")
        return
    
    # Check for existing datasets
    jsonl_files = list(Path(".").glob("*.jsonl"))
    
    if not jsonl_files:
        print(" No existing datasets found. Creating demo dataset...")
        create_demo_data()
        print()
    else:
        print(f" Found {len(jsonl_files)} existing dataset(s):")
        for file in jsonl_files:
            size_mb = file.stat().st_size / (1024*1024)
            print(f"    {file.name} ({size_mb:.1f} MB)")
        print()
    
    # Show dashboard information
    show_dashboard_info()
    
    print(" Dashboard is ready to launch!")
    print("   Run: ./launch_dashboard.sh")
    print("   Or: python3 -m streamlit run neuro_brainwave_dashboard.py --server.port 8502")

if __name__ == "__main__":
    main()
