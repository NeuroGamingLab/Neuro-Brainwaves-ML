#!/usr/bin/env python3
"""
Test script for different brain states and durations
Easy way to test the Dynamic Brainwave Generator with various configurations
"""

from dynamic_brainwave_generator import DynamicBrainwaveGenerator
import numpy as np

def test_brain_state(brain_state, duration, description):
    """
    Test a specific brain state with given duration
    """
    print(f"\n{'='*80}")
    print(f" Testing {brain_state.upper()} state for {duration} seconds")
    print(f" {description}")
    print(f"{'='*80}")
    
    # Initialize generator
    generator = DynamicBrainwaveGenerator()
    
    # Generate session
    samples, brain_state_events = generator.generate_brainwave_session(
        duration=duration,
        brain_state=brain_state,
        start_time_range=(0, 65),
        event_variability=0.3
    )
    
    # Save with descriptive filename
    filename = f"{brain_state}_brainwave_{duration}s.json"
    generator.save_brainwave_samples(samples, filename)
    
    # Create visualization
    generator.create_brainwave_visualization(samples, brain_state_events)
    
    # Print summary
    print(f"\n {brain_state.upper()} SESSION SUMMARY:")
    print(f"• Total samples: {len(samples)}")
    print(f"• Duration: {duration} seconds")
    print(f"• Brain state events: {len(brain_state_events)}")
    print(f"• Average state variation: {np.mean([event[3] for event in brain_state_events]):.2f}")
    print(f"• Max state variation: {max([event[3] for event in brain_state_events]):.2f}")
    print(f"• File saved: {filename}")
    
    return samples, brain_state_events

def main():
    """
    Test different brain states and durations
    """
    print(" AI Dynamic Brainwave Generator - Test Suite")
    print("=" * 80)
    
    # Test configurations
    test_configs = [
        # (brain_state, duration, description)
        ('normal', 60, 'Normal baseline - 1 minute'),
        ('excited', 120, 'Excited state - 2 minutes'),
        ('relaxed', 60, 'Deep relaxation - 1 minute'),
        ('focused', 180, 'Focused attention - 3 minutes'),
        ('stressed', 90, 'Stress/anxiety - 1.5 minutes'),
        ('sleepy', 60, 'Drowsy/sleepy - 1 minute'),
    ]
    
    # Run tests
    results = {}
    for brain_state, duration, description in test_configs:
        try:
            samples, events = test_brain_state(brain_state, duration, description)
            results[brain_state] = {
                'samples': len(samples),
                'duration': duration,
                'events': len(events),
                'avg_variation': np.mean([event[3] for event in events]),
                'max_variation': max([event[3] for event in events])
            }
        except Exception as e:
            print(f" Error testing {brain_state}: {e}")
            results[brain_state] = {'error': str(e)}
    
    # Print overall summary
    print(f"\n{'='*80}")
    print(" OVERALL TEST SUMMARY")
    print(f"{'='*80}")
    
    for brain_state, result in results.items():
        if 'error' not in result:
            print(f" {brain_state:8s}: {result['samples']:4d} samples, "
                  f"{result['duration']:3.0f}s, "
                  f"{result['events']:2d} events, "
                  f"avg_var: {result['avg_variation']:.2f}")
        else:
            print(f" {brain_state:8s}: ERROR - {result['error']}")
    
    print(f"\n All tests completed!")
    print(" Check the generated JSON files and PNG visualizations")

if __name__ == "__main__":
    main()
