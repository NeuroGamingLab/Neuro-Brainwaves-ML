#!/usr/bin/env python3
"""
Demo: Neuro-Brainwave AI System
===============================

Quick demonstration of the complete Neuro-Brainwave AI system
with a smaller dataset for testing purposes.

Author: AI Engineering Team
Purpose: Demonstrate the complete system with manageable data size
"""

import os
import sys
import time
from neuro_brainwave_ai_system import NeuroBrainwaveAISystem

def main():
    """Run a demo of the Neuro-Brainwave AI system"""
    print(" Neuro-Brainwave AI System - DEMO")
    print("=" * 50)
    print("This demo will:")
    print("1. Generate 10,000 neuro-brainwave samples")
    print("2. Perform unsupervised learning")
    print("3. Train an LSTM model")
    print("4. Generate behavior forecasts")
    print("=" * 50)
    
    # Ask for confirmation
    response = input("Continue with demo? (y/N): ").strip().lower()
    if response != 'y':
        print("Demo cancelled.")
        return
    
    # Initialize system with smaller dataset for demo
    print("\n Initializing Neuro-Brainwave AI System...")
    system = NeuroBrainwaveAISystem(
        data_samples=10000,      # Smaller dataset for demo
        sequence_length=30,      # Shorter sequences
        forecast_horizon=5       # Shorter forecast horizon
    )
    
    # Run the complete pipeline
    print("\n Running complete pipeline...")
    start_time = time.time()
    
    try:
        results = system.run_complete_pipeline(force_regenerate=True)
        
        total_time = time.time() - start_time
        
        if 'error' in results:
            print(f"\n Demo failed: {results['error']}")
            return
        
        # Print demo results
        print("\n" + "="*60)
        print(" DEMO COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        print(f"  Total Demo Time: {total_time:.2f} seconds")
        print(f" Samples Generated: {results['data_generation'].get('samples_generated', 0):,}")
        print(f" Clusters Discovered: {results['unsupervised_learning'].get('n_clusters', 0)}")
        print(f" Silhouette Score: {results['unsupervised_learning'].get('silhouette_score', 0):.3f}")
        print(f" Final Training Loss: {results['model_training'].get('final_loss', 0):.4f}")
        print(f" Forecasts Generated: {results['forecasting'].get('forecasts_generated', 0)}")
        
        # Show sample forecast
        if results['forecasting']['forecasts']:
            sample_forecast = results['forecasting']['forecasts'][0]
            forecast_data = sample_forecast['forecast']
            
            print(f"\n Sample Forecast:")
            print(f"    Brain State: {forecast_data['predicted_brain_state']}")
            print(f"    Emotional State: {forecast_data['predicted_emotional_state']}")
            print(f"    Attention: {forecast_data['predicted_behavioral_markers']['attention']:.3f}")
            print(f"    Stress: {forecast_data['predicted_behavioral_markers']['stress']:.3f}")
            print(f"    Fatigue: {forecast_data['predicted_behavioral_markers']['fatigue']:.3f}")
            print(f"    Arousal: {forecast_data['predicted_behavioral_markers']['arousal']:.3f}")
        
        print(f"\n Files Generated:")
        print(f"    Data: neuro_brainwave_dataset.jsonl")
        print(f"    Model: lstm_behavior_forecasting_model.pth")
        print(f"    Results: forecasting_results.json")
        
        print("\n" + "="*60)
        print(" Demo completed! The system is ready for use.")
        print("Run with --help to see all available options.")
        print("="*60)
        
    except Exception as e:
        print(f"\n Demo failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
