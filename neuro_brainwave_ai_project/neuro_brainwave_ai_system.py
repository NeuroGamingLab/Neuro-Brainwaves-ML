#!/usr/bin/env python3
"""
Neuro-Brainwave AI System
=========================

Integrated Agentic AI system combining:
1. Neuro-Brainwave Data Generator Agent (100,000 samples)
2. LSTM Behavior Forecasting Agent (unsupervised learning)
3. 10-minute behavior prediction pipeline

Author: AI Engineering Team
Purpose: Complete neuro-brainwave analysis and forecasting system
"""

import os
import sys
import time
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
import argparse

# Import our custom agents
from neuro_brainwave_data_generator_agent import NeuroBrainwaveDataGeneratorAgent
from lstm_behavior_forecasting_agent import LSTMBehaviorForecastingAgent, ForecastResult

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('neuro_brainwave_ai_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class NeuroBrainwaveAISystem:
    """
    Integrated Agentic AI system for neuro-brainwave analysis and forecasting
    """
    
    def __init__(self, data_samples: int = 100000, sequence_length: int = 60, forecast_horizon: int = 10):
        self.data_samples = data_samples
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        
        # Initialize agents
        self.data_generator = NeuroBrainwaveDataGeneratorAgent(target_samples=data_samples)
        self.forecasting_agent = LSTMBehaviorForecastingAgent(
            sequence_length=sequence_length,
            forecast_horizon=forecast_horizon
        )
        
        # System state
        self.data_file = "neuro_brainwave_dataset.jsonl"
        self.model_file = "lstm_behavior_forecasting_model.pth"
        self.results_file = "forecasting_results.json"
        
        logger.info(f"Neuro-Brainwave AI System initialized")
        logger.info(f"Target samples: {data_samples:,}")
        logger.info(f"Sequence length: {sequence_length} minutes")
        logger.info(f"Forecast horizon: {forecast_horizon} minutes")
    
    def run_complete_pipeline(self, force_regenerate: bool = False) -> Dict[str, Any]:
        """Run the complete pipeline from data generation to forecasting"""
        logger.info(" Starting complete Neuro-Brainwave AI pipeline...")
        
        pipeline_results = {
            'start_time': datetime.now().isoformat(),
            'data_generation': {},
            'unsupervised_learning': {},
            'model_training': {},
            'forecasting': {},
            'system_performance': {}
        }
        
        try:
            # Step 1: Generate neuro-brainwave data
            logger.info(" Step 1: Generating neuro-brainwave data...")
            if force_regenerate or not os.path.exists(self.data_file):
                start_time = time.time()
                generated_file = self.data_generator.generate_dataset(self.data_file)
                generation_time = time.time() - start_time
                
                pipeline_results['data_generation'] = {
                    'status': 'success',
                    'file': generated_file,
                    'samples_generated': self.data_generator.generated_samples,
                    'generation_time_seconds': generation_time,
                    'samples_per_second': self.data_generator.generated_samples / generation_time
                }
                logger.info(f" Data generation complete: {self.data_generator.generated_samples:,} samples in {generation_time:.2f}s")
            else:
                pipeline_results['data_generation'] = {
                    'status': 'skipped',
                    'reason': 'File already exists',
                    'file': self.data_file
                }
                logger.info(" Data generation skipped - file already exists")
            
            # Step 2: Load and prepare data
            logger.info(" Step 2: Loading and preparing data...")
            data = self.forecasting_agent.load_data(self.data_file)
            dataset = self.forecasting_agent.prepare_dataset(data)
            
            # Step 3: Perform unsupervised learning
            logger.info(" Step 3: Performing unsupervised learning...")
            start_time = time.time()
            unsupervised_results = self.forecasting_agent.perform_unsupervised_learning(data)
            unsupervised_time = time.time() - start_time
            
            pipeline_results['unsupervised_learning'] = {
                'status': 'success',
                'silhouette_score': unsupervised_results['silhouette_score'],
                'n_clusters': len(unsupervised_results['cluster_centers']),
                'processing_time_seconds': unsupervised_time,
                'cluster_analysis': unsupervised_results['cluster_analysis']
            }
            logger.info(f" Unsupervised learning complete: {len(unsupervised_results['cluster_centers'])} clusters, silhouette score: {unsupervised_results['silhouette_score']:.3f}")
            
            # Step 4: Train LSTM model
            logger.info(" Step 4: Training LSTM model...")
            start_time = time.time()
            training_history = self.forecasting_agent.train_model(
                dataset, 
                epochs=50, 
                batch_size=32, 
                learning_rate=0.001
            )
            training_time = time.time() - start_time
            
            pipeline_results['model_training'] = {
                'status': 'success',
                'final_loss': training_history['train_loss'][-1],
                'training_time_seconds': training_time,
                'epochs_completed': len(training_history['train_loss']),
                'training_history': training_history
            }
            logger.info(f" Model training complete: Final loss: {training_history['train_loss'][-1]:.4f}")
            
            # Step 5: Save trained model
            logger.info(" Step 5: Saving trained model...")
            self.forecasting_agent.save_model(self.model_file)
            
            # Step 6: Perform behavior forecasting
            logger.info(" Step 6: Performing behavior forecasting...")
            start_time = time.time()
            forecasting_results = self._perform_forecasting_tests(data)
            forecasting_time = time.time() - start_time
            
            pipeline_results['forecasting'] = {
                'status': 'success',
                'forecasts_generated': len(forecasting_results),
                'forecasting_time_seconds': forecasting_time,
                'forecasts': forecasting_results
            }
            logger.info(f" Forecasting complete: {len(forecasting_results)} forecasts generated")
            
            # Step 7: System performance analysis
            logger.info(" Step 7: Analyzing system performance...")
            performance_metrics = self._analyze_system_performance(pipeline_results)
            pipeline_results['system_performance'] = performance_metrics
            
            pipeline_results['end_time'] = datetime.now().isoformat()
            pipeline_results['total_pipeline_time'] = sum([
                pipeline_results['data_generation'].get('generation_time_seconds', 0),
                unsupervised_time,
                training_time,
                forecasting_time
            ])
            
            # Save results
            with open(self.results_file, 'w') as f:
                json.dump(pipeline_results, f, indent=2, default=str)
            
            logger.info(" Complete pipeline finished successfully!")
            self._print_summary(pipeline_results)
            
            return pipeline_results
            
        except Exception as e:
            logger.error(f" Pipeline failed: {e}")
            pipeline_results['error'] = str(e)
            pipeline_results['end_time'] = datetime.now().isoformat()
            return pipeline_results
    
    def _perform_forecasting_tests(self, data: List[Dict]) -> List[Dict]:
        """Perform multiple forecasting tests"""
        forecasting_results = []
        
        # Test forecasting for different users and time periods
        test_cases = [
            {'user_id': 'user_001', 'time_offset': -1000},
            {'user_id': 'user_050', 'time_offset': -500},
            {'user_id': 'user_099', 'time_offset': -100},
        ]
        
        for test_case in test_cases:
            try:
                # Get recent data for specific user
                user_data = [d for d in data if d['user_id'] == test_case['user_id']]
                if len(user_data) < self.sequence_length:
                    continue
                
                # Sort by timestamp and get recent sequence
                user_data.sort(key=lambda x: x['timestamp'])
                recent_data = user_data[test_case['time_offset']:test_case['time_offset'] + self.sequence_length]
                
                # Generate forecast
                forecast = self.forecasting_agent.forecast_behavior(recent_data, forecast_minutes=10)
                
                # Convert to dict for JSON serialization
                forecast_dict = {
                    'test_case': test_case,
                    'forecast': {
                        'timestamp': forecast.timestamp,
                        'forecast_horizon': forecast.forecast_horizon,
                        'predicted_behavioral_markers': forecast.predicted_behavioral_markers,
                        'predicted_brain_state': forecast.predicted_brain_state,
                        'predicted_emotional_state': forecast.predicted_emotional_state,
                        'predicted_cognitive_load': forecast.predicted_cognitive_load,
                        'confidence_scores': forecast.confidence_scores,
                        'attention_trend': forecast.attention_trend,
                        'stress_trend': forecast.stress_trend,
                        'fatigue_trend': forecast.fatigue_trend,
                        'arousal_trend': forecast.arousal_trend
                    }
                }
                
                forecasting_results.append(forecast_dict)
                logger.info(f" Forecast generated for {test_case['user_id']}: {forecast.predicted_brain_state} state")
                
            except Exception as e:
                logger.error(f" Forecasting failed for {test_case['user_id']}: {e}")
        
        return forecasting_results
    
    def _analyze_system_performance(self, pipeline_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze overall system performance"""
        performance = {
            'data_quality': {
                'total_samples': pipeline_results['data_generation'].get('samples_generated', 0),
                'generation_rate': pipeline_results['data_generation'].get('samples_per_second', 0)
            },
            'unsupervised_learning': {
                'silhouette_score': pipeline_results['unsupervised_learning'].get('silhouette_score', 0),
                'clusters_discovered': pipeline_results['unsupervised_learning'].get('n_clusters', 0)
            },
            'model_performance': {
                'final_training_loss': pipeline_results['model_training'].get('final_loss', 0),
                'training_efficiency': pipeline_results['model_training'].get('training_time_seconds', 0)
            },
            'forecasting_capability': {
                'successful_forecasts': pipeline_results['forecasting'].get('forecasts_generated', 0),
                'forecasting_speed': pipeline_results['forecasting'].get('forecasting_time_seconds', 0)
            },
            'overall_system': {
                'total_pipeline_time': pipeline_results.get('total_pipeline_time', 0),
                'success_rate': 1.0 if 'error' not in pipeline_results else 0.0
            }
        }
        
        return performance
    
    def _print_summary(self, results: Dict[str, Any]):
        """Print pipeline summary"""
        print("\n" + "="*60)
        print(" NEURO-BRAINWAVE AI SYSTEM - PIPELINE SUMMARY")
        print("="*60)
        
        # Data Generation
        data_gen = results['data_generation']
        print(f" Data Generation: {data_gen.get('samples_generated', 0):,} samples")
        if 'generation_time_seconds' in data_gen:
            print(f"     Time: {data_gen['generation_time_seconds']:.2f}s")
            print(f"    Rate: {data_gen.get('samples_per_second', 0):.0f} samples/sec")
        
        # Unsupervised Learning
        unsupervised = results['unsupervised_learning']
        print(f" Unsupervised Learning: {unsupervised.get('n_clusters', 0)} clusters discovered")
        print(f"    Silhouette Score: {unsupervised.get('silhouette_score', 0):.3f}")
        
        # Model Training
        training = results['model_training']
        print(f" Model Training: Final loss {training.get('final_loss', 0):.4f}")
        print(f"    Epochs: {training.get('epochs_completed', 0)}")
        
        # Forecasting
        forecasting = results['forecasting']
        print(f" Forecasting: {forecasting.get('forecasts_generated', 0)} forecasts generated")
        
        # Performance
        performance = results['system_performance']
        print(f" System Performance:")
        print(f"    Success Rate: {performance['overall_system']['success_rate']*100:.1f}%")
        print(f"     Total Time: {performance['overall_system']['total_pipeline_time']:.2f}s")
        
        print("="*60)
        print(" Pipeline completed successfully!")
        print("="*60)
    
    def load_trained_system(self) -> bool:
        """Load pre-trained system"""
        try:
            if os.path.exists(self.model_file):
                self.forecasting_agent.load_model(self.model_file)
                logger.info(" Pre-trained system loaded successfully")
                return True
            else:
                logger.warning(" No pre-trained model found")
                return False
        except Exception as e:
            logger.error(f" Failed to load pre-trained system: {e}")
            return False
    
    def forecast_user_behavior(self, user_id: str, minutes_ahead: int = 10) -> Optional[ForecastResult]:
        """Forecast behavior for a specific user"""
        if not os.path.exists(self.data_file):
            logger.error(" No data file found. Please run the complete pipeline first.")
            return None
        
        try:
            # Load data
            data = self.forecasting_agent.load_data(self.data_file)
            
            # Filter for specific user
            user_data = [d for d in data if d['user_id'] == user_id]
            if len(user_data) < self.sequence_length:
                logger.error(f" Insufficient data for user {user_id}")
                return None
            
            # Sort by timestamp and get recent sequence
            user_data.sort(key=lambda x: x['timestamp'])
            recent_data = user_data[-self.sequence_length:]
            
            # Generate forecast
            forecast = self.forecasting_agent.forecast_behavior(recent_data, forecast_minutes=minutes_ahead)
            
            logger.info(f" Behavior forecast generated for {user_id}")
            return forecast
            
        except Exception as e:
            logger.error(f" Forecasting failed for user {user_id}: {e}")
            return None

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Neuro-Brainwave AI System')
    parser.add_argument('--samples', type=int, default=100000, help='Number of data samples to generate')
    parser.add_argument('--sequence-length', type=int, default=60, help='LSTM sequence length in minutes')
    parser.add_argument('--forecast-horizon', type=int, default=10, help='Forecast horizon in minutes')
    parser.add_argument('--force-regenerate', action='store_true', help='Force regeneration of data')
    parser.add_argument('--load-only', action='store_true', help='Only load pre-trained system')
    parser.add_argument('--forecast-user', type=str, help='Forecast behavior for specific user ID')
    
    args = parser.parse_args()
    
    print(" Neuro-Brainwave AI System")
    print("=" * 50)
    print(f" Target samples: {args.samples:,}")
    print(f" Sequence length: {args.sequence_length} minutes")
    print(f" Forecast horizon: {args.forecast_horizon} minutes")
    print("=" * 50)
    
    # Initialize system
    system = NeuroBrainwaveAISystem(
        data_samples=args.samples,
        sequence_length=args.sequence_length,
        forecast_horizon=args.forecast_horizon
    )
    
    if args.load_only:
        # Only load pre-trained system
        if system.load_trained_system():
            print(" Pre-trained system loaded successfully!")
        else:
            print(" Failed to load pre-trained system")
            sys.exit(1)
    elif args.forecast_user:
        # Load system and forecast for specific user
        if system.load_trained_system():
            forecast = system.forecast_user_behavior(args.forecast_user, args.forecast_horizon)
            if forecast:
                print(f"\n Behavior Forecast for {args.forecast_user}:")
                print(f"    Brain State: {forecast.predicted_brain_state}")
                print(f"    Emotional State: {forecast.predicted_emotional_state}")
                print(f"    Attention: {forecast.predicted_behavioral_markers['attention']:.3f}")
                print(f"    Stress: {forecast.predicted_behavioral_markers['stress']:.3f}")
                print(f"    Fatigue: {forecast.predicted_behavioral_markers['fatigue']:.3f}")
                print(f"    Arousal: {forecast.predicted_behavioral_markers['arousal']:.3f}")
                print(f"    Cognitive Load: {forecast.predicted_cognitive_load:.3f}")
            else:
                print(" Failed to generate forecast")
        else:
            print(" Failed to load system")
    else:
        # Run complete pipeline
        results = system.run_complete_pipeline(force_regenerate=args.force_regenerate)
        
        if 'error' in results:
            print(f" Pipeline failed: {results['error']}")
            sys.exit(1)
        else:
            print(" Pipeline completed successfully!")

if __name__ == "__main__":
    main()
