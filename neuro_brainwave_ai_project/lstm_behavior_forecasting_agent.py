#!/usr/bin/env python3
"""
LSTM Behavior Forecasting Agent
===============================

An Agentic AI system using LSTM neural networks for unsupervised learning
to forecast 10-minute behavior patterns from neuro-brainwave data.

Author: AI Engineering Team
Purpose: Unsupervised learning and behavior forecasting from brainwave data
"""

import numpy as np
import pandas as pd
import json
import pickle
import time
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ForecastResult:
    """Result of behavior forecasting"""
    timestamp: float
    forecast_horizon: int  # minutes
    predicted_behavioral_markers: Dict[str, float]
    predicted_brain_state: str
    predicted_emotional_state: str
    predicted_cognitive_load: float
    confidence_scores: Dict[str, float]
    attention_trend: List[float]
    stress_trend: List[float]
    fatigue_trend: List[float]
    arousal_trend: List[float]

class NeuroBrainwaveDataset(Dataset):
    """PyTorch Dataset for neuro-brainwave data"""
    
    def __init__(self, data: List[Dict], sequence_length: int = 60, forecast_horizon: int = 10):
        self.data = data
        self.sequence_length = sequence_length  # minutes of history
        self.forecast_horizon = forecast_horizon  # minutes to forecast
        
        # Prepare sequences
        self.sequences = self._prepare_sequences()
        
    def _prepare_sequences(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Prepare input sequences and targets"""
        sequences = []
        
        # Group data by user_id for proper temporal ordering
        user_data = {}
        for sample in self.data:
            user_id = sample['user_id']
            if user_id not in user_data:
                user_data[user_id] = []
            user_data[user_id].append(sample)
        
        # Sort by timestamp for each user
        for user_id in user_data:
            user_data[user_id].sort(key=lambda x: x['timestamp'])
        
        # Create sequences
        for user_id, user_samples in user_data.items():
            if len(user_samples) < self.sequence_length + self.forecast_horizon:
                continue
                
            for i in range(len(user_samples) - self.sequence_length - self.forecast_horizon):
                # Input sequence (past data)
                input_seq = user_samples[i:i + self.sequence_length]
                
                # Target sequence (future data)
                target_seq = user_samples[i + self.sequence_length:i + self.sequence_length + self.forecast_horizon]
                
                # Convert to feature vectors
                input_features = self._extract_features(input_seq)
                target_features = self._extract_features(target_seq)
                
                sequences.append((input_features, target_features))
        
        return sequences
    
    def _extract_features(self, samples: List[Dict]) -> np.ndarray:
        """Extract feature vector from samples"""
        features = []
        
        for sample in samples:
            # Frequency bands
            freq_bands = sample['frequency_bands']
            features.extend([
                freq_bands['delta'],
                freq_bands['theta'],
                freq_bands['alpha'],
                freq_bands['beta'],
                freq_bands['gamma']
            ])
            
            # Behavioral markers
            behavioral = sample['behavioral_markers']
            features.extend([
                behavioral['attention'],
                behavioral['stress'],
                behavioral['fatigue'],
                behavioral['arousal']
            ])
            
            # Environmental factors
            environmental = sample['environmental_factors']
            features.extend([
                environmental['time_of_day'],
                environmental['activity_level'],
                environmental['social_context']
            ])
            
            # Additional features
            features.extend([
                sample['cognitive_load'],
                sample['task_complexity']
            ])
            
            # Brain state encoding
            brain_state_encoding = self._encode_brain_state(sample['brain_state'])
            features.extend(brain_state_encoding)
            
            # Emotional state encoding
            emotional_state_encoding = self._encode_emotional_state(sample['emotional_state'])
            features.extend(emotional_state_encoding)
        
        return np.array(features, dtype=np.float32)
    
    def _encode_brain_state(self, brain_state: str) -> List[float]:
        """One-hot encode brain state"""
        states = ['focused', 'relaxed', 'stressed', 'sleepy', 'excited']
        encoding = [0.0] * len(states)
        if brain_state in states:
            encoding[states.index(brain_state)] = 1.0
        return encoding
    
    def _encode_emotional_state(self, emotional_state: str) -> List[float]:
        """One-hot encode emotional state"""
        states = ['anxious', 'depressed', 'excited', 'calm', 'concentrated', 'neutral']
        encoding = [0.0] * len(states)
        if emotional_state in states:
            encoding[states.index(emotional_state)] = 1.0
        return encoding
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        input_features, target_features = self.sequences[idx]
        return torch.tensor(input_features), torch.tensor(target_features)

class LSTMNeuralNetwork(nn.Module):
    """
    Agentic LSTM Neural Network for behavior forecasting
    """
    
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 3, 
                 forecast_horizon: int = 10, dropout: float = 0.2):
        super(LSTMBehaviorForecastingAgent, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.forecast_horizon = forecast_horizon
        
        # LSTM layers for temporal modeling
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=True
        )
        
        # Attention mechanism for focusing on important temporal patterns
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,  # *2 for bidirectional
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Output heads for different predictions
        self.behavioral_head = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 4)  # attention, stress, fatigue, arousal
        )
        
        self.brain_state_head = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 5)  # 5 brain states
        )
        
        self.emotional_state_head = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 6)  # 6 emotional states
        )
        
        self.cognitive_load_head = nn.Sequential(
            nn.Linear(hidden_size * 2, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
        
        # Confidence estimation
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_size * 2, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 4)  # confidence for each behavioral marker
        )
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through the network"""
        batch_size, seq_len, _ = x.size()
        
        # LSTM processing
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Apply attention mechanism
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Use the last timestep for predictions
        final_output = attn_out[:, -1, :]  # [batch_size, hidden_size * 2]
        
        # Generate predictions
        behavioral_pred = torch.sigmoid(self.behavioral_head(final_output))
        brain_state_pred = torch.softmax(self.brain_state_head(final_output), dim=-1)
        emotional_state_pred = torch.softmax(self.emotional_state_head(final_output), dim=-1)
        cognitive_load_pred = torch.sigmoid(self.cognitive_load_head(final_output))
        confidence_pred = torch.sigmoid(self.confidence_head(final_output))
        
        return {
            'behavioral': behavioral_pred,
            'brain_state': brain_state_pred,
            'emotional_state': emotional_state_pred,
            'cognitive_load': cognitive_load_pred,
            'confidence': confidence_pred
        }

class UnsupervisedLearningAgent:
    """
    Agent for unsupervised learning and pattern discovery
    """
    
    def __init__(self, n_clusters: int = 10):
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.pca = PCA(n_components=0.95)  # Use variance ratio instead of fixed components
        self.scaler = StandardScaler()
        self.cluster_centers = None
        self.cluster_labels = None
        
    def fit(self, features: np.ndarray) -> Dict[str, Any]:
        """Fit unsupervised learning models"""
        logger.info("Starting unsupervised learning...")
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        
        # Apply PCA for dimensionality reduction
        features_pca = self.pca.fit_transform(features_scaled)
        
        # Perform clustering
        self.cluster_labels = self.kmeans.fit_predict(features_pca)
        self.cluster_centers = self.kmeans.cluster_centers_
        
        # Calculate silhouette score
        silhouette_avg = silhouette_score(features_pca, self.cluster_labels)
        
        # Analyze clusters
        cluster_analysis = self._analyze_clusters(features, self.cluster_labels)
        
        logger.info(f"Unsupervised learning complete. Silhouette score: {silhouette_avg:.3f}")
        
        return {
            'silhouette_score': silhouette_avg,
            'cluster_centers': self.cluster_centers.tolist(),
            'cluster_labels': self.cluster_labels.tolist(),
            'cluster_analysis': cluster_analysis,
            'explained_variance_ratio': self.pca.explained_variance_ratio_.tolist()
        }
    
    def _analyze_clusters(self, features: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
        """Analyze cluster characteristics"""
        cluster_analysis = {}
        
        for cluster_id in range(self.n_clusters):
            cluster_mask = labels == cluster_id
            cluster_features = features[cluster_mask]
            
            if len(cluster_features) == 0:
                continue
                
            # Calculate cluster statistics
            cluster_stats = {
                'size': len(cluster_features),
                'percentage': len(cluster_features) / len(features) * 100,
                'mean_features': np.mean(cluster_features, axis=0).tolist(),
                'std_features': np.std(cluster_features, axis=0).tolist()
            }
            
            cluster_analysis[f'cluster_{cluster_id}'] = cluster_stats
        
        return cluster_analysis
    
    def predict_cluster(self, features: np.ndarray) -> np.ndarray:
        """Predict cluster labels for new features"""
        features_scaled = self.scaler.transform(features)
        features_pca = self.pca.transform(features_scaled)
        return self.kmeans.predict(features_pca)

class LSTMBehaviorForecastingAgent:
    """
    Main Agentic AI system for LSTM-based behavior forecasting
    """
    
    def __init__(self, sequence_length: int = 60, forecast_horizon: int = 10):
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.model = None
        self.unsupervised_agent = UnsupervisedLearningAgent(n_clusters=5)  # Reduced for smaller datasets
        self.scaler = StandardScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Brain state and emotional state mappings
        self.brain_states = ['focused', 'relaxed', 'stressed', 'sleepy', 'excited']
        self.emotional_states = ['anxious', 'depressed', 'excited', 'calm', 'concentrated', 'neutral']
        
        logger.info(f"LSTM Behavior Forecasting Agent initialized on {self.device}")
    
    def load_data(self, data_file: str) -> List[Dict]:
        """Load neuro-brainwave data from JSONL file"""
        logger.info(f"Loading data from {data_file}...")
        
        data = []
        with open(data_file, 'r') as f:
            for line in f:
                sample = json.loads(line.strip())
                data.append(sample)
        
        logger.info(f"Loaded {len(data):,} samples")
        return data
    
    def prepare_dataset(self, data: List[Dict]) -> NeuroBrainwaveDataset:
        """Prepare dataset for training"""
        logger.info("Preparing dataset...")
        
        dataset = NeuroBrainwaveDataset(
            data=data,
            sequence_length=self.sequence_length,
            forecast_horizon=self.forecast_horizon
        )
        
        logger.info(f"Dataset prepared: {len(dataset)} sequences")
        return dataset
    
    def train_model(self, dataset: NeuroBrainwaveDataset, epochs: int = 100, 
                   batch_size: int = 32, learning_rate: float = 0.001) -> Dict[str, Any]:
        """Train the LSTM model"""
        logger.info("Starting model training...")
        
        # Create data loader
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Initialize model
        sample_input, sample_target = dataset[0]
        input_size = sample_input.shape[0] // self.sequence_length
        self.model = LSTMNeuralNetwork(
            input_size=input_size,
            hidden_size=128,
            num_layers=3,
            forecast_horizon=self.forecast_horizon
        ).to(self.device)
        
        # Initialize optimizer and loss functions
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        # Training history
        training_history = {
            'train_loss': [],
            'behavioral_loss': [],
            'brain_state_loss': [],
            'emotional_state_loss': [],
            'cognitive_load_loss': []
        }
        
        # Training loop
        for epoch in range(epochs):
            epoch_losses = {
                'total': 0.0,
                'behavioral': 0.0,
                'brain_state': 0.0,
                'emotional_state': 0.0,
                'cognitive_load': 0.0
            }
            
            self.model.train()
            for batch_idx, (input_seq, target_seq) in enumerate(dataloader):
                input_seq = input_seq.to(self.device)
                target_seq = target_seq.to(self.device)
                
                # Reshape for LSTM
                batch_size = input_seq.shape[0]
                input_seq = input_seq.view(batch_size, self.sequence_length, -1)
                target_seq = target_seq.view(batch_size, self.forecast_horizon, -1)
                
                # Forward pass
                predictions = self.model(input_seq)
                
                # Calculate losses
                behavioral_loss = criterion(
                    predictions['behavioral'], 
                    target_seq[:, :, :4]  # First 4 features are behavioral markers
                )
                
                brain_state_loss = criterion(
                    predictions['brain_state'], 
                    target_seq[:, :, 4:9]  # Brain state encoding
                )
                
                emotional_state_loss = criterion(
                    predictions['emotional_state'], 
                    target_seq[:, :, 9:15]  # Emotional state encoding
                )
                
                cognitive_load_loss = criterion(
                    predictions['cognitive_load'], 
                    target_seq[:, :, 15:16]  # Cognitive load
                )
                
                # Total loss
                total_loss = (behavioral_loss + brain_state_loss + 
                            emotional_state_loss + cognitive_load_loss)
                
                # Backward pass
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                
                # Accumulate losses
                epoch_losses['total'] += total_loss.item()
                epoch_losses['behavioral'] += behavioral_loss.item()
                epoch_losses['brain_state'] += brain_state_loss.item()
                epoch_losses['emotional_state'] += emotional_state_loss.item()
                epoch_losses['cognitive_load'] += cognitive_load_loss.item()
            
            # Average losses
            num_batches = len(dataloader)
            for key in epoch_losses:
                epoch_losses[key] /= num_batches
            
            # Store training history
            training_history['train_loss'].append(epoch_losses['total'])
            training_history['behavioral_loss'].append(epoch_losses['behavioral'])
            training_history['brain_state_loss'].append(epoch_losses['brain_state'])
            training_history['emotional_state_loss'].append(epoch_losses['emotional_state'])
            training_history['cognitive_load_loss'].append(epoch_losses['cognitive_load'])
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}/{epochs}: "
                          f"Loss={epoch_losses['total']:.4f}, "
                          f"Behavioral={epoch_losses['behavioral']:.4f}, "
                          f"Brain State={epoch_losses['brain_state']:.4f}")
        
        logger.info("Model training complete!")
        return training_history
    
    def perform_unsupervised_learning(self, data: List[Dict]) -> Dict[str, Any]:
        """Perform unsupervised learning to discover patterns"""
        logger.info("Performing unsupervised learning...")
        
        # Extract features for clustering
        features = []
        for sample in data:
            feature_vector = self._extract_sample_features(sample)
            features.append(feature_vector)
        
        features = np.array(features)
        
        # Fit unsupervised models
        unsupervised_results = self.unsupervised_agent.fit(features)
        
        return unsupervised_results
    
    def _extract_sample_features(self, sample: Dict) -> np.ndarray:
        """Extract feature vector from a single sample"""
        features = []
        
        # Frequency bands
        freq_bands = sample['frequency_bands']
        features.extend([
            freq_bands['delta'],
            freq_bands['theta'],
            freq_bands['alpha'],
            freq_bands['beta'],
            freq_bands['gamma']
        ])
        
        # Behavioral markers
        behavioral = sample['behavioral_markers']
        features.extend([
            behavioral['attention'],
            behavioral['stress'],
            behavioral['fatigue'],
            behavioral['arousal']
        ])
        
        # Environmental factors
        environmental = sample['environmental_factors']
        features.extend([
            environmental['time_of_day'],
            environmental['activity_level'],
            environmental['social_context']
        ])
        
        # Additional features
        features.extend([
            sample['cognitive_load'],
            sample['task_complexity']
        ])
        
        return np.array(features, dtype=np.float32)
    
    def forecast_behavior(self, recent_data: List[Dict], forecast_minutes: int = 10) -> ForecastResult:
        """Forecast behavior for the next N minutes"""
        if self.model is None:
            raise ValueError("Model not trained. Please train the model first.")
        
        self.model.eval()
        
        # Prepare input sequence
        input_features = []
        for sample in recent_data[-self.sequence_length:]:
            features = self._extract_sample_features(sample)
            input_features.append(features)
        
        input_tensor = torch.tensor(input_features).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(input_tensor)
        
        # Extract predictions
        behavioral_pred = predictions['behavioral'][0].cpu().numpy()
        brain_state_pred = predictions['brain_state'][0].cpu().numpy()
        emotional_state_pred = predictions['emotional_state'][0].cpu().numpy()
        cognitive_load_pred = predictions['cognitive_load'][0].cpu().numpy()
        confidence_pred = predictions['confidence'][0].cpu().numpy()
        
        # Generate forecast result
        forecast_result = ForecastResult(
            timestamp=time.time(),
            forecast_horizon=forecast_minutes,
            predicted_behavioral_markers={
                'attention': float(behavioral_pred[0]),
                'stress': float(behavioral_pred[1]),
                'fatigue': float(behavioral_pred[2]),
                'arousal': float(behavioral_pred[3])
            },
            predicted_brain_state=self.brain_states[np.argmax(brain_state_pred)],
            predicted_emotional_state=self.emotional_states[np.argmax(emotional_state_pred)],
            predicted_cognitive_load=float(cognitive_load_pred[0]),
            confidence_scores={
                'attention': float(confidence_pred[0]),
                'stress': float(confidence_pred[1]),
                'fatigue': float(confidence_pred[2]),
                'arousal': float(confidence_pred[3])
            },
            attention_trend=[float(behavioral_pred[0])] * forecast_minutes,
            stress_trend=[float(behavioral_pred[1])] * forecast_minutes,
            fatigue_trend=[float(behavioral_pred[2])] * forecast_minutes,
            arousal_trend=[float(behavioral_pred[3])] * forecast_minutes
        )
        
        return forecast_result
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        if self.model is None:
            raise ValueError("No model to save")
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'sequence_length': self.sequence_length,
            'forecast_horizon': self.forecast_horizon,
            'brain_states': self.brain_states,
            'emotional_states': self.emotional_states
        }, filepath)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Initialize model
        sample_input = torch.randn(1, self.sequence_length, 20)  # Dummy input for model creation
        self.model = LSTMNeuralNetwork(
            input_size=20,
            hidden_size=128,
            num_layers=3,
            forecast_horizon=self.forecast_horizon
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.brain_states = checkpoint['brain_states']
        self.emotional_states = checkpoint['emotional_states']
        
        logger.info(f"Model loaded from {filepath}")

def main():
    """Main function to run the LSTM behavior forecasting agent"""
    print(" LSTM Behavior Forecasting Agent")
    print("=" * 50)
    
    # Initialize agent
    agent = LSTMBehaviorForecastingAgent(sequence_length=60, forecast_horizon=10)
    
    # Load data (assuming the data generator has been run)
    data_file = "neuro_brainwave_dataset.jsonl"
    try:
        data = agent.load_data(data_file)
        
        # Perform unsupervised learning
        print("\n Performing unsupervised learning...")
        unsupervised_results = agent.perform_unsupervised_learning(data)
        
        print(f" Unsupervised learning complete!")
        print(f" Silhouette Score: {unsupervised_results['silhouette_score']:.3f}")
        print(f" Discovered {len(unsupervised_results['cluster_centers'])} behavioral clusters")
        
        # Prepare dataset
        print("\n Preparing dataset...")
        dataset = agent.prepare_dataset(data)
        
        # Train model
        print("\n Training LSTM model...")
        training_history = agent.train_model(dataset, epochs=50, batch_size=32)
        
        print(f" Model training complete!")
        print(f" Final training loss: {training_history['train_loss'][-1]:.4f}")
        
        # Save model
        model_file = "lstm_behavior_forecasting_model.pth"
        agent.save_model(model_file)
        
        # Test forecasting
        print("\n Testing behavior forecasting...")
        if len(data) >= agent.sequence_length:
            recent_data = data[-agent.sequence_length:]
            forecast = agent.forecast_behavior(recent_data, forecast_minutes=10)
            
            print(f" 10-minute behavior forecast:")
            print(f"   Predicted Brain State: {forecast.predicted_brain_state}")
            print(f"   Predicted Emotional State: {forecast.predicted_emotional_state}")
            print(f"   Predicted Attention: {forecast.predicted_behavioral_markers['attention']:.3f}")
            print(f"   Predicted Stress: {forecast.predicted_behavioral_markers['stress']:.3f}")
            print(f"   Predicted Fatigue: {forecast.predicted_behavioral_markers['fatigue']:.3f}")
            print(f"   Predicted Arousal: {forecast.predicted_behavioral_markers['arousal']:.3f}")
        
        print(f"\n LSTM Behavior Forecasting Agent complete!")
        print(f" Model saved: {model_file}")
        print(f" Training samples: {len(dataset):,}")
        print(f" Forecast horizon: {agent.forecast_horizon} minutes")
        
    except FileNotFoundError:
        print(f" Data file not found: {data_file}")
        print("Please run the Neuro-Brainwave Data Generator Agent first!")
    except Exception as e:
        print(f" Error: {e}")

if __name__ == "__main__":
    main()
