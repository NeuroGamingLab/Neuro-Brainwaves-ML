#!/usr/bin/env python3
"""
Enhanced Neuro-Brainwave Data Generator
======================================

Enhanced version of the data generator that provides more dynamic and varied
brain state and emotional state distributions across different generations.

Key improvements:
- Dynamic brain state distribution based on slider settings
- More varied emotional state patterns
- Temporal variation in state probabilities
- User-controlled state distribution
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import time
from datetime import datetime
from pathlib import Path
import sys
import random

# Add current directory to path for imports
sys.path.append('.')

try:
    from neuro_brainwave_data_generator_agent import NeuroBrainwaveDataGeneratorAgent
except ImportError:
    st.error(" Could not import NeuroBrainwaveDataGeneratorAgent. Please ensure the file exists.")
    st.stop()

# Configure Streamlit page
st.set_page_config(
    page_title="Enhanced Neuro-Brainwave Generator",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

class EnhancedDataGenerator(NeuroBrainwaveDataGeneratorAgent):
    """Enhanced data generator with dynamic state distributions"""
    
    def __init__(self, target_samples: int = 1000, custom_state_distribution: dict = None):
        super().__init__(target_samples)
        self.custom_state_distribution = custom_state_distribution or {}
        self.generation_variation_seed = random.randint(1, 10000)
        
    def _determine_brain_state(self, timestamp: float, user_profile: dict) -> str:
        """Enhanced brain state determination with dynamic distributions"""
        
        # If custom distribution is provided, use it with some variation
        if self.custom_state_distribution:
            return self._get_state_from_custom_distribution(timestamp)
        
        # Enhanced time-based probabilities with more variation
        hour = (timestamp / 3600) % 24
        minute = (timestamp / 60) % 60
        
        # Add random variation to make distributions more dynamic
        variation_factor = 0.3  # 30% variation
        base_variation = np.random.uniform(-variation_factor, variation_factor)
        
        # Enhanced state probabilities with more variation
        if 22 <= hour or hour <= 6:  # Night/sleep hours
            base_probs = {'sleepy': 0.6, 'relaxed': 0.25, 'stressed': 0.08, 'focused': 0.04, 'excited': 0.03}
        elif 7 <= hour <= 9:  # Morning
            base_probs = {'focused': 0.35, 'excited': 0.25, 'relaxed': 0.25, 'stressed': 0.1, 'sleepy': 0.05}
        elif 10 <= hour <= 17:  # Work hours
            base_probs = {'focused': 0.4, 'stressed': 0.25, 'relaxed': 0.2, 'excited': 0.1, 'sleepy': 0.05}
        elif 18 <= hour <= 21:  # Evening
            base_probs = {'relaxed': 0.35, 'focused': 0.25, 'excited': 0.2, 'stressed': 0.15, 'sleepy': 0.05}
        else:
            base_probs = {'relaxed': 0.25, 'focused': 0.25, 'stressed': 0.25, 'excited': 0.15, 'sleepy': 0.1}
        
        # Apply dynamic variation
        state_probs = {}
        for state, prob in base_probs.items():
            # Add variation based on generation seed and timestamp
            variation = np.sin(timestamp * 0.001 + self.generation_variation_seed) * 0.1
            adjusted_prob = prob + base_variation + variation
            state_probs[state] = max(0.01, min(0.8, adjusted_prob))  # Keep probabilities reasonable
        
        # Apply user preferences with less dominance
        preferred_state = user_profile.get('brain_state_preference', 'relaxed')
        if preferred_state in state_probs:
            state_probs[preferred_state] *= 1.3  # Reduced from 1.5 to 1.3
        
        # Normalize probabilities
        total_prob = sum(state_probs.values())
        state_probs = {k: v/total_prob for k, v in state_probs.items()}
        
        # Sample state
        states = list(state_probs.keys())
        probs = list(state_probs.values())
        return np.random.choice(states, p=probs)
    
    def _get_state_from_custom_distribution(self, timestamp: float) -> str:
        """Get brain state from custom distribution with temporal variation"""
        states = list(self.custom_state_distribution.keys())
        base_probs = list(self.custom_state_distribution.values())
        
        # Add temporal variation
        temporal_variation = np.sin(timestamp * 0.001 + self.generation_variation_seed) * 0.15
        
        # Apply variation to probabilities
        varied_probs = []
        for prob in base_probs:
            varied_prob = prob + temporal_variation + np.random.uniform(-0.1, 0.1)
            varied_prob = max(0.05, min(0.7, varied_prob))  # Keep reasonable bounds
            varied_probs.append(varied_prob)
        
        # Normalize
        total_prob = sum(varied_probs)
        normalized_probs = [p/total_prob for p in varied_probs]
        
        return np.random.choice(states, p=normalized_probs)
    
    def _determine_emotional_state(self, behavioral_markers: dict, brain_state: str) -> str:
        """Enhanced emotional state determination with more variety"""
        stress = behavioral_markers['stress']
        arousal = behavioral_markers['arousal']
        attention = behavioral_markers['attention']
        
        # More nuanced emotional state determination
        if stress > 0.75 and arousal > 0.7:
            return 'anxious'
        elif stress > 0.65 and arousal < 0.3:
            return 'depressed'
        elif stress < 0.25 and arousal > 0.75:
            return 'excited'
        elif stress < 0.3 and arousal < 0.4:
            return 'calm'
        elif attention > 0.7 and brain_state == 'focused':
            return 'concentrated'
        elif stress > 0.5 and attention < 0.3:
            return 'frustrated'
        elif arousal > 0.6 and attention > 0.5:
            return 'alert'
        elif stress < 0.4 and arousal > 0.4:
            return 'content'
        else:
            # Add more variety in neutral states
            neutral_options = ['neutral', 'relaxed', 'balanced', 'stable']
            return random.choice(neutral_options)

class EnhancedDynamicGenerator:
    """Enhanced dynamic generator with better state variation"""
    
    def __init__(self):
        self.generator = None
        self.current_data = []
        self.generation_start_time = None
        self.generation_count = 0
        
    def create_parameter_controls(self):
        """Create enhanced parameter controls with state distribution"""
        st.sidebar.header(" Enhanced Generation Parameters")
        
        # Sample size slider
        st.sidebar.markdown("###  Data Generation")
        sample_size = st.sidebar.slider(
            "Number of Samples",
            min_value=10,
            max_value=10000,
            value=1000,
            step=50,
            help="Total number of brainwave samples to generate"
        )
        
        # Generation variation control
        variation_level = st.sidebar.slider(
            "State Variation Level",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1,
            help="How much variation in state distributions across generations"
        )
        
        # Brain state distribution sliders
        st.sidebar.markdown("###  Brain State Distribution")
        
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            focused_ratio = st.slider("Focused", 0.0, 1.0, 0.3, 0.05)
            excited_ratio = st.slider("Excited", 0.0, 1.0, 0.2, 0.05)
            relaxed_ratio = st.slider("Relaxed", 0.0, 1.0, 0.2, 0.05)
        
        with col2:
            stressed_ratio = st.slider("Stressed", 0.0, 1.0, 0.15, 0.05)
            sleepy_ratio = st.slider("Sleepy", 0.0, 1.0, 0.1, 0.05)
        
        # Normalize ratios
        total_ratio = focused_ratio + excited_ratio + relaxed_ratio + stressed_ratio + sleepy_ratio
        if total_ratio > 0:
            focused_ratio /= total_ratio
            excited_ratio /= total_ratio
            relaxed_ratio /= total_ratio
            stressed_ratio /= total_ratio
            sleepy_ratio /= total_ratio
        
        # Emotional state variation
        st.sidebar.markdown("###  Emotional State Variation")
        emotional_variation = st.sidebar.slider(
            "Emotional Variety",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1,
            help="How much variety in emotional states"
        )
        
        # Data quality parameters
        st.sidebar.markdown("###  Data Quality")
        noise_level = st.sidebar.slider(
            "Noise Level",
            min_value=0.0,
            max_value=1.0,
            value=0.1,
            step=0.05,
            help="Amount of noise in EEG signals"
        )
        
        signal_amplitude = st.sidebar.slider(
            "Signal Amplitude (μV)",
            min_value=10,
            max_value=100,
            value=50,
            step=5,
            help="Base amplitude of EEG signals"
        )
        
        return {
            'sample_size': sample_size,
            'variation_level': variation_level,
            'brain_state_ratios': {
                'focused': focused_ratio,
                'excited': excited_ratio,
                'relaxed': relaxed_ratio,
                'stressed': stressed_ratio,
                'sleepy': sleepy_ratio
            },
            'emotional_variation': emotional_variation,
            'noise_level': noise_level,
            'signal_amplitude': signal_amplitude
        }
    
    def generate_data(self, parameters):
        """Generate data with enhanced variation"""
        try:
            # Create enhanced generator with custom distribution
            self.generator = EnhancedDataGenerator(
                target_samples=parameters['sample_size'],
                custom_state_distribution=parameters['brain_state_ratios']
            )
            
            # Update variation settings
            self.generator.generation_variation_seed = random.randint(1, 10000)
            self.generation_count += 1
            
            # Store generation parameters
            self.generation_start_time = time.time()
            
            # Generate data
            output_file = self.generator.generate_dataset('enhanced_generated_data.jsonl')
            
            # Load generated data
            self.current_data = []
            with open(output_file, 'r') as f:
                for line in f:
                    sample = json.loads(line.strip())
                    self.current_data.append(sample)
            
            return True
            
        except Exception as e:
            st.error(f" Error generating data: {e}")
            return False
    
    def create_generation_interface(self):
        """Create the enhanced generation interface"""
        st.markdown('<h1 class="main-header"> Enhanced Neuro-Brainwave Data Generator</h1>', unsafe_allow_html=True)
        
        # Show variation explanation
        st.info("""
        ** Enhanced Features:**
        - **Dynamic State Distributions**: Each generation creates different brain state patterns
        - **Temporal Variation**: States change over time within generations
        - **Custom Distribution Control**: Use sliders to control state ratios
        - **Emotional Variety**: More diverse emotional state patterns
        """)
        
        # Get parameters from sliders
        parameters = self.create_parameter_controls()
        
        # Generation controls
        st.markdown("###  Enhanced Data Generation Controls")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button(" Generate New Data", type="primary"):
                with st.spinner("Generating enhanced data with variation..."):
                    if self.generate_data(parameters):
                        st.success(" Enhanced data generated successfully!")
                        st.rerun()
        
        with col2:
            if st.button(" Randomize Distribution"):
                # Randomize brain state distribution
                total = 1.0
                ratios = {}
                states = ['focused', 'excited', 'relaxed', 'stressed', 'sleepy']
                
                for i, state in enumerate(states[:-1]):
                    max_val = total - sum(ratios.values()) - (len(states) - i - 1) * 0.05
                    ratios[state] = random.uniform(0.05, max_val)
                
                ratios[states[-1]] = total - sum(ratios.values())
                st.session_state.random_distribution = ratios
                st.success(" Distribution randomized!")
                st.rerun()
        
        with col3:
            if st.button(" Save Enhanced Dataset"):
                if self.current_data:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"enhanced_dataset_{timestamp}.jsonl"
                    with open(filename, 'w') as f:
                        for sample in self.current_data:
                            f.write(json.dumps(sample) + '\n')
                    st.success(f" Enhanced dataset saved as {filename}")
                else:
                    st.warning(" No data to save")
        
        with col4:
            if st.button(" Clear Data"):
                self.current_data = []
                st.success(" Data cleared")
                st.rerun()
        
        # Display generation parameters
        st.markdown("###  Current Generation Parameters")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("** Data Parameters**")
            st.write(f"Sample Size: {parameters['sample_size']:,}")
            st.write(f"Variation Level: {parameters['variation_level']:.2f}")
            st.write(f"Generation Count: {self.generation_count}")
            st.write(f"Emotional Variety: {parameters['emotional_variation']:.2f}")
        
        with col2:
            st.markdown("** Brain State Distribution**")
            for state, ratio in parameters['brain_state_ratios'].items():
                if ratio > 0:
                    st.write(f"{state.title()}: {ratio:.1%}")
        
        with col3:
            st.markdown("** Quality Parameters**")
            st.write(f"Noise Level: {parameters['noise_level']:.2f}")
            st.write(f"Signal Amplitude: {parameters['signal_amplitude']} μV")
            
            if self.generation_start_time:
                generation_time = time.time() - self.generation_start_time
                st.write(f"Generation Time: {generation_time:.2f}s")
    
    def create_enhanced_analysis(self):
        """Create enhanced analysis with variation tracking"""
        if not self.current_data:
            st.info(" Generate enhanced data using the controls above to see analysis")
            return
        
        st.markdown("###  Enhanced Data Analysis with Variation")
        
        # Basic statistics
        total_samples = len(self.current_data)
        
        # Brain state distribution with variation analysis
        brain_states = [sample.get('brain_state', 'unknown') for sample in self.current_data]
        state_counts = pd.Series(brain_states).value_counts()
        
        # Emotional state distribution with variety analysis
        emotional_states = [sample.get('emotional_state', 'unknown') for sample in self.current_data]
        emotion_counts = pd.Series(emotional_states).value_counts()
        
        # Display enhanced metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(" Total Samples", f"{total_samples:,}")
        
        with col2:
            st.metric(" Brain States", f"{len(state_counts)}")
        
        with col3:
            st.metric(" Emotions", f"{len(emotion_counts)}")
        
        with col4:
            # Calculate variation score
            state_entropy = -sum((count/total_samples) * np.log2(count/total_samples) for count in state_counts)
            max_entropy = np.log2(len(state_counts))
            variation_score = (state_entropy / max_entropy) * 100 if max_entropy > 0 else 0
            st.metric(" Variation Score", f"{variation_score:.1f}%")
        
        # Enhanced distribution charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("** Brain State Distribution (Enhanced)**")
            fig = px.pie(
                values=state_counts.values,
                names=state_counts.index,
                title=f"Brain States - Generation #{self.generation_count}",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
            
            # Show variation from expected
            if hasattr(self, 'generator') and self.generator.custom_state_distribution:
                st.markdown("** Distribution vs Expected**")
                expected = self.generator.custom_state_distribution
                actual = {state: count/total_samples for state, count in state_counts.items()}
                
                comparison_data = []
                for state in expected.keys():
                    comparison_data.append({
                        'State': state.title(),
                        'Expected': expected[state] * 100,
                        'Actual': actual.get(state, 0) * 100
                    })
                
                df_comparison = pd.DataFrame(comparison_data)
                fig = px.bar(
                    df_comparison.melt(id_vars=['State'], var_name='Type', value_name='Percentage'),
                    x='State',
                    y='Percentage',
                    color='Type',
                    title='Expected vs Actual Distribution',
                    barmode='group'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("** Emotional State Distribution (Enhanced)**")
            fig = px.bar(
                x=emotion_counts.index,
                y=emotion_counts.values,
                title=f"Emotional States - Generation #{self.generation_count}",
                color=emotion_counts.values,
                color_continuous_scale="Viridis"
            )
            fig.update_layout(xaxis_title="Emotional State", yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)
            
            # Show emotional variety score
            emotion_entropy = -sum((count/total_samples) * np.log2(count/total_samples) for count in emotion_counts)
            max_emotion_entropy = np.log2(len(emotion_counts))
            emotion_variety = (emotion_entropy / max_emotion_entropy) * 100 if max_emotion_entropy > 0 else 0
            st.metric(" Emotional Variety", f"{emotion_variety:.1f}%")
        
        # Temporal variation analysis
        if len(self.current_data) > 100:
            st.markdown("###  Temporal Variation Analysis")
            
            # Analyze how states change over time
            time_states = []
            for i, sample in enumerate(self.current_data[::10]):  # Sample every 10th
                time_states.append({
                    'time_index': i * 10,
                    'brain_state': sample.get('brain_state', 'unknown'),
                    'emotional_state': sample.get('emotional_state', 'unknown')
                })
            
            df_temporal = pd.DataFrame(time_states)
            
            # Brain state changes over time
            fig = px.scatter(
                df_temporal,
                x='time_index',
                y='brain_state',
                color='brain_state',
                title='Brain State Changes Over Time',
                labels={'time_index': 'Sample Index', 'brain_state': 'Brain State'}
            )
            st.plotly_chart(fig, use_container_width=True)

def main():
    """Main application function"""
    
    # Initialize the enhanced generator
    if 'enhanced_generator' not in st.session_state:
        st.session_state.enhanced_generator = EnhancedDynamicGenerator()
    
    generator = st.session_state.enhanced_generator
    
    # Create the interface
    generator.create_generation_interface()
    
    # Create enhanced analysis
    generator.create_enhanced_analysis()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p> Enhanced Neuro-Brainwave Data Generator | Dynamic State Variations</p>
        <p>Each generation creates different brain state and emotional patterns</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
