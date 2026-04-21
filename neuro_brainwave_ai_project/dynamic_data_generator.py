#!/usr/bin/env python3
"""
Dynamic Data Generator for Neuro-Brainwave AI System
==================================================

Interactive Streamlit application with sliders for dynamic data generation
and real-time analysis of neuro-brainwave data.

Features:
- Dynamic sample size adjustment with sliders
- Real-time data generation and visualization
- Interactive parameter control
- Live analysis and statistics
- Export capabilities
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

# Add current directory to path for imports
sys.path.append('.')

try:
    from neuro_brainwave_data_generator_agent import NeuroBrainwaveDataGeneratorAgent
except ImportError:
    st.error(" Could not import NeuroBrainwaveDataGeneratorAgent. Please ensure the file exists.")
    st.stop()

# Configure Streamlit page
st.set_page_config(
    page_title="Dynamic Neuro-Brainwave Generator",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .slider-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 4px solid #1f77b4;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
        margin: 0.5rem 0;
    }
    .generation-status {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class DynamicDataGenerator:
    """Dynamic data generator with interactive controls"""
    
    def __init__(self):
        self.generator = None
        self.current_data = []
        self.generation_start_time = None
        
    def create_parameter_controls(self):
        """Create interactive parameter controls"""
        st.sidebar.header(" Generation Parameters")
        
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
        
        # Generation speed slider
        generation_speed = st.sidebar.slider(
            "Generation Speed (samples/sec)",
            min_value=50,
            max_value=500,
            value=250,
            step=25,
            help="Speed of data generation (higher = faster)"
        )
        
        # Brain state distribution sliders
        st.sidebar.markdown("###  Brain State Distribution")
        
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            normal_ratio = st.slider("Normal", 0.0, 1.0, 0.3, 0.05)
            excited_ratio = st.slider("Excited", 0.0, 1.0, 0.2, 0.05)
            relaxed_ratio = st.slider("Relaxed", 0.0, 1.0, 0.2, 0.05)
        
        with col2:
            focused_ratio = st.slider("Focused", 0.0, 1.0, 0.15, 0.05)
            stressed_ratio = st.slider("Stressed", 0.0, 1.0, 0.1, 0.05)
            sleepy_ratio = st.slider("Sleepy", 0.0, 1.0, 0.05, 0.05)
        
        # Normalize ratios
        total_ratio = normal_ratio + excited_ratio + relaxed_ratio + focused_ratio + stressed_ratio + sleepy_ratio
        if total_ratio > 0:
            normal_ratio /= total_ratio
            excited_ratio /= total_ratio
            relaxed_ratio /= total_ratio
            focused_ratio /= total_ratio
            stressed_ratio /= total_ratio
            sleepy_ratio /= total_ratio
        
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
        
        # Environmental factors
        st.sidebar.markdown("###  Environmental Factors")
        
        time_variation = st.sidebar.slider(
            "Time Variation",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1,
            help="Variation in time-of-day effects"
        )
        
        activity_variation = st.sidebar.slider(
            "Activity Variation",
            min_value=0.0,
            max_value=1.0,
            value=0.6,
            step=0.1,
            help="Variation in activity level effects"
        )
        
        return {
            'sample_size': sample_size,
            'generation_speed': generation_speed,
            'brain_state_ratios': {
                'normal': normal_ratio,
                'excited': excited_ratio,
                'relaxed': relaxed_ratio,
                'focused': focused_ratio,
                'stressed': stressed_ratio,
                'sleepy': sleepy_ratio
            },
            'noise_level': noise_level,
            'signal_amplitude': signal_amplitude,
            'time_variation': time_variation,
            'activity_variation': activity_variation
        }
    
    def generate_data(self, parameters):
        """Generate data with given parameters"""
        try:
            # Initialize generator
            self.generator = NeuroBrainwaveDataGeneratorAgent(
                target_samples=parameters['sample_size']
            )
            
            # Store generation parameters
            self.generation_start_time = time.time()
            
            # Generate data
            output_file = self.generator.generate_dataset('dynamic_generated_data.jsonl')
            
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
        """Create the main generation interface"""
        st.markdown('<h1 class="main-header"> Dynamic Neuro-Brainwave Data Generator</h1>', unsafe_allow_html=True)
        
        # Get parameters from sliders
        parameters = self.create_parameter_controls()
        
        # Generation controls
        st.markdown("###  Data Generation Controls")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button(" Generate New Data", type="primary"):
                with st.spinner("Generating data..."):
                    if self.generate_data(parameters):
                        st.success(" Data generated successfully!")
                        st.rerun()
        
        with col2:
            if st.button(" Save Dataset"):
                if self.current_data:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"dynamic_dataset_{timestamp}.jsonl"
                    with open(filename, 'w') as f:
                        for sample in self.current_data:
                            f.write(json.dumps(sample) + '\n')
                    st.success(f" Dataset saved as {filename}")
                else:
                    st.warning(" No data to save")
        
        with col3:
            if st.button(" Analyze Current Data"):
                if self.current_data:
                    st.success(" Data analysis displayed below")
                    st.rerun()
                else:
                    st.warning(" Generate data first")
        
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
            st.write(f"Generation Speed: {parameters['generation_speed']} samples/sec")
            st.write(f"Noise Level: {parameters['noise_level']:.2f}")
            st.write(f"Signal Amplitude: {parameters['signal_amplitude']} μV")
        
        with col2:
            st.markdown("** Brain State Distribution**")
            for state, ratio in parameters['brain_state_ratios'].items():
                if ratio > 0:
                    st.write(f"{state.title()}: {ratio:.1%}")
        
        with col3:
            st.markdown("** Environmental Factors**")
            st.write(f"Time Variation: {parameters['time_variation']:.2f}")
            st.write(f"Activity Variation: {parameters['activity_variation']:.2f}")
            
            if self.generation_start_time:
                generation_time = time.time() - self.generation_start_time
                st.write(f"Generation Time: {generation_time:.2f}s")
    
    def create_data_overview(self):
        """Create data overview section"""
        if not self.current_data:
            st.info(" Generate data using the controls above to see analysis")
            return
        
        st.markdown("###  Generated Data Overview")
        
        # Basic statistics
        total_samples = len(self.current_data)
        
        # Brain state distribution
        brain_states = [sample.get('brain_state', 'unknown') for sample in self.current_data]
        state_counts = pd.Series(brain_states).value_counts()
        
        # Emotional state distribution
        emotional_states = [sample.get('emotional_state', 'unknown') for sample in self.current_data]
        emotion_counts = pd.Series(emotional_states).value_counts()
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(" Total Samples", f"{total_samples:,}")
        
        with col2:
            st.metric(" Brain States", f"{len(state_counts)}")
        
        with col3:
            st.metric(" Emotions", f"{len(emotion_counts)}")
        
        with col4:
            if self.generation_start_time:
                gen_time = time.time() - self.generation_start_time
                rate = total_samples / gen_time if gen_time > 0 else 0
                st.metric(" Generation Rate", f"{rate:.1f} samples/sec")
        
        # Brain state distribution chart
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("** Brain State Distribution**")
            fig = px.pie(
                values=state_counts.values,
                names=state_counts.index,
                title="Distribution of Brain States",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("** Emotional State Distribution**")
            fig = px.bar(
                x=emotion_counts.index,
                y=emotion_counts.values,
                title="Distribution of Emotional States",
                color=emotion_counts.values,
                color_continuous_scale="Viridis"
            )
            fig.update_layout(xaxis_title="Emotional State", yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)
    
    def create_eeg_analysis(self):
        """Create EEG signal analysis"""
        if not self.current_data:
            return
        
        st.markdown("###  EEG Signal Analysis")
        
        # Select sample for detailed analysis
        sample_options = list(range(min(10, len(self.current_data))))
        selected_sample = st.selectbox("Select sample for EEG analysis:", sample_options)
        
        if selected_sample is not None and selected_sample < len(self.current_data):
            sample = self.current_data[selected_sample]
            eeg_channels = sample.get('eeg_channels', {})
            
            if eeg_channels:
                # Create multi-channel EEG plot
                fig = make_subplots(
                    rows=4, cols=2,
                    subplot_titles=list(eeg_channels.keys()),
                    vertical_spacing=0.08,
                    horizontal_spacing=0.08
                )
                
                row_col_positions = [(1,1), (1,2), (2,1), (2,2), (3,1), (3,2), (4,1), (4,2)]
                
                for i, (channel_name, signal_data) in enumerate(eeg_channels.items()):
                    if i < 8:  # Limit to 8 channels
                        row, col = row_col_positions[i]
                        
                        # Create time axis (assuming 250 Hz sampling rate)
                        time_axis = np.arange(len(signal_data)) / 250.0
                        
                        fig.add_trace(
                            go.Scatter(
                                x=time_axis,
                                y=signal_data,
                                mode='lines',
                                name=channel_name,
                                line=dict(width=1),
                                showlegend=False
                            ),
                            row=row, col=col
                        )
                
                fig.update_layout(
                    title=f"EEG Signals - Sample {selected_sample}",
                    height=600,
                    showlegend=False
                )
                
                fig.update_xaxes(title_text="Time (seconds)")
                fig.update_yaxes(title_text="Amplitude (μV)")
                
                st.plotly_chart(fig, use_container_width=True)
    
    def create_frequency_analysis(self):
        """Create frequency band analysis"""
        if not self.current_data:
            return
        
        st.markdown("###  Frequency Band Analysis")
        
        # Extract frequency data
        freq_data = []
        for sample in self.current_data:
            freq_bands = sample.get('frequency_bands', {})
            if freq_bands:
                freq_bands['brain_state'] = sample.get('brain_state', 'unknown')
                freq_data.append(freq_bands)
        
        if not freq_data:
            st.warning("No frequency band data found")
            return
        
        df_freq = pd.DataFrame(freq_data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Frequency Band Power Distribution**")
            
            freq_bands = ['delta', 'theta', 'alpha', 'beta', 'gamma']
            available_bands = [band for band in freq_bands if band in df_freq.columns]
            
            if available_bands:
                fig = go.Figure()
                
                for band in available_bands:
                    fig.add_trace(go.Box(
                        y=df_freq[band],
                        name=band.upper(),
                        boxpoints='outliers'
                    ))
                
                fig.update_layout(
                    title="Frequency Band Power Distribution",
                    xaxis_title="Frequency Bands",
                    yaxis_title="Power",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("**Frequency Bands by Brain State**")
            
            if 'brain_state' in df_freq.columns and available_bands:
                pivot_data = df_freq.groupby('brain_state')[available_bands].mean()
                
                fig = px.imshow(
                    pivot_data.T,
                    labels=dict(x="Brain State", y="Frequency Band", color="Power"),
                    title="Average Frequency Band Power by Brain State",
                    color_continuous_scale="Viridis"
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    def create_behavioral_analysis(self):
        """Create behavioral pattern analysis"""
        if not self.current_data:
            return
        
        st.markdown("###  Behavioral Pattern Analysis")
        
        # Extract behavioral data
        behavioral_data = []
        for sample in self.current_data:
            behavioral = sample.get('behavioral_markers', {})
            if behavioral:
                behavioral['brain_state'] = sample.get('brain_state', 'unknown')
                behavioral['emotional_state'] = sample.get('emotional_state', 'unknown')
                behavioral_data.append(behavioral)
        
        if not behavioral_data:
            st.warning("No behavioral marker data found")
            return
        
        df_behavioral = pd.DataFrame(behavioral_data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Behavioral Markers Correlation**")
            
            behavioral_cols = ['attention', 'stress', 'fatigue', 'arousal']
            available_cols = [col for col in behavioral_cols if col in df_behavioral.columns]
            
            if len(available_cols) > 1:
                corr_matrix = df_behavioral[available_cols].corr()
                
                fig = px.imshow(
                    corr_matrix,
                    text_auto=True,
                    aspect="auto",
                    title="Behavioral Markers Correlation Matrix",
                    color_continuous_scale="RdBu_r"
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("**Behavioral Markers by Brain State**")
            
            if available_cols and 'brain_state' in df_behavioral.columns:
                fig = go.Figure()
                
                for col in available_cols:
                    fig.add_trace(go.Box(
                        y=df_behavioral[col],
                        x=df_behavioral['brain_state'],
                        name=col.title(),
                        boxpoints='outliers'
                    ))
                
                fig.update_layout(
                    title="Behavioral Markers by Brain State",
                    xaxis_title="Brain State",
                    yaxis_title="Intensity",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)

def main():
    """Main application function"""
    
    # Initialize the dynamic generator
    if 'dynamic_generator' not in st.session_state:
        st.session_state.dynamic_generator = DynamicDataGenerator()
    
    generator = st.session_state.dynamic_generator
    
    # Create the interface
    generator.create_generation_interface()
    
    # Create analysis sections
    generator.create_data_overview()
    
    # Create tabs for detailed analysis
    if generator.current_data:
        tab1, tab2, tab3 = st.tabs([" EEG Analysis", " Frequency Analysis", " Behavioral Analysis"])
        
        with tab1:
            generator.create_eeg_analysis()
        
        with tab2:
            generator.create_frequency_analysis()
        
        with tab3:
            generator.create_behavioral_analysis()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p> Dynamic Neuro-Brainwave Data Generator | Real-time Data Generation & Analysis</p>
        <p>Interactive sliders for parameter control and live visualization</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
