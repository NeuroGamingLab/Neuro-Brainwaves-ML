#!/usr/bin/env python3
"""
Simplified Interactive Brain Signal Generator & Visualizer
Fixed version with better real-time updates
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import threading
import json
import sys
import os
from collections import deque

# Add current directory to path to import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dynamic_brainwave_generator import DynamicBrainwaveGenerator

# Page configuration
st.set_page_config(
    page_title="Brain Signal Generator",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

class SimpleBrainSignalApp:
    def __init__(self):
        self.generator = DynamicBrainwaveGenerator()
        
        # Initialize session state
        if 'session_data' not in st.session_state:
            st.session_state.session_data = []
        if 'session_started' not in st.session_state:
            st.session_state.session_started = False
        if 'session_stats' not in st.session_state:
            st.session_state.session_stats = {
                'total_samples': 0,
                'current_brain_state': 'normal',
                'avg_variation': 0.0
            }
    
    def generate_sample_data(self, brain_state, duration, variability):
        """Generate sample brainwave data"""
        # Generate brain state events
        brain_state_events = self.generator.generate_brain_state_events(
            duration, brain_state, variability
        )
        
        # Calculate number of epochs (limit for demo)
        num_epochs = min(100, int(duration / self.generator.epoch_duration))
        start_time = np.random.uniform(0, 65)
        
        samples = []
        
        for epoch_num in range(num_epochs):
            # Generate brainwave EEG data
            signal = self.generator.generate_brainwave_eeg(
                epoch_num, duration, start_time, brain_state_events
            )
            
            # Get current brain state
            current_time = start_time + (epoch_num * self.generator.epoch_duration)
            current_brain_state, state_variation = self.generator.get_brain_state_at_time(
                current_time, brain_state_events
            )
            
            # Create sample data
            sample_data = {
                'timestamp': current_time,
                'data': signal,
                'brain_state': current_brain_state,
                'state_variation': state_variation,
                'epoch_num': epoch_num
            }
            
            samples.append(sample_data)
        
        return samples
    
    def generate_session_with_transitions(self, brain_state, variability, total_duration):
        """Generate a complete session with proper brain state transitions"""
        # Generate brain state events for the entire session duration
        brain_state_events = self.generator.generate_brain_state_events(
            total_duration, brain_state, variability
        )
        
        # Calculate number of epochs (1 epoch per second)
        num_epochs = int(total_duration)
        start_time = np.random.uniform(0, 65)
        
        samples = []
        
        for epoch_num in range(num_epochs):
            # Generate brainwave EEG data
            signal = self.generator.generate_brainwave_eeg(
                epoch_num, total_duration, start_time, brain_state_events
            )
            
            # Get current brain state at this time
            current_time = start_time + (epoch_num * self.generator.epoch_duration)
            current_brain_state, state_variation = self.generator.get_brain_state_at_time(
                current_time, brain_state_events
            )
            
            # Create sample data
            sample_data = {
                'timestamp': current_time,
                'data': signal,
                'brain_state': current_brain_state,
                'state_variation': state_variation,
                'epoch_num': epoch_num
            }
            
            samples.append(sample_data)
        
        return samples
    
    def generate_single_sample(self, brain_state, variability, epoch_num, brain_state_events, start_time, total_duration):
        """Generate a single brain signal sample from pre-generated session"""
        # Generate brainwave EEG data
        signal = self.generator.generate_brainwave_eeg(
            epoch_num, total_duration, start_time, brain_state_events
        )
        
        # Get current brain state at this time
        current_time = start_time + (epoch_num * self.generator.epoch_duration)
        current_brain_state, state_variation = self.generator.get_brain_state_at_time(
            current_time, brain_state_events
        )
        
        # Create sample data
        sample_data = {
            'timestamp': current_time,
            'data': signal,
            'brain_state': current_brain_state,
            'state_variation': state_variation,
            'epoch_num': epoch_num
        }
        
        return sample_data
    
    def create_visualization(self, samples):
        """Create visualization of brain signals"""
        if not samples:
            return None
        
        # Extract data
        timestamps = [s['timestamp'] for s in samples]
        brain_states = [s['brain_state'] for s in samples]
        variations = [s['state_variation'] for s in samples]
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=[
                ' Brainwave Signals (All Channels)',
                ' Brain State Timeline',
                ' State Variation Over Time'
            ],
            vertical_spacing=0.08
        )
        
        # Plot 1: Channel data (averaged for visualization)
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
        
        for ch in range(min(8, samples[0]['data'].shape[0])):
            channel_data = [s['data'][ch, :].mean() for s in samples]
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=channel_data,
                    mode='lines',
                    name=self.generator.channel_names[ch],
                    line=dict(color=colors[ch], width=1),
                    opacity=0.8
                ),
                row=1, col=1
            )
        
        # Plot 2: Brain state timeline
        state_colors = {
            'normal': 'green', 'excited': 'red', 'relaxed': 'blue',
            'focused': 'orange', 'stressed': 'purple', 'sleepy': 'brown'
        }
        
        # Create state segments
        current_state = brain_states[0]
        start_time = timestamps[0]
        segments = []
        
        for i, (time_val, state) in enumerate(zip(timestamps, brain_states)):
            if state != current_state or i == len(timestamps) - 1:
                segments.append({
                    'start': start_time,
                    'end': time_val,
                    'state': current_state,
                    'color': state_colors.get(current_state, 'gray')
                })
                current_state = state
                start_time = time_val
        
        for segment in segments:
            fig.add_trace(
                go.Scatter(
                    x=[segment['start'], segment['end'], segment['end'], segment['start'], segment['start']],
                    y=[0, 0, 1, 1, 0],
                    fill='toself',
                    fillcolor=segment['color'],
                    line=dict(color=segment['color']),
                    mode='lines',
                    name=segment['state'],
                    showlegend=False,
                    opacity=0.7
                ),
                row=2, col=1
            )
        
        # Plot 3: State variation
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=variations,
                mode='lines',
                name='State Variation',
                line=dict(color='red', width=2)
            ),
            row=3, col=1
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            showlegend=True,
            title_text="Brain Signal Visualization",
            title_font_size=20
        )
        
        # Update axes
        fig.update_xaxes(title_text="Time (seconds)", row=3, col=1)
        fig.update_yaxes(title_text="Amplitude (μV)", row=1, col=1)
        fig.update_yaxes(title_text="State", row=2, col=1)
        fig.update_yaxes(title_text="Variation Factor", row=3, col=1)
        
        return fig

def main():
    # Initialize the app
    app = SimpleBrainSignalApp()
    
    # Main header
    st.title(" Brain Signal Generator & Visualizer")
    st.markdown("Generate and visualize realistic brainwave signals in real-time")
    
    # Brain state reference
    with st.expander(" Brain State Reference", expanded=False):
        st.markdown("""
        ###  **EEG Brain States & Frequency Bands**
        
        #### **Normal State**
        - **Alpha waves (8-13 Hz)**: Dominant, eyes closed relaxation
        - **Beta waves (13-30 Hz)**: Light cognitive activity
        - **Theta waves (4-8 Hz)**: Drowsiness, meditation
        - **Delta waves (0.5-4 Hz)**: Deep sleep (minimal in awake state)
        
        #### **Excited State** 
        - **Beta/Gamma dominance**: High cognitive activity, alertness
        - **Reduced Alpha**: Less relaxation, more active thinking
        - **Increased frontal activity**: F5/F6 channels show executive function
        
        #### **Stressed State**
        - **High Beta activity**: Anxious, worried thinking
        - **Gamma bursts**: Stress-related neural activity
        - **Increased muscle tension**: Higher noise levels
        
        #### **Relaxed State**
        - **Alpha dominance**: Calm, meditative state
        - **Theta presence**: Deep relaxation
        - **Reduced Beta**: Less active thinking
        
        #### **Focused State**
        - **Beta concentration**: Sustained attention
        - **Reduced Alpha**: Suppressed relaxation for focus
        - **Frontal engagement**: F5/F6 executive function
        
        #### **Sleepy State**
        - **Theta dominance**: Drowsiness, pre-sleep
        - **Delta appearance**: Early sleep stages
        - **Reduced overall amplitude**: Lower brain activity
        """)
        
        st.markdown("""
        ###  **EEG Channel Locations (10-20 System)**
        
        | Channel | Location | Brain Region | Function |
        |---------|----------|--------------|----------|
        | **F5** | Left Frontal | Prefrontal Cortex | Executive function, decision making |
        | **F6** | Right Frontal | Prefrontal Cortex | Emotional regulation, planning |
        | **C3** | Left Central | Primary Motor Cortex | Right hand movement, motor control |
        | **C4** | Right Central | Primary Motor Cortex | Left hand movement, motor control |
        | **CP3** | Left Central-Parietal | Motor Planning | Movement preparation, spatial planning |
        | **CP4** | Right Central-Parietal | Motor Planning | Movement preparation, spatial planning |
        | **PO3** | Left Parietal-Occipital | Visual Association | Spatial processing, visual attention |
        | **PO4** | Right Parietal-Occipital | Visual Association | Spatial processing, visual attention |
        """)
    
    st.markdown("---")
    
    # Custom CSS for enhanced frequency band styling
    st.markdown("""
    <style>
    .frequency-band {
        padding: 10px;
        margin: 5px 0;
        border-radius: 6px;
        border-left: 4px solid;
        background-color: rgba(255, 255, 255, 0.1);
        transition: all 0.3s ease;
    }
    .frequency-band:hover {
        transform: translateX(5px);
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    .band-delta { border-left-color: #FF6B6B; background-color: #FF6B6B20; }
    .band-theta { border-left-color: #4ECDC4; background-color: #4ECDC420; }
    .band-alpha { border-left-color: #45B7D1; background-color: #45B7D120; }
    .band-beta { border-left-color: #96CEB4; background-color: #96CEB420; }
    .band-gamma { border-left-color: #FFEAA7; background-color: #FFEAA720; }
    .band-activity-card {
        padding: 12px;
        margin: 8px 0;
        border-radius: 8px;
        border-left: 4px solid;
        background-color: rgba(255, 255, 255, 0.05);
        transition: all 0.2s ease;
    }
    .band-activity-card:hover {
        background-color: rgba(255, 255, 255, 0.1);
        transform: scale(1.02);
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Sidebar controls
    with st.sidebar:
        st.header(" Control Panel")
        
        # Brain state selection
        brain_state = st.selectbox(
            " Brain State",
            options=['normal', 'excited', 'relaxed', 'focused', 'stressed', 'sleepy'],
            index=4,  # Default to stressed
            help="Select the type of brain activity to simulate"
        )
        
        # Duration slider
        duration = st.slider(
            " Duration (seconds)",
            min_value=10,
            max_value=300,
            value=60,
            step=10,
            help="How long to generate brain signals"
        )
        
        # Variability slider
        variability = st.slider(
            " Variability",
            min_value=0.1,
            max_value=1.0,
            value=0.7,
            step=0.1,
            help="How much variation in brain state"
        )
        
        # Display brain state info
        if brain_state in app.generator.brain_states:
            state_info = app.generator.brain_states[brain_state]
            st.info(f"""
            **{state_info['name']}**
            
            {state_info['description']}
            """)
        
        # Display dynamic frequency band information based on current brain state
        st.subheader(" Dynamic Active Frequency Bands")
        
        # Get current brain state configuration
        if brain_state in app.generator.brain_states:
            state_config = app.generator.brain_states[brain_state]
            state_frequency_bands = state_config['frequency_bands']
            
            st.markdown(f"**Current State: {state_config['name']}**")
            st.markdown(f"*{state_config['description']}*")
            
            # Display frequency bands with state-specific characteristics
            st.markdown("** State-Specific Active Frequency Bands:**")
            
            for band_name, (low_freq, high_freq, min_amp, max_amp) in state_frequency_bands.items():
                color = app.generator.frequency_band_colors[band_name]
                css_class = f"band-{band_name}"
                
                # Determine if this band is typically active in this state
                is_typically_active = (min_amp > 10) or (max_amp > 30)  # Higher amplitude ranges indicate more activity
                activity_indicator = "" if is_typically_active else ""
                
                # Get band description
                band_descriptions = {
                    'delta': 'Deep sleep, unconsciousness',
                    'theta': 'Drowsiness, meditation',
                    'alpha': 'Relaxed, eyes closed',
                    'beta': 'Active thinking, focus',
                    'gamma': 'High cognitive processing'
                }
                
                st.markdown(f"""
                <div class="frequency-band {css_class}">
                    <strong>{activity_indicator} {band_name.upper()}</strong> ({low_freq}-{high_freq} Hz): <span style="color: {color}">{min_amp}-{max_amp} μV</span><br>
                    <small>{band_descriptions[band_name]}</small>
                </div>
                """, unsafe_allow_html=True)
        
        # Show real-time active bands if session is running
        if st.session_state.get('generation_active', False) and st.session_state.session_data:
            recent_sample = st.session_state.session_data[-1] if st.session_state.session_data else None
            if recent_sample and 'dominant_bands' in recent_sample:
                st.markdown("** Real-Time Active Bands:**")
                if recent_sample['dominant_bands']:
                    # Create a visual activity meter for each active band
                    for band in recent_sample['dominant_bands']:
                        color = app.generator.frequency_band_colors[band]
                        freq_range = app.generator.standard_frequency_bands[band]
                        
                        # Create a progress bar for band activity
                        activity_level = np.random.uniform(0.6, 1.0)  # Simulate activity level
                        
                        st.markdown(f"""
                        <div style="margin: 5px 0; padding: 8px; border-left: 4px solid {color}; background-color: {color}20; border-radius: 4px;">
                            <strong style="color: {color}; font-size: 14px;"> {band.upper()}</strong> ({freq_range[0]}-{freq_range[1]} Hz)<br>
                            <small>Activity Level: {activity_level:.1%}</small>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Add a progress bar for visual feedback
                        st.progress(activity_level)
                else:
                    st.markdown(" *No dominant bands detected*")
            
            # Add frequency band activity timeline
            if len(st.session_state.session_data) > 10:
                st.markdown("** Recent Band Activity:**")
                
                # Get last 10 samples for activity timeline
                recent_samples = st.session_state.session_data[-10:]
                band_activity_counts = {}
                
                for sample in recent_samples:
                    if 'dominant_bands' in sample:
                        for band in sample['dominant_bands']:
                            band_activity_counts[band] = band_activity_counts.get(band, 0) + 1
                
                # Display activity percentages
                total_samples = len(recent_samples)
                for band, count in sorted(band_activity_counts.items(), key=lambda x: x[1], reverse=True):
                    percentage = (count / total_samples) * 100
                    color = app.generator.frequency_band_colors[band]
                    
                    st.markdown(f"""
                    <div style="display: flex; justify-content: space-between; align-items: center; margin: 2px 0;">
                        <span style="color: {color}; font-weight: bold;">{band.upper()}</span>
                        <span>{percentage:.0f}% active</span>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Visual progress bar
                    st.progress(percentage / 100)
        
        # Enhanced frequency band reference section
        with st.expander(" Frequency Band Reference & Analysis"):
            st.markdown("** Standard Frequency Band Ranges:**")
            
            # Interactive frequency band selector
            selected_band = st.selectbox(
                "Select Band for Detailed Analysis:",
                ["DELTA", "THETA", "ALPHA", "BETA", "GAMMA"],
                help="Choose a frequency band to see detailed information"
            )
            
            # Detailed band information
            band_details = {
                "DELTA": {
                    "freq_range": "0.5-4 Hz",
                    "amplitude": "5-20 μV",
                    "description": "Deep sleep, unconsciousness",
                    "color": "#FF6B6B",
                    "brain_states": ["sleepy", "relaxed"],
                    "characteristics": "Slowest brain waves, dominant during deep sleep and unconscious states",
                    "clinical_significance": "Associated with healing, growth hormone release, and deep restorative sleep"
                },
                "THETA": {
                    "freq_range": "4-8 Hz",
                    "amplitude": "8-25 μV", 
                    "description": "Drowsiness, meditation",
                    "color": "#4ECDC4",
                    "brain_states": ["sleepy", "relaxed"],
                    "characteristics": "Associated with deep relaxation, meditation, and light sleep",
                    "clinical_significance": "Important for memory consolidation and emotional processing"
                },
                "ALPHA": {
                    "freq_range": "8-13 Hz",
                    "amplitude": "5-25 μV",
                    "description": "Relaxed, eyes closed",
                    "color": "#45B7D1",
                    "brain_states": ["relaxed", "normal"],
                    "characteristics": "Present when awake but relaxed, eyes closed, calm and peaceful",
                    "clinical_significance": "Associated with creativity, relaxation, and stress reduction"
                },
                "BETA": {
                    "freq_range": "13-30 Hz",
                    "amplitude": "25-60 μV",
                    "description": "Active thinking, focus",
                    "color": "#96CEB4",
                    "brain_states": ["normal", "excited", "focused", "stressed"],
                    "characteristics": "Dominant during active concentration, problem-solving, and alertness",
                    "clinical_significance": "Essential for cognitive function, attention, and decision-making"
                },
                "GAMMA": {
                    "freq_range": "30-45 Hz",
                    "amplitude": "10-30 μV",
                    "description": "High cognitive processing",
                    "color": "#FFEAA7",
                    "brain_states": ["excited", "focused", "stressed"],
                    "characteristics": "Fastest brain waves, associated with high-level cognitive processing",
                    "clinical_significance": "Linked to consciousness, perception, and binding of sensory information"
                }
            }
            
            # Display selected band details
            if selected_band in band_details:
                details = band_details[selected_band]
                
                st.markdown(f"""
                <div style="padding: 15px; border: 2px solid {details['color']}; border-radius: 8px; background-color: {details['color']}10; margin: 10px 0;">
                    <h4 style="color: {details['color']}; margin-top: 0;">{selected_band} WAVES</h4>
                    <p><strong>Frequency Range:</strong> {details['freq_range']}</p>
                    <p><strong>Amplitude Range:</strong> {details['amplitude']}</p>
                    <p><strong>Description:</strong> {details['description']}</p>
                    <p><strong>Characteristics:</strong> {details['characteristics']}</p>
                    <p><strong>Clinical Significance:</strong> {details['clinical_significance']}</p>
                    <p><strong>Common in Brain States:</strong> {', '.join(details['brain_states'])}</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("** All Frequency Bands Overview:**")
            reference_bands = [
                ("DELTA", "0.5-4 Hz", "5-20 μV", "Deep sleep, unconsciousness", "band-delta", "#FF6B6B"),
                ("THETA", "4-8 Hz", "8-25 μV", "Drowsiness, meditation", "band-theta", "#4ECDC4"),
                ("ALPHA", "8-13 Hz", "5-25 μV", "Relaxed, eyes closed", "band-alpha", "#45B7D1"),
                ("BETA", "13-30 Hz", "25-60 μV", "Active thinking, focus", "band-beta", "#96CEB4"),
                ("GAMMA", "30-45 Hz", "10-30 μV", "High cognitive processing", "band-gamma", "#FFEAA7")
            ]
            
            for band_name, freq_range, amplitude, description, css_class, color in reference_bands:
                st.markdown(f"""
                <div class="frequency-band {css_class}" style="opacity: 0.8; margin: 5px 0; padding: 8px; border-radius: 4px;">
                    <strong>{band_name}</strong> ({freq_range}): <span style="color: {color}">{amplitude}</span><br>
                    <small>{description}</small>
                </div>
                """, unsafe_allow_html=True)
            
            # Add frequency band comparison chart
            st.markdown("** Frequency Band Comparison:**")
            
            # Create a simple comparison using metrics
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("DELTA", "0.5-4 Hz", "Sleep")
            with col2:
                st.metric("THETA", "4-8 Hz", "Meditation")
            with col3:
                st.metric("ALPHA", "8-13 Hz", "Relaxed")
            with col4:
                st.metric("BETA", "13-30 Hz", "Active")
            with col5:
                st.metric("GAMMA", "30-45 Hz", "Processing")
        
        st.divider()
        
        # EEG Channel Information
        st.subheader(" EEG Channels")
        st.markdown("""
        **Neurosity EEG Electrode Positions:**
        - **CP3, CP4**: Central-Parietal (motor cortex, planning)
        - **C3, C4**: Central (primary motor cortex, hand movement)
        - **F5, F6**: Frontal (executive function, decision making)
        - **PO3, PO4**: Parietal-Occipital (visual processing, spatial awareness)
        """)
        
        # Channel significance
        st.markdown("""
        **Channel Functions:**
        - **Frontal (F5/F6)**: Attention, working memory, emotional regulation
        - **Central (C3/C4)**: Motor control, sensory processing
        - **Parietal (CP3/CP4)**: Spatial processing, attention
        - **Occipital (PO3/PO4)**: Visual processing, eye movements
        """)
        
        st.divider()
        
        # Generate button
        generate_button = st.button(
            " Generate Brain Signals",
            type="primary",
            help="Generate brain signals with current settings"
        )
        
        # Clear data button
        if st.button(" Clear Data", help="Clear all generated data"):
            st.session_state.session_data = []
            st.session_state.session_stats = {
                'total_samples': 0,
                'current_brain_state': 'normal',
                'avg_variation': 0.0
            }
            st.session_state.generation_active = False
            st.rerun()
        
        # Stop generation button (only show when generating)
        if st.session_state.get('generation_active', False):
            if st.button(" Stop Generation", help="Stop the current generation process"):
                st.session_state.generation_active = False
                st.rerun()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Status display
        if st.session_state.get('generation_active', False):
            progress = st.session_state.current_sample / st.session_state.target_samples
            st.progress(progress)
            st.info(f" Generating sample {st.session_state.current_sample + 1}/{st.session_state.target_samples} - {st.session_state.brain_state} state")
        elif st.session_state.session_data:
            st.success(f" Generated {len(st.session_state.session_data)} brain signal samples!")
        
        # Visualization
        if st.session_state.session_data:
            st.subheader(" Brain Signal Visualization")
            fig = app.create_visualization(st.session_state.session_data)
            if fig:
                st.plotly_chart(fig, width='stretch')
            else:
                st.error(" Could not create visualization")
        else:
            st.info(" No data to display. Click 'Generate Brain Signals' to create visualization.")
    
    with col2:
        # Statistics panel
        st.subheader(" Statistics")
        
        if st.session_state.session_data:
            # Key metrics
            st.metric(
                label="Total Samples",
                value=len(st.session_state.session_data)
            )
            
            # Current brain state
            last_sample = st.session_state.session_data[-1]
            st.metric(
                label="Final State",
                value=last_sample['brain_state'].title()
            )
            
            # Average variation
            avg_variation = np.mean([s['state_variation'] for s in st.session_state.session_data])
            st.metric(
                label="Avg Variation",
                value=f"{avg_variation:.2f}"
            )
            
            # Data summary
            st.subheader(" Data Summary")
            df_data = []
            for sample in st.session_state.session_data[-10:]:  # Last 10 samples
                df_data.append({
                    'Time': f"{sample['timestamp']:.2f}s",
                    'State': sample['brain_state'],
                    'Variation': f"{sample['state_variation']:.2f}",
                    'Epoch': sample['epoch_num']
                })
            
            if df_data:
                df = pd.DataFrame(df_data)
                st.dataframe(df, width='stretch')
            
            # Current brain state explanation
            if st.session_state.session_data:
                current_state = st.session_state.session_data[-1]['brain_state']
                if current_state in app.generator.brain_states:
                    state_info = app.generator.brain_states[current_state]
                    
                    st.subheader(f" Current State: {state_info['name']}")
                    
                    # Frequency band information
                    freq_bands = state_info.get('frequency_bands', {})
                    if freq_bands:
                        st.markdown("**Active Frequency Bands:**")
                        for band, (low, high, min_amp, max_amp) in freq_bands.items():
                            st.markdown(f"- **{band.upper()}** ({low}-{high} Hz): Amplitude {min_amp}-{max_amp} μV")
                    
                    # EEG characteristics by channel
                    st.markdown("**EEG Characteristics by Channel:**")
                    channel_explanations = {
                        'CP3': 'Left motor planning - increased activity in movement preparation',
                        'C3': 'Left motor cortex - primary hand movement control',
                        'F5': 'Left frontal - executive function and attention',
                        'PO3': 'Left parietal-occipital - spatial processing and visual attention',
                        'PO4': 'Right parietal-occipital - spatial processing and visual attention',
                        'F6': 'Right frontal - executive function and emotional regulation',
                        'C4': 'Right motor cortex - primary hand movement control',
                        'CP4': 'Right motor planning - increased activity in movement preparation'
                    }
                    
                    for channel, explanation in channel_explanations.items():
                        st.markdown(f"- **{channel}**: {explanation}")
            
            # Download button
            st.subheader(" Download Data")
            export_data = []
            for sample in st.session_state.session_data:
                export_data.append({
                    'timestamp': sample['timestamp'],
                    'brain_state': sample['brain_state'],
                    'state_variation': sample['state_variation'],
                    'epoch_num': sample['epoch_num'],
                    'data_shape': sample['data'].shape
                })
            
            json_data = json.dumps(export_data, indent=2)
            st.download_button(
                label="Download JSON",
                data=json_data,
                file_name=f"brain_signal_data_{int(time.time())}.json",
                mime="application/json"
            )
        else:
            st.info("No data available. Generate signals to see statistics.")
    
    # Handle generate button
    if generate_button:
        # Initialize generation session with proper state transitions
        st.session_state.generation_active = True
        st.session_state.session_data = []
        st.session_state.current_sample = 0
        st.session_state.target_samples = int(duration)
        st.session_state.brain_state = brain_state
        st.session_state.variability = variability
        
        # Pre-generate the entire session with state transitions
        st.session_state.full_session = app.generate_session_with_transitions(
            brain_state, variability, duration
        )
        
        st.rerun()

    # Auto-generate samples when generation is active
    if st.session_state.get('generation_active', False):
        if st.session_state.current_sample < st.session_state.target_samples:
            # Get the pre-generated sample for this epoch
            sample = st.session_state.full_session[st.session_state.current_sample]
            
            # Add to session data
            st.session_state.session_data.append(sample)
            st.session_state.current_sample += 1
            
            # Auto-refresh after 1 second
            time.sleep(1.0)
            st.rerun()
        else:
            # Generation complete
            st.session_state.generation_active = False
            st.session_state.session_stats = {
                'total_samples': len(st.session_state.session_data),
                'current_brain_state': st.session_state.session_data[-1]['brain_state'] if st.session_state.session_data else 'normal',
                'avg_variation': np.mean([s['state_variation'] for s in st.session_state.session_data]) if st.session_state.session_data else 0.0
            }
            st.success(f" Generated {st.session_state.target_samples} samples with state transitions!")
            st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p> Brain Signal Generator & Visualizer | Built with Streamlit</p>
        <p>Designed by Tuệ Hoàng, Eng.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
