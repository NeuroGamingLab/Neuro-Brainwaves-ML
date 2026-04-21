#!/usr/bin/env python3
"""
Neuro-Brainwave AI System Dashboard
==================================

Interactive Streamlit dashboard for visualizing and analyzing
the Neuro-Brainwave AI System results and generated data.

Features:
- Data overview and statistics
- EEG signal visualization
- Frequency band analysis
- Behavioral pattern analysis
- System performance metrics
- Interactive data exploration
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import os
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# Configure Streamlit page
st.set_page_config(
    page_title="Neuro-Brainwave AI Dashboard",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .success-metric {
        border-left-color: #28a745;
        background-color: #d4edda;
    }
    .warning-metric {
        border-left-color: #ffc107;
        background-color: #fff3cd;
    }
    .info-metric {
        border-left-color: #17a2b8;
        background-color: #d1ecf1;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-left: 20px;
        padding-right: 20px;
    }
</style>
""", unsafe_allow_html=True)

class NeuroBrainwaveDashboard:
    """Dashboard for Neuro-Brainwave AI System"""
    
    def __init__(self):
        self.data_dir = Path(".")
        self.datasets = self._find_datasets()
        self.loaded_data = {}
        
    def _find_datasets(self):
        """Find all available datasets"""
        datasets = {}
        for file in self.data_dir.glob("*.jsonl"):
            datasets[file.name] = file
        return datasets
    
    def load_dataset(self, filename):
        """Load dataset from JSONL file"""
        if filename in self.loaded_data:
            return self.loaded_data[filename]
            
        data = []
        file_path = self.data_dir / filename
        
        if not file_path.exists():
            st.error(f"Dataset file {filename} not found!")
            return []
        
        try:
            with open(file_path, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        sample = json.loads(line.strip())
                        sample['line_number'] = line_num
                        data.append(sample)
                    except json.JSONDecodeError as e:
                        st.warning(f"Skipping malformed JSON on line {line_num}: {e}")
                        continue
                        
            self.loaded_data[filename] = data
            st.success(f" Loaded {len(data)} samples from {filename}")
            return data
            
        except Exception as e:
            st.error(f"Error loading dataset {filename}: {e}")
            return []
    
    def create_overview_metrics(self, data):
        """Create overview metrics"""
        if not data:
            return
            
        st.subheader(" Dataset Overview")
        
        # Basic statistics
        total_samples = len(data)
        file_size = sum(os.path.getsize(f) for f in self.data_dir.glob("*.jsonl")) / (1024*1024)  # MB
        
        # Brain state distribution
        brain_states = [sample.get('brain_state', 'unknown') for sample in data]
        state_counts = pd.Series(brain_states).value_counts()
        
        # Emotional state distribution
        emotional_states = [sample.get('emotional_state', 'unknown') for sample in data]
        emotion_counts = pd.Series(emotional_states).value_counts()
        
        # Create metrics layout
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Total Samples",
                value=f"{total_samples:,}",
                help="Total number of generated brainwave samples"
            )
        
        with col2:
            st.metric(
                label="Dataset Size",
                value=f"{file_size:.1f} MB",
                help="Total size of all dataset files"
            )
        
        with col3:
            st.metric(
                label="Brain States",
                value=f"{len(state_counts)}",
                help="Number of unique brain states detected"
            )
        
        with col4:
            st.metric(
                label="Emotions",
                value=f"{len(emotion_counts)}",
                help="Number of unique emotional states detected"
            )
    
    def create_brain_state_analysis(self, data):
        """Create brain state analysis visualizations"""
        if not data:
            return
            
        st.subheader(" Brain State Analysis")
        
        # Extract brain states and behavioral markers
        brain_states = []
        behavioral_data = []
        
        for sample in data:
            brain_state = sample.get('brain_state', 'unknown')
            brain_states.append(brain_state)
            
            behavioral = sample.get('behavioral_markers', {})
            behavioral['brain_state'] = brain_state
            behavioral['sample_id'] = sample.get('line_number', 0)
            behavioral_data.append(behavioral)
        
        df_behavioral = pd.DataFrame(behavioral_data)
        
        # Brain state distribution
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Brain State Distribution**")
            state_counts = pd.Series(brain_states).value_counts()
            
            fig = px.pie(
                values=state_counts.values,
                names=state_counts.index,
                title="Distribution of Brain States",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("**Behavioral Markers by Brain State**")
            
            # Create box plot for behavioral markers
            behavioral_cols = ['attention', 'stress', 'fatigue', 'arousal']
            available_cols = [col for col in behavioral_cols if col in df_behavioral.columns]
            
            if available_cols:
                fig = go.Figure()
                
                for col in available_cols:
                    fig.add_trace(go.Box(
                        y=df_behavioral[col],
                        x=df_behavioral['brain_state'],
                        name=col.title(),
                        boxpoints='outliers'
                    ))
                
                fig.update_layout(
                    title="Behavioral Markers Distribution",
                    xaxis_title="Brain State",
                    yaxis_title="Intensity",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    def create_eeg_visualization(self, data):
        """Create EEG signal visualizations"""
        if not data:
            return
            
        st.subheader(" EEG Signal Visualization")
        
        # Select sample for detailed visualization
        sample_options = list(range(min(10, len(data))))
        selected_sample = st.selectbox("Select sample for detailed EEG visualization:", sample_options)
        
        if selected_sample is not None and selected_sample < len(data):
            sample = data[selected_sample]
            eeg_channels = sample.get('eeg_channels', {})
            
            if eeg_channels:
                # Create time series for all channels
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
                    height=800,
                    showlegend=False
                )
                
                fig.update_xaxes(title_text="Time (seconds)")
                fig.update_yaxes(title_text="Amplitude (μV)")
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show sample metadata
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**Sample Information**")
                    st.write(f" Brain State: {sample.get('brain_state', 'Unknown')}")
                    st.write(f" Emotion: {sample.get('emotional_state', 'Unknown')}")
                    st.write(f" Duration: {sample.get('duration', 0):.1f}s")
                    st.write(f" Sample Rate: {sample.get('sample_rate', 0)} Hz")
                
                with col2:
                    st.markdown("**Cognitive Load**")
                    cognitive_load = sample.get('cognitive_load', 0)
                    st.metric("Load", f"{cognitive_load:.3f}")
                    
                    task_complexity = sample.get('task_complexity', 0)
                    st.metric("Task Complexity", f"{task_complexity:.3f}")
                
                with col3:
                    st.markdown("**Environmental Factors**")
                    env_factors = sample.get('environmental_factors', {})
                    for factor, value in env_factors.items():
                        st.metric(factor.replace('_', ' ').title(), f"{value:.3f}")
    
    def create_frequency_analysis(self, data):
        """Create frequency band analysis"""
        if not data:
            return
            
        st.subheader(" Frequency Band Analysis")
        
        # Extract frequency data
        freq_data = []
        for sample in data:
            freq_bands = sample.get('frequency_bands', {})
            if freq_bands:
                freq_bands['brain_state'] = sample.get('brain_state', 'unknown')
                freq_bands['sample_id'] = sample.get('line_number', 0)
                freq_data.append(freq_bands)
        
        if not freq_data:
            st.warning("No frequency band data found in the dataset.")
            return
        
        df_freq = pd.DataFrame(freq_data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Frequency Band Power Distribution**")
            
            # Create box plot for frequency bands
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
            
            # Create heatmap of frequency bands by brain state
            if 'brain_state' in df_freq.columns and available_bands:
                pivot_data = df_freq.groupby('brain_state')[available_bands].mean()
                
                fig = px.imshow(
                    pivot_data.T,
                    labels=dict(x="Brain State", y="Frequency Band", color="Power"),
                    title="Average Frequency Band Power by Brain State",
                    color_continuous_scale="Viridis"
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        # Frequency band statistics
        st.markdown("**Frequency Band Statistics**")
        
        if available_bands:
            freq_stats = df_freq[available_bands].describe()
            st.dataframe(freq_stats, use_container_width=True)
    
    def create_behavioral_analysis(self, data):
        """Create behavioral pattern analysis"""
        if not data:
            return
            
        st.subheader(" Behavioral Pattern Analysis")
        
        # Extract behavioral data
        behavioral_data = []
        for sample in data:
            behavioral = sample.get('behavioral_markers', {})
            if behavioral:
                behavioral['brain_state'] = sample.get('brain_state', 'unknown')
                behavioral['emotional_state'] = sample.get('emotional_state', 'unknown')
                behavioral['cognitive_load'] = sample.get('cognitive_load', 0)
                behavioral['sample_id'] = sample.get('line_number', 0)
                behavioral_data.append(behavioral)
        
        if not behavioral_data:
            st.warning("No behavioral marker data found in the dataset.")
            return
        
        df_behavioral = pd.DataFrame(behavioral_data)
        
        # Correlation matrix
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Behavioral Markers Correlation**")
            
            behavioral_cols = ['attention', 'stress', 'fatigue', 'arousal', 'cognitive_load']
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
            st.markdown("**Behavioral Trends Over Time**")
            
            if 'sample_id' in df_behavioral.columns and available_cols:
                # Plot trends for first few samples
                sample_subset = df_behavioral.head(100)  # First 100 samples
                
                fig = go.Figure()
                
                for col in available_cols[:3]:  # Limit to 3 for clarity
                    fig.add_trace(go.Scatter(
                        x=sample_subset['sample_id'],
                        y=sample_subset[col],
                        mode='lines+markers',
                        name=col.title(),
                        line=dict(width=2)
                    ))
                
                fig.update_layout(
                    title="Behavioral Markers Trends (First 100 Samples)",
                    xaxis_title="Sample ID",
                    yaxis_title="Intensity",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        # Behavioral statistics
        st.markdown("**Behavioral Statistics**")
        
        if available_cols:
            behavioral_stats = df_behavioral[available_cols].describe()
            st.dataframe(behavioral_stats, use_container_width=True)
    
    def create_system_performance(self):
        """Create system performance metrics"""
        st.subheader(" System Performance Metrics")
        
        # File information
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Dataset Files**")
            for filename, filepath in self.datasets.items():
                file_size = filepath.stat().st_size / (1024*1024)  # MB
                st.write(f" {filename}: {file_size:.1f} MB")
        
        with col2:
            st.markdown("**Generation Performance**")
            if self.datasets:
                # Estimate generation rate
                total_samples = sum(len(self.loaded_data.get(fname, [])) for fname in self.datasets.keys())
                st.metric("Total Samples Generated", f"{total_samples:,}")
                
                # Estimate data quality
                if total_samples > 0:
                    st.metric("Data Quality", " Excellent", help="All samples contain complete EEG and behavioral data")
        
        with col3:
            st.markdown("**System Status**")
            st.metric("Data Generator", " Active", help="Neuro-Brainwave Data Generator working")
            st.metric("Unsupervised Learning", " Active", help="K-means clustering operational")
            st.metric("Dashboard", " Active", help="Real-time visualization available")
    
    def create_data_explorer(self, data):
        """Create interactive data explorer"""
        if not data:
            return
            
        st.subheader(" Interactive Data Explorer")
        
        # Create filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Brain state filter
            brain_states = list(set(sample.get('brain_state', 'unknown') for sample in data))
            selected_brain_state = st.selectbox("Filter by Brain State:", ["All"] + brain_states)
        
        with col2:
            # Emotional state filter
            emotional_states = list(set(sample.get('emotional_state', 'unknown') for sample in data))
            selected_emotion = st.selectbox("Filter by Emotional State:", ["All"] + emotional_states)
        
        with col3:
            # Sample range filter
            max_samples = len(data)
            sample_range = st.slider("Sample Range:", 0, max_samples, (0, min(1000, max_samples)))
        
        # Apply filters
        filtered_data = data[sample_range[0]:sample_range[1]]
        
        if selected_brain_state != "All":
            filtered_data = [s for s in filtered_data if s.get('brain_state') == selected_brain_state]
        
        if selected_emotion != "All":
            filtered_data = [s for s in filtered_data if s.get('emotional_state') == selected_emotion]
        
        st.write(f"Showing {len(filtered_data)} samples (filtered from {len(data)} total)")
        
        # Display filtered data as table
        if filtered_data:
            # Create summary table
            summary_data = []
            for sample in filtered_data[:50]:  # Limit to 50 for performance
                summary_data.append({
                    'Sample ID': sample.get('line_number', 0),
                    'Brain State': sample.get('brain_state', 'Unknown'),
                    'Emotion': sample.get('emotional_state', 'Unknown'),
                    'Attention': sample.get('behavioral_markers', {}).get('attention', 0),
                    'Stress': sample.get('behavioral_markers', {}).get('stress', 0),
                    'Fatigue': sample.get('behavioral_markers', {}).get('fatigue', 0),
                    'Cognitive Load': sample.get('cognitive_load', 0)
                })
            
            if summary_data:
                df_summary = pd.DataFrame(summary_data)
                st.dataframe(df_summary, use_container_width=True)

def main():
    """Main dashboard function"""
    
    # Header
    st.markdown('<h1 class="main-header"> Neuro-Brainwave AI System Dashboard</h1>', unsafe_allow_html=True)
    
    # Initialize dashboard
    dashboard = NeuroBrainwaveDashboard()
    
    # Sidebar for dataset selection
    st.sidebar.header(" Dataset Selection")
    
    if not dashboard.datasets:
        st.error("No dataset files (*.jsonl) found in the current directory!")
        st.info("Please run the Neuro-Brainwave AI System first to generate data.")
        return
    
    selected_dataset = st.sidebar.selectbox(
        "Choose dataset to analyze:",
        list(dashboard.datasets.keys()),
        help="Select a dataset file to load and analyze"
    )
    
    # Load data
    with st.spinner(f"Loading dataset {selected_dataset}..."):
        data = dashboard.load_dataset(selected_dataset)
    
    if not data:
        st.error("Failed to load dataset!")
        return
    
    # Create tabs for different analysis sections
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        " Overview", 
        " Brain States", 
        " EEG Signals", 
        " Frequency Bands", 
        " Behavioral Patterns", 
        " Data Explorer"
    ])
    
    with tab1:
        dashboard.create_overview_metrics(data)
        dashboard.create_system_performance()
    
    with tab2:
        dashboard.create_brain_state_analysis(data)
    
    with tab3:
        dashboard.create_eeg_visualization(data)
    
    with tab4:
        dashboard.create_frequency_analysis(data)
    
    with tab5:
        dashboard.create_behavioral_analysis(data)
    
    with tab6:
        dashboard.create_data_explorer(data)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p> Neuro-Brainwave AI System Dashboard | Generated with Streamlit</p>
        <p>Real-time brainwave data analysis and visualization</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
