#!/usr/bin/env python3
"""
Interactive Brain Signal Producer & Consumer Streamlit App
Real-time brainwave generation and visualization with interactive controls
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import threading
import queue
import json
import socket
from collections import deque
from datetime import datetime
import sys
import os
from pathlib import Path
from typing import Optional
from datetime import timedelta

# Add current directory to path to import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dynamic_brainwave_generator import DynamicBrainwaveGenerator
from paths import DATA_DIR, ensure_data_dir

# Page configuration
st.set_page_config(
    page_title="Brain Signal Generator & Visualizer",
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
    .status-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .status-running {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .status-stopped {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
    }
    .metric-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

class StreamlitBrainSignalApp:
    def __init__(self):
        self.generator = DynamicBrainwaveGenerator()
        self.data_queue = queue.Queue()
        self.session_data = deque(maxlen=2000)  # Store last 2000 samples
        self.is_generating = False
        self.generation_thread = None
        
        # Initialize session state
        if 'session_started' not in st.session_state:
            st.session_state.session_started = False
        if 'session_data' not in st.session_state:
            st.session_state.session_data = []
        if 'session_stats' not in st.session_state:
            st.session_state.session_stats = {
                'total_samples': 0,
                'session_start_time': None,
                'current_brain_state': 'normal',
                'avg_variation': 0.0
            }
        if 'latest_consumer_png' not in st.session_state:
            st.session_state.latest_consumer_png = None
        if 'generation_log' not in st.session_state:
            st.session_state.generation_log = []
        if 'generation_total_epochs' not in st.session_state:
            st.session_state.generation_total_epochs = 0

    @staticmethod
    def find_latest_consumer_visualization_png(*search_dirs: Path) -> Optional[str]:
        """Return newest consumer_visualization_*.png under any search_dir (or None)."""
        try:
            ensure_data_dir()
            candidates: list = []
            for d in search_dirs:
                if not d.is_dir():
                    continue
                candidates.extend(d.glob("consumer_visualization_*.png"))
            if not candidates:
                return None
            newest = max(candidates, key=lambda p: p.stat().st_mtime)
            return str(newest)
        except Exception:
            return None
    
    def generate_brainwave_data(self, brain_state, duration, variability):
        """Generate brainwave data in a separate thread"""
        try:
            # Generate brain state events
            brain_state_events = self.generator.generate_brain_state_events(
                duration, brain_state, variability
            )
            
            # Calculate number of epochs
            num_epochs = int(duration / self.generator.epoch_duration)
            start_time = np.random.uniform(0, 65)
            
            for epoch_num in range(num_epochs):
                if not self.is_generating:
                    break
                
                # Generate brainwave EEG data
                signal = self.generator.generate_brainwave_eeg(
                    epoch_num, duration, start_time, brain_state_events
                )
                
                # Get current brain state
                current_time = start_time + (epoch_num * self.generator.epoch_duration)
                current_brain_state, state_variation = self.generator.get_brain_state_at_time(
                    current_time, brain_state_events
                )
                
                # Frequency analysis once per epoch (dominant bands reuse FFT results)
                frequency_analysis = self.generator.analyze_frequency_bands(signal)
                dominant_bands = self.generator.get_dominant_frequency_bands(
                    signal, brain_state=current_brain_state, activities=frequency_analysis
                )
                
                # Add timestamp and session info
                sample_data = {
                    'timestamp': current_time,
                    'data': signal,  # This is already a numpy array
                    'brain_state': current_brain_state,
                    'state_variation': state_variation,
                    'epoch_num': epoch_num,
                    'neurosity_data': None,
                    'frequency_analysis': frequency_analysis,
                    'dominant_bands': dominant_bands
                }
                
                # Put data in queue for real-time processing
                self.data_queue.put(sample_data)
                # Tiny yield so the UI thread can drain the queue (was 0.5s, felt sluggish)
                time.sleep(0.02)
            
            # Signal completion
            self.data_queue.put({'type': 'session_complete'})
            
        except Exception as e:
            self.data_queue.put({'type': 'error', 'message': str(e)})
    
    def process_realtime_data(self):
        """Process data from the queue and update session state"""
        processed_count = 0
        max_process = 2000  # Drain backlog quickly when generation outpaces UI
        
        while not self.data_queue.empty() and processed_count < max_process:
            try:
                sample = self.data_queue.get_nowait()
                
                if sample.get('type') == 'session_complete':
                    st.session_state.session_started = False
                    root = Path(os.path.dirname(os.path.abspath(__file__)))
                    st.session_state.latest_consumer_png = self.find_latest_consumer_visualization_png(
                        root,
                        DATA_DIR,
                    )
                    st.session_state.generation_log.append(
                        "Session complete — all epochs generated."
                    )
                    if not st.session_state.get("_completion_toast_shown"):
                        st.toast("Session completed successfully")
                        st.session_state._completion_toast_shown = True
                    break
                elif sample.get('type') == 'error':
                    st.session_state.generation_log.append(f"Error: {sample['message']}")
                    st.error(f"Error: {sample['message']}")
                    st.session_state.session_started = False
                    break
                else:
                    # Add to session data
                    st.session_state.session_data.append(sample)
                    
                    # Update statistics
                    st.session_state.session_stats['total_samples'] += 1
                    st.session_state.session_stats['current_brain_state'] = sample['brain_state']
                    
                    # Calculate average variation
                    if st.session_state.session_data:
                        avg_var = np.mean([s['state_variation'] for s in st.session_state.session_data[-100:]])
                        st.session_state.session_stats['avg_variation'] = avg_var

                    total_e = st.session_state.get('generation_total_epochs') or '?'
                    bands = sample.get('dominant_bands') or []
                    bands_s = ', '.join(bands) if bands else '—'
                    log_line = (
                        f"epoch {sample['epoch_num'] + 1}/{total_e} | t={sample['timestamp']:.3f}s | "
                        f"state={sample['brain_state']} | var={sample['state_variation']:.3f} | "
                        f"bands=[{bands_s}] | shape={tuple(sample['data'].shape)}"
                    )
                    st.session_state.generation_log.append(log_line)
                    if len(st.session_state.generation_log) > 200:
                        st.session_state.generation_log = st.session_state.generation_log[-200:]
                    
                    processed_count += 1
                        
            except queue.Empty:
                break
            except Exception as e:
                st.error(f"Error processing data: {e}")
                break
    
    def create_realtime_visualization(self):
        """Create real-time visualization of brain signals"""
        if not st.session_state.session_data:
            return None
        
        # Last N epochs only — smaller Plotly payloads / faster browser rendering
        recent_data = list(st.session_state.session_data)[-250:]
        
        if len(recent_data) < 2:
            return None
        
        # Extract data
        timestamps = [s['timestamp'] for s in recent_data]
        brain_states = [s['brain_state'] for s in recent_data]
        variations = [s['state_variation'] for s in recent_data]
        
        # Create subplots with frequency band visualization
        fig = make_subplots(
            rows=4, cols=1,
            subplot_titles=[
                'Real-time Brainwave Signals (All Channels)',
                'Brain State Timeline',
                'State Variation Over Time',
                'Active Frequency Bands'
            ],
            vertical_spacing=0.06,
            specs=[[{"secondary_y": False}],
                   [{"secondary_y": False}],
                   [{"secondary_y": False}],
                   [{"secondary_y": False}]]
        )
        
        # Plot 1: All channels overlaid
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
        
        # Ensure we have valid data
        if recent_data and 'data' in recent_data[0]:
            for ch in range(min(8, len(self.generator.channel_names))):
                channel_data = []
                for s in recent_data:
                    if 'data' in s and s['data'] is not None:
                        if isinstance(s['data'], np.ndarray) and s['data'].ndim == 2:
                            channel_data.append(s['data'][ch, :].mean() if ch < s['data'].shape[0] else 0)
                        else:
                            channel_data.append(0)  # Default value if data is invalid
                    else:
                        channel_data.append(0)  # Default value if no data
                
                fig.add_trace(
                    go.Scattergl(
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
            go.Scattergl(
                x=timestamps,
                y=variations,
                mode='lines',
                name='State Variation',
                line=dict(color='red', width=2)
            ),
            row=3, col=1
        )
        
        # Plot 4: Active Frequency Bands
        if recent_data and 'dominant_bands' in recent_data[0]:
            # Create frequency band activity timeline
            band_names = list(self.generator.frequency_band_colors.keys())
            band_activity = {band: [] for band in band_names}
            
            for sample in recent_data:
                if 'dominant_bands' in sample:
                    for band in band_names:
                        band_activity[band].append(1 if band in sample['dominant_bands'] else 0)
                else:
                    for band in band_names:
                        band_activity[band].append(0)
            
            # Add traces for each frequency band
            for band_name in band_names:
                color = self.generator.frequency_band_colors[band_name]
                fig.add_trace(
                    go.Scattergl(
                        x=timestamps,
                        y=band_activity[band_name],
                        mode='lines',
                        name=f'{band_name.upper()} ({self.generator.standard_frequency_bands[band_name][0]}-{self.generator.standard_frequency_bands[band_name][1]} Hz)',
                        line=dict(color=color, width=2),
                        opacity=0.85
                    ),
                    row=4, col=1
                )
        
        # Update layout (uirevision preserves zoom/pan across fragment refreshes)
        fig.update_layout(
            height=780,
            showlegend=True,
            title_text="Real-time Brain Signal Visualization with Frequency Band Analysis",
            title_font_size=18,
            uirevision="brain-live",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        
        # Update axes
        fig.update_xaxes(title_text="Time (seconds)", row=4, col=1)
        fig.update_yaxes(title_text="Amplitude (μV)", row=1, col=1)
        fig.update_yaxes(title_text="State", row=2, col=1)
        fig.update_yaxes(title_text="Variation Factor", row=3, col=1)
        fig.update_yaxes(title_text="Band Activity (0=Inactive, 1=Active)", row=4, col=1)
        
        return fig


@st.fragment(run_every=timedelta(seconds=0.35))
def live_dashboard():
    """Drain the queue & redraw charts on a timer — avoids full-page st.rerun spam."""
    app = st.session_state.brain_signal_app
    if st.session_state.session_started:
        app.process_realtime_data()

    col1, col2 = st.columns([2, 1])

    with col1:
        if st.session_state.session_started:
            st.markdown("""
            <div class="status-box status-running">
                <h3>Session Active</h3>
                <p>Generating brain signals in real-time…</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="status-box status-stopped">
                <h3>Session Stopped</h3>
                <p>Click START to begin generating brain signals</p>
            </div>
            """, unsafe_allow_html=True)

        if st.session_state.session_started or st.session_state.generation_log:
            st.subheader("Generation log")
            log_body = "\n".join(st.session_state.generation_log) if st.session_state.generation_log else (
                "Initializing brainwave generation… (first epoch will appear shortly)"
            )
            st.text_area(
                "Live generation output",
                value=log_body,
                height=200,
                disabled=True,
                label_visibility="collapsed",
            )

        if st.session_state.session_data:
            st.caption(f"Samples in session: **{len(st.session_state.session_data)}**")
            fig = app.create_realtime_visualization()
            if fig:
                st.plotly_chart(
                    fig,
                    config={"displayModeBar": True, "scrollZoom": False, "responsive": True},
                )
            else:
                st.warning("Visualization could not be created. Check data format.")
        else:
            if st.session_state.session_started:
                st.info("Generating data… first chart points appear after 2+ epochs.")
            else:
                st.info("No data to display. Start a session to see real-time visualization.")

    with col2:
        st.header("Session Statistics")

        metrics_col1, metrics_col2 = st.columns(2)

        with metrics_col1:
            st.metric(
                label="Total Samples",
                value=st.session_state.session_stats['total_samples'],
            )

            st.metric(
                label="Current State",
                value=st.session_state.session_stats['current_brain_state'].title(),
            )

        with metrics_col2:
            st.metric(
                label="Avg Variation",
                value=f"{st.session_state.session_stats['avg_variation']:.2f}",
            )

            t0 = st.session_state.get("session_start_time")
            if t0 is not None and st.session_state.session_started:
                elapsed = time.time() - t0
                st.metric(label="Elapsed Time", value=f"{elapsed:.1f}s")

        if st.session_state.session_data:
            st.subheader("Recent Data")
            recent_data = list(st.session_state.session_data)[-10:]

            df_data = []
            for sample in recent_data:
                df_data.append({
                    'Time': f"{sample['timestamp']:.2f}s",
                    'State': sample['brain_state'],
                    'Variation': f"{sample['state_variation']:.2f}",
                    'Epoch': sample['epoch_num'],
                })

            if df_data:
                st.dataframe(pd.DataFrame(df_data), width="stretch")

        if st.session_state.session_data:
            st.subheader("Download Data")

            export_data = []
            for sample in st.session_state.session_data:
                export_data.append({
                    'timestamp': sample['timestamp'],
                    'brain_state': sample['brain_state'],
                    'state_variation': sample['state_variation'],
                    'epoch_num': sample['epoch_num'],
                    'data_shape': list(sample['data'].shape),
                })

            json_data = json.dumps(export_data, indent=2)
            st.download_button(
                label="Download JSON",
                data=json_data,
                file_name=f"brain_signal_data_{int(time.time())}.json",
                mime="application/json",
            )

        st.subheader("Latest Consumer Visualization")
        app_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        latest_png = StreamlitBrainSignalApp.find_latest_consumer_visualization_png(app_dir, DATA_DIR)
        if latest_png and latest_png != st.session_state.latest_consumer_png:
            st.session_state.latest_consumer_png = latest_png

        if st.session_state.latest_consumer_png:
            with st.expander("Show latest consumer_visualization_*.png", expanded=False):
                st.caption(f"File: {os.path.basename(st.session_state.latest_consumer_png)}")
                st.image(st.session_state.latest_consumer_png, width="stretch")
        else:
            st.info(
                "No `consumer_visualization_*.png` found under the project folder or `data/` yet. "
                "That image is created only when you run the **standalone** consumer "
                "(e.g. `./scripts/run_consumer.sh` with `./scripts/run_producer.sh` in another terminal); "
                "the Streamlit START button does not write this file. After a consumer session ends, "
                "the PNG is saved under `data/`."
            )


def main():
    # One app instance per browser session so the generator thread and UI share one queue
    if "brain_signal_app" not in st.session_state:
        st.session_state.brain_signal_app = StreamlitBrainSignalApp()
    app = st.session_state.brain_signal_app

    # Main header
    st.markdown('<h1 class="main-header">Brain Signal Generator & Visualizer</h1>', unsafe_allow_html=True)
    
    # Sidebar controls
    with st.sidebar:
        st.header("Control Panel")
        
        # Brain state selection
        brain_state = st.selectbox(
            "Brain State",
            options=['normal', 'excited', 'relaxed', 'focused', 'stressed', 'sleepy'],
            index=4,  # Default to stressed
            help="Select the type of brain activity to simulate"
        )
        
        # Duration slider
        duration = st.slider(
            "Duration (seconds)",
            min_value=10,
            max_value=300,
            value=180,
            step=10,
            help="How long to generate brain signals"
        )
        
        # Variability slider
        variability = st.slider(
            "Variability",
            min_value=0.1,
            max_value=1.0,
            value=0.7,
            step=0.1,
            help="How much variation in brain state (0.1=stable, 1.0=very dynamic)"
        )
        
        # Display brain state info
        if brain_state in app.generator.brain_states:
            state_info = app.generator.brain_states[brain_state]
            st.info(f"""
            **{state_info['name']}**
            
            {state_info['description']}
            """)
        
        # Display frequency band information
        st.subheader("Frequency Bands")
        st.markdown("""
        **Active Frequency Bands:**
        - **DELTA** (0.5-4 Hz): 5-20 μV - Deep sleep, unconsciousness
        - **THETA** (4-8 Hz): 8-25 μV - Drowsiness, meditation
        - **ALPHA** (8-13 Hz): 5-25 μV - Relaxed, eyes closed
        - **BETA** (13-30 Hz): 25-60 μV - Active thinking, focus
        - **GAMMA** (30-45 Hz): 10-30 μV - High cognitive processing
        """)
        
        # Show current dominant bands if session is active
        if st.session_state.session_started and st.session_state.session_data:
            recent_sample = st.session_state.session_data[-1] if st.session_state.session_data else None
            if recent_sample and 'dominant_bands' in recent_sample:
                st.markdown("**Currently Active Bands:**")
                for band in recent_sample['dominant_bands']:
                    color = app.generator.frequency_band_colors[band]
                    st.markdown(f"- <span style='color: {color}'>{band.upper()}</span>", unsafe_allow_html=True)
        
        st.divider()
        
        # Control buttons
        col1, col2 = st.columns(2)
        
        with col1:
            start_button = st.button(
                "START",
                type="primary",
                disabled=st.session_state.session_started,
                help="Start generating brain signals"
            )
        
        with col2:
            stop_button = st.button(
                "STOP",
                disabled=not st.session_state.session_started,
                help="Stop the current session"
            )
        
        # Clear data button
        if st.button("Clear Data", help="Clear all session data"):
            st.session_state.session_data = []
            st.session_state.generation_log = []
            st.session_state.generation_total_epochs = 0
            st.session_state.session_stats = {
                'total_samples': 0,
                'session_start_time': None,
                'current_brain_state': 'normal',
                'avg_variation': 0.0
            }
            st.rerun()

    live_dashboard()

    # Handle button clicks
    if start_button and not st.session_state.session_started:
        st.session_state.session_started = True
        st.session_state.session_start_time = time.time()
        st.session_state._completion_toast_shown = False
        st.session_state.session_data = []
        st.session_state.generation_total_epochs = max(
            1, int(duration / app.generator.epoch_duration)
        )
        st.session_state.generation_log = [
            f"Starting session: state={brain_state}, duration={duration}s, "
            f"~{st.session_state.generation_total_epochs} epochs "
            f"(@ {app.generator.epoch_duration:.4f}s/epoch)"
        ]

        # Start generation in a separate thread
        app.is_generating = True
        app.generation_thread = threading.Thread(
            target=app.generate_brainwave_data,
            args=(brain_state, duration, variability)
        )
        app.generation_thread.start()
        
        st.success(f"Started {brain_state} session for {duration} seconds!")
        st.rerun()
    
    if stop_button and st.session_state.session_started:
        app.is_generating = False
        st.session_state.session_started = False
        st.session_state.generation_log.append("Session stopped by user.")
        st.warning(" Session stopped by user")
        st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>Brain Signal Generator & Visualizer | Built with Streamlit</p>
        <p>Designed by Tuệ Hoàng, Eng.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
