#!/usr/bin/env python3
"""
AI Agent for Brainwave Visualization
Reads brainwave_samples.json and creates comprehensive graphs and charts
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal
from scipy.fft import fft, fftfreq
import seaborn as sns
from datetime import datetime
import os

class BrainwaveVisualizer:
    def __init__(self, json_file="brainwave_samples.json"):
        self.json_file = json_file
        self.samples = []
        self.channel_names = []
        self.sampling_rate = 256
        self.epoch_size = 16
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def load_data(self):
        """Load brainwave data from JSON file"""
        try:
            with open(self.json_file, 'r') as f:
                self.samples = json.load(f)
            
            if self.samples:
                self.channel_names = self.samples[0]['info']['channelNames']
                self.sampling_rate = self.samples[0]['info']['samplingRate']
                print(f" Loaded {len(self.samples)} brainwave samples")
                print(f" Channels: {', '.join(self.channel_names)}")
                print(f" Sampling rate: {self.sampling_rate} Hz")
            else:
                print(" No samples found in JSON file")
                return False
                
        except FileNotFoundError:
            print(f" File {self.json_file} not found. Please run brainwave_generator.py first.")
            return False
        except Exception as e:
            print(f" Error loading data: {e}")
            return False
            
        return True
    
    def prepare_data_matrix(self):
        """Convert samples to a 3D numpy array: (samples, channels, time_points)"""
        num_samples = len(self.samples)
        num_channels = len(self.channel_names)
        
        # Create 3D array: (samples, channels, time_points)
        data_matrix = np.zeros((num_samples, num_channels, self.epoch_size))
        
        for i, sample in enumerate(self.samples):
            data_matrix[i] = np.array(sample['data'])
        
        return data_matrix
    
    def create_time_series_plot(self, data_matrix):
        """Create time series plots for all channels"""
        print(" Creating time series plots...")
        
        num_samples, num_channels, time_points = data_matrix.shape
        
        # Create time vector (in seconds)
        time_vector = np.arange(time_points) / self.sampling_rate
        
        # Create subplots
        fig, axes = plt.subplots(num_channels, 1, figsize=(15, 2*num_channels))
        fig.suptitle(' AI-Generated Brainwave Time Series', fontsize=16, fontweight='bold')
        
        for ch in range(num_channels):
            ax = axes[ch] if num_channels > 1 else axes
            
            # Plot all samples for this channel
            for sample_idx in range(num_samples):
                # Add sample offset for visualization
                offset = sample_idx * 0.1  # Small offset between samples
                y_data = data_matrix[sample_idx, ch, :] + offset
                
                ax.plot(time_vector, y_data, alpha=0.7, linewidth=1)
            
            ax.set_title(f'Channel {self.channel_names[ch]}', fontweight='bold')
            ax.set_xlabel('Time (seconds)')
            ax.set_ylabel('Amplitude (μV)')
            ax.grid(True, alpha=0.3)
            
            # Add sample labels
            if ch == 0:  # Only on first subplot
                for sample_idx in range(0, num_samples, 2):  # Every other sample
                    ax.text(time_vector[-1], sample_idx * 0.1, f'S{sample_idx+1}', 
                           fontsize=8, alpha=0.7)
        
        plt.tight_layout()
        plt.savefig('brainwave_time_series.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_frequency_analysis(self, data_matrix):
        """Create frequency domain analysis"""
        print(" Creating frequency domain analysis...")
        
        num_samples, num_channels, time_points = data_matrix.shape
        
        # Calculate FFT for each channel
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle(' Frequency Domain Analysis - Power Spectral Density', fontsize=16, fontweight='bold')
        axes = axes.flatten()
        
        for ch in range(num_channels):
            ax = axes[ch]
            
            # Combine all samples for this channel
            channel_data = data_matrix[:, ch, :].flatten()
            
            # Calculate FFT
            fft_data = fft(channel_data)
            freqs = fftfreq(len(channel_data), 1/self.sampling_rate)
            
            # Only plot positive frequencies
            positive_freqs = freqs[:len(freqs)//2]
            power_spectrum = np.abs(fft_data[:len(fft_data)//2])**2
            
            # Plot power spectral density
            ax.semilogy(positive_freqs, power_spectrum, linewidth=2)
            ax.set_title(f'Channel {self.channel_names[ch]}', fontweight='bold')
            ax.set_xlabel('Frequency (Hz)')
            ax.set_ylabel('Power Spectral Density')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 50)  # Focus on relevant EEG frequencies
            
            # Highlight frequency bands
            bands = {
                'Delta': (0.5, 4, 'purple'),
                'Theta': (4, 8, 'blue'),
                'Alpha': (8, 13, 'green'),
                'Beta': (13, 30, 'orange'),
                'Gamma': (30, 45, 'red')
            }
            
            for band_name, (low, high, color) in bands.items():
                ax.axvspan(low, high, alpha=0.2, color=color, label=band_name)
            
            if ch == 0:  # Add legend only to first subplot
                ax.legend(loc='upper right', fontsize=8)
        
        plt.tight_layout()
        plt.savefig('brainwave_frequency_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_statistical_analysis(self, data_matrix):
        """Create statistical analysis plots"""
        print(" Creating statistical analysis...")
        
        num_samples, num_channels, time_points = data_matrix.shape
        
        # Calculate statistics for each channel
        stats_data = []
        for ch in range(num_channels):
            channel_data = data_matrix[:, ch, :].flatten()
            stats_data.append({
                'Channel': self.channel_names[ch],
                'Mean': np.mean(channel_data),
                'Std': np.std(channel_data),
                'Min': np.min(channel_data),
                'Max': np.max(channel_data),
                'Range': np.max(channel_data) - np.min(channel_data)
            })
        
        stats_df = pd.DataFrame(stats_data)
        
        # Create subplots for different statistics
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(' Statistical Analysis of AI-Generated Brainwaves', fontsize=16, fontweight='bold')
        
        # Mean values
        axes[0, 0].bar(stats_df['Channel'], stats_df['Mean'], color='skyblue', alpha=0.7)
        axes[0, 0].set_title('Mean Amplitude by Channel', fontweight='bold')
        axes[0, 0].set_ylabel('Mean (μV)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Standard deviation
        axes[0, 1].bar(stats_df['Channel'], stats_df['Std'], color='lightcoral', alpha=0.7)
        axes[0, 1].set_title('Standard Deviation by Channel', fontweight='bold')
        axes[0, 1].set_ylabel('Std Dev (μV)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Range
        axes[0, 2].bar(stats_df['Channel'], stats_df['Range'], color='lightgreen', alpha=0.7)
        axes[0, 2].set_title('Amplitude Range by Channel', fontweight='bold')
        axes[0, 2].set_ylabel('Range (μV)')
        axes[0, 2].tick_params(axis='x', rotation=45)
        axes[0, 2].grid(True, alpha=0.3)
        
        # Box plot for all channels
        channel_data_list = [data_matrix[:, ch, :].flatten() for ch in range(num_channels)]
        axes[1, 0].boxplot(channel_data_list, labels=self.channel_names)
        axes[1, 0].set_title('Distribution by Channel (Box Plot)', fontweight='bold')
        axes[1, 0].set_ylabel('Amplitude (μV)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Heatmap of channel correlations
        correlation_matrix = np.corrcoef([data_matrix[:, ch, :].flatten() for ch in range(num_channels)])
        im = axes[1, 1].imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        axes[1, 1].set_title('Channel Correlation Matrix', fontweight='bold')
        axes[1, 1].set_xticks(range(num_channels))
        axes[1, 1].set_yticks(range(num_channels))
        axes[1, 1].set_xticklabels(self.channel_names, rotation=45)
        axes[1, 1].set_yticklabels(self.channel_names)
        
        # Add correlation values to heatmap
        for i in range(num_channels):
            for j in range(num_channels):
                axes[1, 1].text(j, i, f'{correlation_matrix[i, j]:.2f}', 
                               ha='center', va='center', fontsize=8)
        
        plt.colorbar(im, ax=axes[1, 1])
        
        # Sample-wise statistics
        sample_means = [np.mean(data_matrix[i, :, :]) for i in range(num_samples)]
        sample_stds = [np.std(data_matrix[i, :, :]) for i in range(num_samples)]
        
        axes[1, 2].plot(range(1, num_samples+1), sample_means, 'o-', label='Mean', linewidth=2)
        axes[1, 2].plot(range(1, num_samples+1), sample_stds, 's-', label='Std Dev', linewidth=2)
        axes[1, 2].set_title('Statistics Across Samples', fontweight='bold')
        axes[1, 2].set_xlabel('Sample Number')
        axes[1, 2].set_ylabel('Amplitude (μV)')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('brainwave_statistical_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return stats_df
    
    def create_multi_channel_comparison(self, data_matrix):
        """Create multi-channel comparison plots"""
        print(" Creating multi-channel comparison...")
        
        num_samples, num_channels, time_points = data_matrix.shape
        
        # Create time vector
        time_vector = np.arange(time_points) / self.sampling_rate
        
        # Plot 1: All channels overlaid for first sample
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(' Multi-Channel Brainwave Analysis', fontsize=16, fontweight='bold')
        
        # First sample - all channels
        ax1 = axes[0, 0]
        for ch in range(num_channels):
            ax1.plot(time_vector, data_matrix[0, ch, :], 
                    label=self.channel_names[ch], linewidth=2, alpha=0.8)
        ax1.set_title('First Sample - All Channels', fontweight='bold')
        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel('Amplitude (μV)')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Average across all samples for each channel
        ax2 = axes[0, 1]
        for ch in range(num_channels):
            avg_signal = np.mean(data_matrix[:, ch, :], axis=0)
            ax2.plot(time_vector, avg_signal, 
                    label=self.channel_names[ch], linewidth=2, alpha=0.8)
        ax2.set_title('Average Signal Across All Samples', fontweight='bold')
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel('Amplitude (μV)')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        # Channel power comparison
        ax3 = axes[1, 0]
        channel_powers = []
        for ch in range(num_channels):
            power = np.mean(data_matrix[:, ch, :]**2)
            channel_powers.append(power)
        
        bars = ax3.bar(self.channel_names, channel_powers, color='lightblue', alpha=0.7)
        ax3.set_title('Average Power by Channel', fontweight='bold')
        ax3.set_ylabel('Power (μV²)')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, power in zip(bars, channel_powers):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(channel_powers)*0.01,
                    f'{power:.1f}', ha='center', va='bottom', fontsize=8)
        
        # Sample progression heatmap
        ax4 = axes[1, 1]
        # Create heatmap data: samples x channels (using mean amplitude)
        heatmap_data = np.mean(data_matrix, axis=2)  # Average across time
        im = ax4.imshow(heatmap_data.T, cmap='viridis', aspect='auto')
        ax4.set_title('Sample Progression Heatmap', fontweight='bold')
        ax4.set_xlabel('Sample Number')
        ax4.set_ylabel('Channel')
        ax4.set_xticks(range(num_samples))
        ax4.set_yticks(range(num_channels))
        ax4.set_yticklabels(self.channel_names)
        
        # Add colorbar
        plt.colorbar(im, ax=ax4, label='Mean Amplitude (μV)')
        
        plt.tight_layout()
        plt.savefig('brainwave_multi_channel_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_summary_report(self, stats_df):
        """Create a summary report"""
        print("\n Generating Summary Report...")
        
        report = f"""
 AI BRAINWAVE GENERATOR - VISUALIZATION REPORT
{'='*60}

 DATA OVERVIEW:
• Total Samples: {len(self.samples)}
• Channels: {len(self.channel_names)} ({', '.join(self.channel_names)})
• Sampling Rate: {self.sampling_rate} Hz
• Epoch Duration: {self.epoch_size/self.sampling_rate:.3f} seconds
• Total Data Points: {len(self.samples) * len(self.channel_names) * self.epoch_size}

 CHANNEL STATISTICS:
"""
        
        for _, row in stats_df.iterrows():
            report += f"• {row['Channel']:4s}: Mean={row['Mean']:6.2f}μV, Std={row['Std']:6.2f}μV, Range={row['Range']:6.2f}μV\n"
        
        report += f"""
 KEY INSIGHTS:
• Voltage Range: {stats_df['Min'].min():.2f} to {stats_df['Max'].max():.2f} μV
• Most Variable Channel: {stats_df.loc[stats_df['Std'].idxmax(), 'Channel']} (σ={stats_df['Std'].max():.2f}μV)
• Most Stable Channel: {stats_df.loc[stats_df['Std'].idxmin(), 'Channel']} (σ={stats_df['Std'].min():.2f}μV)
• Average Signal Strength: {stats_df['Mean'].abs().mean():.2f} μV

 GENERATED FILES:
• brainwave_time_series.png - Time domain analysis
• brainwave_frequency_analysis.png - Frequency domain analysis  
• brainwave_statistical_analysis.png - Statistical summaries
• brainwave_multi_channel_analysis.png - Multi-channel comparisons

 Visualization complete! All charts saved as PNG files.
"""
        
        print(report)
        
        # Save report to file
        with open('brainwave_visualization_report.txt', 'w') as f:
            f.write(report)
        
        print(" Report saved to: brainwave_visualization_report.txt")
    
    def run_visualization(self):
        """Main method to run all visualizations"""
        print(" AI Brainwave Visualizer")
        print("="*50)
        
        # Load data
        if not self.load_data():
            return
        
        # Prepare data matrix
        data_matrix = self.prepare_data_matrix()
        
        # Create all visualizations
        self.create_time_series_plot(data_matrix)
        self.create_frequency_analysis(data_matrix)
        stats_df = self.create_statistical_analysis(data_matrix)
        self.create_multi_channel_comparison(data_matrix)
        
        # Generate summary report
        self.create_summary_report(stats_df)
        
        print("\n All visualizations completed successfully!")

def main():
    """Main function"""
    visualizer = BrainwaveVisualizer()
    visualizer.run_visualization()

if __name__ == "__main__":
    main()
