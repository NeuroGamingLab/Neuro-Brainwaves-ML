#!/usr/bin/env python3
"""
Launcher for Brain Signal Producer and Consumer AI Agents
Starts both agents and manages their communication
"""

import subprocess
import time
import threading
import sys
import os

class BrainAgentsLauncher:
    def __init__(self):
        self.producer_process = None
        self.consumer_process = None
        self.is_running = False
        
    def start_producer(self, brain_state='normal', duration=60.0, variability=0.3):
        """Start the brain signal producer"""
        print(" Starting Brain Signal Producer...")
        
        # Create producer script with configuration
        producer_script = f"""
import sys
sys.path.append('{os.getcwd()}')
from brain_signal_producer import BrainSignalProducer

producer = BrainSignalProducer()
producer.configure_session(
    brain_state='{brain_state}',
    duration={duration},
    variability={variability}
)

if producer.start_server():
    producer.generate_and_transmit_session()
    
    # Keep running for additional sessions
    print(" Producer ready for next session...")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\\n Producer stopped")
    finally:
        producer.stop_producer()
"""
        
        # Write temporary script
        with open('temp_producer.py', 'w') as f:
            f.write(producer_script)
        
        # Start producer process
        self.producer_process = subprocess.Popen(
            [sys.executable, 'temp_producer.py'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        print(f" Producer started (PID: {self.producer_process.pid})")
        return True
    
    def start_consumer(self):
        """Start the brain signal consumer"""
        print(" Starting Brain Signal Consumer...")
        
        # Start consumer process
        self.consumer_process = subprocess.Popen(
            [sys.executable, 'brain_signal_consumer.py'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        print(f" Consumer started (PID: {self.consumer_process.pid})")
        return True
    
    def monitor_processes(self):
        """Monitor both processes"""
        while self.is_running:
            # Check producer
            if self.producer_process and self.producer_process.poll() is not None:
                print(" Producer process ended unexpectedly")
                break
            
            # Check consumer
            if self.consumer_process and self.consumer_process.poll() is not None:
                print(" Consumer process ended unexpectedly")
                break
            
            time.sleep(1)
    
    def stop_all(self):
        """Stop all processes"""
        self.is_running = False
        
        if self.producer_process:
            print(" Stopping producer...")
            self.producer_process.terminate()
            self.producer_process.wait()
        
        if self.consumer_process:
            print(" Stopping consumer...")
            self.consumer_process.terminate()
            self.consumer_process.wait()
        
        # Cleanup
        if os.path.exists('temp_producer.py'):
            os.remove('temp_producer.py')
        
        print(" All processes stopped")
    
    def run_interactive_session(self):
        """Run an interactive session with user input"""
        print(" Brain Signal Producer & Consumer Launcher")
        print("=" * 60)
        
        # Get user configuration
        print(" Available Brain States:")
        print("  1. normal - Normal baseline")
        print("  2. excited - Excited state")
        print("  3. relaxed - Deep relaxation")
        print("  4. focused - Focused attention")
        print("  5. stressed - Stress/anxiety")
        print("  6. sleepy - Drowsy/sleepy")
        
        while True:
            try:
                choice = input("\n Select brain state (1-6): ").strip()
                brain_states = {
                    '1': 'normal', '2': 'excited', '3': 'relaxed',
                    '4': 'focused', '5': 'stressed', '6': 'sleepy'
                }
                if choice in brain_states:
                    brain_state = brain_states[choice]
                    break
                else:
                    print(" Invalid choice. Please select 1-6.")
            except KeyboardInterrupt:
                print("\n Goodbye!")
                return
        
        # Get duration
        while True:
            try:
                duration = float(input("  Enter duration in seconds (default 60): ") or "60")
                if duration > 0:
                    break
                else:
                    print(" Duration must be positive.")
            except ValueError:
                print(" Please enter a valid number.")
            except KeyboardInterrupt:
                print("\n Goodbye!")
                return
        
        # Get variability
        while True:
            try:
                variability = float(input(" Enter variability (0.1-1.0, default 0.3): ") or "0.3")
                if 0.1 <= variability <= 1.0:
                    break
                else:
                    print(" Variability must be between 0.1 and 1.0.")
            except ValueError:
                print(" Please enter a valid number.")
            except KeyboardInterrupt:
                print("\n Goodbye!")
                return
        
        print(f"\n Configuration:")
        print(f"   • Brain State: {brain_state}")
        print(f"   • Duration: {duration} seconds")
        print(f"   • Variability: {variability}")
        print("=" * 60)
        
        # Start both agents
        try:
            self.is_running = True
            
            # Start producer first
            if not self.start_producer(brain_state, duration, variability):
                print(" Failed to start producer")
                return
            
            # Wait a moment for producer to start
            time.sleep(2)
            
            # Start consumer
            if not self.start_consumer():
                print(" Failed to start consumer")
                return
            
            print("\n Both agents started successfully!")
            print(" The consumer GUI should open shortly...")
            print(" Press Ctrl+C to stop all agents")
            
            # Monitor processes
            self.monitor_processes()
            
        except KeyboardInterrupt:
            print("\n Stopping all agents...")
        except Exception as e:
            print(f" Error: {e}")
        finally:
            self.stop_all()

def main():
    """Main function"""
    launcher = BrainAgentsLauncher()
    
    if len(sys.argv) > 1:
        # Command line mode
        brain_state = sys.argv[1] if len(sys.argv) > 1 else 'normal'
        duration = float(sys.argv[2]) if len(sys.argv) > 2 else 60.0
        variability = float(sys.argv[3]) if len(sys.argv) > 3 else 0.3
        
        print(f" Starting agents with: {brain_state}, {duration}s, {variability}")
        
        try:
            launcher.is_running = True
            launcher.start_producer(brain_state, duration, variability)
            time.sleep(2)
            launcher.start_consumer()
            launcher.monitor_processes()
        except KeyboardInterrupt:
            print("\n Stopping...")
        finally:
            launcher.stop_all()
    else:
        # Interactive mode
        launcher.run_interactive_session()

if __name__ == "__main__":
    main()
