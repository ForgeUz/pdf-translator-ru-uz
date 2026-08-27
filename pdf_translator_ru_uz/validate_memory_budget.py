# pdf_translator_ru_uz/validate_memory_budget.py

import psutil
import os
import subprocess
import time
import sys
import argparse

# Intent: Monitor peak RAM during model load + inference to validate ADR-7 (<=8B cap).
# Dependencies: psutil, subprocess.
# Flow Summary: Launch model load script -> Poll RSS -> Report peak.

def monitor_process(pid: int, duration: int = 120) -> float:
    """Monitor peak RSS memory in GB for a given PID."""
    peak_bytes = 0
    start_time = time.time()
    
    print(f"[Monitor] Watching PID {pid} for {duration}s...")
    while time.time() - start_time < duration:
        try:
            p = psutil.Process(pid)
            # Include child processes (e.g., spawned by transformers/ct2)
            mem = p.memory_info().rss
            for child in p.children(recursive=True):
                try:
                    mem += child.memory_info().rss
                except psutil.NoSuchProcess:
                    pass
            
            if mem > peak_bytes:
                peak_bytes = mem
                
        except psutil.NoSuchProcess:
            print("[Monitor] Process terminated early.")
            break
        time.sleep(0.5)
        
    return peak_bytes / (1024 ** 3)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate model memory footprint.")
    parser.add_argument("--script", type=str, required=True, help="Python script to run for benchmarking memory.")
    parser.add_argument("--duration", type=int, default=120, help="Max duration to monitor in seconds.")
    args = parser.parse_args()
    
    # Launch the target script (e.g., a benchmark runner) as a subprocess
    proc = subprocess.Popen([sys.executable, args.script])
    
    try:
        peak_gb = monitor_process(proc.pid, args.duration)
        print("\n" + "="*40)
        print(f"PEAK RSS MEMORY: {peak_gb:.2f} GB")
        print("="*40)
        if peak_gb > 11.0:
            print("[WARNING] Exceeds 11GB safe threshold for 12GB system (ADR-7). Model too large.")
        else:
            print("[SUCCESS] Within 11GB safe threshold.")
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
