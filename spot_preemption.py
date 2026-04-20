import os
import csv
import sys
import time
import subprocess
import argparse

def get_simulated_lifetime(trace_path="data/us-east1a_V100_cdf.csv", threshold=0.8):
    trace_path = os.path.join(os.path.dirname(__file__), trace_path)
    if not os.path.exists(trace_path):
        print(f"Warning: Trace file not found at {trace_path}. Using default 3600s.")
        return 3600.0
    try:
        with open(trace_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            max_life = 0.0
            for row in reader:
                cdf = float(row['CDF'])
                life = float(row['lifetime'])
                max_life = max(max_life, life)
                if cdf >= threshold:
                    return life
            return max_life
    except Exception as e:
        print(f"Error parsing trace file: {e}. Using default 3600s.")
        return 3600.0

def main():
    parser = argparse.ArgumentParser(description="Spot Preemption Simulator")
    parser.add_argument("--dry-run", action="store_true", help="Print simulated lifetime and exit")
    args = parser.parse_args()

    print("Initializing Spot Preemption Simulator...")
    lifetime = get_simulated_lifetime()
    print(f"Simulated spot lifetime: {lifetime:.2f} seconds ({lifetime/3600:.2f} hours)")
    
    if args.dry_run:
        return
        
    script_to_run = os.path.join(os.path.dirname(__file__), "train_and_qat.py")
    
    print(f"Launching {script_to_run} ...")
    process = subprocess.Popen([sys.executable, script_to_run])
    
    start_time = time.time()
    
    try:
        while True:
            # Check if training process naturally exited early
            ret_code = process.poll()
            if ret_code is not None:
                print(f"\\nProcess finished gracefully with exit code {ret_code}.")
                break
                
            elapsed = time.time() - start_time
            if elapsed >= lifetime:
                print(f"\\n[!] Preemption triggered at {elapsed:.2f}s! Revoking capacity.")
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                print("Process hard-killed to simulate Spot Instance interruption.")
                process.wait()
                break
                
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\\nOrchestrator interrupted. Killing child training process.")
        process.terminate()
        process.wait()

if __name__ == "__main__":
    main()
