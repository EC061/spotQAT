import os
import sys
import time
import subprocess
import numpy as np
import argparse

def sample_poisson_lifetime(mttf_hours=2.0):
    u = np.random.uniform(0, 1)
    sampled_hours = -mttf_hours * np.log(1 - u)
    return sampled_hours * 3600.0

def main():
    parser = argparse.ArgumentParser(description="Poisson Preemption Simulator")
    parser.add_argument("--dry-run", action="store_true", help="Print simulated lifetime and exit")
    args = parser.parse_args()

    print("Initializing Poisson Preemption Simulator...")
    lifetime = sample_poisson_lifetime(mttf_hours=2.0)
    print(f"Simulated Poisson spot lifetime: {lifetime:.2f} seconds ({lifetime/3600:.2f} hours)")
    
    if args.dry_run:
        return
        
    script_to_run = os.path.join(os.path.dirname(__file__), "train_and_qat.py")
    
    print(f"Launching {script_to_run} ...")
    process = subprocess.Popen([sys.executable, script_to_run])
    
    start_time = time.time()
    
    try:
        while True:
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
