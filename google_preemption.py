import json
import os
import sys
import time
import subprocess
import numpy as np
import argparse
try:
    from scipy.optimize import curve_fit, root_scalar
except ImportError:
    print("Scipy is not installed. Please run `uv add scipy` or `pip install scipy`.")
    sys.exit(1)

def cdf_model(t, A, tau1, tau2, b):
    # F(t) = A * (1 - e^{-t/t1} + e^{(t-b)/t2})
    return A * (1 - np.exp(-t/tau1) + np.exp((t-b)/tau2))

def fit_and_sample_lifetime(data_path="preemption/goog-preemption-data/data/data.json"):
    data_path = os.path.join(os.path.dirname(__file__), data_path)
    
    with open(data_path, "r") as f:
        data = json.load(f)
        
    lifetimes_sec = []
    for key, val in data.items():
        if "time_in_sec" in val:
            lifetimes_sec.append(val["time_in_sec"])
            
    # Convert to hours
    lifetimes_hours = np.array(lifetimes_sec) / 3600.0
    lifetimes_hours = np.sort(lifetimes_hours)
    
    # Calculate empirical CDF
    n = len(lifetimes_hours)
    empirical_cdf = np.arange(1, n + 1) / n
    
    # Initial guesses based on the paper
    p0 = [0.45, 1.0, 0.8, 24.0]
    
    try:
        popt, _ = curve_fit(cdf_model, lifetimes_hours, empirical_cdf, p0=p0, bounds=([0, 0, 0, 0], [1, 10, 10, 30]), maxfev=10000)
        A, tau1, tau2, b = popt
        print(f"Fitted parameters: A={A:.4f}, tau1={tau1:.4f}, tau2={tau2:.4f}, b={b:.4f}")
    except Exception as e:
        print(f"Failed to fit CDF model: {e}. Falling back to default parameters.")
        A, tau1, tau2, b = p0
        
    # Sample a lifetime using inverse transform sampling
    u = np.random.uniform(0, 1)
    
    def obj(t):
        return cdf_model(t, A, tau1, tau2, b) - u
        
    try:
        res = root_scalar(obj, bracket=[0, 24], method='brentq')
        sampled_hours = res.root
    except ValueError:
        if u > cdf_model(24, A, tau1, tau2, b):
            sampled_hours = 24.0
        else:
            sampled_hours = 0.0
            
    sampled_sec = sampled_hours * 3600.0
    return sampled_sec

def main():
    parser = argparse.ArgumentParser(description="Google Preemption Simulator")
    parser.add_argument("--dry-run", action="store_true", help="Print simulated lifetime and exit")
    args = parser.parse_args()

    print("Initializing Google Preemption Simulator...")
    lifetime = fit_and_sample_lifetime()
    print(f"Simulated spot lifetime: {lifetime:.2f} seconds ({lifetime/3600:.2f} hours)")
    
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
