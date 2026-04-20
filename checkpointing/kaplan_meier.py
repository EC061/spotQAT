import os
import time
import json
import numpy as np
import pandas as pd
from .base import BaseCheckpointWriter
from checkpoint_service.checkpoint_client_async import send_checkpoint_file

class KaplanMeierCheckpointWriter(BaseCheckpointWriter):
    def __init__(self, checkpoint_path, checkpoint_times, record_timing_fn, remote_name, data_source="gcp", risk_threshold=0.05, window_size=600):
        super().__init__(checkpoint_path, checkpoint_times, record_timing_fn)
        self.remote_name = remote_name
        self.data_source = data_source
        self.risk_threshold = risk_threshold  # e.g., 5% risk threshold
        self.window_size = window_size        # Assess risk for the next X seconds (e.g., 10 mins)
        self.start_time = time.time()
        
        self.lifetimes = self._load_data()
        self.km_survival = self._compute_kaplan_meier()

    def _load_data(self):
        lifetimes = []
        if self.data_source == "gcp":
            # Load from gcp JSON
            data_path = os.path.join(os.path.dirname(__file__), "..", "data", "gcp", "data.json")
            if os.path.exists(data_path):
                with open(data_path, "r") as f:
                    data = json.load(f)
                for key, val in data.items():
                    if "time_in_sec" in val:
                        lifetimes.append(val["time_in_sec"])
            else:
                print(f"File {data_path} not found. Fallback to default lifetimes.")
                lifetimes = [3600, 7200, 14400, 86400]
        elif self.data_source == "aws":
            # Load from AWS CSV
            data_path = os.path.join(os.path.dirname(__file__), "..", "Emulator-unified", "us-east-1a-lifetime.csv")
            if os.path.exists(data_path):
                df = pd.read_csv(data_path)
                if 'Duration' in df.columns:
                    lifetimes = df['Duration'].tolist()
            else:
                lifetimes = [3600, 7200, 14400, 86400]
                
        lifetimes = np.sort(lifetimes)
        return lifetimes

    def _compute_kaplan_meier(self):
        # We perform standard KM estimate (without censoring)
        # S(t) = 1 - Empirical_CDF(t)
        n = len(self.lifetimes)
        if n == 0:
            return lambda t: 1.0 # Never fail
        
        empirical_cdf = np.arange(1, n + 1) / n
        survival = 1.0 - empirical_cdf
        
        def survival_fn(t):
            # Return survival probability at time t
            idx = np.searchsorted(self.lifetimes, t, side='right')
            if idx >= n:
                return 0.0
            if idx == 0:
                return 1.0
            return survival[idx-1]
            
        return survival_fn
        
    def get_conditional_survival(self, current_time, window):
        # Probability of surviving until current_time + window GIVEN survived until current_time
        s_current = self.km_survival(current_time)
        if s_current <= 0:
            return 0.0
        s_future = self.km_survival(current_time + window)
        return s_future / s_current
        
    def should_save(self, elapsed_time_since_last_save, total_elapsed_time):
        """
        Calculates whether a checkpoint should be written now based on KM risk analysis.
        """
        survival_prob = self.get_conditional_survival(total_elapsed_time, self.window_size)
        failure_prob = 1.0 - survival_prob
        
        min_interval = 300 # minimum interval of 5 mins between checkpoints
        
        if failure_prob > self.risk_threshold and elapsed_time_since_last_save > min_interval:
            return True
            
        if elapsed_time_since_last_save > 3600: # backup interval of 1h
            return True
            
        return False

    def save_checkpoint(self, payload, epoch_idx, step_idx, phase):
        t0 = time.time()
        self._atomic_save_checkpoint(payload)
        send_checkpoint_file(self.checkpoint_path, remote_name=self.remote_name)
        dt = time.time() - t0
        self.checkpoint_times.append(dt)
        self.record_timing_fn(phase, epoch_idx, step_idx, "checkpoint_adaptive_kaplan_meier", dt)
