"""
Patient Agent - Personalized Baseline & Risk Assessment
Learns what's "normal" for THIS patient and detects meaningful deviations
"""

import json
import os
from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np


class PatientAgent:
    def __init__(self, patient_id, state_file="patient_state.json"):
        self.patient_id = patient_id
        self.state_file = state_file
        self.state = self._load_state()
        
    def _load_state(self):
        """Load or initialize patient state"""
        if os.path.exists(self.state_file):
            with open(self.state_file, 'r') as f:
                return json.load(f)
        
        # Initialize fresh patient state
        return {
            "patient_id": self.patient_id,
            "created_date": datetime.now().isoformat(),
            "days_seen": 0,
            "last_update": None,
            
            # Cardiac baselines (running statistics)
            "baseline": {
                "hr_mean": 75.0,        # Start with population average
                "hr_std": 15.0,
                "hrv_rmssd_mean": 35.0,
                "hrv_rmssd_std": 15.0,
                "hrv_sdnn_mean": 50.0,
                "hrv_sdnn_std": 20.0,
                "arrhythmia_rate": 0.05,  # 5% baseline
                "physio_risk_mean": 0.2,
                "physio_risk_std": 0.15
            },
            
            # Activity baselines
            "activity_profile": {
                "WALKING": 0.15,
                "SITTING": 0.40,
                "STANDING": 0.20,
                "LAYING": 0.20,
                "WALKING_UPSTAIRS": 0.025,
                "WALKING_DOWNSTAIRS": 0.025
            },
            
            # Behavioral state
            "behavioral_state": "STABLE",
            "sensitivity_factor": 1.0,
            "confidence": 0.0,  # Low until we have data
            
            # Trend tracking (for drift detection)
            "recent_trend": {
                "degrading_days": 0,
                "stable_days": 0,
                "improving_days": 0
            }
        }
    
    def _save_state(self):
        """Persist state to disk"""
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)
    
    def daily_update(self, sensor_outputs):
        """
        Process a full day of sensor outputs and update baseline
        
        Args:
            sensor_outputs: List of sensor agent outputs from the entire day
            
        Returns:
            dict: Patient agent output (sensitivity_factor, behavioral_state, confidence)
        """
        if not sensor_outputs:
            return self._get_output()
        
        # Aggregate daily metrics
        daily_metrics = self._aggregate_daily_metrics(sensor_outputs)
        
        # Update baseline (online learning with EMA)
        self._update_baseline(daily_metrics)
        
        # Detect behavioral state
        self._update_behavioral_state(daily_metrics)
        
        # Calculate sensitivity adjustment
        self._calculate_sensitivity()
        
        # Update metadata
        self.state["days_seen"] += 1
        self.state["last_update"] = datetime.now().isoformat()
        
        # Increase confidence as we see more data
        self.state["confidence"] = min(1.0, self.state["days_seen"] / 30.0)
        
        self._save_state()
        
        return self._get_output()
    
    def _aggregate_daily_metrics(self, sensor_outputs):
        """Aggregate sensor outputs into daily statistics"""
        hrs = []
        hrv_rmssds = []
        hrv_sdnns = []
        physio_risks = []
        arrhythmia_detections = []
        activities = defaultdict(int)
        
        for output in sensor_outputs:
            # Cardiac metrics
            if output.get('heart_rate', 0) > 0:
                hrs.append(output['heart_rate'])
            if output.get('hrv_rmssd', 0) > 0:
                hrv_rmssds.append(output['hrv_rmssd'])
            if output.get('hrv_sdnn', 0) > 0:
                hrv_sdnns.append(output['hrv_sdnn'])
            
            # Risk metrics
            physio_risks.append(output.get('physio_risk', 0))
            arrhythmia_detections.append(1 if output.get('arrhythmia_detected', False) else 0)
            
            # Activity
            activity = output.get('activity', 'UNKNOWN')
            if activity != 'UNKNOWN':
                activities[activity] += 1
        
        total_readings = len(sensor_outputs)
        total_activities = sum(activities.values())
        
        return {
            "hr_mean": np.mean(hrs) if hrs else 0,
            "hr_std": np.std(hrs) if len(hrs) > 1 else 0,
            "hrv_rmssd_mean": np.mean(hrv_rmssds) if hrv_rmssds else 0,
            "hrv_sdnn_mean": np.mean(hrv_sdnns) if hrv_sdnns else 0,
            "arrhythmia_rate": np.mean(arrhythmia_detections) if arrhythmia_detections else 0,
            "physio_risk_mean": np.mean(physio_risks),
            "physio_risk_std": np.std(physio_risks) if len(physio_risks) > 1 else 0,
            "physio_risk_max": np.max(physio_risks) if physio_risks else 0,
            "activity_distribution": {
                act: count / total_activities 
                for act, count in activities.items()
            } if total_activities > 0 else {}
        }
    
    def _update_baseline(self, daily_metrics):
        """Update baseline using exponential moving average (stable, no jumps)"""
        baseline = self.state["baseline"]
        
        # EMA weight: 5% for new data, 95% for history (very stable)
        # First week: faster learning (20% new data)
        alpha = 0.2 if self.state["days_seen"] < 7 else 0.05
        
        # Update cardiac baselines
        if daily_metrics["hr_mean"] > 0:
            baseline["hr_mean"] = (1 - alpha) * baseline["hr_mean"] + alpha * daily_metrics["hr_mean"]
            baseline["hr_std"] = (1 - alpha) * baseline["hr_std"] + alpha * daily_metrics["hr_std"]
        
        if daily_metrics["hrv_rmssd_mean"] > 0:
            baseline["hrv_rmssd_mean"] = (1 - alpha) * baseline["hrv_rmssd_mean"] + alpha * daily_metrics["hrv_rmssd_mean"]
            
        if daily_metrics["hrv_sdnn_mean"] > 0:
            baseline["hrv_sdnn_mean"] = (1 - alpha) * baseline["hrv_sdnn_mean"] + alpha * daily_metrics["hrv_sdnn_mean"]
        
        # Update risk baselines
        baseline["arrhythmia_rate"] = (1 - alpha) * baseline["arrhythmia_rate"] + alpha * daily_metrics["arrhythmia_rate"]
        baseline["physio_risk_mean"] = (1 - alpha) * baseline["physio_risk_mean"] + alpha * daily_metrics["physio_risk_mean"]
        baseline["physio_risk_std"] = (1 - alpha) * baseline["physio_risk_std"] + alpha * daily_metrics["physio_risk_std"]
        
        # Update activity profile
        if daily_metrics["activity_distribution"]:
            for activity, proportion in daily_metrics["activity_distribution"].items():
                current = self.state["activity_profile"].get(activity, 0)
                self.state["activity_profile"][activity] = (1 - alpha) * current + alpha * proportion
    
    def _update_behavioral_state(self, daily_metrics):
        """Determine if patient is STABLE, DEGRADED, or AT_RISK"""
        baseline = self.state["baseline"]
        
        # Need at least 7 days before making judgments
        if self.state["days_seen"] < 7:
            self.state["behavioral_state"] = "STABLE"
            return
        
        risk_factors = 0
        
        # Check 1: HR elevation
        if daily_metrics["hr_mean"] > 0:
            hr_deviation = (daily_metrics["hr_mean"] - baseline["hr_mean"]) / (baseline["hr_std"] + 1e-6)
            if hr_deviation > 1.5:  # 1.5 std above normal
                risk_factors += 1
        
        # Check 2: HRV reduction (lower HRV = worse)
        if daily_metrics["hrv_rmssd_mean"] > 0:
            hrv_deviation = (baseline["hrv_rmssd_mean"] - daily_metrics["hrv_rmssd_mean"]) / (baseline["hrv_rmssd_std"] + 1e-6)
            if hrv_deviation > 1.5:  # HRV dropped significantly
                risk_factors += 1
        
        # Check 3: Arrhythmia rate increase
        arrhythmia_increase = daily_metrics["arrhythmia_rate"] - baseline["arrhythmia_rate"]
        if arrhythmia_increase > 0.1:  # 10% absolute increase
            risk_factors += 1
        
        # Check 4: Overall physio risk elevation
        if daily_metrics["physio_risk_mean"] > 0:
            risk_deviation = (daily_metrics["physio_risk_mean"] - baseline["physio_risk_mean"]) / (baseline["physio_risk_std"] + 1e-6)
            if risk_deviation > 2.0:  # 2 std above normal
                risk_factors += 2  # Double weight on this
        
        # Check 5: Activity pattern disruption
        activity_diff = 0
        for activity, baseline_prop in self.state["activity_profile"].items():
            daily_prop = daily_metrics["activity_distribution"].get(activity, 0)
            activity_diff += abs(daily_prop - baseline_prop)
        
        if activity_diff > 0.3:  # 30% change in activity patterns
            risk_factors += 1
        
        # Update trend tracking
        if risk_factors >= 3:
            self.state["recent_trend"]["degrading_days"] += 1
            self.state["recent_trend"]["stable_days"] = 0
            self.state["recent_trend"]["improving_days"] = 0
        elif risk_factors == 0:
            self.state["recent_trend"]["improving_days"] += 1
            self.state["recent_trend"]["stable_days"] = 0
            self.state["recent_trend"]["degrading_days"] = 0
        else:
            self.state["recent_trend"]["stable_days"] += 1
            self.state["recent_trend"]["degrading_days"] = max(0, self.state["recent_trend"]["degrading_days"] - 1)
        
        # State determination (conservative: requires sustained change)
        if self.state["recent_trend"]["degrading_days"] >= 3:
            self.state["behavioral_state"] = "AT_RISK"
        elif self.state["recent_trend"]["degrading_days"] >= 1:
            self.state["behavioral_state"] = "DEGRADED"
        else:
            self.state["behavioral_state"] = "STABLE"
    
    def _calculate_sensitivity(self):
        """
        Adjust sensitivity factor based on behavioral state
        Higher sensitivity = lower threshold for alerts
        """
        state = self.state["behavioral_state"]
        
        if state == "AT_RISK":
            # Increase sensitivity (catch more issues)
            self.state["sensitivity_factor"] = 1.3
        elif state == "DEGRADED":
            # Slightly increased sensitivity
            self.state["sensitivity_factor"] = 1.15
        else:  # STABLE
            # Normal sensitivity
            self.state["sensitivity_factor"] = 1.0
        
        # Confidence modulation: less aggressive adjustments early on
        confidence = self.state["confidence"]
        baseline_factor = 1.0
        adjustment = self.state["sensitivity_factor"] - baseline_factor
        self.state["sensitivity_factor"] = baseline_factor + adjustment * confidence
    
    def _get_output(self):
        """Return patient agent output in standard format"""
        return {
            "day": datetime.now().strftime("%Y-%m-%d"),
            "sensitivity_factor": self.state["sensitivity_factor"],
            "behavioral_state": self.state["behavioral_state"],
            "confidence": self.state["confidence"]
        }
    
    def get_baseline(self):
        """Return current baseline for inspection"""
        return {
            "baseline": self.state["baseline"],
            "activity_profile": self.state["activity_profile"],
            "days_seen": self.state["days_seen"],
            "confidence": self.state["confidence"]
        }
    
    def get_risk_context(self, current_metrics):
        """
        Compare current metrics to baseline and return context
        Used by Decision Agent for explainability
        """
        baseline = self.state["baseline"]
        
        deviations = {}
        
        if current_metrics.get("heart_rate", 0) > 0:
            hr_z = (current_metrics["heart_rate"] - baseline["hr_mean"]) / (baseline["hr_std"] + 1e-6)
            deviations["hr_zscore"] = hr_z
            deviations["hr_status"] = "elevated" if hr_z > 1.5 else "normal"
        
        if current_metrics.get("hrv_rmssd", 0) > 0:
            hrv_z = (baseline["hrv_rmssd_mean"] - current_metrics["hrv_rmssd"]) / (baseline["hrv_rmssd_std"] + 1e-6)
            deviations["hrv_zscore"] = hrv_z
            deviations["hrv_status"] = "reduced" if hrv_z > 1.5 else "normal"
        
        arrhythmia_current = 1 if current_metrics.get("arrhythmia_detected", False) else 0
        deviations["arrhythmia_above_baseline"] = arrhythmia_current > baseline["arrhythmia_rate"]
        
        return {
            "deviations": deviations,
            "behavioral_state": self.state["behavioral_state"],
            "baseline_hr": baseline["hr_mean"],
            "baseline_hrv": baseline["hrv_rmssd_mean"]
        }