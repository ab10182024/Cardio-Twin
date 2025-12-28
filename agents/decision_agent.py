"""
Decision Agent - Risk Fusion & Alert Generation
Combines Sensor, Patient, and Clinical outputs to make alert decisions
"""

import json
import os
from datetime import datetime, timedelta
from collections import deque
import numpy as np


class DecisionAgent:
    def __init__(self, patient_id, state_file="decision_state.json"):
        self.patient_id = patient_id
        self.state_file = state_file
        self.state = self._load_state()
        
        # Fusion weights (must sum to 1.0)
        self.alpha = 0.5   # Sensor/physiological risk weight
        self.beta = 0.3    # Patient/personalization weight
        self.gamma = 0.2   # Clinical/static risk weight
        
        # Alert thresholds
        self.MONITOR_THRESHOLD = 0.5   # Start monitoring
        self.ALERT_THRESHOLD = 0.7     # Trigger alert
        
        # Persistence requirements (prevent single-spike alerts)
        self.MONITOR_PERSISTENCE = 3   # 3 consecutive readings above MONITOR
        self.ALERT_PERSISTENCE = 5     # 5 consecutive readings above ALERT
        
        # Cooldown between alerts (prevent spam)
        self.ALERT_COOLDOWN_MINUTES = 30
        
    def _load_state(self):
        """Load or initialize decision state"""
        if os.path.exists(self.state_file):
            with open(self.state_file, 'r') as f:
                state = json.load(f)
                # Convert alert history back to deque
                state['alert_history'] = deque(state['alert_history'], maxlen=100)
                state['recent_risks'] = deque(state['recent_risks'], maxlen=20)
                return state
        
        return {
            "patient_id": self.patient_id,
            "created_date": datetime.now().isoformat(),
            "last_decision": None,
            "alert_history": deque(maxlen=100),  # Last 100 alerts
            "recent_risks": deque(maxlen=20),    # Last 20 risk scores for persistence check
            "consecutive_monitor": 0,
            "consecutive_alert": 0,
            "last_alert_time": None,
            "total_alerts": 0,
            "total_monitors": 0
        }
    
    def _save_state(self):
        """Persist state to disk"""
        state_copy = self.state.copy()
        # Convert deques to lists for JSON serialization
        state_copy['alert_history'] = list(state_copy['alert_history'])
        state_copy['recent_risks'] = list(state_copy['recent_risks'])
        
        with open(self.state_file, 'w') as f:
            json.dump(state_copy, f, indent=2)
    
    def make_decision(self, sensor_output, patient_output, clinical_output):
        """
        Main decision function: fuse all inputs and decide on alert
        
        Args:
            sensor_output: dict from Sensor Agent
            patient_output: dict from Patient Agent
            clinical_output: dict from Clinical Agent
            
        Returns:
            dict: Decision output with global_risk, decision, explanation
        """
        # Calculate global risk score
        global_risk = self._calculate_global_risk(
            sensor_output, patient_output, clinical_output
        )
        
        # Store for persistence checking
        self.state['recent_risks'].append({
            'timestamp': datetime.now().isoformat(),
            'risk': global_risk
        })
        
        # Determine decision with persistence
        decision, explanation = self._determine_decision(
            global_risk, sensor_output, patient_output, clinical_output
        )
        
        # Log decision
        decision_output = {
            "day": datetime.now().strftime("%Y-%m-%d"),
            "timestamp": datetime.now().isoformat(),
            "global_risk": global_risk,
            "decision": decision,
            "explanation": explanation,
            "components": {
                "sensor_risk": sensor_output.get('physio_risk', 0),
                "patient_sensitivity": patient_output.get('sensitivity_factor', 1.0),
                "clinical_risk": clinical_output.get('clinical_risk', 0.5)
            }
        }
        
        # If alert, add to history
        if decision == "ALERT":
            self._log_alert(decision_output)
        
        self.state['last_decision'] = decision_output
        self._save_state()
        
        return decision_output
    
    def _calculate_global_risk(self, sensor_output, patient_output, clinical_output):
        """
        Fusion formula: combines all three agent outputs
        
        global_risk = α × physio_risk × sensor_confidence
                    + β × sensitivity_factor × patient_confidence  
                    + γ × clinical_risk × clinical_confidence
        """
        # Sensor component
        physio_risk = sensor_output.get('physio_risk', 0)
        sensor_confidence = sensor_output.get('physio_confidence', 0)
        sensor_component = physio_risk * sensor_confidence
        
        # Patient component (sensitivity modulates the weight)
        sensitivity_factor = patient_output.get('sensitivity_factor', 1.0)
        patient_confidence = patient_output.get('confidence', 0)
        # Normalize sensitivity to 0-1 range (1.0 baseline, 1.3 max -> 0 to 1)
        # Clamp to avoid division issues
        normalized_sensitivity = max(0, min(1, (sensitivity_factor - 0.8) / 0.5))  # Maps 0.8-1.3 to 0-1
        patient_component = normalized_sensitivity * patient_confidence
        
        # Clinical component
        clinical_risk = clinical_output.get('clinical_risk', 0.5)
        clinical_confidence = clinical_output.get('confidence', 1.0)
        clinical_component = clinical_risk * clinical_confidence
        
        # Weighted fusion
        global_risk = (
            self.alpha * sensor_component +
            self.beta * patient_component +
            self.gamma * clinical_component
        )
        
        return min(1.0, max(0.0, global_risk))  # Clamp to [0, 1]
    
    def _determine_decision(self, global_risk, sensor_output, patient_output, clinical_output):
        """
        Determine decision with persistence requirement
        
        Rules:
        1. NO_ALERT: Risk below MONITOR_THRESHOLD or insufficient persistence
        2. MONITOR: Risk above MONITOR_THRESHOLD with persistence
        3. ALERT: Risk above ALERT_THRESHOLD with persistence AND not in cooldown
        """
        # Check cooldown
        in_cooldown = self._is_in_cooldown()
        
        # Update persistence counters
        if global_risk >= self.ALERT_THRESHOLD:
            self.state['consecutive_alert'] += 1
            self.state['consecutive_monitor'] = max(self.state['consecutive_monitor'], self.ALERT_PERSISTENCE)
        elif global_risk >= self.MONITOR_THRESHOLD:
            self.state['consecutive_monitor'] += 1
            self.state['consecutive_alert'] = 0
        else:
            self.state['consecutive_monitor'] = 0
            self.state['consecutive_alert'] = 0
        
        # Decision logic
        if (self.state['consecutive_alert'] >= self.ALERT_PERSISTENCE and 
            not in_cooldown):
            decision = "ALERT"
            explanation = self._generate_alert_explanation(
                global_risk, sensor_output, patient_output, clinical_output
            )
            self.state['total_alerts'] += 1
        elif self.state['consecutive_monitor'] >= self.MONITOR_PERSISTENCE:
            decision = "MONITOR"
            explanation = self._generate_monitor_explanation(
                global_risk, sensor_output, patient_output, clinical_output
            )
            self.state['total_monitors'] += 1
        else:
            decision = "NO_ALERT"
            explanation = "Risk levels within normal range for this patient."
        
        return decision, explanation
    
    def _is_in_cooldown(self):
        """Check if we're still in cooldown period after last alert"""
        if not self.state['last_alert_time']:
            return False
        
        last_alert = datetime.fromisoformat(self.state['last_alert_time'])
        time_since = datetime.now() - last_alert
        
        return time_since < timedelta(minutes=self.ALERT_COOLDOWN_MINUTES)
    
    def _generate_alert_explanation(self, global_risk, sensor, patient, clinical):
        """
        Generate human-readable explanation for ALERT
        Must answer: What changed? Compared to what? For how long? Why now?
        """
        explanation_parts = []
        
        # What changed?
        risk_level = "CRITICAL" if global_risk > 0.85 else "HIGH"
        explanation_parts.append(f"🚨 {risk_level} RISK DETECTED ({global_risk*100:.0f}%)")
        
        # Identify main contributors
        contributors = []
        
        # Sensor component
        if sensor.get('physio_risk', 0) > 0.6:
            details = []
            if sensor.get('arrhythmia_detected'):
                details.append(f"arrhythmia probability {sensor.get('arrhythmia_probability', 0)*100:.0f}%")
            if sensor.get('rhythm_irregular'):
                details.append("irregular rhythm detected")
            if details:
                contributors.append(f"Physiological: {', '.join(details)}")
        
        # Patient component
        behavioral_state = patient.get('behavioral_state', 'UNKNOWN')
        if behavioral_state in ['DEGRADED', 'AT_RISK']:
            sensitivity = patient.get('sensitivity_factor', 1.0)
            contributors.append(
                f"Behavioral: Patient state is {behavioral_state} "
                f"(sensitivity {sensitivity:.2f}x normal)"
            )
        
        # Clinical component
        if clinical.get('clinical_risk', 0) > 0.6:
            contributors.append(
                f"Clinical: High baseline risk ({clinical.get('clinical_risk', 0)*100:.0f}%) "
                f"from medical history"
            )
        
        if contributors:
            explanation_parts.append("Contributors: " + " | ".join(contributors))
        
        # For how long?
        explanation_parts.append(
            f"Sustained for {self.state['consecutive_alert']} consecutive readings "
            f"(minimum {self.ALERT_PERSISTENCE} required)"
        )
        
        # Why now?
        if patient.get('behavioral_state') == 'AT_RISK':
            explanation_parts.append(
                "⚠️ Patient has shown concerning behavioral patterns over recent days"
            )
        
        # Recommendation
        explanation_parts.append("📞 RECOMMENDATION: Contact healthcare provider or seek immediate evaluation")
        
        return " • ".join(explanation_parts)
    
    def _generate_monitor_explanation(self, global_risk, sensor, patient, clinical):
        """Generate explanation for MONITOR status"""
        explanation_parts = []
        
        explanation_parts.append(f"⚠️ ELEVATED RISK ({global_risk*100:.0f}%)")
        
        # Key factors
        factors = []
        if sensor.get('physio_risk', 0) > 0.4:
            factors.append("elevated physiological readings")
        if patient.get('behavioral_state') == 'DEGRADED':
            factors.append("declining behavioral patterns")
        if clinical.get('clinical_risk', 0) > 0.5:
            factors.append("pre-existing risk factors")
        
        if factors:
            explanation_parts.append(f"Due to: {', '.join(factors)}")
        
        explanation_parts.append(
            f"Persistent for {self.state['consecutive_monitor']} readings "
            f"(alert at {self.ALERT_PERSISTENCE})"
        )
        
        explanation_parts.append("💡 Monitor symptoms and continue tracking")
        
        return " • ".join(explanation_parts)
    
    def _log_alert(self, decision_output):
        """Log alert to history"""
        alert_record = {
            **decision_output,
            "alert_id": len(self.state['alert_history']) + 1,
            "acknowledged": False
        }
        
        self.state['alert_history'].append(alert_record)
        self.state['last_alert_time'] = datetime.now().isoformat()
        self.state['consecutive_alert'] = 0  # Reset after triggering
    
    def get_alert_history(self, limit=10):
        """Get recent alerts"""
        return list(self.state['alert_history'])[-limit:]
    
    def acknowledge_alert(self, alert_id):
        """Mark an alert as acknowledged"""
        for alert in self.state['alert_history']:
            if alert.get('alert_id') == alert_id:
                alert['acknowledged'] = True
                alert['acknowledged_at'] = datetime.now().isoformat()
                self._save_state()
                return True
        return False
    
    def get_current_status(self):
        """Get current decision system status"""
        in_cooldown = self._is_in_cooldown()
        time_until_ready = None
        
        if in_cooldown:
            last_alert = datetime.fromisoformat(self.state['last_alert_time'])
            ready_time = last_alert + timedelta(minutes=self.ALERT_COOLDOWN_MINUTES)
            time_until_ready = (ready_time - datetime.now()).total_seconds() / 60
        
        return {
            "consecutive_monitor": self.state['consecutive_monitor'],
            "consecutive_alert": self.state['consecutive_alert'],
            "in_cooldown": in_cooldown,
            "time_until_ready_minutes": time_until_ready,
            "total_alerts": self.state['total_alerts'],
            "total_monitors": self.state['total_monitors'],
            "last_decision": self.state.get('last_decision'),
            "alert_history_count": len(self.state['alert_history'])
        }
    
    def get_risk_trend(self):
        """Get recent risk trend for visualization"""
        if not self.state['recent_risks']:
            return []
        
        return [
            {
                'timestamp': r['timestamp'],
                'risk': r['risk']
            }
            for r in self.state['recent_risks']
        ]