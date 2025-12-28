"""
Clinical Agent - Static Medical Risk Assessment
Based on patient's clinical profile using logistic regression
"""

import json
import os
from datetime import datetime
import joblib
import numpy as np


class ClinicalAgent:
    def __init__(self, patient_id, state_file="clinical_profile.json"):
        self.patient_id = patient_id
        self.state_file = state_file
        self.state = self._load_state()
        
        # Load trained model and feature list
        try:
            self.model = joblib.load('models/clinical_agent_model.joblib')
            self.features = joblib.load('models/clinical_agent_features.joblib')
            print("✓ Clinical model loaded successfully")
        except Exception as e:
            print(f"⚠ Could not load clinical model: {e}")
            self.model = None
            self.features = [
                "age", "sex", "BMI", "SBP", "DBP",
                "total_cholesterol", "HDL",
                "diabetes", "smoker", "hypertension",
                "on_beta_blocker", "on_antihypertensive", 
                "on_statin", "on_anticoagulant"
            ]
    
    def _load_state(self):
        """Load or initialize clinical profile"""
        if os.path.exists(self.state_file):
            with open(self.state_file, 'r') as f:
                return json.load(f)
        
        # Initialize with default/unknown values
        return {
            "patient_id": self.patient_id,
            "created_date": datetime.now().isoformat(),
            "last_update": None,
            "clinical_risk": 0.5,  # Default medium risk until profile provided
            "profile_complete": False,
            
            # Clinical features (defaults)
            "age": None,
            "sex": None,  # 0=Female, 1=Male
            "BMI": None,
            "SBP": None,  # Systolic Blood Pressure
            "DBP": None,  # Diastolic Blood Pressure
            "total_cholesterol": None,
            "HDL": None,  # High-density lipoprotein
            "diabetes": 0,  # 0=No, 1=Yes
            "smoker": 0,
            "hypertension": 0,
            "on_beta_blocker": 0,
            "on_antihypertensive": 0,
            "on_statin": 0,
            "on_anticoagulant": 0
        }
    
    def _save_state(self):
        """Persist state to disk"""
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)
    
    def update_profile(self, clinical_data):
        """
        Update clinical profile and recalculate risk
        
        Args:
            clinical_data: dict with clinical features
            
        Returns:
            dict: Clinical agent output (clinical_risk, profile_complete)
        """
        # Update profile
        for key, value in clinical_data.items():
            if key in self.state and key not in ["patient_id", "created_date", "last_update", 
                                                   "clinical_risk", "profile_complete"]:
                self.state[key] = value
        
        # Check if profile is complete
        required_fields = ["age", "sex", "BMI", "SBP", "DBP", "total_cholesterol", "HDL"]
        self.state["profile_complete"] = all(
            self.state.get(field) is not None for field in required_fields
        )
        
        # Calculate risk if model available and profile complete
        if self.model and self.state["profile_complete"]:
            self.state["clinical_risk"] = self._calculate_risk()
        else:
            # Default risk if incomplete
            self.state["clinical_risk"] = 0.5
        
        self.state["last_update"] = datetime.now().isoformat()
        self._save_state()
        
        return self._get_output()
    
    def _calculate_risk(self):
        """Calculate clinical risk using logistic regression model"""
        try:
            # Prepare feature vector in correct order
            feature_vector = []
            for feature in self.features:
                value = self.state.get(feature)
                if value is None:
                    value = 0  # Default for missing values
                feature_vector.append(float(value))
            
            # Reshape for prediction
            X = np.array(feature_vector).reshape(1, -1)
            
            # Get probability (0-1)
            risk_proba = self.model.predict_proba(X)[0][1]  # Probability of positive class
            
            return float(risk_proba)
            
        except Exception as e:
            print(f"Error calculating clinical risk: {e}")
            return 0.5
    
    def get_profile(self):
        """Return current clinical profile"""
        return {
            "patient_id": self.state["patient_id"],
            "last_update": self.state["last_update"],
            "profile_complete": self.state["profile_complete"],
            "clinical_risk": self.state["clinical_risk"],
            "profile": {
                "age": self.state.get("age"),
                "sex": "Male" if self.state.get("sex") == 1 else "Female" if self.state.get("sex") == 0 else None,
                "BMI": self.state.get("BMI"),
                "SBP": self.state.get("SBP"),
                "DBP": self.state.get("DBP"),
                "total_cholesterol": self.state.get("total_cholesterol"),
                "HDL": self.state.get("HDL"),
                "diabetes": bool(self.state.get("diabetes")),
                "smoker": bool(self.state.get("smoker")),
                "hypertension": bool(self.state.get("hypertension")),
                "medications": {
                    "beta_blocker": bool(self.state.get("on_beta_blocker")),
                    "antihypertensive": bool(self.state.get("on_antihypertensive")),
                    "statin": bool(self.state.get("on_statin")),
                    "anticoagulant": bool(self.state.get("on_anticoagulant"))
                }
            }
        }
    
    def _get_output(self):
        """Return clinical agent output in standard format"""
        return {
            "day": datetime.now().strftime("%Y-%m-%d"),
            "clinical_risk": self.state["clinical_risk"],
            "confidence": 1.0  # Clinical data is always 100% confident (static)
        }
    
    def get_risk_factors(self):
        """Return identified risk factors for explainability"""
        risk_factors = []
        
        if self.state.get("age") and self.state["age"] > 65:
            risk_factors.append("Advanced age (>65)")
        
        if self.state.get("BMI"):
            bmi = self.state["BMI"]
            if bmi >= 30:
                risk_factors.append(f"Obesity (BMI {bmi:.1f})")
            elif bmi >= 25:
                risk_factors.append(f"Overweight (BMI {bmi:.1f})")
        
        if self.state.get("SBP") and self.state["SBP"] >= 140:
            risk_factors.append(f"Elevated systolic BP ({self.state['SBP']} mmHg)")
        
        if self.state.get("DBP") and self.state["DBP"] >= 90:
            risk_factors.append(f"Elevated diastolic BP ({self.state['DBP']} mmHg)")
        
        if self.state.get("total_cholesterol") and self.state["total_cholesterol"] >= 240:
            risk_factors.append(f"High cholesterol ({self.state['total_cholesterol']} mg/dL)")
        
        if self.state.get("HDL") and self.state["HDL"] < 40:
            risk_factors.append(f"Low HDL ({self.state['HDL']} mg/dL)")
        
        if self.state.get("diabetes"):
            risk_factors.append("Diabetes mellitus")
        
        if self.state.get("smoker"):
            risk_factors.append("Active smoker")
        
        if self.state.get("hypertension"):
            risk_factors.append("Hypertension")
        
        protective_factors = []
        if self.state.get("on_statin"):
            protective_factors.append("On statin therapy")
        if self.state.get("on_antihypertensive"):
            protective_factors.append("On antihypertensive medication")
        if self.state.get("on_beta_blocker"):
            protective_factors.append("On beta blocker")
        if self.state.get("on_anticoagulant"):
            protective_factors.append("On anticoagulant")
        
        return {
            "risk_factors": risk_factors,
            "protective_factors": protective_factors,
            "risk_level": self._get_risk_level(self.state["clinical_risk"])
        }
    
    def _get_risk_level(self, risk_score):
        """Categorize risk score into levels"""
        if risk_score < 0.3:
            return "LOW"
        elif risk_score < 0.6:
            return "MODERATE"
        elif risk_score < 0.8:
            return "HIGH"
        else:
            return "VERY HIGH"