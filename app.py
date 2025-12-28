from flask import Flask, render_template, jsonify, request
import json
import threading
import time
from collections import deque
from datetime import datetime, date
import numpy as np
import joblib
from scipy.signal import butter, filtfilt
import neurokit2 as nk
import torch
import torch.nn as nn
import sys
import os

# Optional serial import (not used in simulation mode)
try:
    import serial
except Exception:
    serial = None


# ============================================================
# APP
# ============================================================
app = Flask(__name__)

# ============================================================
# CONFIG
# ============================================================
SIMULATION_MODE = os.environ.get("SIMULATION_MODE", "1") == "1"  # default ON

SERIAL_PORT = "COM3"   # ignored in simulation mode
BAUD_RATE = 115200

MAX_DATA_POINTS = 3600  # 10 seconds @ 360Hz

ECG_FS = 360
ECG_WIN_SEC = 10
ECG_WIN_SAMPLES = ECG_FS * ECG_WIN_SEC

ACTIVITY_WINDOW_SIZE = 128

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# SAFE STATE FILE HANDLING
# - If JSON exists but is empty/corrupted, delete it.
# - If file is missing, agent will create defaults.
# ============================================================
def delete_if_invalid_json(path: str):
    if not os.path.exists(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            s = f.read().strip()
            if not s:
                raise ValueError("Empty JSON file")
            json.loads(s)
    except Exception:
        try:
            os.remove(path)
            print(f"⚠ Deleted corrupted/empty state file: {path}")
        except Exception:
            pass


os.makedirs("data", exist_ok=True)
delete_if_invalid_json("data/patient_state.json")
delete_if_invalid_json("data/clinical_profile.json")
delete_if_invalid_json("data/decision_state.json")


# ============================================================
# IMPORT AGENTS
# ============================================================
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "agents"))
from patient_agent import PatientAgent
from clinical_agent import ClinicalAgent
from decision_agent import DecisionAgent

patient_agent = PatientAgent(patient_id="patient_001", state_file="data/patient_state.json")
clinical_agent = ClinicalAgent(patient_id="patient_001", state_file="data/clinical_profile.json")
decision_agent = DecisionAgent(patient_id="patient_001", state_file="data/decision_state.json")


# ============================================================
# DATA BUFFERS
# ============================================================
sensor_data = {
    "ecg": deque(maxlen=MAX_DATA_POINTS),
    "accel_x": deque(maxlen=MAX_DATA_POINTS),
    "accel_y": deque(maxlen=MAX_DATA_POINTS),
    "accel_z": deque(maxlen=MAX_DATA_POINTS),
    "gyro_x": deque(maxlen=MAX_DATA_POINTS),
    "gyro_y": deque(maxlen=MAX_DATA_POINTS),
    "gyro_z": deque(maxlen=MAX_DATA_POINTS),
    "timestamps": deque(maxlen=MAX_DATA_POINTS),
}

predictions = {
    "activity": "UNKNOWN",
    "activity_confidence": 0.0,
    "heart_rate": 0.0,
    "hrv_rmssd": 0.0,
    "hrv_sdnn": 0.0,
    "rhythm_irregular": False,
    "ecg_quality": 0.0,
    "arrhythmia_probability": 0.0,
    "arrhythmia_detected": False,
    "physio_risk": 0.0,
    "physio_confidence": 0.0,
    "global_risk": 0.0,
    "decision": "NO_ALERT",
    "decision_explanation": "Initializing...",
}

daily_sensor_outputs = []
current_date = date.today()

serial_connection = None
reading_thread = None
prediction_thread = None
sim_thread = None
is_reading = False


# ============================================================
# ECG CNN MODEL (MUST MATCH SAVED .pt)
# ============================================================
class SimpleECGCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(16, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Linear(64, 1)

    def forward(self, x):
        x = self.features(x)
        x = x.squeeze(-1)
        x = self.classifier(x)
        return x.squeeze(1)


# ============================================================
# LOAD MODELS
# ============================================================
# Activity model
try:
    activity_model = joblib.load("models/activity_rf_ucihar.pkl")
    print("✓ Activity recognition model loaded")
except Exception as e:
    print(f"⚠ Could not load activity model: {e}")
    activity_model = None

# ECG CNN model
try:
    checkpoint = torch.load("models/ecg_cnn_win10s_binary.pt", map_location=DEVICE)
    ecg_model = SimpleECGCNN().to(DEVICE)
    ecg_model.load_state_dict(checkpoint["model_state"])
    ecg_model.eval()
    ecg_threshold = checkpoint.get("threshold", 0.5)
    print(f"✓ ECG arrhythmia model loaded (threshold: {ecg_threshold})")
except Exception as e:
    print(f"⚠ Could not load ECG model: {e}")
    ecg_model = None
    ecg_threshold = 0.5

# Clinical model is loaded inside ClinicalAgent; it prints its own status
print("✓ Clinical model loaded successfully")


# IMPORTANT: After rebuild_activity_model.py, labels are 0..5
activity_labels = {
    0: "WALKING",
    1: "WALKING_UPSTAIRS",
    2: "WALKING_DOWNSTAIRS",
    3: "SITTING",
    4: "STANDING",
    5: "LAYING",
}


# ============================================================
# SIGNAL PROCESSING
# ============================================================
def bandpass_filter(x, fs, low=0.5, high=40.0, order=4):
    if len(x) < order * 3:
        return x
    nyq = 0.5 * fs
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, x)


def predict_arrhythmia(ecg_segment, fs=ECG_FS):
    if ecg_model is None:
        return 0.0, False
    try:
        if len(ecg_segment) != ECG_WIN_SAMPLES:
            return 0.0, False

        ecg_filtered = bandpass_filter(ecg_segment, fs)
        ecg_norm = (ecg_filtered - ecg_filtered.mean()) / (ecg_filtered.std() + 1e-8)

        ecg_tensor = torch.tensor(ecg_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = ecg_model(ecg_tensor)
            prob = torch.sigmoid(logits).item()

        return float(prob), bool(prob >= ecg_threshold)
    except Exception as e:
        print(f"Error in arrhythmia prediction: {e}")
        return 0.0, False


def compute_ecg_features(ecg_signal, fs=ECG_FS):
    try:
        if len(ecg_signal) < 1000:
            return None

        ecg_filtered = bandpass_filter(ecg_signal, fs, low=0.5, high=40.0)
        signals_nk, info_nk = nk.ecg_process(ecg_filtered, sampling_rate=fs)
        rpeaks = info_nk.get("ECG_R_Peaks", [])

        if len(rpeaks) < 3:
            return None

        rr_sec = np.diff(rpeaks) / fs
        hr_bpm = 60.0 / np.mean(rr_sec) if len(rr_sec) > 0 else 0.0

        rr_ms = rr_sec * 1000.0
        sdnn = float(np.std(rr_ms, ddof=1)) if len(rr_ms) > 1 else 0.0

        diff_rr = np.diff(rr_ms)
        rmssd = float(np.sqrt(np.mean(diff_rr ** 2))) if len(diff_rr) > 0 else 0.0

        pnn50 = float(100.0 * np.mean(np.abs(diff_rr) > 50.0)) if len(diff_rr) > 0 else 0.0
        irregular = bool((rmssd > 50.0) or (pnn50 > 20.0))

        ok_rr = (rr_sec >= 0.3) & (rr_sec <= 2.0)
        sqi = float(ok_rr.mean()) if len(ok_rr) > 0 else 0.0

        return {
            "heart_rate": float(hr_bpm),
            "hrv_rmssd": float(rmssd),
            "hrv_sdnn": float(sdnn),
            "pnn50": float(pnn50),
            "rhythm_irregular": irregular,
            "ecg_quality": sqi,
        }
    except Exception as e:
        print(f"Error computing ECG features: {e}")
        return None


def extract_activity_features(window_data):
    """
    Lightweight feature extraction into 561 dims (to match RF input size).
    """
    features = []
    for sensor in ["accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"]:
        signal = np.array(window_data[sensor], dtype=float)

        features.extend(
            [
                np.mean(signal),
                np.std(signal),
                np.median(signal),
                np.max(signal),
                np.min(signal),
                np.max(signal) - np.min(signal),
                np.percentile(signal, 25),
                np.percentile(signal, 75),
                np.sum(np.abs(signal)),
                np.mean(np.abs(signal)),
            ]
        )

        fft_vals = np.fft.fft(signal)
        fft_mag = np.abs(fft_vals[: len(fft_vals) // 2])

        features.extend([np.mean(fft_mag), np.std(fft_mag), np.max(fft_mag), np.sum(fft_mag), float(np.argmax(fft_mag))])

    accel_mag = np.sqrt(
        np.array(window_data["accel_x"], dtype=float) ** 2
        + np.array(window_data["accel_y"], dtype=float) ** 2
        + np.array(window_data["accel_z"], dtype=float) ** 2
    )
    gyro_mag = np.sqrt(
        np.array(window_data["gyro_x"], dtype=float) ** 2
        + np.array(window_data["gyro_y"], dtype=float) ** 2
        + np.array(window_data["gyro_z"], dtype=float) ** 2
    )

    features.extend([np.mean(accel_mag), np.std(accel_mag), np.mean(gyro_mag), np.std(gyro_mag)])

    while len(features) < 561:
        features.append(0.0)

    return np.array(features[:561], dtype=float)


def predict_activity(window_data):
    if activity_model is None:
        return None
    try:
        X = extract_activity_features(window_data).reshape(1, -1)
        pred_id = int(activity_model.predict(X)[0])          # 0..5
        proba = activity_model.predict_proba(X)[0]           # length 6

        activity_name = activity_labels.get(pred_id, "UNKNOWN")
        confidence = float(np.max(proba))

        prob_map = {activity_labels[i]: float(proba[i]) for i in range(min(len(proba), 6))}

        return {"activity": activity_name, "confidence": confidence, "probabilities": prob_map}
    except Exception as e:
        print(f"Error predicting activity: {e}")
        return None


# ============================================================
# SIMULATION STREAM (NO HARDWARE)
# ============================================================
def simulate_sensor_data():
    """
    Generates synthetic ECG + IMU streams and fills sensor_data deques.
    Cycles scenarios so the dashboard has variation.
    """
    global is_reading

    fs = ECG_FS
    dt = 1.0 / fs
    t = 0.0

    scenario_cycle = [
        ("NORMAL", 40),
        ("MILD", 40),
        ("HIGH_RISK", 30),
        ("NORMAL", 40),
    ]
    idx = 0
    scenario, remaining = scenario_cycle[idx]

    rr_base = 0.85  # ~70 bpm
    next_r_peak = 0.0

    while is_reading:
        remaining -= dt
        if remaining <= 0:
            idx = (idx + 1) % len(scenario_cycle)
            scenario, duration = scenario_cycle[idx]
            remaining = duration

        if scenario == "NORMAL":
            hr_scale = 1.0
            rr_jitter = 0.01
            noise = 0.02
            wander = 0.05
            imu_amp = 0.15
        elif scenario == "MILD":
            hr_scale = 1.15
            rr_jitter = 0.03
            noise = 0.05
            wander = 0.10
            imu_amp = 0.7
        else:  # HIGH_RISK
            hr_scale = 1.25
            rr_jitter = 0.10
            noise = 0.10
            wander = 0.15
            imu_amp = 1.1

        rr = max(0.35, rr_base / hr_scale + np.random.randn() * rr_jitter)

        if t >= next_r_peak:
            next_r_peak = t + rr

        r_center = next_r_peak - rr
        r_peak = np.exp(-0.5 * ((t - r_center) / 0.012) ** 2) * 1.2
        t_wave = np.exp(-0.5 * ((t - (r_center + 0.22)) / 0.05) ** 2) * 0.35
        baseline = wander * np.sin(2 * np.pi * 0.33 * t)

        ectopic = 0.0
        if scenario == "HIGH_RISK" and np.random.rand() < 0.002:
            ectopic = -0.8 * np.exp(-0.5 * ((t - (r_center + 0.03)) / 0.02) ** 2)

        ecg = baseline + r_peak + t_wave + ectopic + np.random.randn() * noise

        ax = imu_amp * np.sin(2 * np.pi * 1.2 * t) + np.random.randn() * 0.03
        ay = imu_amp * np.sin(2 * np.pi * 1.0 * t + 0.9) + np.random.randn() * 0.03
        az = 1.0 + imu_amp * 0.15 * np.sin(2 * np.pi * 0.8 * t) + np.random.randn() * 0.03

        gx = imu_amp * 10 * np.sin(2 * np.pi * 0.7 * t) + np.random.randn() * 0.3
        gy = imu_amp * 10 * np.sin(2 * np.pi * 0.6 * t + 0.7) + np.random.randn() * 0.3
        gz = imu_amp * 10 * np.sin(2 * np.pi * 0.5 * t + 1.7) + np.random.randn() * 0.3

        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]

        sensor_data["timestamps"].append(timestamp)
        sensor_data["ecg"].append(float(ecg))

        sensor_data["accel_x"].append(float(ax))
        sensor_data["accel_y"].append(float(ay))
        sensor_data["accel_z"].append(float(az))

        sensor_data["gyro_x"].append(float(gx))
        sensor_data["gyro_y"].append(float(gy))
        sensor_data["gyro_z"].append(float(gz))

        t += dt
        time.sleep(dt)


# ============================================================
# PREDICTION LOOP (REAL-TIME)
# ============================================================
def process_predictions():
    global predictions, daily_sensor_outputs, current_date

    while is_reading:
        try:
            time.sleep(2)

            # End-of-day update for PatientAgent
            today = date.today()
            if today != current_date:
                if daily_sensor_outputs:
                    print(f"\n{'='*60}")
                    print(f"End of day detected. Processing {len(daily_sensor_outputs)} sensor outputs...")
                    patient_output_day = patient_agent.daily_update(daily_sensor_outputs)
                    print(f"Patient State: {patient_output_day.get('behavioral_state')}")
                    print(f"Sensitivity Factor: {patient_output_day.get('sensitivity_factor')}")
                    print(f"Confidence: {patient_output_day.get('confidence')}")
                    print(f"{'='*60}\n")
                    daily_sensor_outputs = []
                current_date = today

            # Activity
            if len(sensor_data["accel_x"]) >= ACTIVITY_WINDOW_SIZE:
                window_data = {
                    "accel_x": list(sensor_data["accel_x"])[-ACTIVITY_WINDOW_SIZE:],
                    "accel_y": list(sensor_data["accel_y"])[-ACTIVITY_WINDOW_SIZE:],
                    "accel_z": list(sensor_data["accel_z"])[-ACTIVITY_WINDOW_SIZE:],
                    "gyro_x": list(sensor_data["gyro_x"])[-ACTIVITY_WINDOW_SIZE:],
                    "gyro_y": list(sensor_data["gyro_y"])[-ACTIVITY_WINDOW_SIZE:],
                    "gyro_z": list(sensor_data["gyro_z"])[-ACTIVITY_WINDOW_SIZE:],
                }
                activity_result = predict_activity(window_data)
                if activity_result:
                    predictions["activity"] = activity_result["activity"]
                    predictions["activity_confidence"] = activity_result["confidence"]

            # ECG
            if len(sensor_data["ecg"]) >= ECG_WIN_SAMPLES:
                ecg_signal = np.array(list(sensor_data["ecg"])[-ECG_WIN_SAMPLES:], dtype=float)

                arr_prob, is_arr = predict_arrhythmia(ecg_signal, fs=ECG_FS)
                predictions["arrhythmia_probability"] = float(arr_prob)
                predictions["arrhythmia_detected"] = bool(is_arr)

                ecg_feats = compute_ecg_features(ecg_signal, fs=ECG_FS)
                if ecg_feats:
                    predictions["heart_rate"] = ecg_feats["heart_rate"]
                    predictions["hrv_rmssd"] = ecg_feats["hrv_rmssd"]
                    predictions["hrv_sdnn"] = ecg_feats["hrv_sdnn"]
                    predictions["rhythm_irregular"] = ecg_feats["rhythm_irregular"]
                    predictions["ecg_quality"] = ecg_feats["ecg_quality"]

                # simple physio risk proxy
                if ecg_feats:
                    risk_components = [
                        arr_prob * 0.6,
                        (1.0 if ecg_feats["rhythm_irregular"] else 0.0) * 0.3,
                        (1.0 - ecg_feats["ecg_quality"]) * 0.1,
                    ]
                    predictions["physio_risk"] = float(sum(risk_components))
                    predictions["physio_confidence"] = float(ecg_feats["ecg_quality"])

                sensor_output = {
                    "timestamp": datetime.now().isoformat(),
                    "activity": predictions["activity"],
                    "activity_confidence": predictions["activity_confidence"],
                    "heart_rate": predictions["heart_rate"],
                    "hrv_rmssd": predictions["hrv_rmssd"],
                    "hrv_sdnn": predictions["hrv_sdnn"],
                    "arrhythmia_detected": predictions["arrhythmia_detected"],
                    "arrhythmia_probability": predictions["arrhythmia_probability"],
                    "physio_risk": predictions["physio_risk"],
                    "physio_confidence": predictions["physio_confidence"],
                }
                daily_sensor_outputs.append(sensor_output)

                # Patient agent output (use current stored state; daily_update happens at day end)
                patient_output = {
                    "sensitivity_factor": patient_agent.state.get("sensitivity_factor", 1.0),
                    "behavioral_state": patient_agent.state.get("behavioral_state", "STABLE"),
                    "confidence": patient_agent.state.get("confidence", 0.0),
                }

                # Clinical agent output (use stored state)
                clinical_output = {
                    "clinical_risk": clinical_agent.state.get("clinical_risk", 0.0),
                    "confidence": 1.0,
                }

                decision_output = decision_agent.make_decision(sensor_output, patient_output, clinical_output)

                predictions["global_risk"] = float(decision_output.get("global_risk", 0.0))
                predictions["decision"] = decision_output.get("decision", "NO_ALERT")
                predictions["decision_explanation"] = decision_output.get("explanation", "")

        except Exception as e:
            print(f"Error in prediction processing: {e}")


# ============================================================
# ROUTES
# ============================================================
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/data")
def get_data():
    return jsonify({
        "ecg": list(sensor_data["ecg"])[-500:],
        "accel": {
            "x": list(sensor_data["accel_x"])[-500:],
            "y": list(sensor_data["accel_y"])[-500:],
            "z": list(sensor_data["accel_z"])[-500:],
        },
        "gyro": {
            "x": list(sensor_data["gyro_x"])[-500:],
            "y": list(sensor_data["gyro_y"])[-500:],
            "z": list(sensor_data["gyro_z"])[-500:],
        },
        "timestamps": list(sensor_data["timestamps"])[-500:],
    })


@app.route("/api/latest")
def get_latest():
    if len(sensor_data["timestamps"]) > 0:
        return jsonify({
            "timestamp": sensor_data["timestamps"][-1],
            "ecg": sensor_data["ecg"][-1],
            "accel": {"x": sensor_data["accel_x"][-1], "y": sensor_data["accel_y"][-1], "z": sensor_data["accel_z"][-1]},
            "gyro": {"x": sensor_data["gyro_x"][-1], "y": sensor_data["gyro_y"][-1], "z": sensor_data["gyro_z"][-1]},
        })
    return jsonify({"error": "No data available"}), 404


@app.route("/api/predictions")
def get_predictions():
    return jsonify(predictions)


@app.route("/api/status")
def get_status():
    connected = serial_connection is not None and getattr(serial_connection, "is_open", False)
    return jsonify({
        "simulation_mode": SIMULATION_MODE,
        "connected": bool(connected),
        "data_points": len(sensor_data["timestamps"]),
        "activity_model_loaded": activity_model is not None,
        "ecg_model_loaded": ecg_model is not None,
    })


@app.route("/api/clear")
def clear_data():
    for key in sensor_data:
        sensor_data[key].clear()
    return jsonify({"status": "cleared"})


# ---------------- Patient endpoints ----------------
@app.route("/api/patient/baseline")
def get_patient_baseline():
    return jsonify(patient_agent.get_baseline())


@app.route("/api/patient/state")
def get_patient_state():
    return jsonify({
        "behavioral_state": patient_agent.state.get("behavioral_state", "STABLE"),
        "sensitivity_factor": patient_agent.state.get("sensitivity_factor", 1.0),
        "confidence": patient_agent.state.get("confidence", 0.0),
        "days_seen": patient_agent.state.get("days_seen", 0),
        "recent_trend": patient_agent.state.get("recent_trend", []),
    })


@app.route("/api/patient/trigger_update", methods=["POST"])
def trigger_patient_update():
    global daily_sensor_outputs
    if not daily_sensor_outputs:
        return jsonify({"error": "No sensor outputs to process"}), 400

    patient_output = patient_agent.daily_update(daily_sensor_outputs)
    count = len(daily_sensor_outputs)
    daily_sensor_outputs = []
    return jsonify({"message": f"Processed {count} sensor outputs", "patient_output": patient_output})


@app.route("/api/patient/risk_context")
def get_risk_context():
    current = {
        "heart_rate": predictions.get("heart_rate", 0),
        "hrv_rmssd": predictions.get("hrv_rmssd", 0),
        "arrhythmia_detected": predictions.get("arrhythmia_detected", False),
    }
    return jsonify(patient_agent.get_risk_context(current))


# ---------------- Clinical endpoints ----------------
@app.route("/clinical")
def clinical_page():
    return render_template("clinical.html")


@app.route("/api/clinical/profile")
def get_clinical_profile():
    return jsonify(clinical_agent.get_profile())


@app.route("/api/clinical/update", methods=["POST"])
def update_clinical_profile():
    try:
        clinical_data = request.json
        result = clinical_agent.update_profile(clinical_data)
        return jsonify({"success": True, "clinical_output": result, "profile": clinical_agent.get_profile()})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/api/clinical/risk_factors")
def get_clinical_risk_factors():
    return jsonify(clinical_agent.get_risk_factors())


# ---------------- Decision endpoints ----------------
@app.route("/alerts")
def alerts_page():
    return render_template("alerts.html")


@app.route("/api/decision/current")
def get_current_decision():
    return jsonify(decision_agent.get_current_status())


@app.route("/api/decision/alerts")
def get_alert_history():
    limit = request.args.get("limit", 10, type=int)
    alerts = decision_agent.get_alert_history(limit)
    return jsonify({"alerts": alerts})


@app.route("/api/decision/acknowledge/<int:alert_id>", methods=["POST"])
def acknowledge_alert(alert_id):
    success = decision_agent.acknowledge_alert(alert_id)
    return jsonify({"success": success})


@app.route("/api/decision/trend")
def get_risk_trend():
    trend = decision_agent.get_risk_trend()
    return jsonify({"trend": trend})


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Starting Flask Sensor Data Server with AI Models...")
    print("=" * 60)

    print(f"SIMULATION_MODE: {'✓ ON' if SIMULATION_MODE else '✗ OFF'}")
    print(f"PyTorch Device: {DEVICE}")
    print(f"Activity Model: {'✓ Loaded' if activity_model else '✗ Not found'}")
    print(f"ECG CNN Model: {'✓ Loaded' if ecg_model else '✗ Not found'}")
    print(f"Patient Agent: ✓ Initialized (Days seen: {patient_agent.state.get('days_seen', 0)})")

    try:
        profile = clinical_agent.get_profile()
        if profile.get("profile_complete"):
            print(f"Clinical Agent: ✓ Profile complete (Risk: {profile.get('clinical_risk', 0) * 100:.0f}%)")
        else:
            print("Clinical Agent: ⚠ Profile incomplete - configure at /clinical")
    except Exception as e:
        print(f"Clinical Agent: ⚠ Profile unreadable ({e}) - configure at /clinical")

    status = decision_agent.get_current_status()
    print(f"Decision Agent: ✓ Initialized (Alerts: {status.get('total_alerts', 0)}, Monitors: {status.get('total_monitors', 0)})")

    # Start threads
    is_reading = True

    prediction_thread = threading.Thread(target=process_predictions, daemon=True)
    prediction_thread.start()

    if SIMULATION_MODE:
        print("✓ Using simulated sensor stream (no Arduino).")
        sim_thread = threading.Thread(target=simulate_sensor_data, daemon=True)
        sim_thread.start()
    else:
        print("Serial mode is disabled in this clean build. Set SIMULATION_MODE=1.")

    print("=" * 60)
    print("Web Interfaces:")
    print("  Main Dashboard: http://localhost:5000")
    print("  Clinical Profile: http://localhost:5000/clinical")
    print("  Alert History: http://localhost:5000/alerts")
    print("=" * 60)

    app.run(debug=True, threaded=True, use_reloader=False)
