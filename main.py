from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import sqlite3
from datetime import datetime
import pickle
import numpy as np
import pandas as pd

app = FastAPI(title="Predictive Maintenance API with AI")

# --- DATABASE SETUP ---
conn = sqlite3.connect("sensor_data.db", check_same_thread=False)
cursor = conn.cursor()
cursor.execute('''
    CREATE TABLE IF NOT EXISTS vibrations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        device_id TEXT,
        ax REAL, ay REAL, az REAL, temp REAL,
        status TEXT
    )
''')
conn.commit()
print("[+] Database initialized.")

# --- LOAD AI MODEL ---
try:
    with open('motor_ai_model.pkl', 'rb') as f:
        ai_model = pickle.load(f)
    print("[+] AI Model loaded successfully.")
except FileNotFoundError:
    print("[-] WARNING: AI Model not found. Run train_model.py first.")
    ai_model = None

class SensorPayload(BaseModel):
    device_id: str
    ax: float
    ay: float
    az: float
    temp: float

@app.post("/data")
async def receive_vibration_data(data: SensorPayload):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    
    status = "Normal"
    
    # Run AI Prediction if model is loaded
    if ai_model:
        # Extract features from the single incoming data point
        incoming_features = pd.DataFrame([{
            'rms_x': data.ax ** 2,
            'rms_y': data.ay ** 2,
            'rms_z': data.az ** 2,
            'magnitude': np.sqrt(data.ax**2 + data.ay**2 + data.az**2)
        }])
        
        # The AI returns 1 for Normal, -1 for Anomaly
        prediction = ai_model.predict(incoming_features)
        
        if prediction[0] == -1:
            status = "ANOMALY DETECTED"
    
    cursor.execute('''
        INSERT INTO vibrations (timestamp, device_id, ax, ay, az, temp, status)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (timestamp, data.device_id, data.ax, data.ay, data.az, data.temp, status))
    conn.commit()
    
    print(f"[{timestamp}] Status: {status} | Mag: {np.sqrt(data.ax**2 + data.ay**2 + data.az**2):.2f}")
    
    return {"status": "saved", "machine_status": status}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
