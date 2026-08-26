from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import sqlite3
from datetime import datetime
import pickle
import numpy as np
import pandas as pd
from collections import deque

app = FastAPI(title="Predictive Maintenance API with Advanced AI")

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

try:
    with open('motor_ai_model.pkl', 'rb') as f:
        ai_model = pickle.load(f)
    print("[+] AI Model loaded successfully.")
except FileNotFoundError:
    print("[-] WARNING: AI Model not found.")
    ai_model = None

class SensorPayload(BaseModel):
    device_id: str
    ax: float
    ay: float
    az: float
    temp: float

# The Live Buffers
buffer_x = deque(maxlen=10)
buffer_y = deque(maxlen=10)
buffer_z = deque(maxlen=10)

@app.post("/data")
async def receive_vibration_data(data: SensorPayload):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    status = "Normal"
    
    buffer_x.append(data.ax)
    buffer_y.append(data.ay)
    buffer_z.append(data.az)
    
    # Run prediction only when buffer is full
    if ai_model and len(buffer_x) == 10:
        
        # Calculate the 6D fingerprint live
        live_features = pd.DataFrame([{
            'var_x': np.var(buffer_x, ddof=1),
            'var_y': np.var(buffer_y, ddof=1),
            'var_z': np.var(buffer_z, ddof=1),
            'ptp_x': max(buffer_x) - min(buffer_x),
            'ptp_y': max(buffer_y) - min(buffer_y),
            'ptp_z': max(buffer_z) - min(buffer_z)
        }])
        
        prediction = ai_model.predict(live_features)
        
        if prediction[0] == -1:
            status = "ANOMALY DETECTED"
    
    cursor.execute('''
        INSERT INTO vibrations (timestamp, device_id, ax, ay, az, temp, status)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (timestamp, data.device_id, data.ax, data.ay, data.az, data.temp, status))
    conn.commit()
    
    # Only print status to keep the terminal readable
    if len(buffer_x) == 10:
        print(f"[{timestamp}] Status: {status} | PTP_X: {max(buffer_x) - min(buffer_x):.2f}")
    
    return {"status": "saved", "machine_status": status}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
