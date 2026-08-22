from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import sqlite3
from datetime import datetime

app = FastAPI(title="Predictive Maintenance Data Receiver")

# --- DATABASE SETUP ---
# Connect to SQLite. check_same_thread=False allows FastAPI's asynchronous nature to work with SQLite.
conn = sqlite3.connect("sensor_data.db", check_same_thread=False)
cursor = conn.cursor()

# Create the table if it doesn't already exist
cursor.execute('''
    CREATE TABLE IF NOT EXISTS vibrations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        device_id TEXT,
        ax REAL,
        ay REAL,
        az REAL,
        temp REAL
    )
''')
conn.commit()
print("[+] Database initialized successfully.")

# --- DATA SCHEMA ---
class SensorPayload(BaseModel):
    device_id: str
    ax: float
    ay: float
    az: float
    temp: float

# --- ENDPOINTS ---

@app.get("/")
def read_root():
    return {"status": "online", "message": "Predictive Maintenance API is running"}

# 1. The Receiver (ESP32 sends data here)
@app.post("/data")
async def receive_vibration_data(data: SensorPayload):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    
    # Write the incoming data into the database
    cursor.execute('''
        INSERT INTO vibrations (timestamp, device_id, ax, ay, az, temp)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (timestamp, data.device_id, data.ax, data.ay, data.az, data.temp))
    conn.commit()
    
    # Still print to terminal so you can see it working
    print(f"[{timestamp}] Saved to DB -> Ax: {data.ax:6.2f} | Ay: {data.ay:6.2f} | Az: {data.az:6.2f}")
    
    return {"status": "saved"}

# 2. The Provider (Frontend will fetch data from here)
@app.get("/data")
def get_historical_data(limit: int = 100):
    # Fetch the most recent rows, up to the 'limit' requested
    cursor.execute('SELECT * FROM vibrations ORDER BY timestamp DESC LIMIT ?', (limit,))
    rows = cursor.fetchall()
    
    # Package the raw database rows back into a clean list of dictionaries
    results = []
    for row in rows:
        results.append({
            "id": row[0],
            "timestamp": row[1],
            "device_id": row[2],
            "ax": row[3],
            "ay": row[4],
            "az": row[5],
            "temp": row[6]
        })
        
    return {"data": results}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
