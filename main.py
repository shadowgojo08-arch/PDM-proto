from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from datetime import datetime

app = FastAPI(title="Predictive Maintenance Data Receiver")

# Define expected data schema from ESP32
class SensorPayload(BaseModel):
    device_id: str
    ax: float
    ay: float
    az: float
    temp: float

@app.get("/")
def read_root():
    return {"status": "online", "message": "Predictive Maintenance API is running"}

@app.post("/data")
async def receive_vibration_data(data: SensorPayload):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    
    # Simple threshold warning for testing
    total_accel = (data.ax**2 + data.ay**2 + data.az**2) ** 0.5
    
    print(f"[{timestamp}] Device: {data.device_id} | Ax: {data.ax:6.2f} m/s² | Ay: {data.ay:6.2f} m/s² | Az: {data.az:6.2f} m/s² | Total: {total_accel:6.2f} | Temp: {data.temp:.1f}°C")
    
    return {
        "status": "success",
        "device_id": data.device_id,
        "received_at": timestamp
    }

if __name__ == "__main__":
    # Host 0.0.0.0 binds to all network interfaces so ESP32 can connect
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)