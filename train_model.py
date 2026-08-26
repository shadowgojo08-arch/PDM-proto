import sqlite3
import pandas as pd
import pickle
from sklearn.ensemble import IsolationForest

print("[-] Connecting to database...")
conn = sqlite3.connect("sensor_data.db")
query = "SELECT ax, ay, az FROM vibrations ORDER BY id ASC"
df = pd.read_sql_query(query, conn)
conn.close()

print("[-] Extracting advanced 6D features (Variance + Amplitude)...")
features = pd.DataFrame()

# 1. Variance (Total Energy)
features['var_x'] = df['ax'].rolling(window=10).var()
features['var_y'] = df['ay'].rolling(window=10).var()
features['var_z'] = df['az'].rolling(window=10).var()

# 2. Peak-to-Peak (Impact Spikes)
features['ptp_x'] = df['ax'].rolling(window=10).max() - df['ax'].rolling(window=10).min()
features['ptp_y'] = df['ay'].rolling(window=10).max() - df['ay'].rolling(window=10).min()
features['ptp_z'] = df['az'].rolling(window=10).max() - df['az'].rolling(window=10).min()

# Clean up empty rows
features = features.dropna()

print(f"[-] Total rows before cleaning: {len(features)}")

# Strip out data where the motor was off (var_x < 20)
features = features[features['var_x'] > 20]

print(f"[-] Total rows after removing 'Off' states: {len(features)}")

if len(features) < 50:
    print("[-] ERROR: Not enough 'Motor On' data! Run the motor longer.")
    exit()

print("[-] Training advanced Isolation Forest...")
# Contamination at 0.05 creates a very strict boundary
model = IsolationForest(contamination=0.05, random_state=42)
model.fit(features)

with open('motor_ai_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("[+] 6D AI Brain upgraded and saved.")
