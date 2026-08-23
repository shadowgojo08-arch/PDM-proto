import sqlite3
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import IsolationForest

print("[-] Connecting to database...")
conn = sqlite3.connect("sensor_data.db")
query = "SELECT ax, ay, az FROM vibrations ORDER BY id ASC"
df = pd.read_sql_query(query, conn)
conn.close()

print(f"[-] Loaded {len(df)} readings from the database.")

# 1. Feature Engineering (Creating meaningful numbers for the AI)
# We calculate the RMS (Root Mean Square) for each axis. This represents the total "energy" of the vibration.
# We also calculate the total vector magnitude (combining X, Y, and Z).

print("[-] Extracting features...")
features = pd.DataFrame()
features['rms_x'] = df['ax'] ** 2
features['rms_y'] = df['ay'] ** 2
features['rms_z'] = df['az'] ** 2
features['magnitude'] = np.sqrt(df['ax']**2 + df['ay']**2 + df['az']**2)

# 2. Initialize the AI Model
# contamination=0.01 means we expect about 1% of the training data might be random noise.
print("[-] Initializing Isolation Forest model...")
model = IsolationForest(contamination=0.01, random_state=42)

# 3. Train the Model
print("[-] Training AI on baseline motor data...")
model.fit(features)

# 4. Save the Model (Pickling)
# We save the trained model to a file so our FastAPI server can load it and use it live.
with open('motor_ai_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("[+] Model trained and saved successfully as 'motor_ai_model.pkl'.")