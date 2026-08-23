import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

# 1. Connect to the database and pull the last 1000 readings (approx 10 seconds of data at 100Hz)
conn = sqlite3.connect("sensor_data.db")
query = "SELECT * FROM vibrations ORDER BY id DESC LIMIT 1000"
df = pd.read_sql_query(query, conn)
conn.close()

# The database pulls newest first, so we reverse it to chronological order
df = df.iloc[::-1].reset_index(drop=True)

# 2. Clean the Data (Remove Gravity)
# We focus on the Z-axis (or whichever axis faces outward from your motor)
# We subtract the mean (average) to remove the 9.81 m/s^2 static pull of gravity
raw_signal = df['az'].values
clean_signal = raw_signal - np.mean(raw_signal)

# 3. The Math (Fast Fourier Transform)
N = len(clean_signal)             # Number of data points
T = 0.01                          # Sampling interval (10ms = 0.01 seconds)

# Calculate the FFT
fft_values = fft(clean_signal)
# Calculate the corresponding frequencies
frequencies = fftfreq(N, T)[:N//2] 
# Get the true amplitude (magnitude) of the FFT
amplitudes = 2.0/N * np.abs(fft_values[0:N//2])

# 4. Visualization
plt.figure(figsize=(12, 6))

# Top Graph: The Time Domain (What you see with your eyes)
plt.subplot(2, 1, 1)
plt.plot(clean_signal, color='blue')
plt.title("Time Domain: Raw Motor Vibration (Gravity Removed)")
plt.ylabel("Acceleration (m/s²)")
plt.xlabel("Time (Samples)")
plt.grid(True)

# Bottom Graph: The Frequency Domain (What the AI will see)
plt.subplot(2, 1, 2)
plt.plot(frequencies, amplitudes, color='red')
plt.title("Frequency Domain: FFT Spectrum")
plt.ylabel("Amplitude")
plt.xlabel("Frequency (Hz)")
plt.grid(True)
# Limit x-axis to our max detectable frequency (50 Hz)
plt.xlim(0, 50) 

plt.tight_layout()
plt.show()