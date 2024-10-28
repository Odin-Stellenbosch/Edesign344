import numpy as np
import sounddevice as sd
import tkinter as tk
from tkinter import filedialog, messagebox
from scipy.fft import fft, fftfreq
from scipy.signal import windows
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd

# Global variables
calibration_factor = None
sampling_rate = 48000  # Fixed microphone sample rate
calibrated = False  # To prevent output before calibration
stream = None  # For stopping and restarting audio stream

# Audio callback
def audio_callback(indata, frames, time, status):
    global calibration_factor, calibrated
    if not calibrated or calibration_factor is None:
        return  # No output until calibration is done
    
    # Scale the input using the calibration factor
    scaled_input = indata[:, 0] / calibration_factor
    plot_time_domain(scaled_input)
    plot_fft(scaled_input)
    calculate_rms_power(scaled_input)

# Load Calibration File
def load_calibration_file():
    global calibration_factor, calibrated
    file_path = filedialog.askopenfilename()
    if file_path:
        data = pd.read_csv(file_path, skiprows=1, header=None)
        sample_values = data.iloc[:, 1].values
        
        # The calibration signal is 1V peak-to-peak (Vpp = 1V) and frequency = 1kHz
        max_sample = np.max(np.abs(sample_values))
        calibration_factor = max_sample  # Use the peak value to calibrate
        calibrated = True
        messagebox.showinfo("Calibration", "Calibrated successfully!")

# Plot Time Domain
def plot_time_domain(signal):
    plt.subplot(2, 1, 1)
    plt.cla()
    plt.plot(signal)
    plt.title('Time Domain Signal')
    plt.grid(True)
    plt.draw()

# Plot FFT
def plot_fft(signal):
    num_samples = len(signal)
    fft_values = fft(signal)
    fft_magnitude = 2.0 / num_samples * np.abs(fft_values[:num_samples // 2])
    frequencies = fftfreq(num_samples, 1.0 / sampling_rate)[:num_samples // 2]
    
    plt.subplot(2, 1, 2)
    plt.cla()
    plt.plot(frequencies, fft_magnitude)
    plt.title('FFT')
    plt.grid(True)
    plt.draw()

# Calculate RMS Power
def calculate_rms_power(signal):
    rms_value = np.sqrt(np.mean(signal ** 2))
    rms_power = rms_value ** 2 / 8  # Assuming an 8 ohm load
    rms_power_label.config(text=f"RMS Power: {rms_power:.4f} W")

# Start live audio
def start_audio():
    global stream
    if stream is None or not stream.active:
        stream = sd.InputStream(callback=audio_callback, channels=1, samplerate=sampling_rate)
        stream.start()

# Stop live audio
def stop_audio():
    global stream
    if stream and stream.active:
        stream.stop()

# Close the application
def close_application():
    global stream
    if stream and stream.active:
        stream.stop()
    plt.close('all')
    root.quit()
    root.destroy()

# GUI setup
root = tk.Tk()
root.title("Live Audio Analyzer with Calibration")

calibrate_button = tk.Button(root, text="Load Calibration File", command=load_calibration_file)
calibrate_button.grid(row=0, column=0)

start_button = tk.Button(root, text="Start Live Audio", command=start_audio)
start_button.grid(row=1, column=0)

stop_button = tk.Button(root, text="Stop Live Audio", command=stop_audio)
stop_button.grid(row=2, column=0)

rms_power_label = tk.Label(root, text="RMS Power: N/A")
rms_power_label.grid(row=3, column=0)

close_button = tk.Button(root, text="Close Application", command=close_application)
close_button.grid(row=4, column=0)

fig, ax = plt.subplots(2, 1)
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().grid(row=0, column=1, rowspan=10)

root.protocol("WM_DELETE_WINDOW", close_application)
root.mainloop()
