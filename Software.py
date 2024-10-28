import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import windows
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Global variables
calibration_factor = None
sampling_rate = None
calibration_amplitude = 1.0
calibration_frequency = 1000.0

def load_calibration_file():
    global calibration_factor, sampling_rate
    file_path = filedialog.askopenfilename()
    if file_path:
        with open(file_path, 'r') as f:
            lines = f.readlines()

        # Extract sampling rate from the first line
        first_line = lines[0].strip().split()
        sampling_rate = int(first_line[1])

        # Read sample values from subsequent lines
        data = pd.read_csv(file_path, skiprows=1, header=None)
        sample_values = data.iloc[:, 1].values

        # Get the max sample value as the calibration factor
        calibration_factor = np.max(np.abs(sample_values)) / calibration_amplitude

        messagebox.showinfo("Calibration", "Calibrated successfully!")

def load_waveform_file():
    global calibration_factor, sampling_rate
    if calibration_factor is None:
        messagebox.showerror("Error", "Please calibrate before loading a waveform file.")
        return

    file_path = filedialog.askopenfilename()
    if file_path:
        with open(file_path, 'r') as f:
            lines = f.readlines()

        # Extract sampling rate from the first line
        first_line = lines[0].strip().split()
        waveform_sampling_rate = int(first_line[1])

        data = pd.read_csv(file_path, skiprows=1, header=None)
        sample_numbers = data.iloc[:, 0].values
        sample_values = data.iloc[:, 1].values

        scaled_values = sample_values / calibration_factor

        plot_time_domain(sample_numbers, scaled_values, waveform_sampling_rate)
        plot_fft(scaled_values, waveform_sampling_rate)

        calculate_rms_and_thd(scaled_values, waveform_sampling_rate)

def plot_time_domain(sample_numbers, scaled_values, waveform_sampling_rate):
    # Calculate time values
    time = sample_numbers / waveform_sampling_rate

    # Perform FFT to determine the frequency of the input signal
    num_samples = len(scaled_values)
    fft_values = fft(scaled_values)
    fft_magnitude = 2.0 / num_samples * np.abs(fft_values[:num_samples // 2])
    frequencies = fftfreq(num_samples, 1.0 / waveform_sampling_rate)[:num_samples // 2]

    # Find the fundamental frequency (highest peak)
    fundamental_index = np.argmax(fft_magnitude)
    fundamental_frequency = frequencies[fundamental_index]

    # Calculate the number of samples that correspond to 5 cycles of the fundamental frequency
    num_samples_for_5_cycles = int(waveform_sampling_rate / fundamental_frequency * 5)

    # Adjust the number of samples to display exactly 5 cycles
    num_samples_to_plot = min(num_samples_for_5_cycles, len(scaled_values))

    # Plot the time-domain signal for the first 5 cycles
    plt.subplot(2, 1, 1)
    plt.cla()
    plt.plot(time[:num_samples_to_plot], scaled_values[:num_samples_to_plot])
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude [V]')
    plt.title('Time Domain Signal (5 Cycles)')
    plt.grid(True)
    plt.draw()


def plot_fft(scaled_values, waveform_sampling_rate):
    global calibration_frequency
    global peak_voltage_fft
    num_samples = len(scaled_values)
    
    # Use the correct window to reduce the average window size
    # hann_window = windows.hann(num_samples)
    # windowed_signal = scaled_values * hann_window

    fft_values = fft(scaled_values)
    fft_magnitude = 2.0 / num_samples * np.abs(fft_values[:num_samples // 2])
    frequencies = fftfreq(num_samples, 1.0 / waveform_sampling_rate)[:num_samples // 2]

    # Calculate peak voltage from FFT (fundamental harmonic)
    fundamental_index = np.argmax(fft_magnitude)
    peak_voltage_fft = fft_magnitude[fundamental_index]

    plt.subplot(2, 1, 2)
    plt.cla()
    plt.plot(frequencies, fft_magnitude)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Magnitude')
    plt.xscale('log')
    plt.title('FFT')
    plt.grid(True)
    plt.draw()

    # Update Peak Voltage and Fundamental Frequency
    peak_voltage_label.config(text=f"FFT Peak Voltage: {peak_voltage_fft:.2f} V")
    fundamental_freq_label.config(text=f"Fundamental Frequency: {frequencies[fundamental_index]:.2f} Hz")

def calculate_rms_and_thd(scaled_values, waveform_sampling_rate):
    # Calculate RMS
    rms_value = np.sqrt(np.mean(scaled_values**2))

    # Compute FFT
    num_samples = len(scaled_values)
    fft_values = fft(scaled_values)
    fft_magnitude = 2.0 / num_samples * np.abs(fft_values[:num_samples // 2])
    frequencies = fftfreq(num_samples, 1.0 / waveform_sampling_rate)[:num_samples // 2]

    # Find the fundamental frequency
    fundamental_idx = np.argmax(fft_magnitude)
    fundamental_freq = frequencies[fundamental_idx]
    fundamental = fft_magnitude[fundamental_idx]

    # Limit THD calculation to first 5 harmonics
    harmonics = []
    for i in range(2, 6):  # Harmonics 2nd to 5th
        harmonic_freq = fundamental_freq * i
        harmonic_idx = np.argmin(np.abs(frequencies - harmonic_freq))
        harmonic_magnitude = fft_magnitude[harmonic_idx]

        # Apply noise threshold (optional)
        if harmonic_magnitude > fundamental * 0.01:  # 1% of fundamental
            harmonics.append(harmonic_magnitude)

    # Calculate THD using the sum of squares of harmonics
    if harmonics:
        thd_value = np.sqrt(np.sum(np.square(harmonics))) / fundamental * 100
    else:
        thd_value = 0  # If no significant harmonics, THD is zero

    # Display RMS and THD
    rms_label.config(text=f"RMS Voltage: {rms_value:.2f} V")
    thd_label.config(text=f"THD: {thd_value:.2f} %")
    fundamental_freq_label.config(text=f"Fundamental Frequency: {fundamental_freq:.2f} Hz")

    rms_power = peak_voltage_fft**2 / (8*2)
    rms_power_label.config(text=f"RMS Power: {rms_power:.2f} W")


def update_graphs():
    load_waveform_file()

def update_calibration_values():
    global calibration_amplitude, calibration_frequency

    try:
        # Read values from the text boxes
        calibration_amplitude = float(calibration_amplitude_entry.get())
        calibration_frequency = float(calibration_frequency_entry.get())
        messagebox.showinfo("Update", f"Calibration values updated!\nAmplitude: {calibration_amplitude} V\nFrequency: {calibration_frequency} Hz")
    except ValueError:
        messagebox.showerror("Error", "Please enter valid numbers for calibration amplitude and frequency.")


# GUI setup
root = tk.Tk()
root.title("Waveform Analyzer")

calibrate_button = tk.Button(root, text="Load Calibration File", command=load_calibration_file)
calibrate_button.grid(row=0, column=0)

calibration_amplitude_label = tk.Label(root, text="Calibration Amplitude (V)")
calibration_amplitude_label.grid(row=1, column=0)
calibration_amplitude_entry = tk.Entry(root)
calibration_amplitude_entry.grid(row=1, column=1)
calibration_amplitude_entry.insert(0, "1.0")

calibration_frequency_label = tk.Label(root, text="Calibration Frequency (Hz)")
calibration_frequency_label.grid(row=2, column=0)
calibration_frequency_entry = tk.Entry(root)
calibration_frequency_entry.grid(row=2, column=1)
calibration_frequency_entry.insert(0, "1000")

load_waveform_button = tk.Button(root, text="Load Waveform File", command=load_waveform_file)
load_waveform_button.grid(row=3, column=0)

rms_label = tk.Label(root, text="RMS Voltage: N/A")
rms_label.grid(row=4, column=0)

thd_label = tk.Label(root, text="THD: N/A")
thd_label.grid(row=5, column=0)

rms_power_label = tk.Label(root, text="RMS Power: N/A")
rms_power_label.grid(row=6, column=0)

peak_voltage_label = tk.Label(root, text="FFT Peak Voltage: N/A")
peak_voltage_label.grid(row=7, column=0)

fundamental_freq_label = tk.Label(root, text="Fundamental Frequency: N/A")
fundamental_freq_label.grid(row=8, column=0)

update_button = tk.Button(root, text="Update", command=update_calibration_values)
update_button.grid(row=9, column=0)

student_no_label = tk.Label(root, text = "Odin Mostert 260786727")
student_no_label.grid(row=10, column=0)

fig, ax = plt.subplots(2, 1)
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().grid(row=0, column=2, rowspan=10)

root.mainloop()
