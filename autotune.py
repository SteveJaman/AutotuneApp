import librosa
from pathlib import Path
import psola
import numpy as np
import scipy.signal as sig
import sounddevice as sd
import time
import sys
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# --- Pitch correction functions (same as original) ---

def correct(f0, selected_key_degrees):
    """
    Corrects a single fundamental frequency (f0) to the nearest degree in the selected key.
    """
    if np.isnan(f0):
        return np.nan
    midi_note = librosa.hz_to_midi(f0)
    degree = midi_note % 12  # Get the degree within an octave (0-11)
    
    # Find the closest degree in the selected key
    closest_degree_id = np.argmin(np.abs(selected_key_degrees - degree))
    degree_difference = degree - selected_key_degrees[closest_degree_id]
    
    midi_note -= degree_difference # Shift the MIDI note to align with the closest key degree
    return librosa.midi_to_hz(midi_note)

def correct_pitch(f0, selected_key_degrees, progress_callback=None):
    """
    Applies pitch correction to an array of fundamental frequencies (f0).
    Includes a progress callback for GUI updates.
    """
    corrected_f0 = np.zeros_like(f0)
    total = f0.shape[0]

    for i in range(total):
        corrected_f0[i] = correct(f0[i], selected_key_degrees)
        # Update progress periodically to avoid slowing down the loop too much
        if progress_callback and i % 50 == 0: # Update every 50 frames
            progress_callback(i / total * 100)
    
    # Ensure final progress update
    if progress_callback:
        progress_callback(100)

    # Apply median filtering for smoothing, handling NaNs
    smoothed_corrected_f0 = sig.medfilt(corrected_f0, kernel_size=11)
    # Fill back NaNs where median filter might have introduced them or original NaNs
    smoothed_corrected_f0[np.isnan(smoothed_corrected_f0)] = f0[np.isnan(smoothed_corrected_f0)]
    return smoothed_corrected_f0

def autotune(y, sr, selected_key, progress_callback=None):
    """
    Performs the full autotune process: pitch detection, correction, and vocoding.
    """
    frame_length = 2048
    hop_length = frame_length // 4
    fmin = librosa.note_to_hz('C2') # Minimum frequency for pitch detection
    fmax = librosa.note_to_hz('C7') # Maximum frequency for pitch detection

    try:
        # Convert the selected key (e.g., 'C:maj') to an array of MIDI degrees
        selected_key_degrees = librosa.key_to_degrees(selected_key)
        # Add an octave higher for wrapping around (e.g., C major includes C, D, E, F, G, A, B, C+12)
        selected_key_degrees = np.concatenate((selected_key_degrees, [selected_key_degrees[0] + 12]))
    except Exception as e:
        raise ValueError(f"Invalid key '{selected_key}'. Please use format like 'C:maj' or 'A:min'.")

    # Perform pitch detection using pYIN
    # f0: fundamental frequency contour
    # voiced_flag: boolean array indicating voiced segments
    # voiced_prob: probability of being voiced
    f0, voiced_flag, voiced_prob = librosa.pyin(
        y, frame_length=frame_length, hop_length=hop_length,
        sr=sr, fmin=fmin, fmax=fmax
    )

    # Correct the detected pitches
    corrected_f0 = correct_pitch(f0, selected_key_degrees, progress_callback)
    
    # Use PSOLA (Pitch Synchronous Overlap and Add) to vocode the audio
    # This resynthesizes the audio with the corrected pitch contour
    return psola.vocode(y, sample_rate=int(sr), target_pitch=corrected_f0, fmin=fmin, fmax=fmax)

# --- GUI ---

class AutotuneApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Simple Autotune GUI with Spectrum")
        self.filepath = None

        # Configure grid layout for better organization
        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(3, weight=1) # Row for plots

        # Input Frame
        input_frame = ttk.LabelFrame(root, text="Audio Input and Key Selection")
        input_frame.grid(row=0, column=0, columnspan=2, padx=10, pady=10, sticky="ew")
        input_frame.columnconfigure(0, weight=1)
        input_frame.columnconfigure(1, weight=1)

        self.file_btn = tk.Button(input_frame, text="Select Audio File", command=self.select_file)
        self.file_btn.grid(row=0, column=0, padx=5, pady=5, sticky="ew")

        self.key_label = tk.Label(input_frame, text="Enter autotune key (e.g. C:maj, A:min):")
        self.key_label.grid(row=1, column=0, padx=5, pady=5, sticky="w")

        self.key_entry = tk.Entry(input_frame)
        self.key_entry.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        self.key_entry.insert(0, "C:maj") # Default key

        self.run_btn = tk.Button(input_frame, text="Run Autotune", command=self.run_autotune)
        self.run_btn.grid(row=2, column=0, columnspan=2, padx=5, pady=10, sticky="ew")

        self.progress = ttk.Progressbar(input_frame, length=300, mode="determinate")
        self.progress.grid(row=3, column=0, columnspan=2, padx=5, pady=5, sticky="ew")

        # Plotting Setup
        # Frame for original audio spectrum
        original_plot_frame = ttk.LabelFrame(root, text="Original Audio Spectrum")
        original_plot_frame.grid(row=3, column=0, padx=10, pady=10, sticky="nsew")
        original_plot_frame.columnconfigure(0, weight=1)
        original_plot_frame.rowconfigure(0, weight=1)

        self.figure_original = plt.Figure(figsize=(5, 4), dpi=100)
        self.ax_original = self.figure_original.add_subplot(111)
        self.canvas_original = FigureCanvasTkAgg(self.figure_original, master=original_plot_frame)
        self.canvas_original_widget = self.canvas_original.get_tk_widget()
        self.canvas_original_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        
        # Frame for autotuned audio spectrum
        autotuned_plot_frame = ttk.LabelFrame(root, text="Autotuned Audio Spectrum")
        autotuned_plot_frame.grid(row=3, column=1, padx=10, pady=10, sticky="nsew")
        autotuned_plot_frame.columnconfigure(0, weight=1)
        autotuned_plot_frame.rowconfigure(0, weight=1)

        self.figure_autotuned = plt.Figure(figsize=(5, 4), dpi=100)
        self.ax_autotuned = self.figure_autotuned.add_subplot(111)
        self.canvas_autotuned = FigureCanvasTkAgg(self.figure_autotuned, master=autotuned_plot_frame)
        self.canvas_autotuned_widget = self.canvas_autotuned.get_tk_widget()
        self.canvas_autotuned_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        # Initial plot setup (empty plots)
        self.ax_original.set_title("Original Audio Spectrum")
        self.ax_original.set_xlabel("Frequency (Hz)")
        self.ax_original.set_ylabel("Magnitude")
        self.ax_original.set_xscale('log')
        self.ax_original.set_xlim([20, 20000])
        self.ax_original.grid(True)
        self.canvas_original.draw()

        self.ax_autotuned.set_title("Autotuned Audio Spectrum")
        self.ax_autotuned.set_xlabel("Frequency (Hz)")
        self.ax_autotuned.set_ylabel("Magnitude")
        self.ax_autotuned.set_xscale('log')
        self.ax_autotuned.set_xlim([20, 20000])
        self.ax_autotuned.grid(True)
        self.canvas_autotuned.draw()

    def select_file(self):
        """
        Opens a file dialog for the user to select an audio file.
        """
        file = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav"), ("All files", "*.*")])
        if file:
            self.filepath = file
            messagebox.showinfo("File Selected", f"Selected file:\n{Path(file).name}")
        else:
            self.filepath = None # Clear filepath if selection is cancelled

    def plot_spectrum(self, ax, audio_data, sample_rate, title):
        """
        Calculates and plots the magnitude spectrum of the given audio data.
        """
        N = len(audio_data)
        # Compute the Fast Fourier Transform
        yf = np.fft.fft(audio_data)
        # Compute the corresponding frequencies
        xf = np.fft.fftfreq(N, 1 / sample_rate)[:N//2] # Only positive frequencies

        ax.clear() # Clear previous plot
        ax.plot(xf, 2.0/N * np.abs(yf[0:N//2])) # Plot magnitude spectrum
        ax.set_title(title)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Magnitude")
        ax.set_xscale('log') # Use logarithmic scale for frequency
        ax.set_xlim([20, sample_rate / 2]) # Limit x-axis to relevant frequencies (20Hz to Nyquist)
        ax.grid(True)
        self.figure_original.tight_layout()
        self.figure_autotuned.tight_layout()


    def run_autotune(self):
        """
        Executes the autotune process, loads audio, applies autotune,
        plays the result, and plots the spectra.
        """
        key = self.key_entry.get().strip()
        if not key:
            messagebox.showerror("Input Error", "Please enter a musical key (e.g., C:maj).")
            return

        if not self.filepath:
            # Fallback to a default file if no file is selected
            default_path = Path(__file__).parent / "monitoring.wav"
            if default_path.exists():
                self.filepath = str(default_path)
                messagebox.showinfo("Fallback", "No file selected. Using default: monitoring.wav")
            else:
                messagebox.showerror("File Error", "No file selected and 'monitoring.wav' not found in the script directory.")
                return

        try:
            # Load audio file
            y, sr = librosa.load(self.filepath, sr=None, mono=False)
            if y.ndim > 1:
                y = y[0, :]  # Use left channel only for processing
        except Exception as e:
            messagebox.showerror("Load Error", f"Failed to load audio: {e}")
            return

        def update_progress(p):
            """Callback function to update the progress bar."""
            self.progress["value"] = p
            self.root.update_idletasks() # Force GUI update

        try:
            self.run_btn.config(state="disabled") # Disable button during processing
            self.progress["value"] = 0 # Reset progress bar

            start_time = time.time()
            # Run the autotune algorithm
            corrected_y = autotune(y, sr, key, update_progress)
            elapsed = time.time() - start_time

            # Display success message and play audio
            messagebox.showinfo("Success", f"Autotune completed in {elapsed:.2f} seconds.\nPlaying result...")
            sd.play(corrected_y, samplerate=sr)
            sd.wait() # Wait for playback to finish

            # Plot spectra
            self.plot_spectrum(self.ax_original, y, sr, "Original Audio Spectrum")
            self.canvas_original.draw()

            self.plot_spectrum(self.ax_autotuned, corrected_y, sr, "Autotuned Audio Spectrum")
            self.canvas_autotuned.draw()

        except Exception as e:
            messagebox.showerror("Processing Error", str(e))
        finally:
            self.run_btn.config(state="normal") # Re-enable button
            self.progress["value"] = 0 # Reset progress bar

if __name__ == "__main__":
    root = tk.Tk()
    app = AutotuneApp(root)
    root.mainloop()