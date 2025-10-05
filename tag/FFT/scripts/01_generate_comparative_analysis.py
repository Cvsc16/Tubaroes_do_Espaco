
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import spectrogram, welch
import os

# --- Comparative Analysis Script ---

def generate_comparative_plots():
    # --- 1. SETUP PATHS ---
    try:
        fft_root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        songs_dir = os.path.join(fft_root_dir, 'songs')
        original_audio_path = os.path.join(songs_dir, 'final track.wav')
        signature_audio_path = os.path.join(songs_dir, 'assinatura.wav')
        output_plot_path = os.path.join(fft_root_dir, 'comparative_analysis.png')
        print("Starting comparative analysis...")
    except Exception as e:
        print(f"Error setting up paths: {e}")
        return

    # --- 2. LOAD AUDIO ---
    try:
        fs_main, data_main = wavfile.read(original_audio_path)
        if data_main.ndim > 1: data_main = data_main.mean(axis=1)
        time_main = np.arange(len(data_main)) / fs_main

        fs_sig, data_sig = wavfile.read(signature_audio_path)
        if data_sig.ndim > 1: data_sig = data_sig.mean(axis=1)
        time_sig = np.arange(len(data_sig)) / fs_sig
    except Exception as e:
        print(f"Error loading audio files: {e}")
        return

    # --- 3. CREATE PLOTS ---
    try:
        fig, axs = plt.subplots(3, 2, figsize=(18, 15))
        fig.suptitle('Comparative Signal Analysis', fontsize=20)

        # -- LEFT COLUMN: FINAL TRACK --
        # Waveform
        axs[0, 0].plot(time_main, data_main, color='blue')
        axs[0, 0].set_title('Final Track - Waveform')
        axs[0, 0].set_xlabel('Time [sec]')
        axs[0, 0].set_ylabel('Amplitude')
        axs[0, 0].grid(True)

        # Spectrogram
        f_main, t_main, Sxx_main = spectrogram(data_main, fs_main, nperseg=1024)
        axs[1, 0].pcolormesh(t_main, f_main, 10 * np.log10(Sxx_main), cmap='viridis', shading='gouraud')
        axs[1, 0].set_title('Final Track - Spectrogram')
        axs[1, 0].set_ylabel('Frequency [Hz]')
        axs[1, 0].set_xlabel('Time [sec]')

        # PSD
        f_psd_main, Pxx_main = welch(data_main, fs_main, nperseg=1024)
        axs[2, 0].semilogy(f_psd_main, Pxx_main, color='blue')
        axs[2, 0].set_title('Final Track - Power Spectral Density')
        axs[2, 0].set_xlabel('Frequency [Hz]')
        axs[2, 0].set_ylabel('Power/Frequency [dB/Hz]')
        axs[2, 0].grid(True)

        # -- RIGHT COLUMN: SIGNATURE --
        # Waveform
        axs[0, 1].plot(time_sig, data_sig, color='orange')
        axs[0, 1].set_title('Signature - Waveform')
        axs[0, 1].set_xlabel('Time [sec]')
        axs[0, 1].grid(True)

        # Spectrogram
        f_sig, t_sig, Sxx_sig = spectrogram(data_sig, fs_sig, nperseg=1024)
        axs[1, 1].pcolormesh(t_sig, f_sig, 10 * np.log10(Sxx_sig), cmap='viridis', shading='gouraud')
        axs[1, 1].set_title('Signature - Spectrogram')
        axs[1, 1].set_ylabel('Frequency [Hz]')
        axs[1, 1].set_xlabel('Time [sec]')

        # PSD
        f_psd_sig, Pxx_sig = welch(data_sig, fs_sig, nperseg=1024)
        axs[2, 1].semilogy(f_psd_sig, Pxx_sig, color='orange')
        axs[2, 1].set_title('Signature - Power Spectral Density')
        axs[2, 1].set_xlabel('Frequency [Hz]')
        axs[2, 1].grid(True)

        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        plt.savefig(output_plot_path)
        plt.close()

        print(f"\nComparative analysis plot saved to: {output_plot_path}")

    except Exception as e:
        print(f"An error occurred during plotting: {e}")
        return

if __name__ == '__main__':
    generate_comparative_plots()
