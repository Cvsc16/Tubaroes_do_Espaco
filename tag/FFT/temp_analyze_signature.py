
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import welch
import os

# --- Paths ---
fft_root_dir = os.path.abspath(os.path.dirname(__file__))
songs_dir = os.path.join(fft_root_dir, 'songs')
signature_audio_path = os.path.join(songs_dir, 'assinatura.wav')
output_plot_path = os.path.join(fft_root_dir, 'signature_psd_analysis.png')

try:
    # 1. Load the signature file
    fs, data_sig = wavfile.read(signature_audio_path)
    if data_sig.ndim > 1:
        data_sig = data_sig.mean(axis=1)

    # 2. Calculate Power Spectral Density (PSD)
    # The 'welch' method is robust for estimating power spectra.
    freqs, psd = welch(data_sig, fs, nperseg=1024)

    # 3. Analyze the PSD to find the key frequency range
    peak_freq = freqs[np.argmax(psd)]
    # Define a threshold (e.g., 10% of max power) to find the significant band
    threshold = np.max(psd) * 0.10
    significant_indices = np.where(psd > threshold)[0]
    f_min = int(freqs[significant_indices[0]])
    f_max = int(freqs[significant_indices[-1]])

    print(f"Signature Analysis Report:")
    print(f"  - Peak Frequency: {int(peak_freq)} Hz")
    print(f"  - Suggested Filter Range (based on 10% power threshold): {f_min} Hz - {f_max} Hz")

    # 4. Create and save the plot
    plt.figure(figsize=(12, 6))
    plt.semilogy(freqs, psd)
    plt.title('Power Spectral Density (PSD) of Signature')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Power/Frequency [dB/Hz]')
    plt.grid(True)
    # Add annotations for the determined range
    plt.axvspan(f_min, f_max, color='red', alpha=0.2, label=f'Suggested Range: {f_min}-{f_max} Hz')
    plt.legend()
    plt.savefig(output_plot_path)
    plt.close()
    
    print(f"\nPSD analysis plot saved to: {output_plot_path}")

except Exception as e:
    print(f"An error occurred: {e}")
