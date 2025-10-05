
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.io import wavfile
from scipy.signal import find_peaks, butter, filtfilt
import os
import time

# --- Final, Robust Correlation Analysis Script ---

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    """Apply a bandpass Butterworth filter"""
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def run_final_analysis():
    start_time = time.time()
    
    # --- 1. SETUP & PARAMETERS ---
    try:
        fft_root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        songs_dir = os.path.join(fft_root_dir, 'songs')
        original_audio_path = os.path.join(songs_dir, 'final track.wav')
        signature_audio_path = os.path.join(songs_dir, 'assinatura.wav')
        output_plot_path = os.path.join(fft_root_dir, 'final_analysis_results.png')
        
        # Filter & Detection Parameters
        LOWCUT = 43.0
        HIGHCUT = 1464.0
        detection_threshold = 0.6 # Using a slightly lower threshold to be safe
        
        print("=" * 60)
        print("FINAL ROBUST CORRELATION ANALYSIS")
        print("=" * 60)
        print(f"Main Track: {os.path.basename(original_audio_path)}")
        print(f"Signature: {os.path.basename(signature_audio_path)}")
        print(f"Filter Range: {LOWCUT}-{HIGHCUT} Hz")
        print(f"Detection Threshold: {int(detection_threshold*100)}%")
        print("=" * 60)

    except Exception as e:
        print(f"(-) Error setting up paths: {e}")
        return

    # --- 2. LOAD AUDIO ---
    try:
        print("\n[1/4] Loading audio files...")
        fs, data_orig = wavfile.read(original_audio_path)
        fs_sig, data_sig = wavfile.read(signature_audio_path)
        
        if fs != fs_sig:
            raise ValueError("Sample rates must match!")
        
        if data_orig.ndim > 1: data_orig = data_orig.mean(axis=1)
        if data_sig.ndim > 1: data_sig = data_sig.mean(axis=1)
        
        print(f"   (+) Main track loaded: {len(data_orig)/fs:.1f}s")
        print(f"   (+) Signature loaded: {len(data_sig)/fs:.1f}s")
        
    except Exception as e:
        print(f"(-) Error loading audio: {e}")
        return

    # --- 3. FILTER & CORRELATE ---
    try:
        print("\n[2/4] Applying filter and computing correlation...")
        
        # Filter the main track
        data_orig_filtered = bandpass_filter(data_orig, LOWCUT, HIGHCUT, fs)
        
        # Normalize signals
        sig_norm = (data_sig - np.mean(data_sig)) / np.std(data_sig)
        main_norm = (data_orig_filtered - np.mean(data_orig_filtered)) / np.std(data_orig_filtered)
        
        # Standard, robust correlation
        correlation = np.correlate(main_norm, sig_norm, mode='valid')
        
        print("   (+) Correlation complete.")
        
    except Exception as e:
        print(f"(-) Error during correlation: {e}")
        return

    # --- 4. PEAK DETECTION & VISUALIZATION ---
    try:
        print("\n[3/4] Detecting peaks and generating plot...")
        
        corr_threshold = np.max(correlation) * detection_threshold
        peaks, _ = find_peaks(correlation, height=corr_threshold, distance=int(fs * 1.0))
        detection_times = peaks / fs
        
        print(f"   (+) Found {len(peaks)} match(es).")

        # Create Plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
        fig.suptitle('Final Correlation Analysis Results', fontsize=16)

        # Correlation Plot
        time_corr = np.arange(len(correlation)) / fs
        ax1.plot(time_corr, correlation, color='#2c7bb6', label='Correlation')
        ax1.plot(detection_times, correlation[peaks], "x", color='#d7191c', markersize=12, mew=3, label='Detections')
        ax1.axhline(corr_threshold, color='#fdae61', linestyle='--', label=f'{int(detection_threshold*100)}% Threshold')
        ax1.set_title('Cross-Correlation Result')
        ax1.set_ylabel('Correlation Strength')
        ax1.legend()
        ax1.grid(True, alpha=0.5)

        # Spectrogram Plot
        ax2.specgram(data_orig, Fs=fs, NFFT=1024, cmap='viridis')
        ax2.set_title('Original Spectrogram with Detected Events')
        ax2.set_xlabel('Time [sec]')
        ax2.set_ylabel('Frequency [Hz]')

        signature_duration = len(data_sig) / fs
        for t in detection_times:
            rect = patches.Rectangle((t, 0), signature_duration, (fs / 2) * 0.95, 
                                     linewidth=2, edgecolor='r', facecolor='none', linestyle='--')
            ax2.add_patch(rect)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(output_plot_path)
        plt.close()
        
        print(f"   (+) Plot saved to: {output_plot_path}")

        total_time = time.time() - start_time
        print("\n[4/4] Analysis complete.")
        print(f"Total time: {total_time:.1f}s")

    except Exception as e:
        print(f"(-) Error during visualization: {e}")
        return

if __name__ == '__main__':
    run_final_analysis()
