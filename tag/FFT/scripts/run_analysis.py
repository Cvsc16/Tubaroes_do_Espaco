
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import butter, filtfilt, spectrogram
from skimage.feature import peak_local_max
import os

# --- Final Analysis Script based on main.ipynb workflow ---

def analyze_and_isolate():
    # --- 1. SETUP & PARAMETERS ---
    try:
        fft_root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        songs_dir = os.path.join(fft_root_dir, 'songs')
        original_audio_path = os.path.join(songs_dir, 'final track.wav')
        signature_audio_path = os.path.join(songs_dir, 'assinatura.wav')
        
        # Parameters based on Claude's analysis of the signature
        lowcut = 43.0
        highcut = 1464.0
        peak_threshold = 0.5 # Start with a 50% relative threshold for peak detection
        
        print(f"Analysis started using filter range: {lowcut}-{highcut} Hz")
    except Exception as e:
        print(f"Error setting up paths: {e}")
        return

    # --- 2. LOAD AUDIO ---
    try:
        fs, data_orig = wavfile.read(original_audio_path)
        fs_sig, data_sig = wavfile.read(signature_audio_path)
        if fs != fs_sig:
            raise ValueError("Sample rates of main audio and signature must be the same.")
        if data_orig.ndim > 1: data_orig = data_orig.mean(axis=1)
        if data_sig.ndim > 1: data_sig = data_sig.mean(axis=1)
    except Exception as e:
        print(f"Error loading audio files: {e}")
        return

    # --- 3. FILTER MAIN TRACK & DETECT PEAKS ---
    try:
        def butter_bandpass(low, high, fs, order=4):
            nyq = 0.5 * fs
            b, a = butter(order, [low / nyq, high / nyq], btype='band')
            return b, a

        b, a = butter_bandpass(lowcut, highcut, fs)
        filtered_main_audio = filtfilt(b, a, data_orig)

        freqs_spec, times_spec, sxx_filt = spectrogram(filtered_main_audio, fs=fs, nperseg=1024)

        peaks = peak_local_max(sxx_filt, min_distance=5, threshold_rel=peak_threshold)
        
        if len(peaks) == 0:
            print(f"Detection Complete: No significant energy peaks found with current threshold ({peak_threshold}).")
            detection_times = []
        else:
            peak_times = times_spec[peaks[:, 1]]
            detection_times = [peak_times[0]]
            for t in peak_times:
                if (t - detection_times[-1]) > 1.0: # Group peaks within 1s of each other
                    detection_times.append(t)
            print(f"Detection Complete. Found {len(detection_times)} event(s) at: {np.round(detection_times, 2)} seconds.")

    except Exception as e:
        print(f"Error during filtering or peak detection: {e}")
        return

    # --- 4. GENERATE FINAL VISUALIZATIONS ---
    try:
        # Image 1: Full Audio Spectrogram
        plt.figure(figsize=(12, 6))
        plt.specgram(data_orig, Fs=fs, NFFT=1024, cmap='viridis')
        plt.title('Image 1: Spectrogram of Full Audio (final track.wav)')
        plt.xlabel('Time [sec]'); plt.ylabel('Frequency [Hz]'); plt.colorbar().set_label('Intensity [dB]')
        plt.tight_layout()
        plt.savefig(os.path.join(fft_root_dir, 'final_track_spectrogram.png'))
        plt.close()
        print("Generated Image 1: Full audio spectrogram.")

        # Image 2: Filtered Signature Spectrogram
        b_sig, a_sig = butter_bandpass(lowcut, highcut, fs_sig)
        filtered_signature = filtfilt(b_sig, a_sig, data_sig)
        plt.figure(figsize=(12, 6))
        plt.specgram(filtered_signature, Fs=fs_sig, NFFT=1024, cmap='viridis')
        plt.title(f'Image 2: Spectrogram of Filtered Signature ({lowcut}-{highcut} Hz)')
        plt.xlabel('Time [sec]'); plt.ylabel('Frequency [Hz]'); plt.colorbar().set_label('Intensity [dB]')
        plt.tight_layout()
        plt.savefig(os.path.join(fft_root_dir, 'filtered_signature_spectrogram.png'))
        plt.close()
        print("Generated Image 2: Filtered signature spectrogram.")

        # Image 3: Isolated Detection Spectrogram
        signature_duration_sec = len(data_sig) / fs_sig
        mask = np.full(data_orig.shape, 0.1)
        if not detection_times:
             print("Skipping Image 3 as no events were detected.")
        else:
            for t in detection_times:
                start_sample = int(t * fs)
                end_sample = int((t + signature_duration_sec) * fs)
                mask[start_sample:end_sample] = 1.0
            isolated_data = data_orig * mask

            plt.figure(figsize=(12, 6))
            plt.specgram(isolated_data, Fs=fs, NFFT=1024, cmap='viridis')
            plt.title('Image 3: Isolated Signature(s) within Main Track')
            plt.xlabel('Time [sec]'); plt.ylabel('Frequency [Hz]'); plt.colorbar().set_label('Intensity [dB]')
            plt.tight_layout()
            plt.savefig(os.path.join(fft_root_dir, 'isolated_detection_spectrogram.png'))
            plt.close()
            print("Generated Image 3: Isolated detection spectrogram.")
        
        print("\nAnalysis complete. All files generated in the 'FFT' directory.")

    except Exception as e:
        print(f"Error during visualization: {e}")
        return

if __name__ == '__main__':
    analyze_and_isolate()
