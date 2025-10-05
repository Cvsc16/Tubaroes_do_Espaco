# Acoustic Analysis Summary with FFT

## Goal
Detect a specific audio signature (a simulated bite using a cheese bun) inside a longer, noisy audio track (`final track.wav`) using Python signal-processing techniques.

## Process and Findings

### 1. Initial Correlation Detection
- **Action:** First attempt used **cross-correlation** between the main audio (`final track.wav`) and the signature (`assinatura.wav`).
- **Result:** Success. We found **one match** at approximately **2.05 seconds**, proving detection was feasible.

### 2. Low-Frequency Filtering Attempt
- **Action:** To improve results and reduce noise, we analyzed the signature frequency spectrum, which showed a peak around **49 Hz**. We designed a 20-200 Hz band-pass filter to isolate low frequencies.
- **Result:** Failure. Running correlation on the filtered audio yielded **no matches**.
- **Conclusion:** The signature’s key features were not in the low-frequency band. The 49 Hz peak was likely background noise common to both files, and the filter removed the frequencies that mattered.

### 3. Mid-Frequency Filtering (Refinement)
- **Action:** Hypothesized that signature characteristics live in mid frequencies. Designed a new band-pass filter from **300 Hz to 3000 Hz**.
- **Result:** **Complete success.** Correlation on the new filtered audio found **two clear matches** at **1.95 s** and **3.39 s**. The first matched the initial analysis; the second was a new occurrence previously masked by noise.

### 4. Final Visualization
- Generated a spectrogram of the original audio with detections highlighted in red, plus a second spectrogram showing the removed noise band.

## Final Takeaway
Iterative analysis and filtering were crucial. Combining **FFT-based frequency analysis** (to design an effective filter) with **cross-correlation** (to pinpoint the pattern) proved a robust strategy for finding the desired acoustic signature.
