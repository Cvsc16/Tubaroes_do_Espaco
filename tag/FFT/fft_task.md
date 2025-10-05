# Acoustic Analysis Task for Shark Feeding Detection (Revised)

## Approach Summary
Build Python software that analyzes `final track.wav` (main noisy audio) and finds events of interest using `assinatura.wav` (sample signature). We combine two techniques:
1. **Frequency analysis (FFT)** - visualize both audios and design a noise-reduction filter.
2. **Cross-correlation** - precisely locate the moments when the signature occurs in the main audio.

## Suggested Python Workflow

### 1. Load the audio files
- **Main audio:** `final track.wav`.
- **Signature:** `assinatura.wav` (clean bite sample).

### 2. Visual inspection with spectrograms
Generate spectrograms to understand each audio’s characteristics.

- **2.1 Spectrogram of `final track.wav`:** reveals background noise (horizontal lines) and distinct events (bright blobs).
- **2.2 Spectrogram of `assinatura.wav`:** shows the “fingerprint” of the bite, highlighting the frequency band where energy concentrates.

### 3. Use FFT to design a noise filter
- **Step 1:** Compute the FFT of `assinatura.wav` to find the dominant frequency band (F_min-F_max).
- **Step 2:** Design a band-pass filter (`scipy.signal`) allowing only F_min-F_max.
- **Step 3:** Apply the filter to `final track.wav`, producing a cleaned version.
- **Step 4:** Generate a spectrogram of the filtered audio to confirm that noise was reduced.

### 4. Precise detection with cross-correlation
Sliding the signature over the main/filtered audio yields a similarity curve; high peaks mark matches. Running correlation on the filtered audio typically improves accuracy by reducing false positives.

## Recommended Python Libraries
- **scipy** - load WAV files (`scipy.io.wavfile`) and design/apply digital filters (`scipy.signal`).
- **numpy** - numerical operations, FFT (`numpy.fft`), correlation (`numpy.correlate`).
- **matplotlib** - spectrograms and diagnostic plots.

## Next Steps
1. **Signature analysis:** script to save the spectrogram/FFT of `assinatura.wav` and identify its main frequency band.
2. **Filtering:** script to design/apply the band-pass filter on `final track.wav`, saving `final_track_filtrado.wav`.
3. **Verification:** rerun the cross-correlation script (`find_signature.py`) with the filtered audio and compare results.
