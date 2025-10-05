# Acoustic Analysis for Signature Detection

## Project Goal
Find occurrences of a specific sound (the "signature" in `assinatura.wav`) inside a longer, noisy audio file (`final track.wav`). We iterated through multiple methods to arrive at the most robust and efficient approach.

---

## Final Methodology
After exploring different techniques, including energy-peak detection in spectrograms, we concluded that **cross-correlation with filtering** is the most effective for this case.

Workflow of the final script:
1. **Load audio files:** `final track.wav` (main) and `assinatura.wav` (signature).
2. **Band-pass filtering:** apply a Butterworth filter to isolate the frequency band where the signature concentrates energy (identified as **43 Hz - 1464 Hz**). This suppresses irrelevant noise.
3. **Cross-correlation:** slide the signature over the filtered audio and compute similarity at each point-ideal for subtle patterns.
4. **Peak detection:** mark correlation peaks above 60% as valid detections.
5. **Visualization:** save `final_analysis_results.png` with two plots for easy interpretation.

---

## Folder Layout
- `/songs` - original audio files (`final track.wav`, `assinatura.wav`).
- `/scripts` - final analysis script.
- `/data` - reserved for processed audio (currently empty).

---

## How to Run
1. Open a terminal inside the project folder.
2. Execute the analysis script, for example:
   ```
   C:\Users\lucas\OneDrive\Documentos\Projetos\Nasa_space_apps\Tubaroes_do_Espaco\.venv\Scripts\python.exe C:\Users\lucas\OneDrive\Documentos\Projetos\Nasa_space_apps\Devices\FFT\scripts\run_correlation_analysis.py
   ```

---

## Expected Output
Running the script generates `final_analysis_results.png` inside `FFT/`.
- **Top plot (Cross-Correlation Result):** correlation strength over time. Red crosses indicate exact matches with the signature.
- **Bottom plot (Original Spectrogram):** spectrogram of the original audio with red rectangles highlighting detected events.
