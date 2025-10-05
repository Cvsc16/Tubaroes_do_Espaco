# Report and Recommendations: NASA Space Apps 2025 – Sharks from Space

## 1. Introduction
This report reviews the “Sharks from Space” challenge, summarizes the research performed on acoustic detection and shark bioacoustics, and outlines recommendations for designing an innovative tag capable of identifying feeding patterns through underwater sound analysis.

## 2. Challenge Analysis
NASA requests a mathematical framework that predicts shark feeding habitats from satellite data (SWOT, PACE, SST), plus a next-generation tag that reports not only shark location but also what they are eating. The reference document “Ultrasonic Shark-tag Locator System for IVER2 AUV” details a robust tracking solution, yet it does not cover feeding-pattern detection. Our acoustic-based tag approach fills that gap by listening for bite signatures.

## 3. Acoustic Feasibility
- Sharks respond strongly to low-frequency sounds typical of struggling prey.
- Implanted bioacoustic probes have already recorded shark feeding sounds, proving detectability.
- FFT is the standard technique to analyze underwater audio and isolate spectral signatures.

## 4. Tag Design Recommendations
- **Sensors:** low-frequency hydrophone, 3-axis accelerometer (to confirm post-bite low activity), depth/temperature sensors.
- **Processing:** DSP-capable MCU running FFT and a lightweight classifier trained on feeding signatures.
- **Telemetry:** compact event packets transmitted via acoustic modem to buoys, then relayed through satellite.
- **Power & Housing:** lithium-thionyl batteries with deep-sleep management inside a titanium or alumina housing; electronics potted for pressure resistance.

## 5. Data & Model Roadmap
1. Build partnerships or run controlled recordings to collect feeding audio.
2. Label events using spectrogram inspection and correlation analysis.
3. Train ML models (CNN/SVM) to classify feeding vs. non-feeding.
4. Integrate the classifier into the MCU and validate in the field.

## 6. Conclusion
The acoustic-tag concept satisfies NASA’s request for an innovative tag that measures shark feeding behavior. With curated audio datasets, FFT-based feature extraction, and embedded classification, the system can deliver real-time feeding events to support conservation-focused models.
