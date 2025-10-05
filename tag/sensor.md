# B-AMPT: Bio-Acoustic Motion and Profiling Tag

*Version 2.0 - Conceptual Model*

## 1. Summary

B-AMPT (Bio-Acoustic Motion and Profiling Tag) is a conceptual, non-invasive biologging device designed for large marine species. The tag aims to detect and transmit shark feeding events in near real time by fusing multiple sensors: it looks for an acoustic signature followed by a period of low activity, inferring a hunt-to-satiation cycle.

---

## 2. Hardware Components (Conceptual)
- **High-sensitivity hydrophone:** captures the acoustic environment and the feeding signature.
- **3-axis accelerometer:** monitors shark activity; confirms low-motion windows after a feeding sound.
- **Temperature & pressure sensor:** records water temperature and depth to contextualize behavior.
- **Microcontroller unit (MCU):** low-power MCU (e.g., NXP MCX family) capable of running FFT/correlation bursts only when a candidate sound is detected.
- **Communication module (ultrasonic modem):** transmits small data packets underwater; ultrasound is preferred over radio in seawater.
- **Battery and power management:** long-life lithium-thionyl batteries with deep-sleep strategies to maximize endurance.

---

## 3. Detection Methodology
1. **Listening mode:** hydrophone samples audio continuously or at intervals with a very low energy threshold.
2. **Sound trigger:** if the signal exceeds the threshold within the band of interest (for example, 43-1464 Hz), the MCU wakes up.
3. **Correlation detection:** the core algorithm (`run_correlation_analysis.py`) checks the clip against the stored feeding signature.
4. **Movement confirmation:** the accelerometer is monitored; a low-activity period immediately after the sound validates the event.
5. **Data transmission:** a compact packet is sent via ultrasound.

---

## 4. Communication Architecture
- **Stage 1 – Tag to receiver (ultrasound):** acoustic modem converts digital packets into ultrasonic pulses. Bandwidth is 100-2000 bps, so the packet contains only tag ID, timestamp, event type, confidence, depth, and temperature. Range: hundreds of meters to a few kilometers.
- **Stage 2 – Receiver to satellite:** autonomous acoustic receivers (buoys or seafloor nodes) pick up the packets and relay them via satellite (Iridium/Argos) to research centers in near real time.

---

## 5. Physical Design & Materials
- **Enclosure:** Grade 5 titanium or alumina ceramic housing, hydrodynamic to minimize drag.
- **Internal protection:** electronics potted with marine epoxy or polyurethane forming a solid block resistant to high pressure.
- **Attachment:** dorsal-fin clamp designed to release after a preset period, allowing retrieval and minimizing long-term impact on the animal.
