# Analysis of the Provided Documents

## 1. NASA Space Apps 2025 – Sharks from Space
The challenge asks for a mathematical framework that identifies sharks and predicts feeding habitats using NASA satellite data, plus a conceptual tag that reports where sharks are and what they are eating in near real time.

**Key points:**
- Build a predictive framework from satellite variables (SWOT, PACE, SST, currents, depth, temperature).
- Propose an innovative tag capable of capturing feeding patterns.
- Identify hotspots by linking physical oceanography, phytoplankton, and predator movement.
- Solution should be explainable to high-school students and the general public.

## 2. Ultrasonic Shark-tag Locator System for IVER2 AUV
Technical document describing an ultrasonic shark-tag locator integrated into the IVER2 AUV to follow tagged sharks for research.

**System components:**
- Shark tag (Sonotronics CTT-83-3-I) at 73 kHz, emitting 20 ms pulses whose interval depends on water temperature.
- Two omnidirectional hydrophones (Desert Star Systems) converting acoustic vibrations to analog signals.
- Filter/amplifier circuit that boosts weak signals (~40 µV) and applies a 73 kHz band-pass filter.
- Microcontroller (Atmel STK500 / Atmega8) that computes tag azimuth using time delay (?T) and hydrophone spacing, assuming 1,560 m/s sound speed in seawater.
- RS232 link to the IVER2 AUV for navigation.

**Limitations:**
- Designed solely for localization, not behavioral or feeding detection.
- Azimuth accuracy degrades in challenging environments (bubbles, suspended sediment).
- Focused on passive tracking of a predefined acoustic beacon.

## Preliminary Conclusion
NASA seeks a tag that goes beyond localization to identify feeding behavior. The existing ultrasonic system is robust for tracking but does not satisfy this new requirement. Using FFT and a hydrophone to capture shark bite sounds is a promising approach for meeting the challenge’s emphasis on measuring what sharks are eating.
