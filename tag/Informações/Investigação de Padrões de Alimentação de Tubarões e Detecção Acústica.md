# Investigation of Shark Feeding Patterns and Acoustic Detection

## Acoustic Characteristics of Shark Feeding
Research indicates that sharks are sensitive to low-frequency sounds, and detecting feeding acoustically is a promising avenue.
- **Attracted to low frequencies:** sharks respond to irregular pulses below ~80 Hz, suggesting they use sound to locate struggling prey.
- **Feeding sound recordings:** implanted bioacoustic probes have successfully captured shark feeding sounds along with ambient noises (fish vocalizations, boat noise), confirming detectability.
- **Sound production:** some species produce clicks or other sounds, reinforcing the role of acoustics in shark ecology.

## Acoustic Detection Methods
To implement acoustic feeding detection in the tag:
- **Bioacoustics + biologging:** multi-sensor tags (including hydrophones) reveal foraging ecology and behavior.
- **High-resolution acoustic cameras:** effective for reef-shark behavior studies—while large for a tag, they show the viability of acoustic imaging.
- **Acoustic telemetry:** well-established for tracking presence and movement; the proposed tag could integrate with existing networks to transmit feeding events.

## Tag Design Implications
1. **Low-frequency hydrophone** optimized for <1 kHz to capture feeding sounds and prey distress signals.
2. **Signal processing on-board:** MCU must run FFT and other feature extraction methods (event duration, peak frequency, harmonic structure).
3. **Embedded classifier:** machine-learning model trained to recognize acoustic feeding signatures running in real time.
4. **Training data acquisition:** biggest challenge; requires controlled recordings (aquariums) or raw audio logging tags for later labeling.

Next steps: locate datasets and studies that supply the necessary audio for training and validating feeding-pattern recognition models.
