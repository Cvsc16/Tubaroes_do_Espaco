# Analysis of Studies and Datasets Available for the Project

## Challenges in Obtaining Shark Feeding Audio Datasets
Public, shark-feeding–specific audio datasets are scarce. Nevertheless, literature confirms that such sounds can be recorded, which is essential for validating the tag concept.

## Relevant Studies and Potential Data Sources
1. **"Implanted Tags Record Acoustic Environment of Sharks" / "Use of an implanted sound recording device (Bioacoustic Probe) to document the acoustic environment of a blacktip reef shark"** – these studies report an implanted tag on *Carcharhinus melanopterus* that successfully recorded *shark feeding sounds*. The raw audio is not publicly linked, but the existence of these recordings supports the feasibility of detecting feeding events acoustically. *Implication:* collaborating with researchers who deploy bioacoustic tags will be key to building a training dataset.

2. **"Evidence of active sound production by a shark"** – the associated Figshare dataset focuses on active sound production by *Mustelus lenticulatus* when handled. While not feeding-specific, it provides shark-generated audio and methodologies for underwater recording and analysis. *Implication:* useful baseline for shark-related acoustics and methodology replication.

3. **Underwater acoustic repositories (NOAA Passive Acoustic Data, Open Access Underwater Acoustics Data)** – large archives containing marine mammal, fish, and vessel sounds. They are not shark-feeding datasets, but they can supply background/noise profiles and negative examples for classification tasks. *Implication:* train models to differentiate feeding events from general ocean sounds.

4. **Behavior classification studies (e.g., "Feature extraction, selection, and K-nearest neighbors algorithm for shark behavior classification...")** – discuss sensor-based detection of shark behaviors, including feeding, and highlight issues with imbalanced datasets. *Implication:* provide insights into classification techniques and handling rare events.

## Recommendations for Data Collection & Analysis
- **Collaborate with researchers** already using bioacoustic tags to access raw shark audio.
- **Capture controlled recordings** in aquariums or research facilities if collaborations are not possible.
- **Leverage spectrographic analysis** (FFT spectrograms) to visually identify feeding signatures and assist manual labeling.
- **Apply machine-learning techniques** (SVM, neural nets, random forest) while addressing dataset imbalance through oversampling or other strategies.

This research phase confirms the concept’s viability and underscores the need for a focused effort on acquiring and preparing shark-feeding audio to train recognition models.
