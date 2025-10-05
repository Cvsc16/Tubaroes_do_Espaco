# Project: Sharks from Space - NASA Space Apps 2025

## 1. Project Overview

This project answers the "Sharks from Space" challenge of NASA Space Apps 2025. The goal is twofold:

1. **Create a mathematical framework** to identify and predict shark foraging habitats using NASA satellite data.
2. **Propose an innovative tag concept** capable of monitoring not only shark location but also feeding patterns, transmitting data in real time.

This document focuses on the second part of the challenge: designing the innovative tag.

## 2. The Innovative Tag: Acoustic Feeding Detection

Our proposed solution is a shark tag that uses acoustic detection to identify when a shark is feeding.

### 2.1 Concept

The tag is equipped with a hydrophone (underwater microphone) to capture the shark's acoustic environment. An embedded microcontroller analyzes the sound in real time using a **Fast Fourier Transform (FFT)** to identify the acoustic signature of a feeding event (e.g., a bite).

A pre-trained machine learning model classifies the captured sounds as "feeding" or "non-feeding." When a feeding event is detected, the tag transmits this information along with the shark's GPS position to a base station or directly via satellite.

### 2.2 Tag Components

| Component | Suggested Specification | Justification |
|-----------|-------------------------|---------------|
| **Hydrophone** | Wideband piezoelectric sensor with high sensitivity at low frequencies (< 2 kHz). | Feeding and prey sounds occur mainly at low frequencies. |
| **Microcontroller** | DSP-capable board (e.g., ESP32). | Runs FFT and classification algorithms in real time. |
| **Memory** | Flash memory. | Stores the classification model and audio snippets. |
| **Communication** | Low-power module (e.g., LoRaWAN or satellite). | Sends feeding events in real time. |
| **Battery** | Long-lasting, optimized for low consumption. | Maximizes tag lifetime in the marine environment. |
| **GPS** | Embedded GPS module. | Logs shark location during feeding events. |

### 2.3 Workflow

1. **Continuous capture:** the hydrophone records ambient sound.
2. **FFT analysis:** the microcontroller performs the FFT on audio samples.
3. **Feature extraction:** spectral features are computed from the FFT output.
4. **Classification:** a machine learning model labels the event.
5. **Transmission:** if the event is feeding, the tag transmits the data.

## 3. Feasibility and Challenges

### 3.1 Feasibility
The proposal is technically viable. The required technology already exists, and previous studies have captured shark feeding sounds with implanted tags.

### 3.2 Challenges
The main challenge is the **lack of a public dataset of shark feeding sounds**, which is critical for training the machine learning model.

**Proposed solutions:**
- **Collaboration:** partner with researchers who already collect shark acoustic data.
- **Captive recording:** capture feeding sounds in aquariums or research centers.

## 4. Next Steps
1. **Prototype development:** build a tag prototype with the components listed above.
2. **Data collection:** start gathering acoustic data to train the model.
3. **Model training:** develop and train the classifier.
4. **Testing and validation:** conduct controlled experiments and then field trials.

## 5. Reference Documents
- Análise de Estudos e Datasets Disponíveis para o Projeto.md 
- Análise dos Documentos Fornecidos.md 
- Investigação de Padrões de Alimentação de Tubarões e Detecção Acústica.md 
- Pesquisa sobre Tecnologias FFT e Monitoramento Acústico Subaquático.md 
- Relatório e Recomendações - Desafio NASA Space Apps 2025 - Sharks from Space.md 
- UltrasonicShark-tagLocatorSystemforIVER2AUV.pdf
