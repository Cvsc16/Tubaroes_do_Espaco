# Sharks from Space | NASA Space Apps Challenge 2025

Project developed for the **"Sharks from Space"** challenge of the NASA Space Apps Challenge 2025. Our goal is to **predict shark foraging habitats** by combining NASA satellite data (SST, MODIS, PACE, SWOT), mathematical modeling, and AI-together with the concept of an **intelligent electronic tag** that collects real-time acoustic data directly from the shark.

---

## Executive Summary
- **NASA data + ocean fronts = feeding hotspots.**
- Automated pipeline generates **daily probability maps** and **interactive dashboards**.
- The **smart shark tag** adds behavioral context and field validation.

---

## Expected Impact
- **Marine conservation** - prioritize critical habitats and migratory routes.
- **Sustainable fishing** - reduce bycatch and conflicts between fisheries and biodiversity.
- **Scientific outreach** - translate complex science into engaging storytelling.
- **Applied space science** - NASA data powering real-world ocean insights.

---

## Integrated Vision

| Module | Focus | Stack | Purpose |
|--------|-------|-------|---------|
| **Modeling & Prediction** | NASA data, ocean physics, machine learning | Python, XGBoost, Plotly | Predict hotspots based on SST gradients, chlorophyll, phytoplankton, and SSH. |
| **Smart Tag (Shark Tracker)** | Embedded bioacoustic monitoring | C++ / ESP32 | Detect feeding sounds, classify events, and transmit summarized data. |

---

## 1. Modeling with NASA Data

### Key Datasets

| Dataset | Variable | Resolution | Purpose |
|---------|----------|------------|---------|
| **SST (MUR)** | Sea Surface Temperature | 1 km / daily | Identifies thermal fronts-preferred foraging areas. |
| **MODIS L3 CHL** | Chlorophyll-a | 4 km / daily-weekly | Indicates biological productivity (plankton ? fish ? sharks). |
| **PACE OCI / MOANA** | Phytoplankton Functional Types | 1 km / daily | Differentiates nutritious versus poor-quality plankton communities. |
| **SWOT SSH** | Surface Topography | 1 km / 21 days | Detects mesoscale eddies and fronts that concentrate prey. |

**In short:** PACE/MODIS show **where food originates**; SST shows **where it concentrates**; SWOT shows **how it moves**. The model integrates these signals into **actionable probability maps**.

---

### Modeling Pipeline
1. `01_search_download.py` - automatic download of SST, MODIS, PACE and SWOT from Earthdata.
2. `02_preprocess.py` - crop AOI, convert SST to Celsius, compute gradients and masks.
3. `03_feature_engineering.py` - build CSVs per date with aligned variables.
4. `04_train_model.py` - train an **XGBoost** classifier for hotspot detection.
5. `05_export_tiles.py` - export GeoTIFF probability maps and tiles for the web app.

**Outputs:** `dataset.csv`, `model_xgb.pkl`, `metrics.json`, `tiles/hotspots_probability_*.tif`

---

### Ecological Chain Concept

```
Phytoplankton (PACE / MODIS)
    ?
Fish / Prey (currents, structure)
    ?
Fronts & Eddies (SWOT + SST gradient)
    ?
Sharks (model prediction)
```

---

## 2. Smart Tag: Shark Tracker Device

The second component is an **acoustic-based intelligent tag** capable of detecting feeding events by analyzing underwater sound patterns.

### Concept

The tag combines a **hydrophone**, **DSP-capable microcontroller (ESP32)**, and **LoRa/satellite communication**. It records ambient sound, performs FFT in real time, and uses a compact embedded ML model to classify feeding signatures. Upon detection, it transmits **GPS position** and **event metadata**.

### Hardware Components

| Component | Function | Description |
|-----------|----------|-------------|
| **Hydrophone** | Captures low-frequency sounds (<2 kHz). | Feeding and prey sounds occur mainly at low frequencies. |
| **Microcontroller (ESP32)** | Runs FFT + ML model. | Low-power DSP architecture for onboard signal analysis. |
| **Flash Memory** | Stores model and audio snippets. | Enables post-processing and retraining. |
| **Communication Module (LoRa / Satellite)** | Sends detected events. | Argos/Iridium uplink or surface LoRa gateway. |
| **Battery** | Provides long-term power. | Duty cycling ensures energy efficiency. |
| **GPS Module** | Geolocates feeding events. | Merges behavior and position data. |

### Workflow

1. Continuous acoustic capture.
2. Real-time FFT computation.
3. Feature extraction from the spectrum.
4. ML classification (“feeding” / “non-feeding”).
5. Transmission of event data.

### Feasibility & Challenges

- **Feasibility:** Existing underwater recorders already detect shark feeding behavior; miniaturization is viable.
- **Challenge:** Lack of public *feeding-sound datasets* for supervised training.

**Proposed roadmap:** prototype hardware ? collect labeled audio ? train FFT-based classifier ? field-test on tagged animals or AUV simulations.

---

## Repository Structure

```
Tubaroes_do_Espaco/
+-- app/                     # Leaflet web map
+-- config/                  # Configuration (bbox, dataset IDs)
+-- data/                    # Raw, processed, aggregated data
+-- docs/                    # Reports and documentation
+-- scripts/                 # Python modeling pipeline
+-- tag/                     # Smart-tag design, firmware, FFT studies
¦   +-- FFT/
¦   +-- Informations/
+-- requirements.txt
+-- README.md
```

---

## Setup and Execution

### Python Environment
```bash
python -m venv .venv
source .venv/bin/activate      # Linux/macOS
.\.venv\Scripts\Activate.ps1  # Windows
pip install -r requirements.txt
```

### Full Modeling Pipeline
```bash
python scripts/01_search_download.py
python scripts/02_preprocess.py
python scripts/03_feature_engineering.py
python scripts/04_train_model.py
python scripts/05_export_tiles.py
```

---

## Integration between Modules

| Component | Input | Output | Communication |
|-----------|-------|--------|---------------|
| Modeling | SST, CHL, SWOT, PACE | Probability maps | Web app / API |
| Smart Tag | Acoustic signal + GPS | Feeding events | LoRa / Satellite |
| Feedback Loop | Tag data | Retraining & validation | Continuous improvement |

---

## Conclusion

**Sharks from Space** bridges space-based observation and in-field behavior:
- Satellites **see** the ocean.
- The smart tag **hears** the shark.
- AI **connects** both worlds.

Together, they create a unified platform for **marine conservation**, **embedded innovation**, and **data-driven storytelling**.

---

### Team Iron Shark
Interdisciplinary team in **NASA Space Apps Challenge 2025**, combining engineers, data scientists, and marine biologists to decode shark behavior through space and sound.

Special thanks to the **PACE**, **SWOT**, **MODIS**, and **MUR** missions for their open data and inspiration.
