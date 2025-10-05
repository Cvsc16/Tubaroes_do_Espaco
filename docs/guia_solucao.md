# Quick Guide for the Team - Sharks from Space

This document helps non-developers understand how our solution addresses the "Sharks from Space" challenge and where each teammate can contribute.

## 1. What the challenge asks for
- Use NASA satellite data to predict where sharks will be feeding.
- Build a mathematical/predictive model linking ocean conditions to shark behavior.
- Propose an intelligent tag capable of reporting where the shark is, what it is eating, and streaming data almost in real time.

## 2. Our answer in a nutshell
- Focus area: Gulf Stream and Sargasso Sea.
- We integrate water temperature, biological productivity, and currents to map feeding hotspots.
- The automated pipeline turns NASA files into probability maps and ready-to-present visualizations.
- The embedded tag concept reinforces the model with data straight from the animal.

## 3. NASA data sources
- **MUR SST (GHRSST)** - sea surface temperature, foundation for detecting thermal fronts.
- **MODIS Aqua** - historical chlorophyll, shows where prey finds food.
- **PACE** - phytoplankton composition, differentiates food quality.
- **ECCO** - u/v currents, indicate nutrient transport.
- **SWOT** - sea surface topography, reveals eddies that concentrate prey.

More details in `docs/recursos_projeto.md`.

## 4. Workflow and scripts
1. `scripts/01_search_download.py` - downloads raw NASA data based on `config/config.yaml`.
2. `scripts/02_preprocess.py` - crops the area, converts SST to Celsius, and computes gradients.
3. `scripts/03_feature_engineering.py` - builds tables with lat, lon, date, SST, gradient, chlorophyll, and MOANA metrics.
4. `scripts/04_train_model.py` - merges features, labels hotspots (top gradient), and trains the XGBoost model.
5. `scripts/05_export_tiles.py` - applies the model and generates probability GeoTIFFs in `data/tiles/`.
6. `scripts/utils/build_tiles_manifest.py` - produces `data/tiles/tiles_manifest.json` for the web app.
7. Supporting visualization scripts:
   - `scripts/visualization/check_processed.py` and `check_processed_interactive.py` - inspect SST and gradient outputs.
   - `scripts/visualization/compare_modis_truecolor.py` and `compare_probability_vs_truecolor.py` - compare MODIS imagery, scientific data, and model outputs.
   - `app/index.html` - Leaflet map that loads GeoTIFFs and lets the user switch palettes.

## 5. Expected deliverables
- Interactive Leaflet map with available dates and hotspot probability.
- Comparative images/HTML dashboards for storytelling.
- Consolidated dataset in `data/processed/dataset.csv`.
- Technical documentation in `docs/visao_geral_e_melhorias.md` and daily log in `gpt.md`.
- Tag concept documented in `tag/concept.md`.

## 6. Intelligent tag summary
The proposed tag combines IMU, pressure/temperature sensor, hydrophone, and a lightweight gastric sensor. An embedded model detects feeding events and transmits only summaries when the animal surfaces, saving energy and validating satellite hotspots.

## 7. Tasks for non-developers
- Narrative & communication: build video scripts, slides, and explanations for high-school students using the visuals.
- Research support: summarize papers listed in `docs/recursos_projeto.md` to strengthen scientific justification.
- Qualitative validation: review maps and highlight stories (e.g., hotspot aligned with a SWOT eddy or MODIS chlorophyll spike).
- Presentation planning: connect map, model, and tag into a single storytelling arc.

## 8. Where to find more information
- `docs/desafio_projeto.md` - challenge briefing.
- `docs/recursos_projeto.md` - references and links.
- `docs/visao_geral_e_melhorias.md` - pipeline status and backlog.
- `gpt.md` - technical change log.
- `scripts/` directory - code organized by step.

Questions? Check the main README or talk to whoever is maintaining the pipeline.
