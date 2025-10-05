# Sharks from Space - Technical Delivery Dossier (NASA Space Apps 2025)

## Project Overview
Our team built an end-to-end framework to predict shark feeding hotspots from NASA sensors. The pipeline covers automated data download, scientific preprocessing, feature engineering, modeling, visualization, and an intelligent tag concept. The supervised model still depends on high-quality telemetry; we justify the gap with the limited coverage of public tracks (e.g., OCEARCH) and show how the proposed tag closes the loop with near real-time field data.

## NASA Missions and Products Used

| Mission / Product | Variable | Resolution / Cadence | Role |
|-------------------|----------|----------------------|------|
| **GHRSST / MUR SST** | Sea-surface temperature (`sst`) | ~1 km / daily | Physical backbone for gradients and alignment. |
| **GHRSST / MUR SST** | Thermal gradient (`sst_gradient`) | derived | Locates fronts that concentrate prey and oxygen. |
| **MODIS Aqua (L3)** | Chlorophyll-a (`chlor_a_modis`) | ~4 km / daily-weekly | Historical productivity signal feeding the combined `chlor_a`. |
| **PACE OCI (MOANA)** | Phytoplankton functional types (`moana_prococcus_moana`, `moana_syncoccus_moana`, `moana_picoeuk_moana`) | ~1 km / daily | Differentiates plankton communities; converted into biomass and quality metrics. |
| **PACE OCI (MOANA)** | Derived metrics (`moana_total_cells`, `moana_picoeuk_share`, `moana_cyanobacteria_share`, `moana_diversity_index`) | derived | Quantifies food availability and quality for predators. |
| **PACE OCI (L3)** | Chlorophyll-a (`chlor_a_pace`) | ~1 km / daily | Complements MODIS in `chlor_a`. |
| **SWOT L2 LR SSH** | Sea-surface height (`ssh_swot`) | ~1 km / 21 days | Detects mesoscale eddies and fronts that trap prey. |
| **SWOT L2 LR SSH** | SSH gradient / mask (`ssh_swot_gradient`, `swot_mask`) | derived | Measures structure intensity and valid coverage. |

*Planned additions:* ECCO currents and bathymetry for future supervised runs.

## Technical Pipeline
1. **Download** - `scripts/01_search_download.py` (Earthdata + config-driven collections).
2. **Preprocess** - `scripts/02_preprocess.py` crops the BBOX, converts SST to Celsius, computes gradients with Dask, merges MOANA and SWOT.
3. **Feature engineering** - `scripts/03_feature_engineering.py` interpolates all variables onto the SST grid and exports per-day CSVs (`data/features/`).
4. **Models**
   - **Heuristic per species** - `scripts/04_shark_habitat_model.py` applies weighted physical/biological scores and outputs `habitat_score`, `habitat_class`, `is_hotspot`.
   - **Supervised baseline** - `scripts/ml/04_train_model.py` prepares `data/processed/dataset.csv` and trains XGBoost (`model_xgb.pkl`) using the heuristic label for now.
5. **Tiles & visualization** - `scripts/ml/05_export_tiles.py` produces probability GeoTIFFs for `app/index.html`; visualization scripts (e.g., `plot_all_variables.py`, `plot_all_variables_mean.py`) create daily and aggregated panels.
6. **External validation** - `scripts/05_validate_with_obis.py` compares predictions with OBIS/GBIF occurrences (Hit@Hotspot, lift).

## Features Feeding the Prediction
| Feature | Meaning | Source |
|---------|---------|--------|
| `lat`, `lon`, `date` | Pixel coordinates and timestamp | Shared SST grid |
| `sst`, `sst_gradient` | Sea-surface temperature and gradient magnitude | MUR SST |
| `chlor_a_modis`, `chlor_a_pace`, `chlor_a` | Chlorophyll-a from MODIS/PACE and combined mean | MODIS Aqua + PACE OCI |
| `ssh_swot`, `ssh_swot_gradient`, `swot_mask` | SSH, gradient, valid coverage | SWOT L2 LR |
| `moana_*` columns | Phytoplankton abundances and derived metrics | PACE OCI / MOANA |
| `habitat_score`, `habitat_class`, `is_hotspot` | Score, class, and hotspot flag | Heuristic model |

### Mathematical structure of `habitat_score`

\[
H(\text{pixel}, \text{species}) = \frac{\sum_i w_i S_i}{\sum_i w_i}, \quad \{w_i\} = \{0.23, 0.18, 0.15, 0.18, 0.10, 0.08, 0.05, 0.03\}
\]

Components:
- **Thermal gradient (`S_thermal_gradient`)** - piecewise function on \(|\nabla T|\).
- **Temperature (`S_temperature`)** - species-specific piecewise function (`SPECIES_PREFS`).
- **Chlorophyll (`S_chlorophyll`)** - piecewise function on chlorophyll concentration.
- **SWOT structure (`S_swot_structure`)** - quantile normalization of SSH gradient.
- **SWOT polarity (`S_swot_polarity`)** - warm/cold core preference using percentiles.
- **MOANA productivity (`S_moana_prod`)** - normalized log10 of `moana_total_cells` (P40-P90).
- **MOANA diversity (`S_moana_div`)** - entropy of phytoplankton shares, normalized to [0,1].
- **MOANA composition (`S_moana_comp`)** - piecewise function on `moana_picoeuk_share` with species-specific ideal ranges.

Scores are clipped to [0,1]; the top 10% become hotspots. The same formula applies to aggregated averages (`AVG_*.csv`).

### MOANA communities and implications
- `moana_prococcus_moana` (Prochlorococcus) - warm, oligotrophic cyanobacteria; indicates low-nutrient regimes typical of blue shark foraging.
- `moana_syncoccus_moana` (Synechococcus) - moderate productivity, often on front edges where nutrient replenishment is high.
- `moana_picoeuk_moana` (picoeukaryotes) - lipid-rich microalgae; higher values imply high-energy prey, attracting apex sharks.
- Derived columns (total, shares, diversity) distinguish basic versus premium food webs.

### Supervised model (XGBoost)
`scripts/ml/04_train_model.py` uses the same feature vector \([\text{sst}, \text{sst_gradient}, \text{chlor_a}, \text{moana_total_cells}, \ldots]\) to estimate \(P(\text{hotspot}\mid\mathbf{x})\). Currently trained on the heuristic label; it will be recalibrated with tag telemetry for continuous learning.

## Deliverables
- Feature CSVs (`data/features/*.csv`).
- Species predictions (`data/predictions/*_{species}_predictions.csv`) and PNG maps (`*_map.png`, `*_comparative_map.png`).
- Probability GeoTIFFs (`data/tiles/hotspots_probability_*.tif`) plus `tiles_manifest.json`.
- Visualization assets (`data/viz/compare/*.png`, `data/viz/moana/*.png`).
- Reports (`data/processed/metrics.json`, `data/predictions/habitat_model_report.json`, `data/predictions/habitat_model_report_mean.json`).

## Data Gaps & Closing the Loop
- **OCEARCH:** public tracks cover few species and short trajectories-insufficient for robust training.
- **OBIS/GBIF:** useful for point validation, not continuous labeling.
- **Solution:** the intelligent tag (IMU + hydrophone + gastric proxy + Argos) supplies near real-time feeding events, enabling: (1) reliable labels, (2) continuous retraining, (3) reduced uncertainty for managers.

## Next Steps
1. Secure telemetry partnerships or deploy the prototype tag.
2. Retrain XGBoost with real labels using spatial/temporal cross-validation.
3. Integrate ECCO currents and bathymetry into feature engineering.
4. Publish outputs via web app/API for conservation, fisheries, and education stakeholders.

---

Team Iron Shark - aligned with NASA Space Apps 2025.
