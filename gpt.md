# GPT Notes - Sharks from Space

## Current Project View
- Core objective: map shark feeding hotspots by combining NASA satellite data, feature engineering, and machine learning to support conservation.
- Main pipeline: `01_search_download.py` -> `02_preprocess.py` -> `03_feature_engineering.py` -> (future) `04_train_model.py` -> `05_export_tiles.py`, plus visualization scripts and MODIS comparisons.
- Data layout: raw NetCDF files in `data/raw/`, gradient-ready NetCDFs in `data/processed/`, tabular features in `data/features/`, and visuals in `data/compare/` plus `data/viz/`.
- Stack: Python, xarray, pandas, Plotly, earthaccess, XGBoost (planned), and Leaflet for the web app.

## Change Log
- 2025-09-29 - Centralized config loading in `scripts/utils_config.py` and refactored `scripts/compare_side_by_side_slider.py` to use the BBOX from config, handle coordinate names, and embed MODIS images via base64.
- 2025-09-29 - Updated `scripts/02_preprocess.py` to keep standard coordinates, preserve the `time` dimension, and compute gradients per timestep.
- 2025-09-29 - Tweaked `scripts/03_feature_engineering.py` to convert NetCDF files into DataFrames while keeping `lat`, `lon`, and `date` derived from `time`.

> Keep documenting new changes here with date, summary, and touched files.

- 2025-09-29 - Updated `docs/visao_geral_e_melhorias.md` with the latest changes and priorities.
- 2025-09-29 - Revised `scripts/04_train_model.py` to consolidate features, label hotspots, and train the model; `scripts/05_export_tiles.py` now reads NetCDF directly; refreshed `docs/visao_geral_e_melhorias.md` with the new flow.
- 2025-09-29 - Optimized `scripts/compare_side_by_side_slider.py`, reducing MODIS resolution and resampling grids (smaller HTML output).
- 2025-09-29 - Rewrote `scripts/compare_side_by_side_slider.py` entirely in ASCII to avoid encoding errors while keeping optimizations.
- 2025-09-29 - Implemented subplot slider (go.Image + Heatmap) for `scripts/compare_side_by_side_slider.py`, matching the `compare_modis_truecolor` layout.
- 2025-09-29 - Added `rioxarray` to `requirements.txt` for `scripts/05_export_tiles.py`.
- 2025-09-29 - Adjusted `scripts/05_export_tiles.py` to use y/x dimensions and sanitize timestamps (no colons) before writing GeoTIFFs.
- 2025-09-29 - Updated `app/index.html` to load GeoTIFFs with leaflet-geotiff (dropdown + SST/gradient toggle).
- 2025-09-29 - Added `scripts/compare_probability_vs_truecolor.py` to compare MODIS, SST, and generated GeoTIFFs.
- 2025-09-29 - Created `scripts/compare_probability_vs_truecolor_interactive.py` to build Plotly dashboards (MODIS/SST/gradient/probability).
- 2025-09-29 - Reworked the interactive dashboard with a `--date` argument and MODIS downsized to 512 px (lighter HTML).
- 2025-09-29 - Adjusted the dashboard layout (column widths) to avoid overlapping legends.
- 2025-09-30 - Normalized imports to `scripts.utils`, rewrote `scripts/02_preprocess.py`, updated `requirements.txt`, added `scripts/utils/build_tiles_manifest.py`, and improved `app/index.html` for dynamic manifests.
- 2025-09-30 - Added `docs/guia_solucao.md` to guide the non-technical team through deliverables.
- 2025-09-30 - Converted briefs to Markdown (`docs/desafio_projeto.md`, `docs/recursos_projeto.md`) and updated references.
- 2025-09-30 - Updated `config/config.yaml` to 2025-09-20-2025-09-25, reran scripts 01-05, and produced new CSVs/GeoTIFFs plus `tiles_manifest.json`.
- 2025-09-30 - Fixed pipeline/visualization scripts to add `sys.path` fallbacks and a `--date` argument in `compare_probability_vs_truecolor.py` (no ModuleNotFoundError).
- 2025-09-30 - Integrated MODIS CHL across 01-05 (download, preprocess, merge, export).
- 2025-09-30 - Temporarily set `max_granules_per_source = 1` to avoid multiple MODIS granules per day.
- 2025-09-30 - Restored `max_granules_per_source` to 10 and deduplicated MODIS L3 CHL by filename/time window.
- 2025-09-30 - Switched config dates to ISO format (YYYY-MM-DDTHH:MM:SS) to cover specific days.
- 2025-09-30 - Optimized `scripts/02_preprocess.py` (Dask chunks, no `.load()`, float32 + compression) to reduce memory use.
- 2025-10-03 - Updated `scripts/02_preprocess.py` to keep only SWOT points and reworked `scripts/03_feature_engineering.py` to project points onto the grid without extrapolation.
- 2025-10-03 - Updated `scripts/visualization/swot/plot_swot.py` and `plot_swot_gradient.py` to use Cartopy, adding coastlines, gridlines, and masks.
- 2025-10-03 - Tweaked SWOT plots to enforce project BBOX and draw the reference frame.
- 2025-10-03 - Colored SWOT points (SSH and gradient) directly, keeping geographic context.
