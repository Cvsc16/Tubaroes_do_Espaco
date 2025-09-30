# 🦈 Tubarões do Espaço | NASA Space Apps Challenge 2025

Projeto desenvolvido para o desafio **Tubarões do Espaço** do NASA Space Apps Challenge 2025. Nosso objetivo é **prever habitats de alimentação de tubarões** combinando dados de satélite (SST, PACE, SWOT, ECCO), modelagem matemática, machine learning e visualização interativa, além do conceito de **tag eletrônica inteligente**.

---

## 🚀 Resumo Executivo
- Dados NASA → frentes oceânicas → hotspots de alimentação.  
- Pipeline automatizada gera mapas de probabilidade diários e dashboards interativos.  
- Tag proposta reforça o modelo com dados “on shark”.

---

## 🌍 Impacto Esperado
- 🌱 **Conservação marinha** – proteger áreas críticas.  
- 🎣 **Pesca sustentável** – reduzir conflito pesca × biodiversidade.  
- 🧑‍🎓 **Educação científica** – contar a história de forma acessível.  
- 🛰️ **Valorizar dados NASA** – ciência espacial aplicada a desafios costeiros.

---

## 🗂️ Estrutura Principal
```
Tubaroes_do_Espaco/
│
├── config/
│   └── config.yaml               # BBox, janela temporal, nomes de datasets
│
├── data/
│   ├── raw/                      # NetCDF brutos (MUR, MODIS, ...)
│   ├── processed/                # NetCDF recortados, dataset.csv, modelo, métricas
│   ├── features/                 # CSVs tabulares para ML
│   ├── tiles/                    # GeoTIFFs hotspots_probability_*.tif + tiles_manifest.json
│   └── compare/                  # PNG/HTML usados no storytelling
│
├── scripts/
│   ├── 01_search_download.py     # Busca no Earthdata (usa config)
│   ├── 02_preprocess.py          # Recorte, conversão para °C, gradiente (preserva `time`)
│   ├── 03_feature_engineering.py # Gera tabelas (lat, lon, date, sst, grad, chlor_a)
│   ├── 04_train_model.py         # Concatena, rotula hotspots, treina XGBoost
│   ├── 05_export_tiles.py        # Aplica o modelo e exporta GeoTIFFs de probabilidade
│   ├── utils/                    # load_config, project_root, build_tiles_manifest.py
│   └── visualization/            # Inspeções rápidas + comparações MODIS/SST/prob (PNG, slider, dashboard)
│
├── app/                          # Mapa Leaflet (lê o manifest JSON de tiles)
├── docs/                         # Briefing, visão geral/melhorias, guia rápido
└── tag/                          # Conceito de tag embarcada
```

### Novidades recentes
- Janela atual: **2025-09-20 → 2025-09-25** (scripts 01–05 rerodados; GeoTIFFs + manifest atualizados).  
- `app/index.html` carrega `data/tiles/tiles_manifest.json` e permite alternar a paleta (viridis ↔ inferno).  
- Scripts de visualização aceitam `--date` e tratam `sys.path` automaticamente.  
- `compare_probability_vs_truecolor_interactive.py --date YYYY-MM-DD` gera dashboards Plotly leves com MODIS, SST, gradiente e probabilidade.

---

## 🧰 Pipeline
1. **01_search_download.py** – login Earthdata (`~/.netrc`) e download dos granules configurados (SST + MODIS CHL).  
2. **02_preprocess.py** – recorte da bbox, conversão para °C (SST), cálculo de gradiente e exportação `_proc.nc` (inclui `chlor_a` quando disponível).  
3. **03_feature_engineering.py** – gera CSVs por data com `lat`, `lon`, `date`, `sst`, `sst_gradient`, `chlor_a`.  
4. **04_train_model.py** – agrega features, rotula hotspots (top-N% gradiente) e treina XGBoost (`dataset.csv`, `model_xgb.pkl`, `metrics.json`).  
5. **05_export_tiles.py** – interpola `chlor_a` para a grade de SST, aplica o modelo e salva GeoTIFFs (`data/tiles/hotspots_probability_*.tif`).  
6. **scripts/utils/build_tiles_manifest.py** – gera `data/tiles/tiles_manifest.json` consumido pelo app Leaflet.

### Status atual (2025-09-30)
- ✅ Download (SST MUR + MODIS CHL, intervalo configurável)  
- ✅ Pré-processamento (gradiente com `xarray`, preservando `time`)  
- ✅ Features tabulares combinadas (SST, gradiente, clorofila)  
- ✅ Treino baseline (XGBoost, métricas em `data/processed/metrics.json`)  
- ✅ GeoTIFFs + manifest (`data/tiles/*.tif`, `tiles_manifest.json`)  
- 🟡 Integração de novas variáveis (correntes ECCO, SWOT)  
- ⚪ Tag eletrônica (design conceitual – falta protótipo)

---

## 🛰️ Conjuntos de Dados
| Dataset | Variável principal | Resolução | Relevância | Uso atual |
|---------|-------------------|-----------|------------|-----------|
| **MUR SST** | 🌡️ Temperatura da superfície | ~1 km / diário | Detecta frentes térmicas | Feature principal
| **MODIS CHL** | 🟢 Clorofila-a | ~4 km / diário-semanal | Proxy de produtividade | Integrada (feature `chlor_a`)
| **PACE OCI** | 🌈 Composição fitoplâncton | ~1 km / diário | Qualidade do alimento | Planejado
| **ECCO** | 🌀 Correntes u/v | ~10–20 km / diário | Transporte de nutrientes/presas | Planejado
| **SWOT** | 🌊 Topografia / redemoinhos | ~1 km / 21 dias | Concentração de presas | Planejado

---

## 🌐 Cadeia trófica (inspiração)
```
🌱 Fitoplâncton (PACE / MODIS)
   ↓
🐟 Peixes (correntes ECCO)
   ↓
🌀 Frentes/redemoinhos (SWOT + gradiente SST)
   ↓
🦈 Tubarões (modelados via ML)
```

---

## ⚙️ Configuração do ambiente
```powershell
# Ambiente virtual
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Credenciais Earthdata (~/.netrc)
machine urs.earthdata.nasa.gov
login SEU_USUARIO
password SUA_SENHA
```

## ▶️ Execução da pipeline
```powershell
python scripts/01_search_download.py
python scripts/02_preprocess.py
python scripts/03_feature_engineering.py
python scripts/04_train_model.py
python scripts/05_export_tiles.py
python scripts/utils/build_tiles_manifest.py
```

## 🗺️ App web
```powershell
python -m http.server 8000
# abrir http://localhost:8000/app/index.html
```
Dropdown lista as datas disponíveis e o botão alterna a paleta.

---

## 📸 Visualizações úteis
- `scripts/visualization/compare_probability_vs_truecolor.py --date YYYY-MM-DD`
- `scripts/visualization/compare_probability_vs_truecolor_interactive.py --date YYYY-MM-DD`
- `scripts/visualization/compare_side_by_side_slider.py`

---

## 📌 Próximos passos
- Integrar correntes ECCO e SWOT ao pipeline (02→03→04).  
- Refinar rótulo com dados de presença/ausência reais (telemetria, pesca).  
- Adicionar retries/cache aos downloads MODIS (WMS) e testes automatizados (`tests/data/`).  
- Evoluir a tag proposta para protótipo físico e integração com o modelo.

---

## 🧵 Storytelling
Tubarões são **indicadores de saúde oceânica**. A pipeline transforma dados de satélite em mapas acionáveis; a tag proposta fecha o ciclo com validação em campo; e os dashboards ajudam a contar a história de ponta a ponta.
