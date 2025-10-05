# 🦈 Tubarões do Espaço | NASA Space Apps Challenge 2025

Projeto desenvolvido para o desafio **Tubarões do Espaço** do NASA Space Apps Challenge 2025. Nosso objetivo é **prever habitats de alimentação de tubarões** combinando dados de satélite (SST, PACE, SWOT), modelagem matemática, *machine learning* e visualização interativa — junto de um conceito de **tag eletrônica inteligente** que reforça o modelo com dados coletados “no tubarão”.

---

## 🚀 Resumo executivo
- **Dados NASA → frentes oceânicas → hotspots de alimentação.**  
- Pipeline automatizada gera **mapas diários de probabilidade** + **dashboards interativos**.  
- **Tag eletrônica** proposta adiciona contexto comportamental e valida o modelo em campo.

---

## 🌍 Impacto esperado
- 🌱 **Conservação marinha** – priorizar áreas críticas e rotas migratórias.  
- 🎣 **Pesca sustentável** – reduzir captura acidental e conflitos pesca × biodiversidade.  
- 🧑‍🎓 **Educação científica** – contar a história de forma acessível para o público geral.  
- 🛰️ **Valorizar dados NASA** – ciência espacial aplicada a desafios costeiros do dia a dia.

---

## 🧭 Glossário rápido
- **SST (Temperatura da Superfície do Mar):** “quente/frio” da pele do oceano; tubarões usam **frentes térmicas** (mudanças bruscas) como pistas de caça.  
- **Frente oceânica:** linha de encontro entre águas de temperaturas diferentes que **concentra alimento**.  
- **Redemoinho (*eddy*):** “carrossel” de água que **aprisiona nutrientes e presas**; detectável por topografia do mar.  
- **Clorofila‑a:** pigmento das micro‑algas (**fitoplâncton**); indica **produtividade** (comida para toda a cadeia).  
- **PACE / OCI:** satélite + sensor que enxergam **cores do oceano** para estimar **tipos de fitoplâncton** (*PFTs*).  
- **PFTs (Tipos Funcionais de Fitoplâncton):** grupos como **Prochlorococcus, Synechococcus, picoeucariotos**; ajudam a inferir a **qualidade do cardápio** para peixes e, indiretamente, tubarões.  
- **SWOT (SSH):** mede **altura da superfície do mar** em alta resolução ⇒ revela **frentes e redemoinhos**.  
- **Hotspot de alimentação:** pixel com **alta probabilidade** de presença/forrageamento.

---

## 🗂️ Estrutura do repositório
```
Tubaroes_do_Espaco/
│
├── config/
│   └── config.yaml               # BBox, janela temporal, nomes de datasets
│
├── data/
│   ├── raw/                      # NetCDF brutos (MUR, MODIS, PACE, SWOT, ECCO)
│   ├── processed/                # NetCDF recortados, dataset.csv, modelo, métricas
│   ├── features/                 # CSVs tabulares para ML
│   ├── tiles/                    # GeoTIFFs hotspots_probability_*.tif + tiles_manifest.json
│   └── compare/                  # PNG/HTML usados no storytelling
│
├── scripts/
│   ├── 01_search_download.py     # Busca no Earthdata (usa config)
│   ├── 02_preprocess.py          # Recorte, conversões, gradientes (preserva `time`)
│   ├── 03_feature_engineering.py # Tabelas (lat, lon, date, sst, grad, chlor_a, ...)
│   ├── 04_train_model.py         # Concatena, rotula hotspots, treina XGBoost
│   ├── 05_export_tiles.py        # Aplica o modelo e exporta GeoTIFFs de probabilidade
│   ├── utils/                    # load_config, project_root, build_tiles_manifest.py
│   └── visualization/            # Inspeções rápidas + comparações (PNG/Plotly)
│
├── app/                          # Mapa Leaflet (lê o manifest JSON de tiles)
├── docs/                         # Briefing, visão geral/melhorias, guia rápido
└── tag/                          # Conceito de tag embarcada
```
---

## 🧰 Pipeline (de ponta a ponta)
1. **01_search_download.py** – login Earthdata (`~/.netrc`) e download dos *granules* configurados (SST + MODIS CHL; PACE/SWOT/ECCO quando habilitados).  
2. **02_preprocess.py** – recorte da BBox, conversão para °C (SST), cálculo de **gradiente**, *masks* e exportação `_proc.nc` (inclui `chlor_a` quando disponível).  
3. **03_feature_engineering.py** – gera CSVs por data com `lat`, `lon`, `date`, `sst`, `sst_gradient`, `chlor_a` (+ campos PACE/SWOT/ECCO quando habilitados).  
4. **04_train_model.py** – agrega features, rotula **hotspots** (ex.: top‑N% do gradiente) e treina **XGBoost** (`dataset.csv`, `model_xgb.pkl`, `metrics.json`).  

---

## 🛰️ Conjuntos de dados NASA
| Dataset | Variável Principal | Resolução Espacial / Temporal | Por que é importante para tubarões? | Uso no modelo |
|---------|-------------------|-------------------------------|--------------------------------------|---------------|
| **SST (MUR)** | 🌡️ Temperatura da Superfície do Mar | ~1 km / diário | Define preferências térmicas e frentes oceânicas (hotspots de caça). | Base principal para identificar frentes térmicas. |
| **MODIS L3 CHL** | 🟢 Clorofila-a (biomassa fitoplâncton) | ~4 km / diário-semanal | Indica produtividade biológica (cadeia alimentar: plâncton → peixes → tubarões). | Variável biológica chave para prever disponibilidade de presas. |
| **PACE OCI** | 🌈 Composição do fitoplâncton (cores do oceano) | ~1 km / diário | Diferencia tipos de plâncton (nutritivos vs tóxicos). | Enriquecimento do modelo, explicando qualidade da comida disponível. |
| **SWOT** | 🌊 Topografia da superfície / Redemoinhos | ~1 km / repetição 21 dias | Detecta estruturas de mesoescala (eddies) que concentram alimento. | Identifica hotspots estruturais que atraem predadores. |

> **Tradução prática:** PACE/MODIS mostram **onde nasce o alimento**; SST/gradiente indica **onde ele se concentra**; SWOT mostra **como ele é empurrado/retido**; o modelo converte esses sinais em **hotspots**.

---

## 🌐 Cadeia trófica (inspiração conceitual)
```
🌱 Fitoplâncton (PACE / MODIS)
   ↓
🐟 Peixes (correntes)
   ↓
🌀 Frentes/Redemoinhos (SWOT + gradiente SST)
   ↓
🦈 Tubarões (probabilidade via ML)
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

## ▶️ Executando a pipeline
```powershell
python scripts/01_search_download.py
python scripts/02_preprocess.py
python scripts/03_feature_engineering.py
python scripts/04_train_model.py
python scripts/05_export_tiles.py
python scripts/utils/build_tiles_manifest.py
```
---

## 📸 Visualizações úteis
- `scripts/visualization/compare_probability_vs_truecolor.py --date YYYY-MM-DD`  
- `scripts/visualization/compare_probability_vs_truecolor_interactive.py --date YYYY-MM-DD`  
- `scripts/visualization/compare_side_by_side_slider.py`

---

## 📦 Saídas e produtos gerados
Nosso pipeline conecta sensores orbitais, modelagem e storytelling. Principais artefatos:

- **`data/features/*.csv`** – tabelas diárias usadas nos modelos. Cada linha (um ponto `lat`, `lon`) inclui, por exemplo:  
  - `sst`, `sst_gradient`: temperatura da superfície (MUR) e intensidade da frente térmica;  
  - `chlor_a_modis`, `chlor_a_pace`, `chlor_a`: clorofila‑a por MODIS Aqua (L3) e PACE OCI;  
  - `ssh_swot`, `ssh_swot_gradient`, `swot_mask`: topografia e gradiente (SWOT) destacando redemoinhos/estruturas;  
  - `moana_prochlorococcus`, `moana_synechococcus`, `moana_picoeuk`: abundâncias celulares (PACE/MOANA);  
  - `moana_total_cells`, `moana_picoeuk_share`, `moana_cyanobacteria_share`, `moana_diversity_index`: métricas derivadas de **biomassa**, **composição** e **diversidade** fitoplanctônica.  
  > *Obs.: os nomes exatos podem variar conforme a versão do produto; mantemos mapeamento no `config.yaml`.*

- **`data/processed/`** – intermediários e resultados de ML:  
  - `_proc.nc` de SST, CHL, PACE/MOANA e SWOT (recortes comprimidos);  
  - `dataset.csv`, `model_xgb.pkl`, `metrics.json` (AUC, *Average Precision*).

- **`data/predictions/*.csv`** – saídas do modelo heurístico por espécie (quando habilitado):  
  - `habitat_score` (0–1): combina gradiente térmico, temperatura, clorofila, estrutura SWOT e métricas MOANA;  
  - `habitat_class` (poor/moderate/good/excellent);  
  - `is_hotspot` (top 10% dos pixels);  
  - PNGs em `data/predictions/*_map.png` e `*_comparative_map.png` com hotspots marcados.

- **`data/viz/compare`** e **`data/viz/moana`** – painéis PNG gerados pelos scripts de visualização:  
  - `compare_truecolor_moana_YYYY-MM-DD.png`: MODIS True Color + camadas MOANA;  
  - `compare_all_variables_YYYY-MM-DD.png`: True Color, SST, gradiente, clorofila, SWOT e *score* de fitoplâncton.

### Por que cada variável importa?
| Variável | Origem | Papel ecológico |
|---|---|---|
| `sst`, `sst_gradient` | MUR SST (GHRSST) | Identificam **frentes térmicas** que concentram presas e oxigênio. |
| `chlor_a_*` | MODIS Aqua & PACE OCI | **Produtividade primária** (onde a base da cadeia está ativa). |
| `ssh_swot`, `ssh_swot_gradient` | SWOT | **Redemoinhos e meandros** que agregam nutrientes/presas. |
| `moana_*` (Prochlorococcus, Synechococcus, picoeuk.) | PACE OCI / MOANA | Diferenciam **tipos de fitoplâncton** (qualidade nutricional). |
| `moana_total_cells`, `moana_picoeuk_share`, `moana_diversity_index` | Derivadas MOANA | **Biomassa**, **proporção** e **diversidade** (resiliência). |
| `habitat_score`, `habitat_class`, `is_hotspot` | Modelo heurístico/ML | Tradução dos sinais ambientais em **decisão prática** para conservação e planejamento. |

---

## 🛰️ Missões e satélites utilizados
- **MUR SST (GHRSST, JPL/MEaSUREs)** – temperaturas diárias de alta resolução.  
- **MODIS Aqua (NASA EOS PM)** – clorofila‑a histórica (L3) para monitorar produtividade.  
- **PACE OCI (Plankton, Aerosols, Clouds and Ecosystems)** – espectrorradiometria para distinguir **comunidades fitoplanctônicas** (MOANA/PFTs).  
- **SWOT (Surface Water and Ocean Topography)** – altimetria de alta resolução para **frentes/redemoinhos**.  

> Cada missão cobre uma peça da cadeia trófica: PACE/MODIS mostram **a oferta**; SWOT mostra **a física que redistribui**; MUR SST mostra **as pistas**; o modelo junta tudo em **mapas acionáveis**.
---

## 🧵 Storytelling
Tubarões são **indicadores de saúde oceânica**. Nossa pipeline transforma **dados de satélite** em **mapas de decisão**; a **tag eletrônica** fecha o ciclo com validação em campo; e os **dashboards** contam a história, do espaço até o mar, de uma forma que qualquer pessoa consegue acompanhar.

---

### Contatos e créditos
Time **Iron Shark** – NASA Space Apps 2025.  
Agradecimentos às equipes das missões **PACE, SWOT, MODIS, GHRSST/MUR** e ao projeto **ECCO** pelos dados abertos.
