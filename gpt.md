# GPT Notes — Tubarões do Espaço

## Visão Atual do Projeto
- Objetivo central: mapear hotspots de alimentação de tubarões combinando dados de satélite NASA, engenharia de features e ML para suporte à conservação.
- Pipeline principal: `01_search_download.py` → `02_preprocess.py` → `03_feature_engineering.py` → (futuro) `04_train_model.py` → `05_export_tiles.py`, complementado por scripts de visualização e comparações MODIS.
- Estrutura de dados: NetCDF brutos em `data/raw/`, processados com gradiente em `data/processed/`, tabelas em `data/features/`, visualizações em `data/compare/` e `data/viz/`.
- Stack: Python, xarray, pandas, Plotly, earthaccess, XGBoost (planejado) e Leaflet para o app web.

## Registro de Atualizações
- 2025-09-29 — Centralizei carregamento de config em `scripts/utils_config.py:1` e refiz `scripts/compare_side_by_side_slider.py:1` para usar bbox do config, tratar coordenadas e embutir MODIS em base64.
- 2025-09-29 — Atualizei `scripts/02_preprocess.py:1` para manter coordenadas padrão, preservar dimensão `time` e gerar gradiente por passo temporal.
- 2025-09-29 — Ajustei `scripts/03_feature_engineering.py:1` para converter NetCDF em DataFrame mantendo `lat`, `lon` e `date` a partir da coordenada `time`.

> Sempre que novas modificações forem feitas, documentar aqui com data, resumo e arquivos tocados.


- 2025-09-29 � Atualizei docs/visao_geral_e_melhorias.md:1 com resumo das mudancas recentes e novas prioridades.

- 2025-09-29 � Atualizei scripts/04_train_model.py:1 para consolidar features, rotular hotspots e treinar o modelo; scripts/05_export_tiles.py:1 agora gera GeoTIFFs direto dos NetCDF; revisei docs/visao_geral_e_melhorias.md:1 com o novo fluxo.

- 2025-09-29 � Otimizei scripts/compare_side_by_side_slider.py:1 reduzindo resolu��o MODIS e reamostrando os grids (HTML bem menor).

- 2025-09-29 � Reescrevi scripts/compare_side_by_side_slider.py:1 inteiro em ASCII para evitar erros de encoding, mantendo as otimiza��es.

- 2025-09-29 � Reimplementei scripts/compare_side_by_side_slider.py:1 com subplots (go.Image + Heatmap) e slider, replicando o visual do compare_modis_truecolor para todas as datas.

- 2025-09-29 � Inclu� rioxarray em requirements.txt:1 para suportar scripts/05_export_tiles.py.

- 2025-09-29 � Ajustei scripts/05_export_tiles.py:1 para usar dimens�es y/x e sanitizar timestamps (sem dois-pontos) antes de gravar GeoTIFF.

- 2025-09-29 � Atualizei app/index.html para carregar GeoTIFFs com leaflet-geotiff (dropdown + toggle SST/Gradiente).

- 2025-09-29 � Adicionei scripts/compare_probability_vs_truecolor.py:1 para comparar MODIS, SST e os GeoTIFFs gerados.

- 2025-09-29 � Criei scripts/compare_probability_vs_truecolor_interactive.py:1 para gerar dashboard Plotly com MODIS/SST/gradiente/probabilidade (hover interativo).

- 2025-09-29 � Reescrevi scripts/compare_probability_vs_truecolor_interactive.py:1 com argumento --date (um dia) e MODIS reduzido para 512px, gerando HTML leve.

- 2025-09-29 � Ajustei dashboard interativo (colwidths e largura) para legendas n�o sobreporem os mapas.
- 2025-09-30 - Normalizei imports para `scripts.utils`, reescrevi `scripts/02_preprocess.py:1`, atualizei `requirements.txt:1`, criei `scripts/utils/build_tiles_manifest.py:1` e ajustei `app/index.html:1` para manifest dinamico.
- 2025-09-30 - Adicionei `guia_equipa_tubaroes.md:1` para orientar a equipe nao tecnica sobre o fluxo e entregaveis.
- 2025-09-30 - Convertei os briefs para Markdown (`docs/desafio_projeto.md:1`, `docs/recursos_projeto.md:1`) e atualizei referencias.

- 2025-09-30 - Atualizei config/config.yaml:1 para 2025-09-20→2025-09-25, rerodei scripts 01-05 e gerei novos CSVs/GeoTIFFs + tiles_manifest.json.
- 2025-09-30 - Ajustei scripts pipeline/visualizacao com fallback de sys.path e argumento --date em compare_probability_vs_truecolor.py:1 (sem ModuleNotFoundError).


- 2025-09-30 — Integrei MODIS CHL: 01_search_download.py agora baixa clorofila, 02_preprocess.py processa chlor_a, 03_feature_engineering.py combina SST/gradiente/CHL e 05_export_tiles.py interpola CHL na hora da previsão.

- 2025-09-30 — Ajustei config/config.yaml:1 para `max_granules_per_source = 1` a fim de evitar múltiplos granules MODIS por dia.

- 2025-09-30 — Voltei `max_granules_per_source` para 10 e dedupei MODIS L3 CHL no download (um granule por dia via temporal).

- 2025-09-30 — Ajustei deduplicação de MODIS L3 CHL usando `granule.filename`; se não houver `temporal`, gera chave pelo nome do arquivo.

- 2025-09-30 — Ajustei config/config.yaml:1 para usar datas ISO (YYYY-MM-DDTHH:MM:SS) garantindo cobertura exata dos dias 26-27/set.

- 2025-09-30 — Otimizei scripts/02_preprocess.py:1 (uso de chunks Dask, sem .load(), float32 + compressão) para reduzir memória.
