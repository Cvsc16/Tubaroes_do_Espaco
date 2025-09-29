# GPT Notes â€” TubarÃµes do EspaÃ§o

## VisÃ£o Atual do Projeto
- Objetivo central: mapear hotspots de alimentaÃ§Ã£o de tubarÃµes combinando dados de satÃ©lite NASA, engenharia de features e ML para suporte Ã  conservaÃ§Ã£o.
- Pipeline principal: `01_search_download.py` â†’ `02_preprocess.py` â†’ `03_feature_engineering.py` â†’ (futuro) `04_train_model.py` â†’ `05_export_tiles.py`, complementado por scripts de visualizaÃ§Ã£o e comparaÃ§Ãµes MODIS.
- Estrutura de dados: NetCDF brutos em `data/raw/`, processados com gradiente em `data/processed/`, tabelas em `data/features/`, visualizaÃ§Ãµes em `data/compare/` e `data/viz/`.
- Stack: Python, xarray, pandas, Plotly, earthaccess, XGBoost (planejado) e Leaflet para o app web.

## Registro de AtualizaÃ§Ãµes
- 2025-09-29 â€” Centralizei carregamento de config em `scripts/utils_config.py:1` e refiz `scripts/compare_side_by_side_slider.py:1` para usar bbox do config, tratar coordenadas e embutir MODIS em base64.
- 2025-09-29 â€” Atualizei `scripts/02_preprocess.py:1` para manter coordenadas padrÃ£o, preservar dimensÃ£o `time` e gerar gradiente por passo temporal.
- 2025-09-29 â€” Ajustei `scripts/03_feature_engineering.py:1` para converter NetCDF em DataFrame mantendo `lat`, `lon` e `date` a partir da coordenada `time`.

> Sempre que novas modificaÃ§Ãµes forem feitas, documentar aqui com data, resumo e arquivos tocados.


- 2025-09-29 — Atualizei docs/visao_geral_e_melhorias.md:1 com resumo das mudancas recentes e novas prioridades.

- 2025-09-29 — Atualizei scripts/04_train_model.py:1 para consolidar features, rotular hotspots e treinar o modelo; scripts/05_export_tiles.py:1 agora gera GeoTIFFs direto dos NetCDF; revisei docs/visao_geral_e_melhorias.md:1 com o novo fluxo.

- 2025-09-29 — Otimizei scripts/compare_side_by_side_slider.py:1 reduzindo resolução MODIS e reamostrando os grids (HTML bem menor).

- 2025-09-29 — Reescrevi scripts/compare_side_by_side_slider.py:1 inteiro em ASCII para evitar erros de encoding, mantendo as otimizações.

- 2025-09-29 — Reimplementei scripts/compare_side_by_side_slider.py:1 com subplots (go.Image + Heatmap) e slider, replicando o visual do compare_modis_truecolor para todas as datas.

- 2025-09-29 — Incluí rioxarray em requirements.txt:1 para suportar scripts/05_export_tiles.py.

- 2025-09-29 — Ajustei scripts/05_export_tiles.py:1 para usar dimensões y/x e sanitizar timestamps (sem dois-pontos) antes de gravar GeoTIFF.

- 2025-09-29 — Atualizei app/index.html para carregar GeoTIFFs com leaflet-geotiff (dropdown + toggle SST/Gradiente).

- 2025-09-29 — Adicionei scripts/compare_probability_vs_truecolor.py:1 para comparar MODIS, SST e os GeoTIFFs gerados.

- 2025-09-29 — Criei scripts/compare_probability_vs_truecolor_interactive.py:1 para gerar dashboard Plotly com MODIS/SST/gradiente/probabilidade (hover interativo).

- 2025-09-29 — Reescrevi scripts/compare_probability_vs_truecolor_interactive.py:1 com argumento --date (um dia) e MODIS reduzido para 512px, gerando HTML leve.

- 2025-09-29 — Ajustei dashboard interativo (colwidths e largura) para legendas não sobreporem os mapas.
