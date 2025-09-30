# Tubaroes do Espaco - Visao Geral e Melhorias

Este documento resume, de forma pratica, o que o projeto faz hoje, como a pipeline esta organizada e quais melhorias priorizadas podem elevar a robustez, utilidade cientifica e a experiencia de uso.

---

## Atualizacoes recentes (2025-09-30)

- Padronizei scripts para usar `scripts.utils` (load_config/project_root) e evitar caminhos relativos duplicados.
- Reescrevi `scripts/02_preprocess.py:1` preservando a dimensao `time`, gradiente com `xr.apply_ufunc` e metadados de bbox.
- Atualizei `requirements.txt:1` em ASCII e inclui `requests`, `pillow`, `cartopy` e `joblib`.
- Criei `scripts/utils/build_tiles_manifest.py:1` para gerar `data/tiles/tiles_manifest.json` e alimentar o app.
- Ajustei `app/index.html:1` para carregar tiles dinamicamente e lidar com ausencia de GeoTIFFs.
- Convertei os briefs para Markdown (`docs/desafio_projeto.md:1`, `docs/recursos_projeto.md:1`).
- Criei `guia_equipa_tubaroes.md:1` como explicacao nao tecnica para a equipe.

---

## Atualizacoes recentes (2025-09-29)

- Criei `scripts/utils_config.py:1` para centralizar o carregamento de `config/config.yaml`.
- Refatorei `scripts/compare_side_by_side_slider.py:1` para usar a bbox do config, embutir imagens MODIS em base64 e aceitar diferentes nomes de coordenadas.
- Ajustei `scripts/02_preprocess.py:1` para padronizar lat/lon, preservar a dimensao temporal e calcular gradiente por timestep.
- Atualizei `scripts/03_feature_engineering.py:1` para gerar DataFrames com colunas `lat`, `lon` e `date` a partir do eixo `time` quando presente.
- Revisei `scripts/04_train_model.py:1` para consolidar as features, rotular hotspots via top-percent de gradiente e treinar o modelo.
- Reescrevi `scripts/05_export_tiles.py:1` para aplicar o modelo diretamente aos NetCDF processados e gerar GeoTIFFs em `data/tiles/`.

---

## Visao Geral

- Objetivo: prever e visualizar hotspots de alimentacao de tubaroes a partir de dados de satelite da NASA, unindo processamento cientifico, feature engineering, ML e visualizacao interativa.
- Entradas: colecoes NASA (ex.: MUR SST, MODIS CHL, PACE, ECCO, SWOT), baixadas via Earthdata.
- Saidas: arquivos processados (NetCDF), tabelas de features (CSV), graficos de pre-visualizacao (PNG/HTML) e raster/tiles de probabilidade (GeoTIFF).
- Publico-alvo: cientistas, gestores ambientais, educadores e publico geral interessado em conservacao marinha.

---

## Como Funciona (Pipeline)

1) Busca e download
- `scripts/01_search_download.py:1` faz login no Earthdata (netrc) e busca/baixa granules conforme `config/config.yaml:1` (bbox e periodo). Hoje habilitado principalmente para SST MUR.

2) Pre-processamento cientifico
- `scripts/02_preprocess.py:1` recorta por `bbox`, converte SST para graus Celsius quando necessario e calcula gradiente espacial (proxy de frentes oceanicas). Preserva a dimensao `time` quando existente e salva em `data/processed/` (NetCDF) com variaveis `sst` e `sst_gradient`.

3) Feature engineering tabular
- `scripts/03_feature_engineering.py:1` converte os NetCDF processados em tabelas (lat, lon, date, sst, sst_gradient) e salva em `data/features/`.

4) Treinamento de modelo (baseline)
- `scripts/04_train_model.py:1` agrega as features, rotula hotspots como o top-N por cento de gradiente por dia, salva `data/processed/dataset.csv` e treina XGBoost (fallback para GradientBoosting).

5) Exportacao de produtos para mapas
- `scripts/05_export_tiles.py:1` carrega o modelo treinado, aplica aos NetCDF processados (por data/time) e gera GeoTIFFs de probabilidade em `data/tiles/`.

6) Visualizacao e inspecoes
- Estatico: `scripts/check_processed.py:1` gera PNGs (`data/sst_preview.png`, `data/sst_gradient_preview.png`).
- Interativo: `scripts/check_processed_interactive.py:1` e `scripts/extras/plot_features_interactive.py:1` geram HTMLs.
- Comparacoes MODIS vs cientifico: `scripts/compare_modis_truecolor.py:1`, `scripts/compare_side_by_side_slider.py:1` e backups em `scripts/backups/`.
- Web app (esqueleto): `app/index.html:1` com Leaflet para sobrepor tiles (quando disponiveis).

---

## Estrutura de Pastas (resumo)

- `config/config.yaml:1` - parametros de bbox, datas e colecoes.
- `data/raw/` - NetCDF brutos baixados.
- `data/processed/` - NetCDF recortados + gradiente, dataset consolidado, modelo e metricas.
- `data/features/` - CSVs tabulares para ML.
- `data/tiles/` - GeoTIFFs de probabilidade gerados pelo modelo.
- `data/viz/` e `data/compare/` - saidas HTML/PNG interativas e comparativas.
- `app/index.html:1` - base para o mapa web (Leaflet).
- `requirements.txt:1` - dependencias Python.

---

## O que ja funciona

- Download automatizado de MUR SST conforme bbox/tempo.
- Recorte e calculo de gradiente com previews (PNG/HTML) preservando a dimensao temporal.
- Geracao de features tabulares (SST + gradiente) por arquivo/dia com coluna de data vinda do NetCDF.
- Consolida o dataset para treino com rotulo heuristico baseado em gradiente e treina o modelo base (AUC/AP salvos).
- Comparacoes MODIS vs cientifico com slider `scripts/compare_side_by_side_slider.py:1` usando configuracao centralizada e imagens embutidas offline.
- Exporta GeoTIFFs de probabilidade diretamente dos NetCDF processados em `data/tiles/`.

---

## Lacunas e pontos de atencao (observados no codigo)

- Rotulo heuristico simplificado
  - `scripts/04_train_model.py:1` usa top-N por cento do gradiente como proxy de hotspot. Ideal substituir por dados reais de presenca ou outra logica fisica.

- Features limitadas
  - O modelo usa apenas `sst` e `sst_gradient`. Faltam variaveis biologicas/dinamicas (CHL, correntes, SWOT) ja previstas na configuracao.

- Gradiente em graus (nao em metros)
  - `scripts/02_preprocess.py:1` calcula `np.gradient` no grid geografico. Para comparar intensidade fisica de frentes, considerar correcao pela escala graus-km ou reamostrar para projecao metrica antes do gradiente.

- Fail-safe de rede para MODIS
  - Scripts que baixam WMS (MODIS True Color) usam cache, mas ainda nao tem retry exponencial nem tempo de espera configuravel.

- Normalizacao de coordenadas legado
  - Outros scripts (ex.: `scripts/backups/compare_scientific_vs_truecolor_slider.py:1`) ainda assumem `lon/lat` e nao usam `utils_config.py:1`.

---

## Melhorias recomendadas (priorizadas)

Curto prazo (alto impacto, baixo/medio esforco)
- Refinar o rotulo de treino com dados reais de presenca/ausencia ou outra heuristica fisica (ex.: combinar gradiente + SST + CHL).
- Atualizar scripts de comparacao remanescentes para ler configuracao via `utils_config.py:1`, tratar nomes de coordenadas e aplicar base64 real.
- Adicionar retry/backoff configuravel nos downloads MODIS e opcao de reutilizar cache sem erro.

Medio prazo
- Integrar variaveis adicionais: incorporar CHL (MODIS/PACE), correntes (ECCO) e metricas de mesoescala (SWOT) na cadeia `02_preprocess.py:1` -> `03_feature_engineering.py:1` -> `04_train_model.py:1`.
- Ajustar gradiente para escala metrica: reprojetar temporariamente para CRS metrica ou corrigir pela latitude ao calcular gradiente.
- Parametrizar scripts via CLI: permitir `--start`, `--end`, `--bbox`, `--maxn` para facilitar reuso.
- Gerar produtos para o app web (tiles vectorizados, legendas) e integrar no `app/index.html:1`.

Longo prazo
- Modelagem avancada e incerteza: calibrar/validar com trilhas reais de tubaroes, adicionar incerteza e avaliacao espacial/temporal.
- Processamento distribuido: Dask para chunking de arrays e paralelizacao em multiplos dias/variaveis.
- Testes automatizados: pequenos arquivos de exemplo em `tests/data/` e suites de regressao (ex.: conferir dimensoes, ranges, existencia de saidas esperadas).
- Empacotamento e reprodutibilidade: `pyproject.toml`, travamento de versoes e/ou container leve (ex.: conda-lock, uv, micromamba ou Docker slim).
- Web app completo: servidor de tiles (ou uso de bibliotecas como `georaster-layer-for-leaflet`), seletores de data/variavel e camadas tematicas.

---

## Dicas de uso e execucao

- Ambiente: ver `README.md:1` para criacao do venv e instalacao.
- Ordem tipica: `01_search_download.py` -> `02_preprocess.py` -> `03_feature_engineering.py` -> inspeccoes/plots -> `04_train_model.py` -> `05_export_tiles.py`.
- Saidas prontas no repositorio: ha exemplos em `data/compare/`, `data/viz/` e `data/tiles/` que ajudam a validar a experiencia interativa sem rodar toda a pipeline.

---

## Observacoes finais

- O projeto esta bem encaminhado na ingestao/pre-processamento, consolidacao de dataset e visualizacoes iniciais. As maiores lacunas agora estao na qualidade do rotulo, na incorporacao de variaveis adicionais e na robustez operacional.
- A priorizacao sugerida foca em robustez (config centralizada, rotulos melhores, coordenacao de variaveis) e em aproximar o produto final (GeoTIFF/tiles + app Leaflet) de um fluxo pronto para uso publico.
