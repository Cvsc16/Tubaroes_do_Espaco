# Tubaroes do Espaco – Dossie Tecnico de Entrega (NASA Space Apps 2025)

## Visao Geral

Nossa equipe construiu uma estrutura completa para prever hotspots de alimentacao de tubaroes a partir de sensores da NASA. O fluxo vai do download automatico dos granulos ate a geracao de mapas interativos e um modelo heuristico por especie. A etapa de treinamento supervisionado ainda depende de telemetria confiavel; justificamos o gap com a pouca representatividade dos rastros publicos (por exemplo, OCEARCH recente cobre poucas especies) e mostramos como a tag proposta vai fechar o ciclo com dados em tempo quase real.

## Satelites e Produtos NASA Utilizados

| Missao / Produto | Variavel extraida | Resolucao / Cadencia | Uso na pipeline |
|------------------|-------------------|----------------------|-----------------|
| **GHRSST / MUR SST** | Temperatura da superficie (`sst`) | ~1 km, diaria | Base fisica do modelo, pivot para gradiente e co-registro das demais camadas. |
| **GHRSST / MUR SST** | Gradiente termico (`sst_gradient`) | derivado | Localiza frentes que concentram presas e oxigenio. |
| **MODIS Aqua (L3)** | Clorofila-a (`chlor_a_modis`) | ~4 km, diario/semanal | Proxy historico de produtividade primária; compoe a coluna `chlor_a`. |
| **PACE OCI (MOANA)** | Abundancia celular por grupo (`moana_prococcus_moana`, `moana_syncoccus_moana`, `moana_picoeuk_moana`) | ~1 km, diaria | Diferencia comunidades fitoplanctonicas; traduzimos em biomassa e composicao. |
| **PACE OCI (MOANA)** | Derivados (`moana_total_cells`, `moana_picoeuk_share`, `moana_cyanobacteria_share`, `moana_diversity_index`) | derivado | Quantifica a qualidade do “cardapio” disponivel aos predadores. |
| **PACE OCI (L3)** | Clorofila-a (`chlor_a_pace`) | ~1 km, diaria | Complementa MODIS; alimenta a coluna `chlor_a`. |
| **SWOT L2 LR SSH** | Altura da superficie do mar (`ssh_swot`) | ~1 km, ciclo 21 dias | Identifica redemoinhos e estruturas de mesoescala que agregam presas. |
| **SWOT L2 LR SSH** | Gradiente de SSH (`ssh_swot_gradient`) e mascara de cobertura (`swot_mask`) | derivado | Mede intensidade das estruturas e garante que apenas pontos com observacao sejam usados. |

> **Planejados**: ECCO (correntes u/v) e batimetria para enriquecer o modelo supervisionado em versoes futuras.

## Pipeline Tecnica

1. **Busca e download** – `scripts/01_search_download.py` usa `earthaccess` com configuracoes centralizadas (`config/config.yaml`).
2. **Pre-processamento** – `scripts/02_preprocess.py` recorta o BBOX, converte SST para Celsius, calcula gradiente com Dask, extrai produtos MOANA e agrega pontos SWOT.
3. **Engenharia de features** – `scripts/03_feature_engineering.py` interpola todos os produtos na grade da SST, combina MODIS/PACE, deriva metricas MOANA e exporta CSVs (`data/features/YYYYMMDD_features.csv`).
4. **Modelos**
   - **Heuristico por especie** – `scripts/04_shark_habitat_model.py` aplica pesos fisico-biologicos (gradiente, temperatura, clorofila, SWOT, biomassa MOANA) e gera `habitat_score`, `habitat_class` e `is_hotspot` para white/blue/tiger sharks.
   - **Baseline supervisionado** – `scripts/ml/04_train_model.py` prepara `data/processed/dataset.csv` e um XGBoost (`model_xgb.pkl`), treinado hoje com rotulo heuristico. Mantemos o codigo pronto para receber rótulos reais quando a telemetria estiver disponivel.
5. **Tiles e visualizacoes** – `scripts/ml/05_export_tiles.py` cria GeoTIFFs de probabilidade para o app Leaflet (`app/index.html`); scripts em `scripts/visualization/` geram paineis comparativos, incluindo MOANA vs True Color (`plot_moana_truecolor.py`) e o painel consolidado (`compare/plot_all_variables.py`).
6. **Validacao externa** – `scripts/05_validate_with_obis.py` cruza previsoes com OBIS/GBIF para relatorios exploratorios (Hit@hotspot, lift). Reforcamos que a amostragem ainda e escassa.

## Variaveis que alimentam a Predicao

| Coluna | Significado | Origem |
|--------|-------------|--------|
| `lat`, `lon`, `date` | Coordenadas e data da observacao. | grade comum (MUR SST) |
| `sst` | Temperatura da superficie do mar (Celsius). | MUR SST |
| `sst_gradient` | Modulo do gradiente termico (proxy de frentes). | derivado de MUR |
| `chlor_a_modis`, `chlor_a_pace`, `chlor_a` | Clorofila-a (mg m^-3); `chlor_a` e a media ponderada. | MODIS Aqua + PACE OCI |
| `ssh_swot`, `ssh_swot_gradient`, `swot_mask` | Altura da superficie do mar, gradiente e mascara de cobertura SWOT. | SWOT L2 LR |
| `moana_prococcus_moana`, `moana_syncoccus_moana`, `moana_picoeuk_moana` | Abundancia (cells ml^-1) de grupos-chave de fitoplancton. | PACE OCI (MOANA) |
| `moana_total_cells` | Biomassa total estimada (soma das colunas MOANA). | derivado |
| `moana_picoeuk_share`, `moana_cyanobacteria_share` | Fracao relativa de picoeucariotos e cianobacterias. | derivado |
| `moana_diversity_index` | Entropia normalizada (0-1) da comunidade fitoplanctonica. | derivado |
| `habitat_score` | Score final (0-1) por especie. | modelo heuristico |
| `habitat_class` | Faixa qualitativa (poor/moderate/good/excellent). | modelo heuristico |
| `is_hotspot` | Top 10% com maior `habitat_score`. | modelo heuristico |

### Comunidades fitoplanctonicas MOANA e implicacoes para a predicao

- `moana_prococcus_moana` (Prochlorococcus): cianobacteria adaptada a aguas quentes e pobres em nutrientes. Valores altos sinalizam regioes oligotroficas com cardapio basico; combinamos com gradiente e clorofila para diferenciar hotspots que dependem de eficiencia termica, tipicos de tubaroes azuis.
- `moana_syncoccus_moana` (Synechococcus): prefere aguas um pouco mais ricas e misturadas. Picos desse grupo indicam margens de frentes onde ha reposicao rapida de nutrientes, funcionando como zonas de transicao para presas pelagicas.
- `moana_picoeuk_moana` (picoeucariotos): microalgas eucariontes menores que 2 um ricas em lipideos. Altas densidades reforcam alimento de maior qualidade energetica, atraindo presas maiores e, por consequencia, tubaroes de topo como white e tiger sharks.
- `moana_total_cells`: biomassa total estimada; hotspots com altos valores descartam falsos positivos de gradiente sem produtividade.
- `moana_picoeuk_share`, `moana_cyanobacteria_share`: proporcoes relativas que distinguem cardapios energeticos (mais picoeucariotos) de cardapios basicos (dominados por cianobacterias).
- `moana_diversity_index`: entropia da comunidade fitoplanctonica. Valores altos indicam ecossistemas resilientes com cadeia trofica longa, que sustentam forrageamento repetitivo.


## Saidas Entregues

- **CSV de features** (`data/features/*.csv`) – prontos para reuso ou treino de modelos adicionais.
- **Predicoes por especie** (`data/predictions/*_{especie}_predictions.csv`) e PNGs correspondentes (`*_map.png`, `*_comparative_map.png`).
- **GeoTIFFs** (`data/tiles/hotspots_probability_*.tif`) + `tiles_manifest.json` para o app web.
- **Visualizacoes** (`data/viz/compare/*.png`, `data/viz/moana/*.png`) – incluem paineis com True Color, MOANA e camadas fisicas.
- **Relatorios** (`data/processed/metrics.json`, `data/predictions/habitat_model_report.json`) e, quando executado, `data/validation/validation_report.json`.

## Limitacoes de Dados e Plano de Fechamento do Ciclo

- **OCEARCH**: avaliamos a API publica; a amostragem recente tem poucas especies e trajetorias curtas, tornando inviavel um treino supervisionado generalista.
- **OBIS/GBIF**: servem para validacao pontual, mas nao oferecem densidade temporal suficiente.
- **Solucao proposta**: a **tag inteligente** (IMU, hidrofone, sensor gastrico, Argos) gera ocorrencias de alimentacao em tempo quase real. Ao integrar essas telemetrias:
  1. Obtemos rotulos de presenca/ausencia confiaveis.
  2. Alimentamos continuamente o pipeline, recalibrando tanto o heuristico quanto o XGBoost.
  3. Reduzimos incerteza e produzimos previsoes realmente operacionais para gestores e cientistas.

## Proxima Etapa

1. Firmar parceria para coleta de telemetria (tag proprietaria ou bases de pesquisa). 
2. Usar `scripts/ml/04_train_model.py` com rótulos reais, validando em folds temporais/espaciais. 
3. Incorporar ECCO (correntes) e batimetria, ampliando o vetor de features. 
4. Distribuir as previsoes via app web e APIs simples para apoio a conservacao, pesca responsavel e educacao.

---

Equipe Iron Shark– alinhados ao desafio NASA Space Apps 2025.
