# Guia rapido para equipe - Tubaroes do Espaco

Este arquivo ajuda quem nao programa a entender como nossa solucao atende ao desafio "Tubaroes do Espaco" e onde cada parte da equipe pode colaborar.

## 1. O que o desafio pede
- Usar dados de satelite da NASA para prever onde tubaroes estarao se alimentando.
- Construir um modelo matematico/preditivo que ligue condicoes do oceano com o comportamento dos tubaroes.
- Propor um conceito de tag inteligente capaz de indicar onde o tubarao esta, o que ele come e enviar dados quase em tempo real.

Resumo oficial em ``docs/desafio_projeto.md``.
## 2. Nossa resposta em poucas palavras
- Zona foco: Corrente do Golfo e Mar dos Sargacos.
- Integramos temperatura da agua, produtividade biologica e correntes para mapear hotspots de alimentacao.
- Pipeline automatizada transforma arquivos da NASA em mapas de probabilidade e visualizacoes prontas para apresentacao.
- Conceito de tag embarcada reforca o modelo com dados diretamente do animal.

## 3. Fontes de dados da NASA
- MUR SST (GHRSST) - temperatura da superficie do mar, base para detectar frentes termicas.
- MODIS Aqua - clorofila historica, mostra onde ha alimento para as presas dos tubaroes.
- PACE - composicao do fitoplancton, diferencia qualidade do alimento.
- ECCO - correntes oceanicas u/v, indicam transporte de nutrientes.
- SWOT - topografia da superficie, revela redemoinhos que concentram presas.

Mais detalhes em `docs/recursos_projeto.md`.

## 4. Fluxo de trabalho e scripts
1. `scripts/01_search_download.py` - baixa dados brutos da NASA conforme `config/config.yaml`.
2. `scripts/02_preprocess.py` - recorta a area, converte SST para Celsius e calcula gradiente para cada data.
3. `scripts/03_feature_engineering.py` - cria tabelas com lat, lon, data, sst e gradiente.
4. `scripts/04_train_model.py` - junta as tabelas, rotula hotspots (topo do gradiente) e treina modelo XGBoost.
5. `scripts/05_export_tiles.py` - aplica o modelo e gera GeoTIFFs de probabilidade em `data/tiles/`.
6. `scripts/utils/build_tiles_manifest.py` - cria `data/tiles/tiles_manifest.json` para o app web.
7. Visualizacoes de apoio:
   - `scripts/visualization/check_processed.py` e `check_processed_interactive.py` - conferem SST e gradiente.
   - `scripts/visualization/compare_modis_truecolor.py` e `compare_probability_vs_truecolor.py` - comparam MODIS vs dados cientificos vs modelo.
   - `app/index.html` - mapa Leaflet que carrega os GeoTIFFs e permite alternar paletas.

## 5. Entregaveis previstos
- Mapa interativo (Leaflet) com lista de datas e probabilidade de hotspots.
- Imagens e HTMLs comparativos para storytelling.
- Dataset consolidado em `data/processed/dataset.csv`.
- Documentacao tecnica em `docs/visao_geral_e_melhorias.md` e log em `gpt.md`.
- Conceito de tag descrito em `tag/concept.md`.

## 6. Tag inteligente
A tag proposta combina IMU, sensor de pressao/temperatura, hidrofone e um sensor gastrico leve. Um modelo simples embarcado detecta eventos de alimentacao e envia apenas resumos quando o animal emerge, economizando energia e validando os hotspots identificados pelos satelites.

## 7. Tarefas para quem nao programa
- Narrativa e comunicacao: montar roteiro de video, slides e explicacoes voltadas a estudantes de ensino medio usando as visualizacoes.
- Pesquisa de apoio: resumir artigos listados em `docs/recursos_projeto.md` para reforcar as justificativas cientificas.
- Validacao qualitativa: analisar mapas e destacar historias (ex.: hotspot alinhado a redemoinho SWOT ou pico de clorofila MODIS).
- Planejamento da apresentacao: conectar o mapa, o modelo e a tag em um storytelling unico.

## 8. Onde achar mais informacoes
- `docs/desafio_projeto.md` - briefing do desafio.
- `docs/recursos_projeto.md` - links e referencias.
- `docs/visao_geral_e_melhorias.md` - estado da pipeline e backlog.
- `gpt.md` - registro diario das mudancas tecnicas.
- Diretorio `scripts/` - codigo organizado por etapa.

Ficou com duvida tecnica? Procure o README principal ou fale com quem esta mantendo a pipeline.
