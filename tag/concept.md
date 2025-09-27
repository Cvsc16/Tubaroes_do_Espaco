
# Conceito de Tag – Dieta & Caça em Tempo (quase) Real

**Sensores:**
- IMU (acel/giroscópio) → detecção de rajadas/ataque
- Pressão + temperatura → profundidade/termoclina
- Hidrofone de banda limitada → sons de mastigação
- Proxy gástrico (bioimpedância/termistor interno leve) → ingestão

**Edge AI/Duty Cycle:**
- Classificador leve detecta possível evento de alimentação (pela IMU) → ativa hidrofone por janelas curtas e registra proxy gástrico.
- Apenas **eventos** são transmitidos (economia de energia).

**Transmissão e Energia:**
- Buffer local; uplink via Argos/Iridium quando em superfície.
- Energia: duty-cycling agressivo; considerar micro painéis solares e otimização de firmware.

**Dados enviados (compactos):**
- timestamp, lat/lon, profundidade, prob_evento_alimentação, resumo de features.
