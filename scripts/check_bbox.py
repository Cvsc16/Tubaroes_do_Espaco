#!/usr/bin/env python3
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import yaml
from pathlib import Path

# Lê bbox direto do config.yaml
ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load(open(ROOT/"config"/"config.yaml"))

bbox = CFG["aoi"]["bbox"]
west, south, east, north = bbox

# Criar figura
fig = plt.figure(figsize=(8,6))
ax = plt.axes(projection=ccrs.PlateCarree())

# Adiciona continentes, linhas de costa e grade
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linestyle=":")
ax.add_feature(cfeature.LAND, facecolor="lightgray")
ax.add_feature(cfeature.OCEAN, facecolor="lightblue")
ax.gridlines(draw_labels=True)

# Define limites do mapa
ax.set_extent([west-10, east+10, south-10, north+10])

# Desenha o retângulo da bbox
ax.plot(
    [west, east, east, west, west],
    [south, south, north, north, south],
    color="red", linewidth=2, transform=ccrs.PlateCarree()
)

plt.title("Bounding Box de Recorte", fontsize=14)
plt.savefig(ROOT/"data"/"bbox_preview.png", dpi=150)
plt.show()
