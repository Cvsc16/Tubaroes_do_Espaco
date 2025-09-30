#!/usr/bin/env python3
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature


from scripts.utils import get_bbox, load_config, project_root

# Le BBOX direto do config.yaml
ROOT = project_root()
CFG = load_config()

BBOX = get_bbox(CFG) or [-80.0, 25.0, -60.0, 40.0]
west, south, east, north = BBOX

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

# Desenha o ret??ngulo da BBOX
ax.plot(
    [west, east, east, west, west],
    [south, south, north, north, south],
    color="red", linewidth=2, transform=ccrs.PlateCarree()
)

plt.title("Bounding Box de Recorte", fontsize=14)
plt.savefig(ROOT/"data"/"BBOX_preview.png", dpi=150)
plt.show()
