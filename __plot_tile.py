from pathlib import Path
import rasterio
import matplotlib.pyplot as plt

fp = Path("data/tiles/hotspots_probability_2025-08-20T09-00-00.tif")
with rasterio.open(fp) as src:
    arr = src.read(1)
    extent = (src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top)

plt.imshow(arr, cmap="viridis", extent=extent)
plt.colorbar(label="Probabilidade prevista")
plt.title(fp.name)
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()
