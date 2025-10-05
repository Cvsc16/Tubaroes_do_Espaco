import xarray as xr
from pathlib import Path

file = Path("C:/Users/DS Caio/Repositorios/java/libs/Tubaroes_do_Espaco/data/raw/swot/20250926T053606_SSH-SWOT.nc")

# abrir o feixe esquerdo
ds_left = xr.open_dataset(file, group="left", decode_times=False)
print("\n=== Grupo LEFT ===")
print(ds_left)

# abrir o feixe direito
ds_right = xr.open_dataset(file, group="right", decode_times=False)
print("\n=== Grupo RIGHT ===")
print(ds_right)
