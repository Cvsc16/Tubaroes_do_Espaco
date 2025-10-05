import xarray as xr
ds = xr.open_dataset("20250926T053606_SSH-SWOT.nc", group="left")
print(ds["ssh_karin"].min().values, ds["ssh_karin"].max().values)
print(ds["ssh_karin"].attrs)
