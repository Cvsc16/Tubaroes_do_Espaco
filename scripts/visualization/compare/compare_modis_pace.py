#!/usr/bin/env python3
"""Comparação visual MODIS vs PACE (clorofila).
Gera mapas lado a lado e scatter plot, salvando em data/viz/.
"""

from pathlib import Path
import xarray as xr
import matplotlib.pyplot as plt

# Diretórios
ROOT = Path(__file__).resolve().parents[3]
PROC_DIR = ROOT / "data" / "processed"
VIZ_DIR = ROOT / "data" / "viz"
VIZ_DIR.mkdir(parents=True, exist_ok=True)

def compare_modis_pace(date_str: str) -> None:
    """Gera mapas e scatter para a data YYYYMMDD."""

    modis_file = PROC_DIR / f"{date_str}_CHL-MODIS_proc.nc"
    pace_file  = PROC_DIR / f"{date_str}_CHL-PACE_proc.nc"

    if not modis_file.exists() or not pace_file.exists():
        print(f"❌ Arquivos de {date_str} não encontrados em {PROC_DIR}")
        return

    ds_modis = xr.open_dataset(modis_file)
    ds_pace  = xr.open_dataset(pace_file)

    chl_modis = ds_modis.get("chlor_a_modis")
    chl_pace  = ds_pace.get("chlor_a_pace")

    # Interpola PACE no grid do MODIS
    chl_pace_interp = chl_pace.interp(lat=chl_modis.lat, lon=chl_modis.lon)

    # Diferença
    diff = chl_pace_interp - chl_modis

    # -------------------------
    # Mapas lado a lado
    # -------------------------
    fig, axs = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)

    im0 = axs[0].pcolormesh(chl_modis.lon, chl_modis.lat, chl_modis,
                            cmap="viridis", shading="auto")
    axs[0].set_title("MODIS Clorofila (mg/m³)")
    plt.colorbar(im0, ax=axs[0], orientation="horizontal", fraction=0.046, pad=0.07)

    im1 = axs[1].pcolormesh(chl_pace_interp.lon, chl_pace_interp.lat, chl_pace_interp,
                            cmap="viridis", shading="auto")
    axs[1].set_title("PACE Clorofila (mg/m³)")
    plt.colorbar(im1, ax=axs[1], orientation="horizontal", fraction=0.046, pad=0.07)

    im2 = axs[2].pcolormesh(diff.lon, diff.lat, diff,
                            cmap="RdBu", shading="auto", vmin=-0.05, vmax=0.05)
    axs[2].set_title("Diferença PACE − MODIS")
    plt.colorbar(im2, ax=axs[2], orientation="horizontal", fraction=0.046, pad=0.07)

    for ax in axs:
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

    fig.suptitle(f"Comparação MODIS vs PACE — {date_str}", fontsize=14)

    out_map = VIZ_DIR / f"{date_str}_modis_pace_maps.png"
    fig.savefig(out_map, dpi=200)
    plt.close(fig)

    print(f"✅ Mapas salvos em {out_map}")

    # -------------------------
    # Scatter plot
    # -------------------------
    plt.figure(figsize=(6,6))
    plt.scatter(chl_modis.values.flatten(),
                chl_pace_interp.values.flatten(),
                s=5, alpha=0.5, c="blue")
    plt.xlabel("MODIS chlor_a (mg/m³)")
    plt.ylabel("PACE chlor_a (mg/m³)")
    plt.title(f"Scatter MODIS vs PACE — {date_str}")
    plt.grid(True)

    out_scatter = VIZ_DIR / f"{date_str}_modis_pace_scatter.png"
    plt.savefig(out_scatter, dpi=200)
    plt.close()

    print(f"✅ Scatter salvo em {out_scatter}")

    ds_modis.close()
    ds_pace.close()


def main():
    print(f"Verificando pasta de processamento: {PROC_DIR.resolve()}")
    # Itera sobre datas encontradas em processed/
    modis_files = sorted(PROC_DIR.glob("*CHL-MODIS_proc.nc"))
    pace_files  = sorted(PROC_DIR.glob("*CHL-PACE_proc.nc"))

    print(f"Arquivos MODIS encontrados: {len(modis_files)}")
    print(f"Arquivos PACE encontrados: {len(pace_files)}")

    modis_dates = {f.name[:8] for f in modis_files}
    pace_dates  = {f.name[:8] for f in pace_files}
    common_dates = sorted(modis_dates & pace_dates)

    if not common_dates:
        print("❌ Nenhuma data comum encontrada entre MODIS e PACE")
        return

    for date_str in common_dates:
        compare_modis_pace(date_str)


if __name__ == "__main__":
    main()
