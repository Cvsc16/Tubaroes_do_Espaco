from pathlib import Path
path = Path("scripts/compare_probability_vs_truecolor.py")
text = path.read_text()
text = text.replace('fig.suptitle(f"Comparacao MODIS x Cientifico x Modelo — {date_iso}", fontsize=14)', 'fig.suptitle(f"Comparacao MODIS x Cientifico x Modelo - {date_iso}", fontsize=14)')
path.write_text(text)
