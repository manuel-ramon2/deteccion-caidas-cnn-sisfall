"""Convierte el .docx a PDF con Microsoft Word y exporta cada página como PNG (para revisar).

    python articulo/docx_a_pdf.py ruta.docx [carpeta_png]
"""
import subprocess
import sys
from pathlib import Path

import fitz  # pymupdf

docx = Path(sys.argv[1]).resolve()
pdf = docx.with_suffix(".pdf")
ps = (f"$w=New-Object -ComObject Word.Application; $w.Visible=$false; "
      f"$d=$w.Documents.Open('{docx}', $false, $true); $d.SaveAs2('{pdf}', 17); $d.Close($false); $w.Quit()")
subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=True)
out = Path(sys.argv[2]) if len(sys.argv) > 2 else None
doc = fitz.open(pdf)
print("paginas:", doc.page_count)
if out:
    out.mkdir(parents=True, exist_ok=True)
    for i, p in enumerate(doc, 1):
        p.get_pixmap(dpi=80).save(out / f"pag-{i:02d}.png")
