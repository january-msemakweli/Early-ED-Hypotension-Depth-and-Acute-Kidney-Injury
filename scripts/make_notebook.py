"""Build final_script.ipynb from analysis.py (jupytext-style # %% cell markers)."""
from pathlib import Path
import nbformat as nbf

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
src = (SCRIPT_DIR / "analysis.py").read_text(encoding="utf-8").splitlines()

cells = []
cur_type = None
cur = []


def flush():
    global cur_type, cur
    if cur_type is None:
        return
    if cur_type == "markdown":
        md = []
        for l in cur:
            if l.startswith("# "):
                md.append(l[2:])
            elif l.strip() == "#":
                md.append("")
            else:
                md.append(l)
        text = "\n".join(md).strip("\n")
        if text:
            cells.append(nbf.v4.new_markdown_cell(text))
    else:
        text = "\n".join(cur).strip("\n")
        if text:
            cells.append(nbf.v4.new_code_cell(text))
    cur = []


for line in src:
    if line.startswith("# %% [markdown]"):
        flush(); cur_type = "markdown"
    elif line.startswith("# %%"):
        flush(); cur_type = "code"
    else:
        cur.append(line)
flush()

nb = nbf.v4.new_notebook()
nb.cells = cells
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}
out = PROJECT_DIR / "final_script.ipynb"
nbf.write(nb, out)
print(f"Wrote {out} with {len(cells)} cells "
      f"({sum(c.cell_type=='code' for c in cells)} code, "
      f"{sum(c.cell_type=='markdown' for c in cells)} markdown)")
