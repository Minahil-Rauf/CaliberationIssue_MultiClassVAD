# ICEET 2026 paper

*Right but Unsure: How Focal Loss and Label Smoothing Miscalibrate Multi-Class Voice Activity Detection*

- `main.pdf`: the compiled paper (IEEE conference format, 6 pages, no author block yet)
- `main.tex`, `refs.bib`: LaTeX source (IEEEtran). Every reference was checked against its arXiv, Crossref or publisher record.
- `make_figures.py`: builds every figure in `figures/` from `../outputs/tables/*.csv`. It also writes `figures/key_numbers.txt`, which lists every number quoted in the text.

## Rebuild
```
python paper/make_figures.py        # needs pandas and matplotlib
cd paper && tectonic main.tex       # or upload main.tex, refs.bib and figures/ to Overleaf
```
The source uses `newtxtext`/`newtxmath` for Times, so it compiles with pdfLaTeX, XeLaTeX or Tectonic.

## Before submission
- Add the author block (`\author{...}` after `\title`).
- Replace the closing sentence about code release with the repository link once the review is no longer anonymous.
