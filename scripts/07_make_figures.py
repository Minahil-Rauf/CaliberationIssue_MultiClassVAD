"""Stage 7: paper figures (PNG + PDF) and LaTeX tables -> outputs/figures/."""
from _common import parser, setup

from vadcal import plots


def main() -> None:
    args = parser(__doc__).parse_args()
    _, paths = setup(args)
    plots.reliability_figure(paths.tables, paths.figures)
    plots.imbalance_figure(paths.tables, paths.figures)
    plots.smoothing_figure(paths.tables, paths.figures)
    plots.threshold_figure(paths.tables, paths.figures)
    plots.latex_tables(paths.tables, paths.figures)


if __name__ == "__main__":
    main()
