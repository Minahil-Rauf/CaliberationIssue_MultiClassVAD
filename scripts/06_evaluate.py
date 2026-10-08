"""Stage 6: experiments E1-E5 on the saved logits of every finished run -> outputs/tables/."""
from _common import parser, setup

from vadcal.experiments import evaluate


def main() -> None:
    args = parser(__doc__).parse_args()
    cfg, paths = setup(args)
    evaluate(cfg, paths)


if __name__ == "__main__":
    main()
