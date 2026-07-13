import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import stage1_pipeline as pipeline


def main(argv=None):
    cli_args = list(sys.argv[1:] if argv is None else argv)
    if "--mode" not in cli_args:
        cli_args = ["--mode", "train", *cli_args]
    return pipeline.main(cli_args)


if __name__ == "__main__":
    main()
