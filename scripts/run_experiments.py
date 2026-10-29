"""Train, tune and evaluate (on validation) the models defined in configs/models.yaml.

Usage (from the project root):
    python scripts/run_experiments.py                      # all models
    python scripts/run_experiments.py --models logreg dummy_prior
    python scripts/run_experiments.py --resume   # continue an interrupted run
"""

from __future__ import annotations

import argparse

from credit_risk.experiment import run_experiments


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="*", help="subset of model names from the config")
    parser.add_argument("--config", default="models", help="config name in configs/ (no .yaml)")
    parser.add_argument(
        "--resume", action="store_true", help="skip models already evaluated with the same spec"
    )
    args = parser.parse_args()
    run_experiments(args.models, config_name=args.config, resume=args.resume)


if __name__ == "__main__":
    main()
