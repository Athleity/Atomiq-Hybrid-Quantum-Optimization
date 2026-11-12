"""Command-line entry point for the classical Stage 1 benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path

from atomiq.evaluation import load_evaluation_config, run_nested_repeated_cv
from atomiq.schema import load_training_data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workbook",
        type=Path,
        default=Path("data/raw/bank_campaign_participant_release.xlsx"),
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/phase1_classical.yaml")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results/stage1_classical")
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Optional subset of model identifiers; defaults to the full portfolio.",
    )
    args = parser.parse_args()

    data = load_training_data(args.workbook)
    config = load_evaluation_config(args.config)
    print(
        f"Training-only rows={len(data.target)} positives={int(data.target.sum())} "
        f"features={len(data.features.columns)}"
    )
    result = run_nested_repeated_cv(
        data,
        config,
        args.output,
        model_names=args.models,
    )
    print(f"Classical CV champion: {result['model']}")
    print(f"Artifacts: {args.output.resolve()}")


if __name__ == "__main__":
    main()
