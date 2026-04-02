from __future__ import annotations

from pathlib import Path

import typer

from imc2025.config import load_config
from imc2025.pipeline import run_pipeline
from imc2025.scoring import score_submission
from imc2025.utils.logging import get_logger

app = typer.Typer(add_completion=False, help="IMC 2025 modular pipeline CLI.")
LOGGER = get_logger(__name__)


@app.command()
def run(
    config: str = typer.Option(..., help="Path to a YAML config file."),
    submission_path: str = typer.Option("outputs/submission.csv", help="Where to save the submission CSV."),
) -> None:
    cfg = load_config(config)
    out_path = run_pipeline(cfg, submission_path=Path(submission_path))
    LOGGER.info("Saved submission to %s", out_path)


@app.command()
def score(
    config: str = typer.Option(..., help="Path to a YAML config file."),
    submission_path: str = typer.Option(..., help="Submission CSV to evaluate."),
) -> None:
    cfg = load_config(config)
    final_score, dataset_scores = score_submission(cfg, submission_path)
    LOGGER.info("Final score: %s", final_score)
    LOGGER.info("Dataset scores: %s", dataset_scores)
