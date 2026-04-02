from pathlib import Path

from imc2025.config import load_config


def test_load_config(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
experiment_name: demo
runtime:
  device: cpu
dataset:
  is_train: true
paths:
  data_dir: /tmp/data
pairing:
  similarity_threshold: 0.4
""",
        encoding="utf-8",
    )

    cfg = load_config(config_path)
    assert cfg.experiment_name == "demo"
    assert cfg.runtime.device == "cpu"
    assert cfg.dataset.is_train is True
    assert cfg.paths.data_dir == "/tmp/data"
    assert cfg.pairing.similarity_threshold == 0.4
