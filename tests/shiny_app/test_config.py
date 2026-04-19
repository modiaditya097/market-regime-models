import pytest
import yaml
from pathlib import Path
import tempfile
from shiny_app.utils.config import load_config

SAMPLE_CONFIG = """
models:
  - id: model1
    name: "Test Model"
    output_dir: "outputs/"
    module: "shiny_app.modules.model1"
    te_targets: [0.01, 0.03]
    run_command: ["python", "main.py"]
  - id: model2
    name: "Model 2"
    output_dir: "outputs/model2/"
    module: "shiny_app.modules.model2"
    te_targets: [0.03]
    run_command: null
"""

def test_load_config_returns_models_list(tmp_path):
    cfg_file = tmp_path / "app_config.yaml"
    cfg_file.write_text(SAMPLE_CONFIG)
    cfg = load_config(cfg_file)
    assert "models" in cfg
    assert len(cfg["models"]) == 2

def test_load_config_resolves_output_dir(tmp_path):
    cfg_file = tmp_path / "app_config.yaml"
    cfg_file.write_text(SAMPLE_CONFIG)
    cfg = load_config(cfg_file, project_root=tmp_path)
    assert cfg["models"][0]["output_dir"] == tmp_path / "outputs"

def test_load_config_null_run_command(tmp_path):
    cfg_file = tmp_path / "app_config.yaml"
    cfg_file.write_text(SAMPLE_CONFIG)
    cfg = load_config(cfg_file, project_root=tmp_path)
    assert cfg["models"][1]["run_command"] is None

def test_load_config_te_targets_as_list(tmp_path):
    cfg_file = tmp_path / "app_config.yaml"
    cfg_file.write_text(SAMPLE_CONFIG)
    cfg = load_config(cfg_file, project_root=tmp_path)
    assert cfg["models"][0]["te_targets"] == [0.01, 0.03]
