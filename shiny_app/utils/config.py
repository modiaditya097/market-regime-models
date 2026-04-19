import yaml
from pathlib import Path


def load_config(config_path: Path, project_root: Path | None = None) -> dict:
    """Load app_config.yaml and resolve output_dir paths.

    Args:
        config_path: Path to app_config.yaml.
        project_root: Base directory for resolving relative output_dir values.
                      Defaults to the current working directory.
    """
    config_path = Path(config_path)
    root = Path(project_root) if project_root else Path.cwd()

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    for model in cfg["models"]:
        model["output_dir"] = (root / model["output_dir"]).resolve()
        model.setdefault("run_command", None)
        model.setdefault("te_targets", [0.03])

    return cfg
