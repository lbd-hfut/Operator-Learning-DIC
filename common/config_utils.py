"""YAML config loading and merging utilities."""
from dataclasses import fields, is_dataclass
from pathlib import Path


def load_yaml(path: str) -> dict:
    """Load a YAML config file."""
    import yaml
    with open(path, "r") as f:
        return yaml.safe_load(f)


def apply_yaml_overrides(dataclass_instance, yaml_cfg: dict, section: str = None):
    """Apply YAML overrides to a dataclass instance in-place.

    Matches YAML keys to dataclass field names. For nested sections
    (e.g. yaml 'training.learning_rate' -> field 'learning_rate'),
    pass the section name.

    Args:
        dataclass_instance: an instance of a @dataclass
        yaml_cfg: loaded YAML dict
        section: optional top-level section name to read from yaml_cfg
    """
    if section:
        overrides = yaml_cfg.get(section, {})
    else:
        overrides = yaml_cfg

    field_names = {f.name for f in fields(dataclass_instance)}
    for key, value in overrides.items():
        if key in field_names:
            if isinstance(value, list):
                value = tuple(value)
            setattr(dataclass_instance, key, value)
