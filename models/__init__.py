"""Edge Device GUI Agent — Research, Benchmark & Deploy GUI Interaction Models Locally."""

__version__ = "0.1.0"

from models.deploy.model_registry import ModelRegistry
from models.deploy.ui_tars_deploy import UITARSDeployer

__all__ = ["ModelRegistry", "UITARSDeployer"]
