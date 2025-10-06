import logging
import argparse
from pathlib import Path
from typing import List

import hydra
from omegaconf import DictConfig, OmegaConf

from src import create_logger
from src import Trainer


def main(cfg: DictConfig) -> None:
    logger = logging.getLogger("root")
    logger.debug("Resolved configuration as YAML:\n%s", OmegaConf.to_yaml(cfg))

    env_params = cfg.env_params
    model_params = cfg.model_params
    optimizer_params = cfg.optimizer_params
    trainer_params = cfg.trainer_params
    logger_params = cfg.logger_params

    create_logger(logger_params)
    _print_config(
        env_params, model_params, optimizer_params, trainer_params, logger_params
    )

    trainer = Trainer(
        env_params=env_params,
        model_params=model_params,
        optimizer_params=optimizer_params,
        trainer_params=trainer_params,
        logger_params=logger_params,
    )

    trainer.run()


def _print_config(
    env_params, model_params, optimizer_params, trainer_params, logger_params
) -> None:
    logger = logging.getLogger("root")

    # Convert params to YAML and add 2-space indentation
    def format_yaml(params):
        yaml_str = OmegaConf.to_yaml(params)
        return "\n".join("  " + line for line in yaml_str.splitlines())

    config_str = (
        f"{'=' * 80}\n"
        f"Configuration\n"
        f"{'=' * 80}\n"
        f"env_params:\n{format_yaml(env_params)}\n"
        f"model_params:\n{format_yaml(model_params)}\n"
        f"optimizer_params:\n{format_yaml(optimizer_params)}\n"
        f"trainer_params:\n{format_yaml(trainer_params)}\n"
        f"logger_params:\n{format_yaml(logger_params)}\n"
        f"{'=' * 80}"
    )
    logger.info(config_str)


def _parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train model with config file")
    parser.add_argument(
        "config", type=str, help="Path to the config file relative to 'configs/train'"
    )
    parser.add_argument(
        "overrides", nargs="*", help="Override configuration parameters (key=value)"
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()

    # Resolve Hydra config path
    config_file = Path("configs") / "train" / args.config
    config_dir = str(config_file.parent)
    config_name = config_file.name

    with hydra.initialize(config_path=config_dir, version_base=None):
        cfg = hydra.compose(config_name=config_name, overrides=args.overrides)

    main(cfg)
