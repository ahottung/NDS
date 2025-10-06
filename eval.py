import logging
import argparse
from pathlib import Path
from typing import List

import hydra
from omegaconf import DictConfig, OmegaConf

from src import create_logger


def main(cfg: DictConfig) -> None:
    logger = logging.getLogger("root")
    logger.debug("Resolved configuration as YAML:\n%s", OmegaConf.to_yaml(cfg))

    env_params = cfg.env_params
    tester_params = cfg.tester_params
    logger_params = cfg.logger_params

    create_logger(logger_params)
    _print_config(env_params, tester_params, logger_params)

    from src.search_sa import Search

    tester = Search(
        env_params=env_params,
        tester_params=tester_params,
    )

    tester.run()


def _print_config(env_params, tester_params, logger_params) -> None:
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
        f"tester_params:\n{format_yaml(tester_params)}\n"
        f"logger_params:\n{format_yaml(logger_params)}\n"
        f"{'=' * 80}"
    )
    logger.info(config_str)


def _parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test model with config file")
    parser.add_argument(
        "config", type=str, help="Path to the config file relative to 'configs/eval'"
    )
    parser.add_argument(
        "overrides", nargs="*", help="Override configuration parameters (key=value)"
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()

    # Resolve Hydra config path
    config_file = Path("configs") / "eval" / args.config
    config_dir = str(config_file.parent)
    config_name = config_file.name

    with hydra.initialize(config_path=config_dir, version_base=None):
        cfg = hydra.compose(config_name=config_name, overrides=args.overrides)

    main(cfg)
