import argparse
import logging

from gymnasium_2048.agents.config import ConfigError
from gymnasium_2048.agents.registry import TRAIN_AGENTS, run_train_command


logging.basicConfig(
    filename="train.log",
    filemode="w",
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train 2048 agents from YAML configs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--agent",
        choices=sorted(TRAIN_AGENTS),
        help="agent to train; optional when --config contains an agent field",
    )
    parser.add_argument(
        "--config",
        help="YAML config path; defaults to the selected agent's bundled config",
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="print the resolved config and exit without training",
    )
    return parser.parse_args()


def train() -> None:
    args = parse_args()
    try:
        run_train_command(
            agent=args.agent,
            config_path=args.config,
            print_config=args.print_config,
        )
    except ConfigError as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    train()
