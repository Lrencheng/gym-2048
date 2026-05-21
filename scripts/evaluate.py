import argparse

from gymnasium_2048.agents.config import ConfigError
from gymnasium_2048.agents.evaluation import (
    EvaluationConfig,
    evaluate_config,
    evaluate_from_yaml,
    load_evaluation_config,
    make_env,
    make_policy,
    plot_statistics,
    print_summary,
    run_episodes,
    summarize_statistics,
)
from gymnasium_2048.agents.registry import EVALUATE_AGENTS, run_evaluate_command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate 2048 agents from YAML configs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--agent",
        choices=sorted(EVALUATE_AGENTS),
        help="agent to evaluate; optional when --config contains an agent field",
    )
    parser.add_argument(
        "--config",
        help="YAML config path; defaults to the selected agent's bundled config",
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="print the resolved config and exit without evaluation",
    )
    return parser.parse_args()


def evaluate() -> None:
    args = parse_args()
    try:
        run_evaluate_command(
            agent=args.agent,
            config_path=args.config,
            print_config=args.print_config,
        )
    except ConfigError as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    evaluate()
