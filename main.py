#!/usr/bin/env python
"""
Main entry point for the ML pipeline.
Usage:
    python main.py train_field --data datasets/field/data.yaml --epochs 100
    python main.py train_player --data datasets/player/data.yaml --epochs 100
    python main.py analyze
"""

import argparse
import sys
import v1test
from config import Config


def main():
    """Main entry point"""
    config = Config()

    parser = argparse.ArgumentParser(
        description="Machine Learning Pipeline CLI"
    )

    subparsers = parser.add_subparsers(
        dest="command",
        help="Available commands"
    )

    subparsers.add_parser(
        "train_field",
        help="Train a model"
    )

    subparsers.add_parser(
        "train_player",
        help="Train a model"
    )
    subparsers.add_parser(
        "analyze",
        help="Analyze a video"
    )

    args, remaining = parser.parse_known_args()

    match args.command:
        case "train_player":
            v1test.train_model_player(remaining, config)
        case "train_field":
            v1test.train_model_field(remaining, config)
        case "analyze":
            v1test.run_analyzer(remaining, config)
        case _:
            print(f"Unknown command: {args.command}")
            parser.print_help()
            sys.exit(1)

if __name__ == "__main__":
    main()
