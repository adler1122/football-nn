#!/usr/bin/env python
"""
Main entry point for the ML pipeline.
Usage:
    python main.py train --data custom_data.csv
    python main.py predict --model my_model.pt
"""

import argparse
import sys
import cmd
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
        "train",
        help="Train a model"
    )

    subparsers.add_parser(
        "analyze",
        help="Analyze a video"
    )

    args, remaining = parser.parse_known_args()

    match args.command:
        case "train":
            cmd.train_model(remaining, config)
        case "analyze":
            cmd.run_analyzer(remaining, config)
        case _:
            print(f"Unknown command: {args.command}")
            parser.print_help()
            sys.exit(1)

if __name__ == "__main__":
    main()
