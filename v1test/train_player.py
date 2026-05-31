
"""
Train player  Detection Model

Usage:
    python main.py train_player --data datasets/player/data.yaml --epochs 100
"""
import os
import shutil
import argparse
import logging
from ultralytics import YOLO
from config import Config

logger = logging.getLogger(__name__)


def parse_args(argv, config: Config):
    parser = argparse.ArgumentParser(
        description="Train football player detection model"
    )

    parser.add_argument(
        "--data", "-d",
        type=str,
        required=True,
        help="Path to player dataset yaml"
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        default="models/player_best.pt",
    )

    parser.add_argument(
        "--epochs", "-e",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--device",
        type=str,
        default=config.device,
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
    )

    return parser.parse_args(argv)


def train_player_model(args, config: Config):
    args = parse_args(args, config)

    log_level = logging.DEBUG if args.verbose else logging.INFO

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    logger.info("Starting player model training")
    logger.info("Dataset: %s", args.data)

    
    model = YOLO("yolo11n.pt")

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch_size,
        device=args.device,

        imgsz=640,

        cache="ram",
        workers=6,

        half=False,
        amp=False,

        lr0=0.01,
        lrf=0.01,

        warmup_epochs=3,

        save=True,
        save_period=5,

        resume=args.resume,
        exist_ok=True,
    )

    logger.info("Training completed")



    try:
        best_weight = os.path.join(
            str(results.save_dir),
            "weights",
            "best.pt"
        )

        shutil.copy2(
            best_weight,
            args.output
        )

        logger.info(
            "Best model copied to %s",
            args.output
        )

    except Exception as e:
        logger.warning(
            "Could not copy model: %s",
            e
        )


if __name__ == "__main__":
    import sys

    config = Config()

    train_player_model(
        sys.argv[1:],
        config
    )