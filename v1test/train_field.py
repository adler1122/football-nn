
"""
Train Football Field Keypoint Detection Model

Usage:
    python main.py train_field --data datasets/field/data.yaml --epochs 100
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
        description="Train football field keypoint model"
    )

    parser.add_argument(
        "--data", "-d",
        type=str,
        required=True,
        help="Path to field dataset yaml"
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        default="models/field_best.pt",
        help="Output model path"
    )

    parser.add_argument(
        "--epochs", "-e",
        type=int,
        default=100,
        help="Training epochs"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size"
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


def train_field_model(args, config: Config):
    args = parse_args(args, config)

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    logger.info("Starting field model training")
    logger.info("Dataset: %s", args.data)
    model = YOLO("yolo11n-pose.pt")

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
    best_model_path = getattr(results, "save_dir", None)

    logger.info("Training completed")
    logger.info("Results saved in: %s", best_model_path)



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
    config = Config()
    import sys
    train_field_model(sys.argv[1:], config)