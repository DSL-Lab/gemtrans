import argparse
import os
import yaml
import shutil
from src.engine import Engine
import logging


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_dir",
        default="./logs/temp",
        help="Directory to save config and model checkpoint",
    )
    parser.add_argument(
        "--config_path",
        default="./configs/default.yml",
        help="Path to config YAML file",
    )
    parser.add_argument(
        "--test",
        default=False,
        action="store_true",
        help="indicates whether we are only testing",
    )
    parser.add_argument(
        "--sweep",
        default=False,
        action="store_true",
        help="indicates whether this is a sweep run",
    )
    args = parser.parse_args()

    # Create the save directory
    for patch_str in ["patch_level", "frame_level", "vid_level", "test_vis"]:
        os.makedirs(
            os.path.join(args.save_dir, "visualizations", patch_str), exist_ok=True
        )

    with open(args.config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Copy the provided config file into save_dir
    shutil.copyfile(args.config_path, os.path.join(args.save_dir, "config.yml"))

    # Create the logger
    logging.basicConfig(
        filename=os.path.join(args.save_dir, "log.log"),
        filemode="a",
        format="%(asctime)s,%(msecs)d %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        level=logging.INFO,
    )
    logger = logging.getLogger("heart-transformer")

    # Create the engine taking care of building different components and starting training/inference
    engine = Engine(
        config=config,
        save_dir=args.save_dir,
        logger=logger,
        train=not args.test,
        sweep=args.sweep,
    )

    if args.test:
        engine.evaluate()
    else:
        engine.train_model()


if __name__ == "__main__":
    run()
