import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description="Random agent for Isaac Lab environments.")
    parser.add_argument(
        "--disable_fabric",
        action="store_true",
        default=False,
        help="Disable fabric and use USD I/O operations.",
    )
    parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
    parser.add_argument("--task", type=str, default=None, help="Name of the task.")
    parser.add_argument(
        "--create_dataset",
        action="store_true",
        default=True,
        help="Whether to create a dataset by saving RGB-D images and segmentation masks.",
    )
    parser.add_argument(
        "--dataset_file",
        type=str,
        default="datasets/random_demo.hdf5",
        help="Path to the output dataset file (relative or absolute).",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=1000,
        help="Maximum number of simulation steps before exiting.",
    )
    parser.add_argument(
        "--store_videos",
        action="store_true",
        default=False,
        help="Whether to store videos of the episodes in the dataset.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Whether to print debug information.",
    )
    return parser
