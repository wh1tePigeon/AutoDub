import argparse
from pathlib import Path
import gdown
import os


def download(root):
    link = ""
    checkpoints_path = os.path.join(root, "checkpoints")
    os.makedirs(checkpoints_path, exist_ok=True)
    gdown.download_folder(url=link, output=checkpoints_path)


if __name__ == "__main__":
    root = os.getcwd()
    download(root)