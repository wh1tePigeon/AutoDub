import os
import gdown


def download(root):
    link = "https://drive.google.com/drive/folders/1Kc56V4eaJsIYLA-zWJMt7UzJuQfVecRW?usp=sharing"
    checkpoints_path = os.path.join(root, "checkpoints")
    os.makedirs(checkpoints_path, exist_ok=True)
    gdown.download_folder(url=link, output=checkpoints_path)


if __name__ == "__main__":
    root = os.getcwd()
    download(root)