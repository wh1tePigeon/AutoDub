import argparse
import requests
import zipfile
import os
from pathlib import Path
from urllib.parse import urlencode

from speechbrain.utils.data_utils import download_file


YANDEX_DISK_BASE_URL = 'https://cloud-api.yandex.net/v1/disk/public/resources/download?'


def download_from_yandex_disk(url: str, dst: Path):
    """
    Download file from yandex disk url to dst
    """
    # Get url to download file from disk
    response = requests.get(YANDEX_DISK_BASE_URL + urlencode(dict(public_key=url)))
    response.raise_for_status()
    # Download file using obtained url
    download_file(response.json()['href'], dst)


def main(download_directory: str, config_url: str, model_url: str, index_url: str):
    # Create model directory

    download_directory = Path(download_directory)
    download_directory.mkdir(exist_ok=True, parents=True)

    # Download model and its config
    config_path = download_directory / 'config.json'
    download_from_yandex_disk(config_url, config_path)

    model_path = download_directory / 'model_checkpoint.pth'
    download_from_yandex_disk(model_url, model_path)

    # Download index
    index_directory = download_directory / 'index/'
    index_directory.mkdir(exist_ok=True, parents=True)
    existing_files = list(index_directory.iterdir())
    if len(list(filter(
            lambda x: x.name.endswith('_index.json') and ('train' in x.name or 'dev' in x.name),
            existing_files)
    )) == 5:
        print('Index already exists: Skipping download')
    else:
        # Download folder in .zip format
        index_zip_path = index_directory / 'index.zip'
        download_from_yandex_disk(index_url, index_zip_path)

        # Extract index
        with zipfile.ZipFile(index_zip_path, 'r') as zip_ref:
            zip_ref.extractall(index_directory.parent)
        # Remove zip file
        os.remove(index_zip_path)

    print('Success!')


if __name__ == '__main__':
    args = argparse.ArgumentParser(description="Download pretrained model")

    args.add_argument(
        "-d",
        "--download_directory",
        default="pretrained_model/",
        type=str,
        help="Directory to save the model and config"
    )

    args.add_argument(
        "-c",
        "--config_url",
        default="https://disk.yandex.ru/d/FJd4nuh59orwbw",
        type=str,
        help="Directory to save the model and config"
    )

    args.add_argument(
        "-m",
        "--model_url",
        default="https://disk.yandex.ru/d/GRkXW-zKSt5k7g",
        type=str,
        help="Directory to save the model and config"
    )

    args.add_argument(
        "-i",
        "--index_url",
        default="https://disk.yandex.ru/d/DpMq8DMatAxmlg",
        type=str,
        help="Directory to save the model and config"
    )

    args = args.parse_args()
    main(args.download_directory, args.config_url, args.model_url, args.index_url)
