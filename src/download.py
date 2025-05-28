import logging
from pathlib import Path
import tarfile
import zipfile
from typing import Optional
import requests


def download_and_extract(
    url: str, save_dir: str, filename: Optional[str] = None
) -> str:
    target = Path(save_dir)
    target.mkdir(parents=True, exist_ok=True)

    name = filename or Path(url).name
    file = target / name
    output = str(file)

    if not file.exists():
        logging.info(f"Downloading '{name}' from '{url}'...")
        try:
            response = requests.get(url)
            response.raise_for_status()
            file.write_bytes(response.content)
            logging.info(f"Saved to '{file}'")

            if name.endswith(".zip"):
                with zipfile.ZipFile(file, "r") as zipf:
                    zipf.extractall(target)
                file.unlink()
                output = str(target)
            elif any(name.endswith(ext) for ext in [".tar.gz", ".tgz", ".gz"]):
                with tarfile.open(file, "r:gz") as tarf:
                    tarf.extractall(target)
                file.unlink()
                output = str(target)
            elif name.endswith(".tar"):
                with tarfile.open(file, "r") as tarf:
                    tarf.extractall(target)
                file.unlink()
                output = str(target)
            else:
                logging.info(f"'{name}' is not an archive.")
        except requests.RequestException as err:
            logging.error(f"Download failed: {err}")
            raise
        except (tarfile.TarError, zipfile.BadZipFile) as err:
            logging.error(f"Extraction failed for '{name}': {err}")
            raise
    else:
        logging.info(f"File already exists: '{file}'")

    return output