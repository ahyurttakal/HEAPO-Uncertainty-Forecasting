from __future__ import annotations

import argparse
import hashlib
import zipfile
from pathlib import Path

import requests


DEFAULT_URL = "https://zenodo.org/records/15056919/files/heapo_data.zip?download=1"


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with temporary.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1 << 20):
                if chunk:
                    handle.write(chunk)
    temporary.replace(destination)


def safe_extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(archive) as zipped:
        for member in zipped.infolist():
            target = (destination / member.filename).resolve()
            if root not in target.parents and target != root:
                raise RuntimeError(f"Unsafe archive member: {member.filename}")
        zipped.extractall(destination)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--sha256", default=None)
    parser.add_argument("--keep-zip", action="store_true")
    args = parser.parse_args()
    archive = args.output_dir / "heapo_data.zip"
    download(args.url, archive)
    if args.sha256:
        digest = sha256_file(archive)
        if digest.lower() != args.sha256.lower():
            raise RuntimeError(f"Checksum mismatch: {digest}")
    safe_extract(archive, args.output_dir)
    if not args.keep_zip:
        archive.unlink()
    print(args.output_dir.resolve())


if __name__ == "__main__":
    main()
