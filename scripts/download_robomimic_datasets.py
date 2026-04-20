import importlib.metadata
import json
import re
from pathlib import Path
import urllib.request

import tyro
from robomimic import DATASET_REGISTRY
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parent.parent
SUPPORTED_ENVS = ("lift", "can", "square", "tool_hang")
SUPPORTED_SPLITS = ("ph",)
HDF5_TYPE = "low_dim"
ROBOMIMIC_GIT_URL = "https://github.com/ARISE-Initiative/robomimic"


def _read_expected_robomimic_rev(pyproject_path):
    match = re.search(
        r'^robomimic\s*=\s*\{[^}]*git\s*=\s*"https://github\.com/ARISE-Initiative/robomimic"[^}]*rev\s*=\s*"([0-9a-f]{40})"[^}]*\}',
        pyproject_path.read_text(),
        flags=re.MULTILINE,
    )
    assert match is not None, f"Could not find pinned robomimic revision in {pyproject_path}"
    return match.group(1)


def _read_installed_robomimic_rev():
    dist = importlib.metadata.distribution("robomimic")
    direct_url_path = Path(dist._path) / "direct_url.json"
    assert direct_url_path.is_file(), f"Missing {direct_url_path}"
    payload = json.loads(direct_url_path.read_text())
    assert payload["url"] == ROBOMIMIC_GIT_URL, payload
    return payload["vcs_info"]["commit_id"]


def _normalize_requested(values, allowed, label):
    if len(values) == 1 and values[0] == "all":
        return allowed
    unknown = tuple(value for value in values if value not in allowed)
    assert len(unknown) == 0, f"Unknown {label}: {unknown}. Allowed: {allowed}"
    return values


def _resolve_root(root):
    root = Path(root)
    if root.is_absolute():
        return root
    return REPO_ROOT / root


def _iter_downloads(envs, splits):
    for env_name in envs:
        for split in splits:
            if split not in DATASET_REGISTRY[env_name]:
                print(f"Skipping unsupported split: env={env_name} split={split}")
                continue
            url = DATASET_REGISTRY[env_name][split][HDF5_TYPE]["url"]
            assert url is not None, f"Missing URL for env={env_name} split={split} hdf5_type={HDF5_TYPE}"
            yield env_name, split, url


def _download_dataset(url, download_dir, dry_run):
    target = download_dir / Path(url).name
    if target.is_file():
        print(f"Already present: {target}")
        return target
    action = "Would download" if dry_run else "Downloading"
    print(f"{action} {url} -> {target}")
    if dry_run:
        return target
    download_dir.mkdir(parents=True, exist_ok=True)
    partial_target = target.with_suffix(target.suffix + ".part")
    if partial_target.exists():
        partial_target.unlink()
    request = urllib.request.Request(url, headers={"User-Agent": "sample_rank_public downloader"})
    with urllib.request.urlopen(request) as response:
        total = None
        if "Content-Length" in response.headers:
            total = int(response.headers["Content-Length"])
        with open(partial_target, "wb") as handle:
            with tqdm(total=total, unit="B", unit_scale=True, desc=target.name) as progress:
                while True:
                    chunk = response.read(1 << 20)
                    if chunk == b"":
                        break
                    handle.write(chunk)
                    progress.update(len(chunk))
    partial_target.rename(target)
    assert target.is_file(), f"Missing downloaded file {target}"
    return target


def main(
    envs: tuple[str, ...] = SUPPORTED_ENVS,
    splits: tuple[str, ...] = SUPPORTED_SPLITS,
    root: Path = Path("datasets/robomimic"),
    dry_run: bool = False,
):
    expected_rev = _read_expected_robomimic_rev(REPO_ROOT / "pyproject.toml")
    installed_rev = _read_installed_robomimic_rev()
    assert installed_rev == expected_rev, (
        f"Installed robomimic revision {installed_rev} does not match pyproject.toml pin {expected_rev}. "
        "Run `uv sync` first."
    )
    requested_envs = _normalize_requested(envs, SUPPORTED_ENVS, "envs")
    requested_splits = _normalize_requested(splits, SUPPORTED_SPLITS, "splits")
    download_root = _resolve_root(root)
    print(f"Using robomimic revision: {installed_rev}")
    print(f"Download root: {download_root}")
    targets = tuple(_iter_downloads(requested_envs, requested_splits))
    assert len(targets) > 0, "No robomimic datasets selected for download"
    for env_name, split, url in targets:
        _download_dataset(url, download_root / env_name / split, dry_run=dry_run)


if __name__ == "__main__":
    tyro.cli(main)
