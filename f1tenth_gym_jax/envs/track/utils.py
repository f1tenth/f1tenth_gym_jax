import inspect
import os
import pathlib
import tarfile
import tempfile

import requests

_MAP_DIR_ENV = "F1TENTH_GYM_JAX_MAP_DIR"


def _safe_extractall(tar_file: tarfile.TarFile, path: pathlib.Path) -> None:
    """Extract a map archive without allowing paths outside the target directory."""
    target_dir = path.resolve()
    for member in tar_file.getmembers():
        member_path = (target_dir / member.name).resolve()
        if member_path != target_dir and target_dir not in member_path.parents:
            raise ValueError(f"Unsafe path in map archive: {member.name}")
        if member.issym() or member.islnk():
            raise ValueError(f"Links are not allowed in map archives: {member.name}")
    if "filter" in inspect.signature(tar_file.extractall).parameters:
        tar_file.extractall(target_dir, filter="data")
    else:
        tar_file.extractall(target_dir)


def _bundled_map_dir() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[3] / "maps"


def _download_map_dir() -> pathlib.Path:
    configured_dir = os.environ.get(_MAP_DIR_ENV)
    if configured_dir:
        return pathlib.Path(configured_dir).expanduser()
    cache_root = pathlib.Path(os.environ.get("XDG_CACHE_HOME", "~/.cache")).expanduser()
    return cache_root / "f1tenth_gym_jax" / "maps"


def get_map_search_dirs() -> tuple[pathlib.Path, ...]:
    """Return map directories in lookup order without creating them."""
    search_dirs = []
    for path in (_bundled_map_dir(), _download_map_dir()):
        if path not in search_dirs:
            search_dirs.append(path)
    return tuple(search_dirs)


def _find_existing_track_dir(
    track_name: str, map_dirs: tuple[pathlib.Path, ...]
) -> pathlib.Path | None:
    for map_dir in map_dirs:
        if not map_dir.exists():
            continue
        for subdir in map_dir.iterdir():
            if track_name == str(subdir.stem).replace(" ", ""):
                return subdir
    return None


def find_track_dir(track_name: str) -> pathlib.Path:
    """
    Find the directory of the track map corresponding to the given track name.

    Parameters
    ----------
    track_name : str
        name of the track

    Returns
    -------
    pathlib.Path
        path to the track map directory

    Raises
    ------
    FileNotFoundError
        if no map directory matching the track name is found
    """
    map_dirs = get_map_search_dirs()
    track_dir = _find_existing_track_dir(track_name, map_dirs)
    if track_dir is not None:
        return track_dir

    map_dir = _download_map_dir()
    map_dir.mkdir(parents=True, exist_ok=True)
    print("Downloading Files for: " + track_name)
    tracks_url = "http://api.f1tenth.org/" + track_name + ".tar.xz"
    tracks_r = requests.get(url=tracks_url, allow_redirects=True, timeout=30)
    if tracks_r.status_code == 404:
        raise FileNotFoundError(f"No maps exists for {track_name}.")
    tracks_r.raise_for_status()

    archive_path = pathlib.Path(tempfile.gettempdir()) / f"{track_name}.tar.xz"

    with archive_path.open("wb") as f:
        f.write(tracks_r.content)

    print("Extracting Files for: " + track_name)
    with tarfile.open(archive_path) as tracks_file:
        _safe_extractall(tracks_file, map_dir)

    track_dir = _find_existing_track_dir(track_name, (map_dir,))
    if track_dir is not None:
        return track_dir

    raise FileNotFoundError(f"no mapdir matching {track_name} in {map_dirs}")
