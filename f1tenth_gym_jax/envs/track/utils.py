import pathlib
import tarfile
import tempfile

import requests


def _safe_extractall(tar_file: tarfile.TarFile, path: pathlib.Path) -> None:
    """Extract a map archive without allowing paths outside the target directory."""
    target_dir = path.resolve()
    for member in tar_file.getmembers():
        member_path = (target_dir / member.name).resolve()
        if member_path != target_dir and target_dir not in member_path.parents:
            raise ValueError(f"Unsafe path in map archive: {member.name}")
        if member.issym() or member.islnk():
            raise ValueError(f"Links are not allowed in map archives: {member.name}")
    tar_file.extractall(target_dir)


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
    map_dir = pathlib.Path(__file__).parent.parent.parent.parent / "maps"

    if not (map_dir / track_name).exists():
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

    # search for map in the map directory
    for subdir in map_dir.iterdir():
        if track_name == str(subdir.stem).replace(" ", ""):
            return subdir

    raise FileNotFoundError(f"no mapdir matching {track_name} in {[map_dir]}")
