import io
import os
import pathlib
import tarfile
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import yaml
from PIL import Image

from f1tenth_gym_jax.envs.track import Track, find_track_dir
from f1tenth_gym_jax.envs.track.utils import _safe_extractall


class TestTrack(unittest.TestCase):
    def _track_archive(self, track_name: str) -> bytes:
        archive = io.BytesIO()
        with tarfile.open(fileobj=archive, mode="w:xz") as tar:
            payload = b"placeholder"
            info = tarfile.TarInfo(f"{track_name}/placeholder.txt")
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
        return archive.getvalue()

    def _write_minimal_map(self, map_root: pathlib.Path, map_dirname: str):
        track_dir = map_root / map_dirname
        track_dir.mkdir()

        image_name = f"{map_dirname}.png"
        Image.fromarray(np.full((16, 16), 255, dtype=np.uint8)).save(
            track_dir / image_name
        )
        with (track_dir / f"{map_dirname}.yaml").open("w") as yaml_file:
            yaml.safe_dump(
                {
                    "image": image_name,
                    "resolution": 0.1,
                    "origin": [0.0, 0.0, 0.0],
                    "negate": 0,
                    "occupied_thresh": 0.45,
                    "free_thresh": 0.196,
                },
                yaml_file,
            )
        centerline = np.array(
            [
                [0.0, 0.0, 1.0, 1.0],
                [1.0, 0.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0],
                [0.0, 1.0, 1.0, 1.0],
            ]
        )
        np.savetxt(
            track_dir / f"{map_dirname}_centerline.csv",
            centerline,
            delimiter=",",
        )
        return track_dir

    def test_error_handling(self):
        wrong_track_name = "i_dont_exists"
        response = Mock(status_code=404)

        with patch("f1tenth_gym_jax.envs.track.utils.requests.get") as request_get:
            request_get.return_value = response
            self.assertRaises(
                FileNotFoundError, Track.from_track_name, wrong_track_name
            )

        request_get.assert_called_once()

    def test_downloaded_track_uses_configured_map_cache(self):
        track_name = "DownloadedTrackForTestNoTempArchive"
        response = Mock(status_code=200, content=self._track_archive(track_name))
        response.raise_for_status = Mock()

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"F1TENTH_GYM_JAX_MAP_DIR": tmpdir}):
                with patch(
                    "f1tenth_gym_jax.envs.track.utils.requests.get"
                ) as request_get:
                    request_get.return_value = response
                    track_dir = find_track_dir(track_name)

            self.assertEqual(track_dir, pathlib.Path(tmpdir) / track_name)
            self.assertTrue((track_dir / "placeholder.txt").exists())
            self.assertFalse(
                (pathlib.Path(tempfile.gettempdir()) / f"{track_name}.tar.xz").exists()
            )

        request_get.assert_called_once()
        response.raise_for_status.assert_called_once()

    def test_local_track_with_spaces_uses_actual_file_stem(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            track_dir = self._write_minimal_map(pathlib.Path(tmpdir), "Space Map")

            with patch.dict(os.environ, {"F1TENTH_GYM_JAX_MAP_DIR": tmpdir}):
                track = Track.from_track_name("SpaceMap")

        self.assertEqual(track.filepath, track_dir / "Space Map.png")
        self.assertGreater(track.centerline.s[-1], 0.0)

    def test_map_lookup_ignores_non_directory_name_matches(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            map_root = pathlib.Path(tmpdir)
            (map_root / "FileOnlyMap").write_text("not a map directory")

            with patch.dict(os.environ, {"F1TENTH_GYM_JAX_MAP_DIR": tmpdir}):
                with patch(
                    "f1tenth_gym_jax.envs.track.utils.requests.get"
                ) as request_get:
                    request_get.return_value = Mock(status_code=404)

                    with self.assertRaisesRegex(FileNotFoundError, "No maps exists"):
                        find_track_dir("FileOnlyMap")

            request_get.assert_called_once()

    def test_map_image_path_must_stay_inside_track_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            map_root = pathlib.Path(tmpdir)
            track_dir = self._write_minimal_map(map_root, "UnsafeMap")
            Image.fromarray(np.full((16, 16), 255, dtype=np.uint8)).save(
                map_root / "outside.png"
            )
            with (track_dir / "UnsafeMap.yaml").open("w") as yaml_file:
                yaml.safe_dump(
                    {
                        "image": "../outside.png",
                        "resolution": 0.1,
                        "origin": [0.0, 0.0, 0.0],
                        "negate": 0,
                        "occupied_thresh": 0.45,
                        "free_thresh": 0.196,
                    },
                    yaml_file,
                )

            with patch.dict(os.environ, {"F1TENTH_GYM_JAX_MAP_DIR": tmpdir}):
                with self.assertRaisesRegex(ValueError, "map image path escapes"):
                    Track.from_track_name("UnsafeMap")

    def test_safe_archive_extraction_rejects_path_traversal(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            archive_path = pathlib.Path(tmpdir) / "bad_map.tar"
            target_dir = pathlib.Path(tmpdir) / "maps"
            target_dir.mkdir()

            with tarfile.open(archive_path, "w") as tar:
                payload = b"bad"
                info = tarfile.TarInfo("../bad.txt")
                info.size = len(payload)
                tar.addfile(info, io.BytesIO(payload))

            with tarfile.open(archive_path) as tar:
                with self.assertRaises(ValueError):
                    _safe_extractall(tar, target_dir)

            self.assertFalse((pathlib.Path(tmpdir) / "bad.txt").exists())

    def test_raceline(self):
        track_name = "Spielberg"
        track = Track.from_track_name(track_name)

        # check raceline is not None
        self.assertNotEqual(track.raceline, None)

        # check loaded raceline match the one in the csv file
        track_dir = find_track_dir(track_name)
        assert track_dir is not None and track_dir.exists(), "track_dir does not exist"

        raceline = np.loadtxt(track_dir / f"{track_name}_raceline.csv", delimiter=";")
        s_idx, x_idx, y_idx, psi_idx, kappa_idx, vx_idx, ax_idx = range(7)

        self.assertTrue(
            np.isclose(track.raceline.s, raceline[:, s_idx], atol=5e-3).all()
        )
        self.assertTrue(np.isclose(track.raceline.xs, raceline[:, x_idx]).all())
        self.assertTrue(np.isclose(track.raceline.ys, raceline[:, y_idx]).all())
        self.assertTrue(np.isclose(track.raceline.psis, raceline[:, psi_idx]).all())
        self.assertTrue(np.isclose(track.raceline.ks, raceline[:, kappa_idx]).all())
        self.assertTrue(np.isclose(track.raceline.vxs, raceline[:, vx_idx]).all())
        self.assertTrue(np.isclose(track.raceline.axs, raceline[:, ax_idx]).all())

    def test_map_dir_structure(self):
        """
        Check that the map dir structure is correct:
        - maps/
            - Trackname/
                - Trackname.*                   # map image
                - Trackname.yaml                # map specification
                - [Trackname_raceline.csv]      # raceline (optional)
                - [Trackname_centerline.csv]    # centerline (optional)
        """
        mapdir = pathlib.Path(__file__).parent.parent / "maps"
        for trackdir in mapdir.iterdir():
            if trackdir.is_file():
                continue

            # check subdir is capitalized (at least first letter is capitalized)
            trackdirname = trackdir.stem
            self.assertTrue(
                trackdirname[0].isupper(), f"trackdir {trackdirname} is not capitalized"
            )

            # check map spec file exists
            file_spec = trackdir / f"{trackdirname}.yaml"
            self.assertTrue(
                file_spec.exists(),
                f"map spec file {file_spec} does not exist in {trackdir}",
            )

            # read map image file from spec
            with file_spec.open() as f:
                map_spec = yaml.safe_load(f)
            file_image = trackdir / map_spec["image"]

            # check map image file exists
            self.assertTrue(
                file_image.exists(),
                f"map image file {file_image} does not exist in {trackdir}",
            )

            # check raceline and centerline files
            file_raceline = trackdir / f"{trackdir.stem}_raceline.csv"
            file_centerline = trackdir / f"{trackdir.stem}_centerline.csv"

            if file_raceline.exists():
                Track.from_raceline_file(file_raceline)

            if file_centerline.exists():
                centerline = np.loadtxt(file_centerline, delimiter=",")
                self.assertEqual(centerline.shape[1], 4)

    def test_from_numpy_respects_s_frame_max(self):
        track_name = "Spielberg"
        track_dir = find_track_dir(track_name)
        waypoints = np.loadtxt(
            track_dir / f"{track_name}_raceline.csv", delimiter=";"
        ).astype(np.float32)

        track = Track.from_numpy(waypoints, s_frame_max=123.0, downsample_step=50)

        self.assertEqual(track.s_frame_max, 123.0)
        self.assertEqual(track.waypoints.shape, waypoints.shape)

    def test_invalid_track_inputs_raise_explicit_errors(self):
        with self.assertRaises(ValueError):
            Track(np.zeros(2), np.zeros(3))

        with self.assertRaisesRegex(ValueError, "2-dimensional"):
            Track.from_numpy(np.zeros(7), s_frame_max=1.0)

        with self.assertRaisesRegex(ValueError, "at least two rows"):
            Track.from_numpy(np.zeros((1, 7)), s_frame_max=1.0)

        with self.assertRaisesRegex(ValueError, "expected waypoints columns"):
            Track.from_numpy(np.zeros((2, 6)), s_frame_max=1.0)

        with self.assertRaisesRegex(ValueError, "s_frame_max"):
            Track.from_numpy(np.zeros((2, 7)), s_frame_max=0.0)

        with self.assertRaisesRegex(ValueError, "distinct points"):
            Track.from_numpy(np.zeros((2, 7)), s_frame_max=1.0)

        waypoints = np.zeros((2, 7))
        with self.assertRaisesRegex(ValueError, "downsample_step"):
            Track.from_numpy(waypoints, s_frame_max=1.0, downsample_step=0)

        with self.assertRaises(FileNotFoundError):
            Track.from_raceline_file(pathlib.Path("missing_raceline.csv"))

    def test_track_file_loaders_reject_malformed_tables(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = pathlib.Path(tmpdir) / "single_row_raceline.csv"
            np.savetxt(filepath, np.zeros((1, 7)), delimiter=";")

            with self.assertRaisesRegex(ValueError, "2-dimensional"):
                Track.from_raceline_file(filepath)

        with tempfile.TemporaryDirectory() as tmpdir:
            map_root = pathlib.Path(tmpdir)
            track_dir = self._write_minimal_map(map_root, "BadCenterline")
            np.savetxt(
                track_dir / "BadCenterline_centerline.csv",
                np.zeros((1, 4)),
                delimiter=",",
            )

            with patch.dict(os.environ, {"F1TENTH_GYM_JAX_MAP_DIR": tmpdir}):
                with self.assertRaisesRegex(ValueError, "2-dimensional"):
                    Track.from_track_name("BadCenterline")

    def test_from_raceline_file_rejects_invalid_downsample_step(self):
        waypoints = np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [2.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = pathlib.Path(tmpdir) / "raceline.csv"
            np.savetxt(filepath, waypoints, delimiter=";")

            with self.assertRaisesRegex(ValueError, "downsample_step"):
                Track.from_raceline_file(filepath, downsample_step=-1)

    def test_frenet_to_cartesian(self):
        track_name = "Spielberg"
        track = Track.from_track_name(track_name)

        # Check frenet to cartesian conversion
        # using the track's xs, ys
        for s, x, y in zip(
            track.centerline.s, track.centerline.xs, track.centerline.ys
        ):
            x_, y_, _ = track.frenet_to_cartesian(s, 0, 0)
            self.assertAlmostEqual(x, x_, places=2)
            self.assertAlmostEqual(y, y_, places=2)

    def test_frenet_to_cartesian_to_frenet(self):
        track_name = "Spielberg"
        track = Track.from_track_name(track_name)

        # check frenet to cartesian conversion
        s_ = 0
        for s in np.linspace(0, 1, 10):
            x, y, psi = track.frenet_to_cartesian(s, 0, 0)
            s_, d, _ = track.cartesian_to_frenet(x, y, psi, s_guess=s_)
            self.assertAlmostEqual(s, s_, places=2)
            self.assertAlmostEqual(d, 0, places=2)

        # check frenet to cartesian conversion
        # with non-zero lateral offset
        s_ = 0
        offsets = np.linspace(-0.5, 0.5, 10)
        for s, d in zip(np.linspace(0.1, 1.0, 10), offsets):
            x, y, psi = track.frenet_to_cartesian(s, d, 0)
            s_, d_, _ = track.cartesian_to_frenet(x, y, psi, s_guess=s_)
            self.assertAlmostEqual(s, s_, places=2)
            self.assertAlmostEqual(d, d_, places=2)
