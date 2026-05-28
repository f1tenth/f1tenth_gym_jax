from __future__ import annotations

import base64
import io
import json
import pathlib
import webbrowser
from typing import Any, Literal, Optional

import numpy as np
from PIL import Image

from ..f110_env import F110Env

TrajectoryLayout = Literal["auto", "step_major", "batch_major"]
ArtifactType = Literal["paths", "sample_paths"]

_DEFAULT_OUTPUT = pathlib.Path("f1tenth_gym_jax_rollout.html")
_COLORS = [
    "#2f6fed",
    "#d84343",
    "#178a5c",
    "#8a4bd6",
    "#d68a1c",
    "#008a9a",
    "#c23b82",
    "#5f7f1f",
]


def _finite_float(value: Any) -> float:
    return float(np.asarray(value, dtype=float))


def _finite_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    return _finite_float(value)


def _series_xy(xs: Any, ys: Any) -> list[list[float]]:
    x_arr = np.asarray(xs, dtype=float)
    y_arr = np.asarray(ys, dtype=float)
    points = np.column_stack((x_arr, y_arr))
    return np.round(points, 6).tolist()


def _artifact_xy_points(
    name: str,
    points: Any,
    expected_dims: int,
    x_index: int,
    y_index: int,
) -> np.ndarray:
    arr = np.asarray(points, dtype=float)
    if arr.ndim != expected_dims:
        raise ValueError(
            f"Artifact '{name}' expected {expected_dims} dimensions, got {arr.ndim}."
        )
    if arr.shape[-1] <= max(x_index, y_index):
        raise ValueError(
            f"Artifact '{name}' does not contain x/y columns " f"{x_index}/{y_index}."
        )
    xy = np.stack((arr[..., x_index], arr[..., y_index]), axis=-1)
    if not np.isfinite(xy).all():
        raise ValueError(f"Artifact '{name}' contains non-finite coordinates.")
    return np.round(xy, 6)


def _normalize_artifacts(artifacts: Optional[dict[str, Any]]) -> dict[str, Any]:
    if artifacts is None:
        return {"overlays": []}
    if not isinstance(artifacts, dict):
        raise ValueError("artifacts must be a dictionary when provided.")

    raw_overlays = artifacts.get("overlays", [])
    if raw_overlays is None:
        raw_overlays = []
    if not isinstance(raw_overlays, list):
        raise ValueError("artifacts['overlays'] must be a list.")

    overlays = []
    for index, raw_overlay in enumerate(raw_overlays):
        if not isinstance(raw_overlay, dict):
            raise ValueError("Each artifact overlay must be a dictionary.")

        overlay_type: ArtifactType = raw_overlay.get("type", "paths")
        if overlay_type not in {"paths", "sample_paths"}:
            raise ValueError("Artifact overlay type must be 'paths' or 'sample_paths'.")

        overlay_id = str(raw_overlay.get("id", f"artifact-{index}"))
        label = str(raw_overlay.get("label", overlay_id))
        x_index = int(raw_overlay.get("x_index", 0))
        y_index = int(raw_overlay.get("y_index", 1))
        expected_dims = 5 if overlay_type == "paths" else 6
        points = _artifact_xy_points(
            overlay_id,
            raw_overlay.get("points"),
            expected_dims,
            x_index,
            y_index,
        )
        step_indices = raw_overlay.get("step_indices")
        if step_indices is None:
            step_indices = list(range(points.shape[0]))
        step_indices_arr = np.asarray(step_indices, dtype=int)
        if step_indices_arr.shape != (points.shape[0],):
            raise ValueError(
                f"Artifact '{overlay_id}' step_indices must have shape "
                f"({points.shape[0]},)."
            )

        values = None
        value_min = None
        value_max = None
        if "values" in raw_overlay and raw_overlay["values"] is not None:
            values_arr = np.asarray(raw_overlay["values"], dtype=float)
            expected_shape = points.shape[:-1]
            if values_arr.shape != expected_shape:
                raise ValueError(
                    f"Artifact '{overlay_id}' values shape {values_arr.shape} "
                    f"does not match expected shape {expected_shape}."
                )
            if not np.isfinite(values_arr).all():
                raise ValueError(f"Artifact '{overlay_id}' contains non-finite values.")
            values = np.round(values_arr, 6).tolist()
            value_min = float(values_arr.min())
            value_max = float(values_arr.max())

        overlays.append(
            {
                "id": overlay_id,
                "label": label,
                "type": overlay_type,
                "scope": str(raw_overlay.get("scope", "playback")),
                "visible": bool(raw_overlay.get("visible", True)),
                "color": str(raw_overlay.get("color", "#6b7280")),
                "lineWidth": _finite_float(raw_overlay.get("line_width", 1.6)),
                "pointRadius": _finite_float(raw_overlay.get("point_radius", 2.4)),
                "opacity": _finite_float(raw_overlay.get("opacity", 0.74)),
                "valueLabel": str(raw_overlay.get("value_label", "value")),
                "valueMode": str(raw_overlay.get("value_mode", "lower_better")),
                "valueMin": _finite_float_or_none(
                    raw_overlay.get("value_min", value_min)
                ),
                "valueMax": _finite_float_or_none(
                    raw_overlay.get("value_max", value_max)
                ),
                "stepIndices": step_indices_arr.tolist(),
                "points": points.tolist(),
                "values": values,
            }
        )

    return {"overlays": overlays}


def _normalize_trajectory(
    trajectory: np.ndarray,
    num_agents: int,
    layout: TrajectoryLayout,
) -> np.ndarray:
    traj = np.asarray(trajectory, dtype=float)

    if traj.ndim == 2:
        traj = traj[None, :, None, :]
    elif traj.ndim == 3:
        traj = traj[None, :, :, :]
    elif traj.ndim == 4:
        if layout == "step_major":
            traj = np.transpose(traj, (1, 0, 2, 3))
        elif layout == "batch_major":
            pass
        elif layout == "auto":
            if traj.shape[2] != num_agents:
                raise ValueError(
                    "Expected trajectory agent axis at index 2 for 4D rollouts."
                )
            if traj.shape[0] >= traj.shape[1]:
                traj = np.transpose(traj, (1, 0, 2, 3))
        else:
            raise ValueError(
                "trajectory_layout must be 'auto', 'step_major', or 'batch_major'."
            )
    else:
        raise ValueError(
            "Expected trajectory shape (steps, envs, agents, states), "
            "(envs, steps, agents, states), (steps, agents, states), "
            "or (steps, states)."
        )

    if traj.shape[0] < 1 or traj.shape[1] < 1:
        raise ValueError("Trajectory must contain at least one rollout and one step.")
    if traj.shape[2] != num_agents:
        raise ValueError(
            f"Trajectory has {traj.shape[2]} agents, but env has {num_agents}."
        )
    if traj.shape[3] < 5:
        raise ValueError("Trajectory state vectors must contain x, y, steer, v, yaw.")
    if not np.isfinite(traj).all():
        raise ValueError("Trajectory contains non-finite values.")
    return traj


def _map_image_data_url(env: F110Env) -> str:
    image_array = np.asarray(env.track.occ_map)
    if image_array.ndim != 2:
        return ""

    if image_array.max(initial=0) <= 1:
        image_array = image_array * 255
    image_array = np.clip(image_array, 0, 255).astype(np.uint8)
    image = Image.fromarray(image_array).transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _rollout_stats(
    traj: np.ndarray, dt: float
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    positions = traj[:, :, :, :2]
    deltas = np.diff(positions, axis=1)
    distances = np.linalg.norm(deltas, axis=-1)
    speeds = np.abs(traj[:, :, :, 3])

    duration = max(traj.shape[1] - 1, 0) * dt
    per_rollout = []
    for rollout_index in range(traj.shape[0]):
        rollout_distances = distances[rollout_index].sum(axis=0)
        per_rollout.append(
            {
                "rollout": rollout_index,
                "totalDistance": round(float(rollout_distances.sum()), 4),
                "meanAgentDistance": round(float(rollout_distances.mean()), 4),
                "meanSpeed": round(float(speeds[rollout_index].mean()), 4),
                "maxSpeed": round(float(speeds[rollout_index].max()), 4),
                "finalStep": traj.shape[1] - 1,
            }
        )

    summary = {
        "rollouts": traj.shape[0],
        "steps": traj.shape[1],
        "agents": traj.shape[2],
        "duration": round(float(duration), 4),
        "meanSpeed": round(float(speeds.mean()), 4),
        "maxSpeed": round(float(speeds.max()), 4),
        "totalDistance": round(float(distances.sum()), 4),
    }
    return summary, per_rollout


def _bounds(
    env: F110Env, traj: np.ndarray, artifact_payload: dict[str, Any]
) -> dict[str, float]:
    xs = [traj[:, :, :, 0].reshape(-1)]
    ys = [traj[:, :, :, 1].reshape(-1)]

    for spline in (env.track.centerline, env.track.raceline):
        xs.append(np.asarray(spline.xs, dtype=float))
        ys.append(np.asarray(spline.ys, dtype=float))

    for overlay in artifact_payload["overlays"]:
        points = np.asarray(overlay["points"], dtype=float)
        xs.append(points[..., 0].reshape(-1))
        ys.append(points[..., 1].reshape(-1))

    map_width = env.track.occ_map.shape[1] * env.track.resolution
    map_height = env.track.occ_map.shape[0] * env.track.resolution
    xs.append(np.asarray([env.track.ox, env.track.ox + map_width], dtype=float))
    ys.append(np.asarray([env.track.oy, env.track.oy + map_height], dtype=float))

    all_x = np.concatenate(xs)
    all_y = np.concatenate(ys)
    min_x = float(np.min(all_x))
    max_x = float(np.max(all_x))
    min_y = float(np.min(all_y))
    max_y = float(np.max(all_y))

    if min_x == max_x:
        min_x -= 1.0
        max_x += 1.0
    if min_y == max_y:
        min_y -= 1.0
        max_y += 1.0

    pad_x = (max_x - min_x) * 0.04
    pad_y = (max_y - min_y) * 0.04
    return {
        "minX": min_x - pad_x,
        "maxX": max_x + pad_x,
        "minY": min_y - pad_y,
        "maxY": max_y + pad_y,
    }


def _payload(
    env: F110Env,
    traj: np.ndarray,
    artifact_payload: dict[str, Any],
    dt: float,
    title: str,
    metadata: Optional[dict[str, Any]],
    canvas_width: int,
    canvas_height: int,
    playing: bool,
) -> dict[str, Any]:
    summary, per_rollout = _rollout_stats(traj, dt)
    map_height, map_width = env.track.occ_map.shape
    return {
        "title": title,
        "metadata": metadata or {},
        "env": {
            "name": env.name,
            "map": env.params.map_name,
            "reward": env.params.reward_type,
            "numAgents": env.num_agents,
            "agents": list(env.agents),
            "dt": dt,
            "trackLength": round(float(env.track_length), 4),
            "maxSteps": int(env.params.max_steps),
        },
        "vehicle": {
            "length": _finite_float(env.params.length),
            "width": _finite_float(env.params.width),
        },
        "map": {
            "image": _map_image_data_url(env),
            "origin": [
                _finite_float(env.track.ox),
                _finite_float(env.track.oy),
                _finite_float(env.track.oyaw),
            ],
            "resolution": _finite_float(env.track.resolution),
            "width": int(map_width),
            "height": int(map_height),
        },
        "track": {
            "centerline": _series_xy(env.track.centerline.xs, env.track.centerline.ys),
            "raceline": _series_xy(env.track.raceline.xs, env.track.raceline.ys),
        },
        "trajectory": np.round(traj, 6).tolist(),
        "artifacts": artifact_payload,
        "summary": summary,
        "rolloutStats": per_rollout,
        "bounds": _bounds(env, traj, artifact_payload),
        "colors": _COLORS,
        "canvas": {"width": int(canvas_width), "height": int(canvas_height)},
        "playing": bool(playing),
    }


class WebRenderer:
    """Generate a self-contained browser dashboard for F110 rollout playback."""

    def __init__(
        self,
        env: F110Env,
        render_fps: Optional[float] = None,
        window_width: int = 1200,
        window_height: int = 760,
        render_mode: str = "html",
        output_path: pathlib.Path | str = _DEFAULT_OUTPUT,
        open_browser: bool = False,
        trajectory_layout: TrajectoryLayout = "step_major",
    ):
        if render_mode != "html":
            raise ValueError(
                "Only render_mode='html' is supported. Use the web dashboard output "
                "instead."
            )
        if trajectory_layout not in {"auto", "step_major", "batch_major"}:
            raise ValueError(
                "trajectory_layout must be 'auto', 'step_major', or 'batch_major'."
            )
        if render_fps is not None and render_fps <= 0:
            raise ValueError("render_fps must be positive when provided.")
        if window_width < 1 or window_height < 1:
            raise ValueError("window_width and window_height must be positive.")

        self.env = env
        self.render_mode = render_mode
        self.output_path = pathlib.Path(output_path)
        self.open_browser = open_browser
        self.trajectory_layout: TrajectoryLayout = trajectory_layout
        self.window_width = window_width
        self.window_height = window_height
        self.dt = (
            1.0 / float(render_fps)
            if render_fps is not None
            else float(env.params.timestep * env.params.timestep_ratio)
        )
        self.playing = True
        self.last_output_path: pathlib.Path | None = None

    def play_pause(self) -> None:
        """Toggle the default playback state stored in generated dashboards."""
        self.playing = not self.playing

    def render(
        self,
        trajectory: np.ndarray,
        output_path: pathlib.Path | str | None = None,
        *,
        title: str | None = None,
        metadata: Optional[dict[str, Any]] = None,
        artifacts: Optional[dict[str, Any]] = None,
    ) -> pathlib.Path:
        """Write a standalone HTML dashboard for a rollout or batched rollouts.

        Parameters
        ----------
        trajectory
            Rollout states to visualize.
        output_path
            Destination HTML path. Uses the renderer default when omitted.
        title
            Optional dashboard title.
        metadata
            Optional JSON-serializable metadata included in the payload.
        artifacts
            Optional overlay payload with an ``overlays`` list. Supported overlay
            types are ``paths`` and ``sample_paths``.
        """
        traj = _normalize_trajectory(
            trajectory,
            num_agents=self.env.num_agents,
            layout=self.trajectory_layout,
        )
        artifact_payload = _normalize_artifacts(artifacts)
        path = (
            pathlib.Path(output_path) if output_path is not None else self.output_path
        )
        path.parent.mkdir(parents=True, exist_ok=True)

        page_payload = _payload(
            self.env,
            traj,
            artifact_payload,
            self.dt,
            title or "F1TENTH Gym JAX Rollout Dashboard",
            metadata,
            self.window_width,
            self.window_height,
            self.playing,
        )
        payload_json = json.dumps(page_payload, separators=(",", ":"), allow_nan=False)
        html = _HTML_TEMPLATE.replace("__PAYLOAD__", payload_json)
        path.write_text(html, encoding="utf-8")
        self.last_output_path = path

        if self.open_browser:
            webbrowser.open(path.resolve().as_uri())
        return path

    def close(self) -> None:
        """No-op compatibility hook for the former desktop renderer API."""
        self.playing = False


TrajRenderer = WebRenderer


_HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>F1TENTH Gym JAX Rollout Dashboard</title>
<style>
:root {
  color-scheme: light;
  --bg: #f4f6f8;
  --panel: #ffffff;
  --ink: #17202a;
  --muted: #647180;
  --border: #d9e0e7;
  --accent: #1f6feb;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--ink);
  font: 14px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
header {
  padding: 20px 24px 12px;
  border-bottom: 1px solid var(--border);
  background: var(--panel);
}
h1 { margin: 0 0 6px; font-size: 26px; letter-spacing: 0; }
h2 { margin: 0 0 12px; font-size: 18px; letter-spacing: 0; }
main { padding: 20px 24px 28px; display: grid; gap: 18px; }
.meta { color: var(--muted); display: flex; flex-wrap: wrap; gap: 12px; }
.stats {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 10px;
}
.stat, .panel {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 8px;
}
.stat { padding: 12px 14px; }
.stat span { display: block; color: var(--muted); font-size: 12px; }
.stat strong { display: block; margin-top: 4px; font-size: 22px; }
.panel { padding: 14px; min-width: 0; }
.grid { display: grid; grid-template-columns: minmax(0, 1fr); gap: 18px; }
canvas {
  display: block;
  width: 100%;
  max-height: 70vh;
  background: #fff;
  border: 1px solid var(--border);
  border-radius: 6px;
  cursor: grab;
  touch-action: none;
}
canvas.is-panning { cursor: grabbing; }
.canvas-toolbar {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: flex-end;
  gap: 10px;
  margin-bottom: 10px;
}
.controls {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 10px;
  align-items: center;
  margin-bottom: 12px;
}
.control-pair {
  display: grid;
  grid-template-columns: auto minmax(0, 1fr);
  gap: 8px;
  align-items: center;
}
.option-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 8px;
}
.option-group + .option-group { margin-top: 12px; }
.option-group h3 {
  margin: 0 0 8px;
  color: var(--muted);
  font-size: 13px;
  letter-spacing: 0;
}
.check-option {
  display: flex;
  align-items: center;
  gap: 8px;
  min-height: 34px;
  padding: 0 10px;
  border: 1px solid var(--border);
  border-radius: 6px;
  background: #fff;
}
button, select, input {
  font: inherit;
}
button, select {
  min-height: 34px;
  border: 1px solid var(--border);
  background: #fff;
  border-radius: 6px;
  padding: 0 10px;
}
button { color: var(--accent); font-weight: 650; cursor: pointer; }
input[type="range"] { width: 100%; }
label { color: var(--muted); font-weight: 650; }
table { width: 100%; border-collapse: collapse; }
th, td { padding: 8px 10px; border-bottom: 1px solid var(--border); text-align: right; }
th:first-child, td:first-child { text-align: left; }
th { color: var(--muted); font-weight: 650; }
.legend { display: flex; flex-wrap: wrap; gap: 10px 16px; margin-top: 10px; color: var(--muted); }
.swatch { display: inline-block; width: 11px; height: 11px; border-radius: 2px; margin-right: 6px; vertical-align: -1px; }
</style>
</head>
<body>
<script id="rollout-data" type="application/json">__PAYLOAD__</script>
<header>
  <h1 id="title">F1TENTH Gym JAX Rollout Dashboard</h1>
  <div class="meta" id="meta"></div>
</header>
<main>
  <section class="stats" id="summary"></section>
  <section class="panel">
    <h2>Batched Rollout Overview</h2>
    <div class="canvas-toolbar">
      <span id="overviewZoom">100%</span>
      <button id="resetOverviewView">Reset view</button>
    </div>
    <canvas id="overview"></canvas>
    <div class="legend" id="legend"></div>
  </section>
  <section class="panel">
    <h2>Visualization Options</h2>
    <div class="option-group">
      <h3>Layers</h3>
      <div class="option-grid" id="layerOptions"></div>
    </div>
    <div class="option-group">
      <h3>Agents</h3>
      <div class="option-grid" id="agentOptions"></div>
    </div>
    <div class="option-group" id="artifactOptionsGroup">
      <h3>Artifact overlays</h3>
      <div class="option-grid" id="artifactOptions"></div>
    </div>
  </section>
  <section class="panel">
    <h2>Trajectory Playback</h2>
    <div class="controls">
      <div class="control-pair">
        <label for="rolloutSelect">Rollout</label>
        <select id="rolloutSelect"></select>
      </div>
      <div class="control-pair">
        <label for="cameraSelect">Camera</label>
        <select id="cameraSelect"></select>
      </div>
      <div class="control-pair">
        <label for="stepRange">Timestep scrubber</label>
        <input id="stepRange" type="range" min="0" value="0" step="1">
      </div>
      <div class="control-pair">
        <label for="speedRange">Speed multiplier</label>
        <input id="speedRange" type="range" min="0.1" max="4" step="0.1" value="1">
      </div>
      <button id="playPause">Pause</button>
      <button id="resetPlaybackView">Reset view</button>
      <span id="speedLabel">1.0x actual real time</span>
      <span id="stepLabel">step 0</span>
      <span id="playbackZoom">100%</span>
    </div>
    <canvas id="playback"></canvas>
    <div class="legend" id="artifactLegend"></div>
  </section>
  <section class="panel">
    <h2>Rollout Stats</h2>
    <table>
      <thead>
        <tr>
          <th>Rollout</th>
          <th>Total distance (m)</th>
          <th>Mean agent distance (m)</th>
          <th>Mean speed (m/s)</th>
          <th>Max speed (m/s)</th>
          <th>Final step</th>
        </tr>
      </thead>
      <tbody id="statsRows"></tbody>
    </table>
  </section>
</main>
<script>
const payload = JSON.parse(document.getElementById("rollout-data").textContent);
const mapImage = new Image();
if (payload.map.image) {
  mapImage.src = payload.map.image;
}

let rolloutIndex = 0;
let stepIndex = 0;
let speedMultiplier = 1.0;
let playing = Boolean(payload.playing);
let lastTimestamp = 0;
let carriedMs = 0;
let cameraTarget = "free";
const viewState = {
  overview: { zoom: 1, panX: 0, panY: 0, dragging: false, lastX: 0, lastY: 0 },
  playback: { zoom: 1, panX: 0, panY: 0, dragging: false, lastX: 0, lastY: 0 },
};
const layerOptions = {
  map: true,
  centerline: true,
  raceline: true,
  labels: true,
  markers: true,
  overviewOtherRollouts: true,
  playbackFullTrace: true,
  playbackHistory: true,
  vehicles: true,
};
const layerLabels = {
  map: "Map",
  centerline: "Centerline",
  raceline: "Raceline",
  labels: "Labels",
  markers: "Start/end markers",
  overviewOtherRollouts: "Other rollouts",
  playbackFullTrace: "Full trace",
  playbackHistory: "History trace",
  vehicles: "Vehicles",
};
const visibleAgents = payload.env.agents.map(() => true);
const artifactVisibility = payload.artifacts.overlays.map(overlay => Boolean(overlay.visible));

function fmt(value, digits = 2) {
  return Number(value).toFixed(digits);
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function canvasHeight() {
  return Math.max(360, Math.floor(payload.canvas.height * 0.68));
}

function setCanvasSize(canvas) {
  const ratio = window.devicePixelRatio || 1;
  const width = payload.canvas.width;
  const height = canvasHeight();
  canvas.width = Math.floor(width * ratio);
  canvas.height = Math.floor(height * ratio);
  canvas.style.maxWidth = width + "px";
  canvas.style.height = height + "px";
  const ctx = canvas.getContext("2d");
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  return { width, height, ctx };
}

function makeBaseTransform(width, height) {
  const b = payload.bounds;
  const margin = 32;
  const rangeX = b.maxX - b.minX;
  const rangeY = b.maxY - b.minY;
  const scale = Math.min((width - 2 * margin) / rangeX, (height - 2 * margin) / rangeY);
  const usedX = rangeX * scale;
  const usedY = rangeY * scale;
  const extraX = width - 2 * margin - usedX;
  const extraY = height - 2 * margin - usedY;
  return {
    scale,
    x: value => margin + extraX / 2 + (value - b.minX) * scale,
    y: value => height - margin - extraY / 2 - (value - b.minY) * scale,
  };
}

function makeTransform(width, height, view, focusPoint = null) {
  const base = makeBaseTransform(width, height);
  if (focusPoint) {
    view.panX = width / 2 - base.x(focusPoint[0]) * view.zoom;
    view.panY = height / 2 - base.y(focusPoint[1]) * view.zoom;
  }
  return {
    scale: base.scale * view.zoom,
    x: value => base.x(value) * view.zoom + view.panX,
    y: value => base.y(value) * view.zoom + view.panY,
    length: value => value * base.scale * view.zoom,
  };
}

function canvasPoint(event, canvas) {
  const rect = canvas.getBoundingClientRect();
  return {
    x: (event.clientX - rect.left) * (payload.canvas.width / rect.width),
    y: (event.clientY - rect.top) * (canvasHeight() / rect.height),
  };
}

function zoomView(view, point, deltaY) {
  const nextZoom = clamp(view.zoom * (deltaY < 0 ? 1.14 : 1 / 1.14), 0.2, 50);
  const factor = nextZoom / view.zoom;
  view.panX = point.x - (point.x - view.panX) * factor;
  view.panY = point.y - (point.y - view.panY) * factor;
  view.zoom = nextZoom;
}

function resetView(kind) {
  viewState[kind].zoom = 1;
  viewState[kind].panX = 0;
  viewState[kind].panY = 0;
  if (kind === "playback") {
    cameraTarget = "free";
    document.getElementById("cameraSelect").value = cameraTarget;
  }
  drawAll();
}

function updateZoomLabels() {
  document.getElementById("overviewZoom").textContent = `${Math.round(viewState.overview.zoom * 100)}%`;
  document.getElementById("playbackZoom").textContent = `${Math.round(viewState.playback.zoom * 100)}%`;
}

function artifactStepIndex(overlay) {
  if (!overlay.stepIndices || overlay.stepIndices.length === 0) {
    return Math.min(stepIndex, overlay.points.length - 1);
  }
  let best = 0;
  let bestDistance = Math.abs(stepIndex - overlay.stepIndices[0]);
  for (let i = 1; i < overlay.stepIndices.length; i += 1) {
    const distance = Math.abs(stepIndex - overlay.stepIndices[i]);
    if (distance < bestDistance) {
      best = i;
      bestDistance = distance;
    }
  }
  return best;
}

function valueFraction(value, overlay) {
  if (value === null || value === undefined || overlay.valueMin === null || overlay.valueMax === null) {
    return 0.5;
  }
  const span = Math.max(overlay.valueMax - overlay.valueMin, 1e-9);
  const fraction = clamp((value - overlay.valueMin) / span, 0, 1);
  return overlay.valueMode === "higher_better" ? 1 - fraction : fraction;
}

function heatColor(fraction) {
  const stops = [
    [37, 99, 235],
    [34, 197, 94],
    [245, 158, 11],
    [220, 38, 38],
  ];
  const scaled = clamp(fraction, 0, 1) * (stops.length - 1);
  const index = Math.min(stops.length - 2, Math.floor(scaled));
  const t = scaled - index;
  const a = stops[index];
  const b = stops[index + 1];
  const rgb = a.map((channel, i) => Math.round(channel + (b[i] - channel) * t));
  return `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})`;
}

function meanValue(values) {
  if (!values || values.length === 0) {
    return null;
  }
  return values.reduce((total, value) => total + Number(value), 0) / values.length;
}

function drawMap(ctx, tr) {
  if (!layerOptions.map || !payload.map.image || !mapImage.complete) {
    return;
  }
  const x = tr.x(payload.map.origin[0]);
  const y = tr.y(payload.map.origin[1] + payload.map.height * payload.map.resolution);
  const width = tr.length(payload.map.width * payload.map.resolution);
  const height = tr.length(payload.map.height * payload.map.resolution);
  ctx.save();
  ctx.globalAlpha = 0.36;
  ctx.drawImage(mapImage, x, y, width, height);
  ctx.restore();
}

function drawPolyline(ctx, tr, points, color, width, label) {
  if (!points || points.length < 2) {
    return;
  }
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.beginPath();
  ctx.moveTo(tr.x(points[0][0]), tr.y(points[0][1]));
  for (let i = 1; i < points.length; i += 1) {
    ctx.lineTo(tr.x(points[i][0]), tr.y(points[i][1]));
  }
  ctx.stroke();
  if (label) {
    const index = Math.min(points.length - 1, Math.floor(points.length * 0.08));
    ctx.fillStyle = color;
    ctx.font = "12px system-ui, sans-serif";
    ctx.fillText(label, tr.x(points[index][0]) + 6, tr.y(points[index][1]) - 6);
  }
  ctx.restore();
}

function drawBase(ctx, tr) {
  ctx.clearRect(0, 0, payload.canvas.width, payload.canvas.height);
  drawMap(ctx, tr);
  if (layerOptions.centerline) {
    drawPolyline(ctx, tr, payload.track.centerline, "#546a7b", 1.2, layerOptions.labels ? "centerline" : "");
  }
  if (layerOptions.raceline) {
    drawPolyline(ctx, tr, payload.track.raceline, "#111827", 1.8, layerOptions.labels ? "raceline" : "");
  }
}

function overlayScopeMatches(overlay, scope) {
  return overlay.scope === scope || overlay.scope === "both";
}

function drawArtifactPoint(ctx, tr, point, color, radius) {
  ctx.save();
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.arc(tr.x(point[0]), tr.y(point[1]), radius, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();
}

function drawArtifactOverlays(ctx, tr, scope) {
  payload.artifacts.overlays.forEach((overlay, overlayIndex) => {
    if (!artifactVisibility[overlayIndex] || !overlayScopeMatches(overlay, scope)) {
      return;
    }
    const overlayStep = artifactStepIndex(overlay);
    const rolloutData = overlay.points[overlayStep]?.[rolloutIndex];
    const valueData = overlay.values ? overlay.values[overlayStep]?.[rolloutIndex] : null;
    if (!rolloutData) {
      return;
    }

    for (let a = 0; a < payload.env.numAgents; a += 1) {
      if (!visibleAgents[a] || !rolloutData[a]) {
        continue;
      }

      if (overlay.type === "paths") {
        ctx.globalAlpha = overlay.opacity;
        drawPolyline(ctx, tr, rolloutData[a], overlay.color, overlay.lineWidth, layerOptions.labels ? overlay.label : "");
        continue;
      }

      const samples = rolloutData[a];
      const sampleValues = valueData ? valueData[a] : null;
      for (let sampleIndex = 0; sampleIndex < samples.length; sampleIndex += 1) {
        const points = samples[sampleIndex];
        const values = sampleValues ? sampleValues[sampleIndex] : null;
        const value = meanValue(values);
        const color = values ? heatColor(valueFraction(value, overlay)) : overlay.color;
        ctx.globalAlpha = overlay.opacity;
        drawPolyline(ctx, tr, points, color, overlay.lineWidth, "");
        if (values) {
          for (let i = 0; i < points.length; i += 1) {
            ctx.globalAlpha = overlay.opacity;
            drawArtifactPoint(ctx, tr, points[i], heatColor(valueFraction(values[i], overlay)), overlay.pointRadius);
          }
        }
      }
    }
  });
  ctx.globalAlpha = 1;
}

function drawOverview() {
  const canvas = document.getElementById("overview");
  const { width, height, ctx } = setCanvasSize(canvas);
  const tr = makeTransform(width, height, viewState.overview);
  drawBase(ctx, tr);
  ctx.save();
  for (let r = 0; r < payload.trajectory.length; r += 1) {
    if (!layerOptions.overviewOtherRollouts && r !== rolloutIndex) {
      continue;
    }
    const rollout = payload.trajectory[r];
    for (let a = 0; a < payload.env.numAgents; a += 1) {
      if (!visibleAgents[a]) {
        continue;
      }
      const color = payload.colors[a % payload.colors.length];
      const points = rollout.map(step => [step[a][0], step[a][1]]);
      ctx.globalAlpha = r === rolloutIndex ? 0.92 : 0.34;
      drawPolyline(ctx, tr, points, color, r === rolloutIndex ? 2.2 : 1.1, "");
      const first = points[0];
      const last = points[points.length - 1];
      if (layerOptions.markers) {
        ctx.globalAlpha = 0.95;
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(tr.x(first[0]), tr.y(first[1]), 3.5, 0, Math.PI * 2);
        ctx.fill();
        ctx.beginPath();
        ctx.arc(tr.x(last[0]), tr.y(last[1]), 3.5, 0, Math.PI * 2);
        ctx.strokeStyle = color;
        ctx.lineWidth = 1.4;
        ctx.stroke();
      }
      if (layerOptions.labels) {
        ctx.globalAlpha = 0.95;
        ctx.fillStyle = color;
        ctx.fillText(`start r${r} ${payload.env.agents[a]}`, tr.x(first[0]) + 5, tr.y(first[1]) + 12);
        ctx.fillText(`r${r} ${payload.env.agents[a]}`, tr.x(last[0]) + 5, tr.y(last[1]) - 5);
      }
    }
  }
  ctx.restore();
  drawArtifactOverlays(ctx, tr, "overview");
}

function vehicleCorners(state) {
  const x = state[0];
  const y = state[1];
  const yaw = state[4];
  const halfLength = payload.vehicle.length / 2;
  const halfWidth = payload.vehicle.width / 2;
  const local = [
    [halfLength, halfWidth],
    [halfLength, -halfWidth],
    [-halfLength, -halfWidth],
    [-halfLength, halfWidth],
  ];
  const cosYaw = Math.cos(yaw);
  const sinYaw = Math.sin(yaw);
  return local.map(([lx, ly]) => [
    x + lx * cosYaw - ly * sinYaw,
    y + lx * sinYaw + ly * cosYaw,
  ]);
}

function drawVehicle(ctx, tr, state, agentName, color) {
  const corners = vehicleCorners(state);
  ctx.save();
  ctx.fillStyle = color;
  ctx.strokeStyle = "#111827";
  ctx.lineWidth = 1.2;
  ctx.beginPath();
  ctx.moveTo(tr.x(corners[0][0]), tr.y(corners[0][1]));
  for (let i = 1; i < corners.length; i += 1) {
    ctx.lineTo(tr.x(corners[i][0]), tr.y(corners[i][1]));
  }
  ctx.closePath();
  ctx.fill();
  ctx.stroke();

  const nose = [(corners[0][0] + corners[1][0]) / 2, (corners[0][1] + corners[1][1]) / 2];
  ctx.strokeStyle = "#ffffff";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(tr.x(state[0]), tr.y(state[1]));
  ctx.lineTo(tr.x(nose[0]), tr.y(nose[1]));
  ctx.stroke();

  ctx.fillStyle = "#111827";
  ctx.font = "12px system-ui, sans-serif";
  if (layerOptions.labels) {
    ctx.fillText(
      `${agentName} x=${fmt(state[0], 1)} y=${fmt(state[1], 1)} v=${fmt(state[3], 2)}`,
      tr.x(state[0]) + 7,
      tr.y(state[1]) - 7,
    );
  }
  ctx.restore();
}

function drawPlayback() {
  const canvas = document.getElementById("playback");
  const { width, height, ctx } = setCanvasSize(canvas);
  const rollout = payload.trajectory[rolloutIndex];
  const current = rollout[stepIndex];
  const focusAgent = cameraTarget.startsWith("agent:") ? Number(cameraTarget.split(":")[1]) : null;
  const focusPoint = focusAgent !== null && visibleAgents[focusAgent]
    ? [current[focusAgent][0], current[focusAgent][1]]
    : null;
  const tr = makeTransform(width, height, viewState.playback, focusPoint);
  drawBase(ctx, tr);
  drawArtifactOverlays(ctx, tr, "playback");

  for (let a = 0; a < payload.env.numAgents; a += 1) {
    if (!visibleAgents[a]) {
      continue;
    }
    const color = payload.colors[a % payload.colors.length];
    const fullTrace = rollout.map(step => [step[a][0], step[a][1]]);
    const history = rollout.slice(0, stepIndex + 1).map(step => [step[a][0], step[a][1]]);
    if (layerOptions.playbackFullTrace) {
      ctx.globalAlpha = 0.22;
      drawPolyline(ctx, tr, fullTrace, color, 1.2, "");
    }
    if (layerOptions.playbackHistory) {
      ctx.globalAlpha = 0.95;
      drawPolyline(ctx, tr, history, color, 2.6, layerOptions.labels ? `${payload.env.agents[a]} path` : "");
    }
    if (layerOptions.vehicles) {
      ctx.globalAlpha = 1;
      drawVehicle(ctx, tr, current[a], payload.env.agents[a], color);
    }
  }

  ctx.globalAlpha = 1;
  ctx.fillStyle = "#111827";
  ctx.font = "13px system-ui, sans-serif";
  ctx.fillText(
    `rollout ${rolloutIndex} | step ${stepIndex}/${payload.summary.steps - 1} | t=${fmt(stepIndex * payload.env.dt, 2)}s | ${fmt(speedMultiplier, 1)}x actual real time`,
    16,
    24,
  );
}

function renderStats() {
  document.title = payload.title;
  document.getElementById("title").textContent = payload.title;
  document.getElementById("meta").innerHTML = [
    `env ${payload.env.name}`,
    `map ${payload.env.map}`,
    `reward ${payload.env.reward}`,
    `${payload.env.numAgents} agents`,
    `dt ${fmt(payload.env.dt, 3)}s`,
    `track ${fmt(payload.env.trackLength, 1)}m`,
  ].map(item => `<span>${item}</span>`).join("");

  const cards = [
    ["Rollouts", payload.summary.rollouts],
    ["Steps", payload.summary.steps],
    ["Agents", payload.summary.agents],
    ["Duration", `${fmt(payload.summary.duration, 2)} s`],
    ["Mean speed", `${fmt(payload.summary.meanSpeed, 2)} m/s`],
    ["Max speed", `${fmt(payload.summary.maxSpeed, 2)} m/s`],
    ["Total distance", `${fmt(payload.summary.totalDistance, 2)} m`],
  ];
  document.getElementById("summary").innerHTML = cards.map(([label, value]) =>
    `<div class="stat"><span>${label}</span><strong>${value}</strong></div>`
  ).join("");

  document.getElementById("statsRows").innerHTML = payload.rolloutStats.map(row => `
    <tr>
      <td>rollout ${row.rollout}</td>
      <td>${fmt(row.totalDistance, 2)}</td>
      <td>${fmt(row.meanAgentDistance, 2)}</td>
      <td>${fmt(row.meanSpeed, 2)}</td>
      <td>${fmt(row.maxSpeed, 2)}</td>
      <td>${row.finalStep}</td>
    </tr>
  `).join("");

  document.getElementById("legend").innerHTML = payload.env.agents.map((agent, index) =>
    `<span style="opacity:${visibleAgents[index] ? 1 : 0.38}"><i class="swatch" style="background:${payload.colors[index % payload.colors.length]}"></i>${agent}</span>`
  ).join("") + '<span><i class="swatch" style="background:#111827"></i>raceline</span><span><i class="swatch" style="background:#546a7b"></i>centerline</span>';

  document.getElementById("artifactLegend").innerHTML = payload.artifacts.overlays.map((overlay, index) => {
    const opacity = artifactVisibility[index] ? 1 : 0.38;
    const range = overlay.values
      ? ` (${overlay.valueLabel}: ${fmt(overlay.valueMin, 2)} to ${fmt(overlay.valueMax, 2)})`
      : "";
    return `<span style="opacity:${opacity}"><i class="swatch" style="background:${overlay.color}"></i>${overlay.label}${range}</span>`;
  }).join("");
}

function renderOptions() {
  const layerContainer = document.getElementById("layerOptions");
  layerContainer.innerHTML = Object.keys(layerOptions).map(key => `
    <label class="check-option">
      <input type="checkbox" data-layer="${key}" ${layerOptions[key] ? "checked" : ""}>
      <span>${layerLabels[key]}</span>
    </label>
  `).join("");

  const agentContainer = document.getElementById("agentOptions");
  agentContainer.innerHTML = payload.env.agents.map((agent, index) => `
    <label class="check-option">
      <input type="checkbox" data-agent="${index}" ${visibleAgents[index] ? "checked" : ""}>
      <span><i class="swatch" style="background:${payload.colors[index % payload.colors.length]}"></i>${agent}</span>
    </label>
  `).join("");

  const artifactGroup = document.getElementById("artifactOptionsGroup");
  const artifactContainer = document.getElementById("artifactOptions");
  if (payload.artifacts.overlays.length === 0) {
    artifactGroup.style.display = "none";
  } else {
    artifactGroup.style.display = "";
    artifactContainer.innerHTML = payload.artifacts.overlays.map((overlay, index) => `
      <label class="check-option">
        <input type="checkbox" data-artifact="${index}" ${artifactVisibility[index] ? "checked" : ""}>
        <span><i class="swatch" style="background:${overlay.color}"></i>${overlay.label}</span>
      </label>
    `).join("");
  }

  layerContainer.querySelectorAll("input[data-layer]").forEach(input => {
    input.addEventListener("change", event => {
      layerOptions[event.target.dataset.layer] = event.target.checked;
      renderStats();
      drawAll();
    });
  });
  agentContainer.querySelectorAll("input[data-agent]").forEach(input => {
    input.addEventListener("change", event => {
      visibleAgents[Number(event.target.dataset.agent)] = event.target.checked;
      renderStats();
      drawAll();
    });
  });
  artifactContainer.querySelectorAll("input[data-artifact]").forEach(input => {
    input.addEventListener("change", event => {
      artifactVisibility[Number(event.target.dataset.artifact)] = event.target.checked;
      renderStats();
      drawAll();
    });
  });
}

function syncControls() {
  const rolloutSelect = document.getElementById("rolloutSelect");
  rolloutSelect.innerHTML = "";
  payload.trajectory.forEach((unused, index) => {
    const option = document.createElement("option");
    option.value = String(index);
    option.textContent = `rollout ${index}`;
    rolloutSelect.appendChild(option);
  });
  const cameraSelect = document.getElementById("cameraSelect");
  cameraSelect.innerHTML = "";
  const freeOption = document.createElement("option");
  freeOption.value = "free";
  freeOption.textContent = "Free camera";
  cameraSelect.appendChild(freeOption);
  payload.env.agents.forEach((agent, index) => {
    const option = document.createElement("option");
    option.value = `agent:${index}`;
    option.textContent = `Center ${agent}`;
    cameraSelect.appendChild(option);
  });
  cameraSelect.value = cameraTarget;
  const stepRange = document.getElementById("stepRange");
  stepRange.max = String(payload.summary.steps - 1);
  stepRange.value = String(stepIndex);
  document.getElementById("stepLabel").textContent = `step ${stepIndex} / ${payload.summary.steps - 1}`;
  document.getElementById("playPause").textContent = playing ? "Pause" : "Play";
}

function drawAll() {
  document.getElementById("stepRange").value = String(stepIndex);
  document.getElementById("stepLabel").textContent = `step ${stepIndex} / ${payload.summary.steps - 1}`;
  document.getElementById("speedLabel").textContent = `${fmt(speedMultiplier, 1)}x actual real time`;
  updateZoomLabels();
  drawOverview();
  drawPlayback();
}

function bindCanvasPanZoom(canvasId, kind) {
  const canvas = document.getElementById(canvasId);
  const view = viewState[kind];

  canvas.addEventListener("wheel", event => {
    event.preventDefault();
    zoomView(view, canvasPoint(event, canvas), event.deltaY);
    drawAll();
  }, { passive: false });

  canvas.addEventListener("pointerdown", event => {
    canvas.setPointerCapture(event.pointerId);
    const point = canvasPoint(event, canvas);
    view.dragging = true;
    view.lastX = point.x;
    view.lastY = point.y;
    canvas.classList.add("is-panning");
    if (kind === "playback" && cameraTarget !== "free") {
      cameraTarget = "free";
      document.getElementById("cameraSelect").value = cameraTarget;
    }
  });

  canvas.addEventListener("pointermove", event => {
    if (!view.dragging) {
      return;
    }
    const point = canvasPoint(event, canvas);
    view.panX += point.x - view.lastX;
    view.panY += point.y - view.lastY;
    view.lastX = point.x;
    view.lastY = point.y;
    drawAll();
  });

  const stopDragging = event => {
    view.dragging = false;
    canvas.classList.remove("is-panning");
    if (canvas.hasPointerCapture(event.pointerId)) {
      canvas.releasePointerCapture(event.pointerId);
    }
  };
  canvas.addEventListener("pointerup", stopDragging);
  canvas.addEventListener("pointercancel", stopDragging);
  canvas.addEventListener("dblclick", () => resetView(kind));
}

document.getElementById("rolloutSelect").addEventListener("change", event => {
  rolloutIndex = Number(event.target.value);
  stepIndex = Math.min(stepIndex, payload.summary.steps - 1);
  drawAll();
});
document.getElementById("cameraSelect").addEventListener("change", event => {
  cameraTarget = event.target.value;
  drawAll();
});
document.getElementById("stepRange").addEventListener("input", event => {
  stepIndex = Number(event.target.value);
  drawAll();
});
document.getElementById("speedRange").addEventListener("input", event => {
  speedMultiplier = Number(event.target.value);
  drawAll();
});
document.getElementById("playPause").addEventListener("click", event => {
  playing = !playing;
  event.target.textContent = playing ? "Pause" : "Play";
});
document.getElementById("resetOverviewView").addEventListener("click", () => resetView("overview"));
document.getElementById("resetPlaybackView").addEventListener("click", () => resetView("playback"));

function animate(timestamp) {
  if (!lastTimestamp) {
    lastTimestamp = timestamp;
  }
  const elapsed = timestamp - lastTimestamp;
  lastTimestamp = timestamp;
  if (playing && payload.summary.steps > 1) {
    carriedMs += elapsed;
    const interval = Math.max(1, (payload.env.dt * 1000) / speedMultiplier);
    while (carriedMs >= interval) {
      stepIndex = (stepIndex + 1) % payload.summary.steps;
      carriedMs -= interval;
    }
    drawAll();
  }
  window.requestAnimationFrame(animate);
}

renderStats();
renderOptions();
syncControls();
bindCanvasPanZoom("overview", "overview");
bindCanvasPanZoom("playback", "playback");
if (mapImage.complete || !payload.map.image) {
  drawAll();
} else {
  mapImage.addEventListener("load", drawAll, { once: true });
}
window.addEventListener("resize", drawAll);
window.requestAnimationFrame(animate);
</script>
</body>
</html>
"""
