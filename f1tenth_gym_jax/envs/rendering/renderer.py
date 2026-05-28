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


def _series_xy(xs: Any, ys: Any) -> list[list[float]]:
    x_arr = np.asarray(xs, dtype=float)
    y_arr = np.asarray(ys, dtype=float)
    points = np.column_stack((x_arr, y_arr))
    return np.round(points, 6).tolist()


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


def _bounds(env: F110Env, traj: np.ndarray) -> dict[str, float]:
    xs = [traj[:, :, :, 0].reshape(-1)]
    ys = [traj[:, :, :, 1].reshape(-1)]

    for spline in (env.track.centerline, env.track.raceline):
        xs.append(np.asarray(spline.xs, dtype=float))
        ys.append(np.asarray(spline.ys, dtype=float))

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
    dt: float,
    title: str,
    metadata: Optional[dict[str, Any]],
    canvas_width: int,
    canvas_height: int,
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
        "summary": summary,
        "rolloutStats": per_rollout,
        "bounds": _bounds(env, traj),
        "colors": _COLORS,
        "canvas": {"width": int(canvas_width), "height": int(canvas_height)},
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
        trajectory_layout: TrajectoryLayout = "auto",
    ):
        if render_mode == "rgb_array":
            raise ValueError(
                "rgb_array rendering was removed. Use the web dashboard output instead."
            )
        if trajectory_layout not in {"auto", "step_major", "batch_major"}:
            raise ValueError(
                "trajectory_layout must be 'auto', 'step_major', or 'batch_major'."
            )

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
    ) -> pathlib.Path:
        """Write a standalone HTML dashboard for a rollout or batched rollouts."""
        traj = _normalize_trajectory(
            trajectory,
            num_agents=self.env.num_agents,
            layout=self.trajectory_layout,
        )
        path = (
            pathlib.Path(output_path) if output_path is not None else self.output_path
        )
        path.parent.mkdir(parents=True, exist_ok=True)

        page_payload = _payload(
            self.env,
            traj,
            self.dt,
            title or "F1TENTH Gym JAX Rollout Dashboard",
            metadata,
            self.window_width,
            self.window_height,
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
}
.controls {
  display: grid;
  grid-template-columns: auto minmax(180px, 1fr) auto minmax(180px, 1fr) auto;
  gap: 10px;
  align-items: center;
  margin-bottom: 12px;
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
@media (max-width: 780px) {
  .controls { grid-template-columns: 1fr; }
}
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
    <canvas id="overview"></canvas>
    <div class="legend" id="legend"></div>
  </section>
  <section class="panel">
    <h2>Trajectory Playback</h2>
    <div class="controls">
      <label for="rolloutSelect">Rollout</label>
      <select id="rolloutSelect"></select>
      <label for="stepRange">Timestep scrubber</label>
      <input id="stepRange" type="range" min="0" value="0" step="1">
      <button id="playPause">Pause</button>
      <label for="speedRange">Speed multiplier</label>
      <input id="speedRange" type="range" min="0.1" max="4" step="0.1" value="1">
      <span id="speedLabel">1.0x actual real time</span>
      <span id="stepLabel">step 0</span>
    </div>
    <canvas id="playback"></canvas>
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
let playing = true;
let lastTimestamp = 0;
let carriedMs = 0;

function fmt(value, digits = 2) {
  return Number(value).toFixed(digits);
}

function setCanvasSize(canvas) {
  const ratio = window.devicePixelRatio || 1;
  const width = payload.canvas.width;
  const height = Math.max(360, Math.floor(payload.canvas.height * 0.68));
  canvas.width = Math.floor(width * ratio);
  canvas.height = Math.floor(height * ratio);
  canvas.style.maxWidth = width + "px";
  canvas.style.height = height + "px";
  const ctx = canvas.getContext("2d");
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  return { width, height, ctx };
}

function makeTransform(width, height) {
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
    length: value => value * scale,
  };
}

function drawMap(ctx, tr) {
  if (!payload.map.image || !mapImage.complete) {
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
  drawPolyline(ctx, tr, payload.track.centerline, "#546a7b", 1.2, "centerline");
  drawPolyline(ctx, tr, payload.track.raceline, "#111827", 1.8, "raceline");
}

function drawOverview() {
  const canvas = document.getElementById("overview");
  const { width, height, ctx } = setCanvasSize(canvas);
  const tr = makeTransform(width, height);
  drawBase(ctx, tr);
  ctx.save();
  for (let r = 0; r < payload.trajectory.length; r += 1) {
    const rollout = payload.trajectory[r];
    for (let a = 0; a < payload.env.numAgents; a += 1) {
      const color = payload.colors[a % payload.colors.length];
      const points = rollout.map(step => [step[a][0], step[a][1]]);
      ctx.globalAlpha = r === rolloutIndex ? 0.92 : 0.34;
      drawPolyline(ctx, tr, points, color, r === rolloutIndex ? 2.2 : 1.1, "");
      const first = points[0];
      const last = points[points.length - 1];
      ctx.globalAlpha = 0.95;
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(tr.x(first[0]), tr.y(first[1]), 3.5, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillText(`start r${r} ${payload.env.agents[a]}`, tr.x(first[0]) + 5, tr.y(first[1]) + 12);
      ctx.fillText(`r${r} ${payload.env.agents[a]}`, tr.x(last[0]) + 5, tr.y(last[1]) - 5);
    }
  }
  ctx.restore();
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
  ctx.fillText(
    `${agentName} x=${fmt(state[0], 1)} y=${fmt(state[1], 1)} v=${fmt(state[3], 2)}`,
    tr.x(state[0]) + 7,
    tr.y(state[1]) - 7,
  );
  ctx.restore();
}

function drawPlayback() {
  const canvas = document.getElementById("playback");
  const { width, height, ctx } = setCanvasSize(canvas);
  const tr = makeTransform(width, height);
  drawBase(ctx, tr);
  const rollout = payload.trajectory[rolloutIndex];
  const current = rollout[stepIndex];

  for (let a = 0; a < payload.env.numAgents; a += 1) {
    const color = payload.colors[a % payload.colors.length];
    const fullTrace = rollout.map(step => [step[a][0], step[a][1]]);
    const history = rollout.slice(0, stepIndex + 1).map(step => [step[a][0], step[a][1]]);
    ctx.globalAlpha = 0.22;
    drawPolyline(ctx, tr, fullTrace, color, 1.2, "");
    ctx.globalAlpha = 0.95;
    drawPolyline(ctx, tr, history, color, 2.6, `${payload.env.agents[a]} path`);
    drawVehicle(ctx, tr, current[a], payload.env.agents[a], color);
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
    `<span><i class="swatch" style="background:${payload.colors[index % payload.colors.length]}"></i>${agent}</span>`
  ).join("") + "<span><i class=\"swatch\" style=\"background:#111827\"></i>raceline</span><span><i class=\"swatch\" style=\"background:#546a7b\"></i>centerline</span>";
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
  const stepRange = document.getElementById("stepRange");
  stepRange.max = String(payload.summary.steps - 1);
  stepRange.value = String(stepIndex);
  document.getElementById("stepLabel").textContent = `step ${stepIndex} / ${payload.summary.steps - 1}`;
}

function drawAll() {
  document.getElementById("stepRange").value = String(stepIndex);
  document.getElementById("stepLabel").textContent = `step ${stepIndex} / ${payload.summary.steps - 1}`;
  document.getElementById("speedLabel").textContent = `${fmt(speedMultiplier, 1)}x actual real time`;
  drawOverview();
  drawPlayback();
}

document.getElementById("rolloutSelect").addEventListener("change", event => {
  rolloutIndex = Number(event.target.value);
  stepIndex = Math.min(stepIndex, payload.summary.steps - 1);
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
syncControls();
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
