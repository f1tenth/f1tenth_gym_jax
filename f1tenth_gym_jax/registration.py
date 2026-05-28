import pathlib

from .envs import F110Env
from .envs.utils import Param

_VALID_SCAN_MODES = {"scan", "noscan"}
_VALID_COLLISION_MODES = {"collision", "nocollision"}
_VALID_LONGITUDINAL_CONTROLS = {"acceleration", "velocity"}
_VALID_STEERING_CONTROLS = {"steeringangle", "steeringvelocity"}
_VALID_REWARDS = {"time", "progress", "alive"}


# scenario patterns
# {map}_{num_agents}_{scan|noscan}_{collision|nocollision}_{rewards}_{longitudinal+steering}_{timestep_ratio}_{max_steps}_v0
def _parse_scenario(scenario: str):
    scenario_parts = scenario.split("_")
    if len(scenario_parts) < 9:
        raise ValueError(
            "Environment ID must follow "
            "{map}_{num_agents}_{scan|noscan}_{collision|nocollision}_{rewards}_"
            "{longitudinal+steering}_{timestep_ratio}_{max_steps}_v0."
        )
    if scenario_parts[-1] != "v0":
        raise ValueError(f"Invalid environment version: {scenario_parts[-1]}.")

    map_name = "_".join(scenario_parts[:-8])
    if not map_name:
        raise ValueError("Environment ID is missing a map name.")

    (
        num_agents_raw,
        scan_mode,
        collision_mode,
        reward_type,
        control_type,
        timestep_ratio_raw,
        max_steps_raw,
    ) = scenario_parts[-8:-1]

    try:
        num_agents = int(num_agents_raw)
    except ValueError as exc:
        raise ValueError(
            f"Invalid number of agents: {num_agents_raw}, must be an integer."
        ) from exc

    # check whether num_agents is valid
    if num_agents < 1:
        raise ValueError(f"Invalid number of agents: {num_agents}")

    if scan_mode not in _VALID_SCAN_MODES:
        raise ValueError(
            f"Invalid scan mode: {scan_mode}, must be from {sorted(_VALID_SCAN_MODES)}."
        )
    produce_scan = scan_mode == "scan"

    if collision_mode not in _VALID_COLLISION_MODES:
        raise ValueError(
            f"Invalid collision mode: {collision_mode}, "
            f"must be from {sorted(_VALID_COLLISION_MODES)}."
        )
    collision_on = collision_mode == "collision"

    controls = control_type.split("+")
    if len(controls) != 2:
        raise ValueError(
            f"Invalid control type: {control_type}, expected longitudinal+steering."
        )
    long_type, steer_type = controls
    if long_type not in _VALID_LONGITUDINAL_CONTROLS:
        raise ValueError(
            f"Invalid longitudinal control type: {long_type}, "
            f"must be from {sorted(_VALID_LONGITUDINAL_CONTROLS)}."
        )
    if steer_type not in _VALID_STEERING_CONTROLS:
        raise ValueError(
            f"Invalid steering control type: {steer_type}, "
            f"must be from {sorted(_VALID_STEERING_CONTROLS)}."
        )
    control_type = [long_type, steer_type]

    all_reward_function = set(reward_type.split("+"))
    if not all_reward_function or not all_reward_function.issubset(_VALID_REWARDS):
        raise ValueError(
            f"Invalid reward type list: {all_reward_function}, "
            f"must be from {sorted(_VALID_REWARDS)}."
        )

    timestep_ratio = timestep_ratio_raw
    if timestep_ratio == "v0":
        timestep_ratio = 1
    else:
        try:
            timestep_ratio = int(timestep_ratio)
        except ValueError as exc:
            raise ValueError(
                f"Invalid timestep ratio: {timestep_ratio}, must be an integer."
            ) from exc
        if timestep_ratio <= 0:
            raise ValueError(
                f"Invalid timestep ratio: {timestep_ratio}, must be a positive integer."
            )

    max_steps = max_steps_raw
    if max_steps == "v0":
        max_steps = None
    else:
        try:
            max_steps = int(max_steps)
        except ValueError as exc:
            raise ValueError(
                f"Invalid max steps: {max_steps}, must be an integer."
            ) from exc
        if max_steps <= 0:
            raise ValueError(
                f"Invalid max steps: {max_steps}, must be a positive integer."
            )

    return (
        map_name,
        num_agents,
        produce_scan,
        collision_on,
        reward_type,
        control_type,
        timestep_ratio,
        max_steps,
    )


def make(env_id: str, **env_kwargs):
    (
        map_name,
        num_agents,
        produce_scan,
        collision_on,
        reward_type,
        control_type,
        timestep_ratio,
        max_steps,
    ) = _parse_scenario(env_id)
    available_maps = list_available_maps()
    if map_name not in available_maps:
        raise ValueError(
            f"{map_name} is not a registered map, choose from {available_maps}."
        )
    if max_steps is None:
        max_steps = int(90 / (0.1 * timestep_ratio))

    env = F110Env(
        num_agents=num_agents,
        params=Param(
            map_name=map_name,
            produce_scans=produce_scan,
            collision_on=collision_on,
            reward_type=reward_type,
            longitudinal_action_type=control_type[0],
            steering_action_type=control_type[1],
            timestep_ratio=timestep_ratio,
            max_steps=max_steps,
            **env_kwargs,
        ),
    )

    return env


def list_available_maps() -> list[str]:
    """Return built-in downloadable maps plus map folders available locally."""
    map_dir = pathlib.Path(__file__).parent.parent / "maps"
    local_maps = []
    if map_dir.exists():
        local_maps = [path.name for path in map_dir.iterdir() if path.is_dir()]
    return sorted(set(registered_maps + local_maps))


registered_maps = [
    "Austin",
    "BrandsHatch",
    "Budapest",
    "Catalunya",
    "Hockenheim",
    "IMS",
    "Melbourne",
    "MexicoCity",
    "Montreal",
    "Monza",
    "MoscowRaceway",
    "Nuerburgring",
    "Oschersleben",
    "Sakhir",
    "SaoPaulo",
    "Sepang",
    "Shanghai",
    "Silverstone",
    "Sochi",
    "Spa",
    "Spielberg",
    "Spielberg_blank",
    "YasMarina",
    "Zandvoort",
]
