import pathlib

from .envs import F110Env
from .envs.utils import VALID_REWARDS, Param

_VALID_SCAN_MODES = {"scan", "noscan"}
_VALID_COLLISION_MODES = {"collision", "nocollision"}
_VALID_LONGITUDINAL_CONTROLS = {"acceleration", "velocity"}
_VALID_STEERING_CONTROLS = {"steeringangle", "steeringvelocity"}


def _parse_positive_int(raw_value: str, field_name: str) -> int:
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(
            f"Invalid {field_name}: {raw_value}, must be an integer."
        ) from exc
    if value <= 0:
        raise ValueError(f"Invalid {field_name}: {value}, must be a positive integer.")
    return value


def _parse_scenario_fields(
    map_name: str,
    num_agents_raw: str,
    scan_mode: str,
    collision_mode: str,
    reward_type: str,
    control_type_raw: str,
    timestep_ratio_raw: str,
    max_steps_raw: str,
):
    if not map_name:
        raise ValueError("Environment ID is missing a map name.")

    num_agents = _parse_positive_int(num_agents_raw, "number of agents")

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

    controls = control_type_raw.split("+")
    if len(controls) != 2:
        raise ValueError(
            f"Invalid control type: {control_type_raw}, expected longitudinal+steering."
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
    if not all_reward_function or not all_reward_function.issubset(VALID_REWARDS):
        raise ValueError(
            f"Invalid reward type list: {all_reward_function}, "
            f"must be from {sorted(VALID_REWARDS)}."
        )

    timestep_ratio = _parse_positive_int(timestep_ratio_raw, "timestep ratio")

    if max_steps_raw == "v0":
        max_steps = None
    else:
        max_steps = _parse_positive_int(max_steps_raw, "max steps")

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


def _candidate_scenario_fields(scenario_parts: list[str]):
    # Current canonical form:
    # {map}_{num_agents}_{scan|noscan}_{collision|nocollision}_{rewards}_{controls}_{ratio}_{max_steps}_v0
    if len(scenario_parts) >= 9:
        yield (
            "_".join(scenario_parts[:-8]),
            *scenario_parts[-8:-2],
            scenario_parts[-2],
        )

    # Current default-length shorthand:
    # {map}_{num_agents}_{scan|noscan}_{collision|nocollision}_{rewards}_{controls}_{ratio}_v0
    if len(scenario_parts) >= 8:
        yield (
            "_".join(scenario_parts[:-7]),
            *scenario_parts[-7:-2],
            scenario_parts[-2],
            "v0",
        )

    # Legacy shorthand accepted for old PPO model filenames:
    # {map}_{num_agents}_{scan|noscan}_{collision|nocollision}_{rewards}_{controls}_v0
    if len(scenario_parts) >= 7:
        yield (
            "_".join(scenario_parts[:-6]),
            *scenario_parts[-6:-1],
            "1",
            "v0",
        )


def _parse_scenario(scenario: str):
    scenario_parts = scenario.split("_")
    if len(scenario_parts) < 7:
        raise ValueError(
            "Environment ID must follow "
            "{map}_{num_agents}_{scan|noscan}_{collision|nocollision}_{rewards}_"
            "{longitudinal+steering}[_timestep_ratio][_max_steps]_v0."
        )
    if scenario_parts[-1] != "v0":
        raise ValueError(f"Invalid environment version: {scenario_parts[-1]}.")

    parse_errors = []
    for candidate in _candidate_scenario_fields(scenario_parts):
        try:
            return _parse_scenario_fields(*candidate)
        except ValueError as exc:
            parse_errors.append(exc)

    raise parse_errors[0]


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
    param_kwargs = {
        "map_name": map_name,
        "produce_scans": produce_scan,
        "collision_on": collision_on,
        "reward_type": reward_type,
        "longitudinal_action_type": control_type[0],
        "steering_action_type": control_type[1],
        "timestep": Param().timestep,
        "timestep_ratio": timestep_ratio,
        "max_steps": max_steps,
    }
    param_kwargs.update(env_kwargs)

    available_maps = list_available_maps()
    if param_kwargs["map_name"] not in available_maps:
        raise ValueError(
            f"{param_kwargs['map_name']} is not a registered map, choose from {available_maps}."
        )
    if param_kwargs["max_steps"] is None:
        param_kwargs["max_steps"] = int(
            90 / (param_kwargs["timestep"] * param_kwargs["timestep_ratio"])
        )

    env = F110Env(
        num_agents=num_agents,
        params=Param(**param_kwargs),
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
