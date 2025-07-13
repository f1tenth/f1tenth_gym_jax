from .envs import F110Env
from .envs.utils import Param


# scenario patterns
# {map_name}_{num_agent}_{produce_scan}_{collision_on}_{reward_type}_{control_type(optional)}_{tsratio}_v0
# {str}_{int}_{"scan"/"noscan"}_{"collision"/"nocollision"}_{"time+/progress+/"}_{accleration+steeringvel/}_{int}_v0
def _parse_scenario(scenario: str):
    scenario = scenario.split("_")
    map_name = scenario[0]
    try:
        num_agents = int(scenario[1])
        index_bump = 0
    except ValueError:
        map_name += "_" + scenario[1]
        num_agents = int(scenario[2])
        index_bump = 1
    # check whether num_agents is valid
    if num_agents < 1:
        raise ValueError(f"Invalid number of agents: {num_agents}")
    produce_scan = scenario[2 + index_bump] == "scan"
    collision_on = scenario[3 + index_bump] == "collision"
    reward_type = scenario[4 + index_bump]
    control_type = scenario[5 + index_bump]
    long_type, steer_type = control_type.split("+")
    assert all(
        c in ["acceleration", "velocity"] for c in [long_type]
    ), f"Invalid longitudinal control type: {long_type}, must be from ['acceleration', 'velocity']"
    assert all(
        c in ["steeringangle", "steeringvelocity"] for c in [steer_type]
    ), f"Invalid steering control type: {steer_type}, must be from ['steeringangle', 'steeringvelocity']"
    control_type = [long_type, steer_type]

    all_reward_function = set(reward_type.split("+"))
    assert all(
        r in ["time", "progress", "alive"] for r in all_reward_function
    ), f"Invalid reward type list: {all_reward_function}, must be from ['time', 'progress', 'alive']"

    timestep_ratio = scenario[6 + index_bump]
    if timestep_ratio == "v0":
        timestep_ratio = 1
    else:
        try:
            timestep_ratio = int(timestep_ratio)
            assert (
                timestep_ratio > 0
            ), f"Invalid timestep ratio: {timestep_ratio}, must be a positive integer."
        except ValueError as exc:
            raise ValueError(
                f"Invalid timestep ratio: {timestep_ratio}, must be an integer."
            ) from exc

    return (
        map_name,
        num_agents,
        produce_scan,
        collision_on,
        reward_type,
        control_type,
        timestep_ratio,
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
    ) = _parse_scenario(env_id)
    if map_name not in registered_maps:
        raise ValueError(
            f"{map_name} is not a registered map, choose from {registered_maps}."
        )

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
            **env_kwargs,
        ),
    )

    return env


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
