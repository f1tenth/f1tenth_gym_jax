from .envs import F110Env
from .envs.utils import Param


# scenario patterns
# {map_name}_{num_agent}_{produce_scan}_{reward_type}_v0
# {str}_{int}_{"scan"/"noscan"}_{"time+/progress+/"}_v0
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
    reward_type = scenario[3 + index_bump]
    all_reward_function = reward_type.split("+")
    assert reward_type in [
        "time",
        "",
    ], f"Invalid reward type: {reward_type}, must be one of ['time', '']"
    return map_name, num_agents, produce_scan, reward_type


def make(env_id: str, **env_kwargs):
    map_name, num_agents, produce_scan, reward_type = _parse_scenario(env_id)
    if map_name not in registered_maps:
        raise ValueError(f"{map_name} is not a registered map, choose from {registered_maps}.")
    
    env = F110Env(
        num_agents=num_agents,
        params=Param(
            map_name=map_name,
            produce_scans=produce_scan,
            reward_type=reward_type,
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
