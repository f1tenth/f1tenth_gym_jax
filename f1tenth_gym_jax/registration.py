from .envs import F110Env

# scenario patterns
# {num_agent}_{produce_scan}_{reward_type}_v0
# {int}_{"scan"/"noscan"}_{"time+/progress+/"}_v0
def _parse_scenario(scenario: str):
    scenario = scenario.split("_")
    num_agents = int(scenario[0])
    # check whether num_agents is valid
    if num_agents < 1:
        raise ValueError(f"Invalid number of agents: {num_agents}")
    produce_scan = scenario[1] == "scan"
    reward_type = scenario[2]
    all_reward_function = reward_type.split("+")
    assert reward_type in ["time", ""], f"Invalid reward type: {reward_type}, must be one of ['time', '']"
    return num_agents, produce_scan, reward_type


def make(env_id: str, **env_kwargs):
    if env_id not in registered_envs:
        raise ValueError(f"{env_id} is not a registered environment.")

    # 1. MPE PettingZoo Environments
    if env_id == "default_v0":
        env = F110Env(**env_kwargs)
    elif env_id == "two_car_racing_v0":
        env = F110Env(**env_kwargs)
    elif env_id == "time_trial_v0":
        env = F110Env(**env_kwargs)
    elif env_id == "three_car_v0":
        env = F110Env(**env_kwargs)
    elif env_id == "time_trial_v0":
        env = F110Env(**env_kwargs)

    return env

registered_envs = [
    "default_v0",
    "two_car_racing_v0",
    "time_trial_v0",
    "three_car_v0",
    "time_trial_v0",
]