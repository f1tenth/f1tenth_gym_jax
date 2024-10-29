from .envs import F110Env

def make(env_id: str, **env_kwargs):
    """A JAX-version of OpenAI's env.make(env_name), built off Gymnax"""
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