import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".99"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import jax
import numpy as np
from train_ppo_example import (
    GaussianActorFF,
    TrainConfig,
    batchify,
    load_params,
    unbatchify,
)

from f1tenth_gym_jax import make
from f1tenth_gym_jax.envs.rendering.renderer import TrajRenderer


def main():
    # config
    config = TrainConfig()
    # load trained model
    actor_params = load_params(
        f"./trained_models_ppo/{config.run_name}/{config.env_name}_actor_params.safetensors"
    )
    actor_network = GaussianActorFF(config=config)

    # Create the environment
    env = make(config.env_name)
    num_envs = 10
    num_actors = num_envs * env.num_agents
    num_steps = 12000

    def _init_env():
        key = jax.random.key(config.seed)
        rng, _rng = jax.random.split(key)
        reset_rng = jax.random.split(_rng, num_envs)
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)
        return (env_state, obsv, rng)

    def _env_step(runner_state, unused):
        env_state, last_obsv, rng = runner_state
        rng, action_rng, _rng = jax.random.split(rng, 3)
        step_rngs = jax.random.split(_rng, num_envs)
        batched_obs = batchify(last_obsv, env.agents, num_actors)
        pi = actor_network.apply(actor_params, batched_obs)
        batched_actions = pi.sample(seed=action_rng)
        actions = unbatchify(batched_actions, env.agents, num_envs, env.num_agents)
        obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0))(
            step_rngs, env_state, actions
        )

        runner_state = (env_state, obsv, rng)
        results_out = (env_state, obsv, batchify(reward, env.agents, num_actors), batchify(done, env.agents, num_actors), info)
        return runner_state, results_out

    final_runner, results_traj = jax.lax.scan(_env_step, _init_env(), length=num_steps)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(10, 10))
    for i in range(num_envs):
        ax[0].plot(results_traj[2][:, i, 0], label=f"env {i}, agent 0")
    ax[0].legend()
    ax[0].set_title("Reward per step")
    ax[0].set_xlabel("Step")
    ax[0].set_ylabel("Reward")

    for i in range(num_envs):
        ax[1].plot(results_traj[3][:, i, 0], label=f"env {i}, agent 0")
    ax[1].legend()
    ax[1].set_title("Done per step")
    ax[1].set_xlabel("Step")
    ax[1].set_ylabel("Done")
    plt.show()

    player = TrajRenderer(env, 1/(env.params.timestep * env.params.timestep_ratio))
    player.render(np.array(results_traj[0].cartesian_states))


if __name__ == "__main__":
    main()