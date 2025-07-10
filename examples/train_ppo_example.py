import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".99"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from typing import Callable, List, NamedTuple, Dict, Union

import chex
import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
import wandb
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from flax.traverse_util import flatten_dict, unflatten_dict
from safetensors.flax import load_file, save_file
from tqdm.auto import tqdm

from f1tenth_gym_jax import make
from f1tenth_gym_jax.envs.utils import batchify, unbatchify, LogWrapper


class TrainConfig(NamedTuple):
    # rng
    seed: int = 420

    # sim params
    # {map_name}_{num_agents}_{scan/noscan}_{reward_type}_v0
    num_agents: int = 1
    action_dim: int = 2
    env_name: str = f"Spielberg_{num_agents}_scan_collision_progress+alive_v0"
    num_envs: int = 1024
    num_steps: int = 2048
    total_timesteps: int = int(1.0e8)
    num_minibatches: int = 64
    num_actors: int = num_agents * num_envs
    num_updates: int = total_timesteps // num_steps // num_envs
    minibatch_size: int = num_actors * num_steps // num_minibatches

    # model params
    hidden_dim: int = 256
    activation: str = "tanh"

    # train params
    update_epochs: int = 10
    lr: float = 3.0e-4
    anneal_lr: bool = True
    max_grad_norm: float = 0.5

    # ppo params
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    scale_clip_eps: bool = False
    ent_coef: float = 0.0
    vf_coef: float = 0.5

    # logging
    project: str = "f1tenth_ppo_example"
    run_name: str = "gaussianactor"
    tags: List[str] = ["ppo", "gaussian actor"]


class Transition(NamedTuple):
    global_done: jnp.ndarray
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    next_obs: jnp.ndarray
    world_state: jnp.ndarray
    samples: jnp.ndarray = None
    z: jnp.ndarray = None
    info: jnp.ndarray = None
    which_in_team: jnp.ndarray = None


class GaussianActorFF(nn.Module):
    config: NamedTuple

    @nn.compact
    def __call__(self, obs):
        if self.config.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        actor_mean = nn.Dense(
            self.config.hidden_dim,
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(obs)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.config.hidden_dim,
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(actor_mean)
        actor_mean = activation(actor_mean)
        action_mean = nn.Dense(
            self.config.action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
        )(actor_mean)

        log_std = self.param("log_std", constant(0.0), (self.config.action_dim,))
        std = jnp.exp(log_std)

        pi = distrax.Normal(action_mean, std)

        return pi


class CriticFF(nn.Module):
    config: NamedTuple

    @nn.compact
    def __call__(self, x):
        if self.config.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        critic = nn.Dense(
            self.config.hidden_dim,
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        critic = activation(critic)
        critic = nn.Dense(
            self.config.hidden_dim,
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(critic)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(
            critic
        )

        return jnp.squeeze(critic, axis=-1)


def save_params(params: Dict, filename: Union[str, os.PathLike]) -> None:
    flattened_dict = flatten_dict(params, sep=",")
    save_file(flattened_dict, filename)


def load_params(filename: Union[str, os.PathLike]) -> Dict:
    flattened_dict = load_file(filename)
    return unflatten_dict(flattened_dict, sep=",")


def make_train(config: TrainConfig) -> Callable:
    # make environment
    env = LogWrapper(make(config.env_name))

    pbar = tqdm(total=config.num_updates)

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config.num_minibatches * config.update_epochs))
            / config.num_updates
        )
        return config.lr * frac

    def train(rng: chex.PRNGKey):
        actor_network = GaussianActorFF(config=config)
        critic_network = CriticFF(config=config)

        # init rng
        rng, _rng_actor, _rng_critic = jax.random.split(rng, 3)

        # init x
        ac_init_x = jnp.zeros(
            (
                config.num_actors,
                env.observation_space(env.agents[0]).shape[0],
            )
        )
        actor_network_params = actor_network.init(_rng_actor, ac_init_x)
        cr_init_x = jnp.zeros(
            (
                config.num_envs,
                env.observation_space(env.agents[0]).shape[0],
            )
        )
        critic_network_params = critic_network.init(_rng_critic, cr_init_x)

        # optimizer
        if config.anneal_lr:
            actor_tx = optax.chain(
                optax.clip_by_global_norm(config.max_grad_norm),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
            critic_tx = optax.chain(
                optax.clip_by_global_norm(config.max_grad_norm),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )

        else:
            actor_tx = optax.chain(
                optax.clip_by_global_norm(config.max_grad_norm),
                optax.adam(config.lr, eps=1e-5),
            )
            critic_tx = optax.chain(
                optax.clip_by_global_norm(config.max_grad_norm),
                optax.adam(config.lr, eps=1e-5),
            )

        # train states
        actor_train_state = TrainState.create(
            apply_fn=actor_network.apply,
            params=actor_network_params,
            tx=actor_tx,
        )
        critic_train_state = TrainState.create(
            apply_fn=critic_network.apply,
            params=critic_network_params,
            tx=critic_tx,
        )

        # reset env
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config.num_envs)
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)

        # train loop
        def _update_step(update_runner_state, unused):
            (runner_state, update_steps) = update_runner_state

            def _env_step(runner_state, unused):
                (
                    train_states,
                    env_state,
                    last_obs,
                    last_done,
                    rng,
                ) = runner_state

                # select action
                obs_batch = batchify(last_obs, env.agents, config.num_actors)
                pi = actor_network.apply(
                    train_states[0].params,
                    obs_batch,
                )
                rng, action_rng = jax.random.split(rng)
                current_actions = pi.sample(seed=action_rng)
                log_prob = pi.log_prob(current_actions)
                actions = unbatchify(
                    current_actions, env.agents, config.num_envs, config.num_agents
                )

                # value
                value = critic_network.apply(
                    train_states[1].params,
                    obs_batch,
                )

                # step env
                rng, _rng = jax.random.split(rng)
                step_rngs = jax.random.split(_rng, config.num_envs)
                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0)
                )(step_rngs, env_state, actions)
                info = jax.tree.map(lambda x: x.flatten(), info)
                done_batch = batchify(done, env.agents, config.num_actors).squeeze()
                next_obs_batch = batchify(obsv, env.agents, config.num_actors)

                # action/logprob should be from cost network here
                transition = Transition(
                    global_done=jnp.tile(done["__all__"], env.num_agents),
                    done=last_done,
                    action=current_actions.squeeze(),
                    value=value,
                    reward=batchify(reward, env.agents, config.num_actors).squeeze(),
                    log_prob=log_prob.squeeze(),
                    obs=obs_batch,
                    next_obs=next_obs_batch,
                    world_state=obs_batch,
                    info=info,
                )

                # update states
                next_runner_state = (
                    train_states,
                    env_state,
                    obsv,
                    done_batch,
                    rng,
                )
                return next_runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step,
                runner_state,
                None,
                length=config.num_steps,
            )

            # calculate advantage
            (
                train_states,
                env_state,
                last_obs,
                last_done,
                rng,
            ) = runner_state

            last_world_state = batchify(last_obs, env.agents, config.num_actors)
            last_value = critic_network.apply(
                train_states[1].params,
                last_world_state,
            ).squeeze()

            def _calculate_gae(traj_batch, last_val):
                def _get_advantage(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + config.gamma * next_value * (1.0 - done) - value
                    gae = delta + config.gamma * config.gae_lambda * (1.0 - done) * gae
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantage,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=True,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_value)

            # update networks
            def _update_epoch(update_state, unused):
                def _update_minibatch(train_states, batch_info):
                    (actor_train_state, critic_train_state) = (
                        train_states
                    )
                    (
                        traj_batch_inside,
                        advantages,
                        targets,
                    ) = batch_info

                    def _actor_loss_fn(
                        actor_params,
                        loss_obs,
                        loss_action,
                        loss_log_prob,
                        gae,
                    ):
                        pi = actor_network.apply(
                            actor_params,
                            loss_obs,
                        )
                        log_prob = pi.log_prob(loss_action)

                        logratio = log_prob - loss_log_prob

                        ratio = jnp.exp(logratio)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)

                        loss_actor1 = ratio * gae[..., None]
                        loss_actor2 = (
                            jnp.clip(ratio, 1 - config.clip_eps, 1 + config.clip_eps)
                        ) * gae[..., None]

                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        # debug
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfrac = jnp.mean(jnp.abs(ratio - 1) > config.clip_eps)
                        actor_loss = loss_actor - config.ent_coef * entropy

                        return actor_loss, (
                            loss_actor,
                            entropy,
                            ratio,
                            approx_kl,
                            clipfrac,
                        )

                    def _critic_loss_fn(
                        critic_params, loss_world_state, loss_value, targets
                    ):
                        value = critic_network.apply(
                            critic_params,
                            loss_world_state,
                        )
                        value_pred_clipped = loss_value + (value - loss_value).clip(
                            -config.clip_eps, config.clip_eps
                        )
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )
                        critic_loss = config.vf_coef * value_loss
                        return critic_loss, (value_loss)

                    actor_grad_fn = jax.value_and_grad(
                        _actor_loss_fn,
                        has_aux=True,
                    )
                    critic_grad_fn = jax.value_and_grad(
                        _critic_loss_fn,
                        has_aux=True,
                    )

                    actor_loss, actor_grads = actor_grad_fn(
                        actor_train_state.params,
                        traj_batch_inside.obs,
                        traj_batch_inside.action,
                        traj_batch_inside.log_prob,
                        advantages,
                    )
                    critic_loss, critic_grads = critic_grad_fn(
                        critic_train_state.params,
                        traj_batch_inside.world_state,
                        traj_batch_inside.value,
                        targets,
                    )

                    actor_train_state = actor_train_state.apply_gradients(
                        grads=actor_grads
                    )
                    critic_train_state = critic_train_state.apply_gradients(
                        grads=critic_grads
                    )

                    total_loss = actor_loss[0] + critic_loss[0]

                    loss_info = {
                        "mean_step_returns": traj_batch_inside.reward.mean(),
                        "actor_loss": actor_loss[0],
                        "critic_loss": critic_loss[0],
                        "total_loss": total_loss,
                        "actor_entropy": actor_loss[1][1],
                        "actor_ratio": actor_loss[1][2],
                        "actor_approx_kl": actor_loss[1][3],
                        "actor_clipfrac": actor_loss[1][4],
                    }
                    return (
                        (
                            actor_train_state,
                            critic_train_state,
                        ),
                        loss_info,
                    )

                (
                    train_states,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                ) = update_state
                rng, _rng = jax.random.split(rng)

                batch = (
                    traj_batch,
                    advantages.squeeze(),
                    targets.squeeze(),
                )

                permutation = jax.random.permutation(_rng, config.num_actors)
                shuffled_batch = jax.tree.map(
                    lambda x: jnp.take(x, permutation, axis=1), batch
                )
                minibatches = jax.tree.map(
                    lambda x: jnp.swapaxes(
                        jnp.reshape(
                            x,
                            [x.shape[0], config.num_minibatches, -1]
                            + list(x.shape[2:]),
                        ),
                        1,
                        0,
                    ),
                    shuffled_batch,
                )

                train_states, loss_info = jax.lax.scan(
                    _update_minibatch,
                    train_states,
                    minibatches,
                    unroll=False,
                )
                update_state = (
                    train_states,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                )

                return update_state, loss_info

            update_state = (
                train_states,
                traj_batch,
                advantages,
                targets,
                rng,
            )
            update_state, loss_info = jax.lax.scan(
                _update_epoch,
                update_state,
                None,
                length=config.update_epochs,
                unroll=False,
            )
            loss_info["actor_ratio_0"] = loss_info["actor_ratio"].at[0, 0].get()
            loss_info = jax.tree.map(lambda x: x.mean(), loss_info)

            train_states = update_state[0]
            metric = traj_batch.info
            metric["loss"] = loss_info
            rng = update_state[-1]

            def callback(metric):
                env_step = metric["update_steps"] * config.num_envs * config.num_steps
                wandb.log(
                    {
                        "episode_returns": metric["returned_episode_returns"][-1, :].mean(),
                        "episode_lengths": metric["returned_episode_lengths"][-1, :].mean(),
                        "env_step": env_step,
                        **metric["loss"],
                    },
                )
                pbar.update(1)

            metric["update_steps"] = update_steps
            jax.experimental.io_callback(callback, None, metric)
            update_steps = update_steps + 1
            runner_state = (
                train_states,
                env_state,
                last_obs,
                last_done,
                rng,
            )
            return (runner_state, update_steps), metric

        rng, _rng = jax.random.split(rng)
        runner_state = (
            (actor_train_state, critic_train_state),
            env_state,
            obsv,
            jnp.zeros((config.num_actors), dtype=bool),
            _rng,
        )
        runner_state, metric = jax.lax.scan(
            _update_step,
            (runner_state, 0),
            None,
            length=config.num_updates,
        )

        return {"runner_state": runner_state}

    return train


def main():
    config = TrainConfig()
    saved_model_dir = f"./trained_models_ppo/{config.run_name}"
    os.makedirs(saved_model_dir, exist_ok=True)
    wandb.init(
        # mode="disabled",
        project=config.project,
        name=config.run_name,
        config=config._asdict(),
        tags=config.tags,
    )
    rng = jax.random.key(config.seed)
    train_jit = jax.jit(make_train(config))
    out = train_jit(rng)

    final_runner_state = out["runner_state"]
    final_train_states = final_runner_state[0][0]
    (
        actor_final_train_state,
        critic_final_train_state,
    ) = final_train_states
    save_params(
        actor_final_train_state.params,
        saved_model_dir + f"/{config.env_name}_actor_params.safetensors",
    )
    save_params(
        critic_final_train_state.params,
        saved_model_dir + f"/{config.env_name}_critic_params.safetensors",
    )


if __name__ == "__main__":
    main()
