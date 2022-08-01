from dataclasses import dataclass
from time import clock_settime
import jax
from jax import numpy as jnp
from evojax.task.base import TaskState, VectorizedTask
import gym

@dataclass
class State(TaskState):
    obs: jnp.ndarray
    state: jnp.ndarray
    steps: jnp.int32
    key: jnp.ndarray

class BipedalWalker(VectorizedTask):
    def __init__(self, harder=False, max_steps=1000, test=False):
        self.env = gym.make("BipedalWalker-v3", hardcore=harder, new_step_api=True)
        self.max_steps = max_steps
        self.obs_shape = self.env.observation_space.shape
        self.act_shape = self.env.action_space.shape

    def step_fn(self, state, action):
        next_state, reward, done, info = self.env.step(action)
        steps += state.steps + 1
        next_key, key = jax.random.split(state.key)
        print(next_key, key)
        return State(
            state=next_state,
            obs=next_state,
            steps=steps,
            key=next_key
        ), reward, done

        #self._step_fn = jax.jit(jax.vmap(step_fn))

    def reset_fn(self, key):
        next_key, key = jax.random.split(key)
        print(next_key)
        print(key)
        next_state, info = self.env.reset(seed=key)
        return State(
            obs=next_state,
            state=next_state,
            steps=jnp.zeros((), dtype=int),
            key=next_key
        )

        #self._reset_fn = jax.jit(jax.vmap(reset_fn))


    def step(self, state, action):
        return self.step_fn(state, action)

    def reset(self, key):
        return self.reset_fn(key)

    @staticmethod
    def render(state, task_id):
        print(state, task_id)