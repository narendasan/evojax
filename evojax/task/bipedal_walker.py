from typing import Tuple
from PIL import Image
from PIL import ImageDraw
import numpy as np
from evojax.task.cartpole import out_of_screen

import jax
import jax.numpy as jnp
from jax import random
from flax.struct import dataclass

from evojax.task.base import TaskState
from evojax.task.base import VectorizedTask

from gym import spaces
import math


FPS = 50
SCALE = 30.0  # affects how fast-paced the game is, forces should be adjusted as well

MOTORS_TORQUE = 80
SPEED_HIP = 4
SPEED_KNEE = 6
LIDAR_RANGE = 160 / SCALE

INITIAL_RANDOM = 5

HULL_POLY = [(-30, +9), (+6, +9), (+34, +1), (+34, -8), (-30, -8)]
LEG_DOWN = -8 / SCALE
LEG_W, LEG_H = 8 / SCALE, 34 / SCALE

VIEWPORT_W = 600
VIEWPORT_H = 400

TERRAIN_STEP = 14 / SCALE
TERRAIN_LENGTH = 200  # in steps
TERRAIN_HEIGHT = VIEWPORT_H / SCALE / 4
TERRAIN_GRASS = 10  # low long are grass spots, in steps
TERRAIN_STARTPAD = 20  # in steps
FRICTION = 2.5

@dataclass
class State(TaskState):
    obs: jnp.ndarray
    state: jnp.ndarray
    steps: jnp.int32
    keys: jnp.ndarray

class BipedalWalker(VectorizedTask):
    """Port of BipedalWalker-V2 from OpenAI Gym"""

    def __init__(self,
                 max_steps: int = 1000,
                 harder: bool = False,
                 test: bool = False):

        self.max_steps = max_steps
        self.hardcore = harder

        # we use 5.0 to represent the joints moving at maximum
        # 5 x the rated speed due to impulses from ground contact etc.
        low = np.array(
            [
                -math.pi,
                -5.0,
                -5.0,
                -5.0,
                -math.pi,
                -5.0,
                -math.pi,
                -5.0,
                -0.0,
                -math.pi,
                -5.0,
                -math.pi,
                -5.0,
                -0.0,
            ]
            + [-1.0] * 10
        ).astype(np.float32)
        high = np.array(
            [
                math.pi,
                5.0,
                5.0,
                5.0,
                math.pi,
                5.0,
                math.pi,
                5.0,
                5.0,
                math.pi,
                5.0,
                math.pi,
                5.0,
                5.0,
            ]
            + [1.0] * 10
        ).astype(np.float32)

        self.obs_space = spaces.Box(low, high)
        self.act_space = spaces.Box(
            np.array([-1, -1, -1, -1]).astype(np.float32),
            np.array([1, 1, 1, 1]).astype(np.float32),
        )

        self.obs_shape = self.obs_space.shape
        self.act_shape = self.act_space.shape

        self.test = test


        def reset_fn(key):
            next_key, key = random.split(key)
            state = get_init_state_fn(key)
            return State(state=state,
                         obs=get_obs(state),
                         steps=jnp.zeros((), dtype=int),
                         key=next_key)
        self._reset_fn = jax.jit(jax.vmap(reset_fn))

        def step_fn(state, action):
            cur_state = update_state(action=action, state=state.state)
            reward = get_reward(state=cur_state)
            steps = state.steps + 1
            done = jnp.bitwise_or(out_of_screen(cur_state), steps >= max_steps)
            steps = jnp.where(done, jnp.zeros((), jnp.int32), steps)
            next_key, key = random.split(state.key)
            cur_state = jax.lax.cond(
                done, lambda x: get_init_state_fn(key), lambda x: x, cur_state)
            return State(state=cur_state, obs=get_obs(state=cur_state),
                        steps=steps, key=next_key), reward, done
        self._step_fn = jax.jit(jax.vmap(step_fn))

        def generate_terrain_fn(hardcore):
            GRASS, STUMP, STAIRS, PIT, _STATES_ = range(5)
            state = GRASS
            velocity = 0.0
            y = TERRAIN_HEIGHT
            counter = TERRAIN_STARTPAD
            oneshot = False
            terrain = []
            terrain_x = []
            terrain_y = []

            for i in range(TERRAIN_LENGTH):
                x = i * TERRAIN_STEP
                terrain_x.append(x)

                if state == GRASS and not oneshot:
                    velocity = 0.8 * velocity + 0.01 * jnp.sign(TERRAIN_HEIGHT - y)
                    if i > TERRAIN_STARTPAD:
                        velocity += random.uniform(-1, 1) / SCALE
                    y += velocity

                elif state == PIT and oneshot:
                    counter = random.integers(3,5)
                    poly = [
                        (x, y),
                        (x + TERRAIN_STEP, y),
                        (x + TERRAIN_STEP, y - 4 * TERRAIN_STEP),
                        (x, y - 4 * TERRAIN_STEP)
                    ]

                    # WORLD GETS POLYGON
                    #fd_polygon.shape.vertices = poly
                    #t = world.CreateStaticBody(fixtures=self.fd_polygon)
                    #t.color1, t.color2 = (255, 255, 255), (153, 153, 153)
                    #terrain.append(t)

                    #fd_polygon.shape.vertices = [
                    #    (p[0] + TERRAIN_STEP * counter, p[1]) for p in poly
                    #]
                    #t = world.CreateStaticBody(fixtures=self.fd_polygon)
                    #t.color1, t.color2 = (255, 255, 255), (153, 153, 153)

                    terrain.append(t)
                    counter += 2
                    original_y = y

                elif state == PIT and not oneshot:
                    y = original_y
                    if counter > 1:
                        y -= 4 * TERRAIN_STEP

                elif state == STUMP and oneshot:
                    counter = random.integers(1,3)
                    poly = [
                        (x, y),
                        (x + counter * TERRAIN_STEP, y),
                        (x + counter * TERRAIN_STEP, y + counter * TERRAIN_STEP),
                        (x, y + counter * TERRAIN_STEP),
                    ]
                    #fd_polygon.shape.vertices = poly
                    #t = world.CreateStaticBody(fixtures=self.fd_polygon)
                    #t.color1, t.color2 = (255, 255, 255), (153, 153, 153)
                    terrain.append(t)

                elif state == STAIRS and oneshot:
                    stair_height = +1 if random.random() > 0.5 else -1
                    stair_width = random.integers(4,5)
                    stair_steps = random.integers(3,5)
                    original_y = y

                    for s in range(stair_steps):
                        poly = [
                            (
                                x + (s * stair_width) * TERRAIN_STEP,
                                y + (s * stair_height) * TERRAIN_STEP,
                            ),
                            (
                                x + ((1 + s) * stair_width) * TERRAIN_STEP,
                                y + (s * stair_height) * TERRAIN_STEP,
                            ),
                            (
                                x + ((1 + s) * stair_width) * TERRAIN_STEP,
                                y + (-1 + s * stair_height) * TERRAIN_STEP,
                            ),
                            (
                                x + (s * stair_width) * TERRAIN_STEP,
                                y + (-1 + s * stair_height) * TERRAIN_STEP,
                            ),
                        ]
                        #fd_polygon.shape.vertices = poly
                        #t = world.CreateStaticBody(fixtures=self.fd_polygon)
                        #t.color1, t.color2 = (255, 255, 255), (153, 153, 153)
                        terrain.append(t)
                    counter = stair_steps * stair_width

                elif state == STAIRS and not oneshot:
                    s = stair_steps * stair_width - counter - stair_height
                    n = s / stair_width
                    y = original_y + (n * stair_height) * TERRAIN_STEP

                oneshot = False
                terrain_y.append(y)
                counter -= 1

                if counter == 0:
                    counter = random.integers(TERRAIN_GRASS / 2, TERRAIN_GRASS)
                    if state == GRASS and hardcore:
                        state = random.integers(1, _STATES_)
                        oneshot = True
                    else:
                        state = GRASS
                        oneshot = True

            terrain_poly = []
            for i in range(TERRAIN_LENGTH - 1):
                poly = [
                    (terrain_x[i], terrain_y[i]),
                    (terrain_x[i + 1], terrain_y[i + 1]),
                ]
                fd_edge.shape.vertices = poly
                t = world.CreateStaticBody(fixtures=fd_edge)
                color = (76, 255 if i % 2 == 0 else 204, 76)
                t.color1 = color
                t.color2 = color
                terrain.append(t)
                color = (102, 153, 76)
                poly += [(poly[1][0], 0), (poly[0][0], 0)]
                terrain_poly.append((poly, color))
            terrain.reverse()
            return (terrain, terrain_poly, terrain_x, terrain_y)
        self._generate_terrain_fn = jax.jit(jax.vmap(generate_terrain_fn))



    def reset(self, key: jnp.ndarray) -> State:
        return self._reset_fn(key)

    def step(self, state: TaskState, action: jnp.ndarray) -> Tuple[TaskState, jnp.ndarray, jnp.ndarray]:
        return self._step_fn(state, action)

    @staticmethod
    def render(state: State, task_id: int) -> Image:

