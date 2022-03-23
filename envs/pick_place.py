import numpy as np
from gym.spaces import Box as SamplingSpace

from alr_sim.controllers.Controller import ControllerBase
from alr_sim.core.Scene import Scene
from alr_sim.gyms.gym_env_wrapper import GymEnvWrapper
from alr_sim.gyms.gym_utils.helpers import obj_distance
from alr_sim.sims.universal_sim.PrimitiveObjects import Box, Sphere


class PickAndPlaceEnv(GymEnvWrapper):
    """
    Reach task: The agent is asked to go to a certain object and gets reward when getting closer
    to the object. Once the object is reached, the agent gets a high reward.
    """

    def __init__(
        self,
        scene: Scene,
        controller: ControllerBase,
        n_substeps: int = 1,
        max_steps_per_episode: int = 2e3,
        debug: bool = False,
        random_env: bool = True,
    ):
        super().__init__(
            scene=scene,
            controller=controller,
            max_steps_per_episode=max_steps_per_episode,
            n_substeps=n_substeps,
            debug=debug,
        )

        self.random_env = random_env

        self.goal = Sphere(
            name="goal",
            size=[0.01],
            init_pos=[0.5, 0, 0.2],
            init_quat=[1, 0, 0, 0],
            rgba=[1, 0, 0, 1],
            static=True,
        )
        self.goal_space = SamplingSpace(
            low=np.array([0.2, -0.3, 0.1]), high=np.array([0.5, 0.3, 0.5])
        )
        self.box = Box(
            name="box",
            mass=1,
            size=[0.02, 0.02, 0.02],
            init_pos=[0.5, 0, 0.1],
            init_quat=[1, 0, 0, 0],
            rgba=[0.32, 0.32, 0.32, 1],
            static=True,
        )
        self.box_space = SamplingSpace(
            low=np.array([0.2, -0.3, 0.0]), high=np.array([0.5, 0.3, 0.0])
        )

        self.scene.add_object(self.goal)
        self.scene.add_object(self.box)

        self.target_min_dist = 0.02

    def get_observation(self) -> np.ndarray:
        goal_pos = self.scene.get_obj_pos(self.goal)[0]
        box_pos = self.scene.get_obj_pos(self.box)[0]
        env_state = np.concatenate([goal_pos, box_pos])
        return np.concatenate([self.robot_state(), env_state])

    def get_reward(self):
        goal_pos = self.scene.get_obj_pos(self.goal)[0]
        box_pos = self.scene.get_obj_pos(self.box)[0]
        tcp_pos = self.robot.current_c_pos
        dist_tcp_goal, _ = obj_distance(goal_pos, tcp_pos)
        dist_box_goal, _ = obj_distance(box_pos, goal_pos)
        return (dist_tcp_goal + dist_box_goal) / 2.0

    def _check_early_termination(self) -> bool:
        goal_pos = self.scene.get_obj_pos(self.goal)[0]
        box_pos = self.scene.get_obj_pos(self.box)[0]

        dist_box_goal, _ = obj_distance(box_pos, goal_pos)

        if dist_box_goal <= self.target_min_dist:
            return True
        return False

    def _reset_env(self):
        if self.random_env:
            new_goal = [self.goal, self.goal_space.sample()]
            new_box = [self.box, self.box_space.sample()]
            self.scene.reset([new_goal, new_box])
        else:
            self.scene.reset()
