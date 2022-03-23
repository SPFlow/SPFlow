import numpy as np

import envs.nut_assembly_env.objects.nuts as nuts
import envs.nut_assembly_env.objects.pegs as pegs
from alr_sim.controllers.Controller import ControllerBase
from alr_sim.core import Scene
from alr_sim.gyms.gym_env_wrapper import GymEnvWrapper


class NutAssemblyEnv(GymEnvWrapper):
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
        self.scene = scene
        self.scene.add_object(pegs.SquarePeg())
        self.scene.add_object(pegs.RoundPeg())
        self.scene.add_object(nuts.SquareNut())
        self.scene.add_object(nuts.RoundNut())

    def get_observation(self) -> np.ndarray:
        return self.robot_state()

    def get_reward(self):
        return 0

    def _check_early_termination(self) -> bool:
        return False

    def _reset_env(self):
        pass
