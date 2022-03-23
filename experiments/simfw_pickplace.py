import time

import numpy as np

from alr_sim.gyms.gym_controllers import GymJointVelController
from alr_sim.sims.SimFactory import SimRepository
from envs.pick_place import PickAndPlaceEnv

if __name__ == "__main__":
    # simulator = 'pybullet'
    simulator = "mujoco"

    sim_factory = SimRepository.get_factory(simulator)

    r = sim_factory.create_robot()
    scene = sim_factory.create_scene(r)

    env = PickAndPlaceEnv(
        scene=scene, controller=GymJointVelController(), random_env=True
    )

    env.start()

    for i in range(10000):
        env.step(np.array([0, 0, 0, 0, 0, 0, 0]))
        time.sleep(1.0 / 240.0)

        if i % 50 == 0:
            env.reset()
