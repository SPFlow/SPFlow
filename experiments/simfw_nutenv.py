import time

import numpy as np

from alr_sim.gyms.gym_controllers import GymJointVelController
from alr_sim.sims.SimFactory import SimRepository
from envs.nut_assembly_env.nut_assembly import NutAssemblyEnv

if __name__ == "__main__":
    # simulator = "pybullet"
    simulator = 'mujoco'

    sim_factory = SimRepository.get_factory(simulator)

    r = sim_factory.create_robot()
    scene = sim_factory.create_scene(r)

    env = NutAssemblyEnv(scene=scene, controller=GymJointVelController())
    env.start()
    for _ in range(10000):
        env.step(np.array([0, 0, 0, 0, 0, 0, 0]))
        time.sleep(1.0 / 240.0)
