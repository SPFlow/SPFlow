import os
from abc import ABC

import pybullet as p

from alr_sim.core.SimObject import SimObject
from alr_sim.sims.mujoco.MujocoLoadable import MujocoXmlLoadable
from alr_sim.sims.pybullet.PybulletLoadable import PybulletLoadable
from alr_sim.utils.sim_path import sim_framework_path


class NutObject(SimObject, MujocoXmlLoadable, PybulletLoadable, ABC):
    def __init__(self, type_str: str, pos_sign: float = 1.0):
        name = type_str + "_nut"
        super(NutObject, self).__init__(name)
        self.type = type_str
        self.pos_sign = pos_sign
        self._pois = [name]

    @staticmethod
    def _file_path():
        return os.path.dirname(os.path.abspath(__file__))

    @property
    def xml_file_path(self):
        return sim_framework_path(
            self._file_path(), "mujoco/assets/{}_nut.xml".format(self.type)
        )

    def pb_load(self, sim):
        obj_id = p.loadURDF(
            basePosition=[0.4, self.pos_sign * 0.2, 0.0],
            fileName=sim_framework_path(
                self._file_path(), "pybullet/assets/{}_nut.urdf".format(self.type)
            ),
            useFixedBase=0,
            physicsClientId=sim,
        )
        return obj_id


class RoundNut(NutObject):
    def __init__(self):
        super().__init__("round", -1.0)
        self._pois = []

    def get_poi(self):
        return self._pois


class SquareNut(NutObject):
    def __init__(self):
        super(SquareNut, self).__init__("square")

    def get_poi(self):
        return self._pois
