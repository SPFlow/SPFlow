import os
from abc import ABC

import pybullet as p

from alr_sim.core.SimObject import SimObject
from alr_sim.sims.mujoco.MujocoLoadable import MujocoXmlLoadable
from alr_sim.sims.pybullet.PybulletLoadable import PybulletLoadable
from alr_sim.utils.sim_path import sim_framework_path


class PegObject(SimObject, MujocoXmlLoadable, PybulletLoadable, ABC):
    def __init__(self, type_str: str):
        name = type_str + "_peg"
        super(PegObject, self).__init__(name)
        self.type = type_str
        self._pois = [name]

    @staticmethod
    def _file_path():
        return os.path.dirname(os.path.abspath(__file__))

    @property
    def xml_file_path(self):
        return sim_framework_path(
            self._file_path(), "mujoco/assets/{}_peg.xml".format(self.type)
        )

    def pb_load(self, sim):
        obj_id = p.loadURDF(
            fileName=sim_framework_path(
                self._file_path(), "pybullet/assets/{}_peg.urdf".format(self.type)
            ),
            useFixedBase=1,
            physicsClientId=sim,
        )
        return obj_id


class RoundPeg(PegObject):
    def __init__(self):
        super().__init__("round")
        self._pois = []

    def get_poi(self):
        return self._pois


class SquarePeg(PegObject):
    def __init__(self):
        super().__init__("square")

    def get_poi(self):
        return self._pois
