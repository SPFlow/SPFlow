from alr_sim.core.Logger import RobotPlotFlags
from alr_sim.sims.SimFactory import SimRepository
from alr_sim.sims.universal_sim.PrimitiveObjects import Box
from alr_sim.core.Camera import Camera


def build_scene(sim):
    box1 = Box(
        name="box1",
        init_pos=[0.5, -0.2, 0.35],
        init_quat=[0, 1, 0, 0],
        rgba=[0.1, 0.25, 0.3, 1],
    )
    box2 = Box(
        name="box2",
        init_pos=[0.6, -0.1, 0.35],
        init_quat=[0, 1, 0, 0],
        rgba=[0.2, 0.3, 0.7, 1],
    )
    box3 = Box(
        name="box3",
        init_pos=[0.4, -0.1, 0.35],
        init_quat=[0, 1, 0, 0],
        rgba=[1, 0, 0, 1],
    )
    box4 = Box(
        name="box4",
        init_pos=[0.6, -0.0, 0.35],
        init_quat=[0, 1, 0, 0],
        rgba=[1, 0, 0, 1],
    )
    box5 = Box(
        name="box5",
        init_pos=[0.6, 0.1, 0.35],
        init_quat=[0, 1, 0, 0],
        rgba=[1, 1, 1, 1],
    )
    box6 = Box(
        name="box6",
        init_pos=[0.6, 0.2, 0.35],
        init_quat=[0, 1, 0, 0],
        rgba=[1, 0, 0, 1],
    )

    table = Box(
        name="table0",
        init_pos=[0.5, 0.0, 0.2],
        init_quat=[0, 1, 0, 0],
        size=[0.25, 0.35, 0.2],
        static=True,
    )

    object_list = [box1, box2, box3, box4, box5, box6, table]

    s = sim.create_scene(robot, object_list=object_list, surrounding='kit_lab_surrounding')
    return s


if __name__ == "__main__":
    # Setup the scene
    sim = SimRepository.get_factory("mujoco")
    robot = sim.create_robot()
    scene = build_scene(sim)
    scene.start()

    duration = (
        2  # you can specify how long a trajectory can be executed witht the duration
    )

    import matplotlib.pyplot as plt
    cage_cam = scene.get_inhand_cam()
    img = cage_cam.get_image()
    plt.imshow(img[0])
    plt.show()

    robot.startLogging()  # this will start logging robots internal state
    robot.set_gripper_width = 0.0  # we set the gripper to clos at the beginning

    home_position = robot.current_c_pos.copy()  # store home position
    home_orientation = robot.current_c_quat.copy()  # store initial orientation

    # execute the pick and place movements
    robot.gotoCartPositionAndQuat(
        [0.5, -0.2, 0.6 - 0.1], [0, 1, 0, 0], duration=duration
    )
    robot.set_gripper_width = 0.04
    robot.gotoCartPositionAndQuat(
        [0.5, -0.2, 0.52 - 0.1], [0, 1, 0, 0], duration=duration
    )
    robot.set_gripper_width = 0.00
    robot.gotoCartPositionAndQuat(home_position, home_orientation, duration=duration)
    robot.gotoCartPositionAndQuat(
        [0.5, 0.2, 0.6 - 0.1], [0, 1, 0, 0], duration=duration
    )
    robot.set_gripper_width = 0.04
    robot.gotoCartPositionAndQuat(
        [0.6, -0.1, 0.6 - 0.1], [0, 1, 0, 0], duration=duration
    )
    robot.gotoCartPositionAndQuat(
        [0.6, -0.1, 0.52 - 0.1], [0, 1, 0, 0], duration=duration
    )
    robot.set_gripper_width = 0.00
    robot.gotoCartPositionAndQuat(home_position, home_orientation, duration=duration)
    robot.gotoCartPositionAndQuat(
        [0.5, 0.2, 0.6 - 0.1], [0, 1, 0, 0], duration=duration
    )
    robot.set_gripper_width = 0.04
    robot.gotoCartPositionAndQuat(
        [0.4, -0.1, 0.6 - 0.1], [0, 1, 0, 0], duration=duration
    )
    robot.gotoCartPositionAndQuat(
        [0.4, -0.1, 0.52 - 0.1], [0, 1, 0, 0], duration=duration
    )
    robot.set_gripper_width = 0.00
    robot.gotoCartPositionAndQuat(home_position, home_orientation, duration=duration)
    robot.gotoCartPositionAndQuat(
        [0.5, 0.2, 0.65 - 0.1], [0, 1, 0, 0], duration=duration
    )
    robot.set_gripper_width = 0.04
    robot.gotoCartPositionAndQuat(home_position, home_orientation, duration=duration)

    robot.stopLogging()
    robot.robot_logger.plot(RobotPlotFlags.JOINTS | RobotPlotFlags.END_EFFECTOR)
