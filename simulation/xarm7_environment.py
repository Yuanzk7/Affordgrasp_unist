"""ThinkGrasp-compatible PyBullet environment for xArm7 + xArm Gripper."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pybullet as pb
from scipy.spatial.transform import Rotation

from environment_sim import Environment


class XArm7Environment(Environment):
    """Replace ThinkGrasp's UR5e/Robotiq execution with an xArm7/G1 model."""

    HOME_JOINTS = np.array(
        [0.0, -0.4670, 0.0, 1.3760, 0.0, 1.8430, 0.0],
        dtype=np.float64,
    )
    GRIPPER_JOINT_NAMES = (
        "drive_joint",
        "left_finger_joint",
        "left_inner_knuckle_joint",
        "right_outer_knuckle_joint",
        "right_finger_joint",
        "right_inner_knuckle_joint",
    )
    CONTACT_LINK_NAMES = {
        "left_outer_knuckle",
        "left_finger",
        "left_inner_knuckle",
        "right_outer_knuckle",
        "right_finger",
        "right_inner_knuckle",
    }

    def __init__(
        self,
        urdf_path: Path,
        gui: bool = True,
        time_step: float = 1 / 240,
        simulation_speed: float = 0.5,
    ) -> None:
        self.urdf_path = urdf_path.expanduser().resolve()
        if not self.urdf_path.is_file():
            raise FileNotFoundError(f"xArm7 URDF does not exist: {self.urdf_path}")
        if not np.isfinite(simulation_speed) or simulation_speed <= 0.0:
            raise ValueError("simulation_speed must be a positive finite number")
        self.simulation_speed = float(simulation_speed)
        super().__init__(gui=gui, time_step=time_step)
        self.home_joints = self.HOME_JOINTS.copy()
        self.ik_rest_joints = self.HOME_JOINTS.copy()

    def _step_simulation(self, steps: int = 1) -> None:
        """Advance physics and pace GUI playback relative to real time."""
        for _ in range(steps):
            pb.stepSimulation(physicsClientId=self._client_id)
            if self.gui:
                time.sleep(self.time_step / self.simulation_speed)

    def _wall_timeout(self, simulation_timeout: float) -> float:
        if not self.gui:
            return simulation_timeout
        return simulation_timeout / min(self.simulation_speed, 1.0)

    def reset(self) -> None:
        self.obj_ids = {"fixed": [], "rigid": []}
        pb.resetSimulation(physicsClientId=self._client_id)
        pb.setGravity(0, 0, -9.8, physicsClientId=self._client_id)
        pb.setTimeStep(self.time_step, physicsClientId=self._client_id)
        if self.gui:
            pb.configureDebugVisualizer(
                pb.COV_ENABLE_RENDERING, 0, physicsClientId=self._client_id
            )

        self.plane = pb.loadURDF(
            "plane.urdf",
            basePosition=(0, 0, -0.0005),
            useFixedBase=True,
            physicsClientId=self._client_id,
        )
        self.workspace = pb.loadURDF(
            "assets/workspace/workspace.urdf",
            basePosition=(0.5, 0, 0),
            useFixedBase=True,
            physicsClientId=self._client_id,
        )
        for body in (self.plane, self.workspace):
            pb.changeDynamics(
                body,
                -1,
                lateralFriction=1.1,
                restitution=0.2,
                linearDamping=0.5,
                angularDamping=0.5,
                physicsClientId=self._client_id,
            )

        self.robot = pb.loadURDF(
            str(self.urdf_path),
            basePosition=(0, 0, 0),
            useFixedBase=True,
            physicsClientId=self._client_id,
        )
        self.ur5e = self.robot  # Compatibility with inherited helpers.
        self.ee = self.robot
        self._index_robot_joints()
        self.ur5e_joints = self.arm_joint_indices
        self.ur5e_ee_id = self.tcp_link_index
        self.ee_tip_id = self.tcp_link_index

        for joint, angle in zip(self.arm_joint_indices, self.home_joints):
            pb.resetJointState(
                self.robot, joint, float(angle), physicsClientId=self._client_id
            )
        for joint in self.gripper_joint_indices:
            pb.resetJointState(
                self.robot, joint, 0.0, physicsClientId=self._client_id
            )
        self._hold_arm(self.home_joints)
        self.open_gripper()

        if self.gui:
            pb.configureDebugVisualizer(
                pb.COV_ENABLE_RENDERING, 1, physicsClientId=self._client_id
            )

    def _index_robot_joints(self) -> None:
        joint_by_name: Dict[str, int] = {}
        link_by_name: Dict[str, int] = {}
        movable = []
        for index in range(pb.getNumJoints(self.robot, physicsClientId=self._client_id)):
            info = pb.getJointInfo(self.robot, index, physicsClientId=self._client_id)
            joint_by_name[info[1].decode("utf-8")] = index
            link_by_name[info[12].decode("utf-8")] = index
            if info[2] != pb.JOINT_FIXED:
                movable.append(index)
        required_joints = [f"joint{number}" for number in range(1, 8)]
        required_joints.extend(self.GRIPPER_JOINT_NAMES)
        missing = [name for name in required_joints if name not in joint_by_name]
        if "link_tcp" not in link_by_name:
            missing.append("link_tcp")
        if missing:
            raise RuntimeError("xArm7 URDF is missing: " + ", ".join(missing))

        self.joint_by_name = joint_by_name
        self.link_by_name = link_by_name
        self.movable_joint_indices = movable
        self.arm_joint_indices = [joint_by_name[name] for name in required_joints[:7]]
        self.gripper_joint_indices = [
            joint_by_name[name] for name in self.GRIPPER_JOINT_NAMES
        ]
        self.drive_joint_index = joint_by_name["drive_joint"]
        self.tcp_link_index = link_by_name["link_tcp"]
        self.contact_link_indices = {
            link_by_name[name]
            for name in self.CONTACT_LINK_NAMES
            if name in link_by_name
        }
        self.left_contact_link_indices = {
            link_by_name[name]
            for name in self.CONTACT_LINK_NAMES
            if name.startswith("left") and name in link_by_name
        }
        self.right_contact_link_indices = {
            link_by_name[name]
            for name in self.CONTACT_LINK_NAMES
            if name.startswith("right") and name in link_by_name
        }
        self.grasp_constraint: Optional[int] = None
        for link in self.contact_link_indices:
            pb.changeDynamics(
                self.robot,
                link,
                lateralFriction=1.5,
                spinningFriction=0.05,
                rollingFriction=0.01,
                physicsClientId=self._client_id,
            )

    def add_object_push_from_file(
        self, file_name: str, switch: Optional[Any] = None
    ) -> Tuple[bool, str]:
        success, instruction = super().add_object_push_from_file(file_name, switch)
        lines = Path(file_name).read_text(encoding="utf-8").splitlines()
        target_indices = [int(value) for value in lines[1].split()]
        self.target_obj_ids = [self.obj_ids["rigid"][index] for index in target_indices]
        return success, instruction

    def _hold_arm(self, target_joints: Sequence[float]) -> None:
        pb.setJointMotorControlArray(
            bodyUniqueId=self.robot,
            jointIndices=self.arm_joint_indices,
            controlMode=pb.POSITION_CONTROL,
            targetPositions=np.asarray(target_joints, dtype=np.float64),
            positionGains=np.full(7, 0.15),
            velocityGains=np.full(7, 1.0),
            forces=np.full(7, 200.0),
            physicsClientId=self._client_id,
        )

    def go_home(self) -> bool:
        return self.move_joints(self.home_joints, speed=0.025, timeout=8.0)

    def move_joints(
        self,
        target_joints: Sequence[float],
        speed: float = 0.02,
        timeout: float = 8.0,
    ) -> bool:
        target = np.asarray(target_joints, dtype=np.float64)
        if target.shape != (7,) or not np.all(np.isfinite(target)):
            return False
        start = time.time()
        while time.time() - start < self._wall_timeout(timeout):
            current = np.asarray(
                [
                    pb.getJointState(
                        self.robot, joint, physicsClientId=self._client_id
                    )[0]
                    for joint in self.arm_joint_indices
                ],
                dtype=np.float64,
            )
            difference = target - current
            if float(np.max(np.abs(difference))) < 0.012:
                self._hold_arm(target)
                self._step_simulation(8)
                return True
            norm = float(np.linalg.norm(difference))
            step = target if norm <= speed else current + difference / norm * speed
            self._hold_arm(step)
            self._step_simulation()
        print("Warning: xArm7 move_joints timeout")
        return False

    def solve_ik(self, pose: Tuple[Sequence[float], Sequence[float]]) -> np.ndarray:
        rest = []
        lower = []
        upper = []
        ranges = []
        for joint in self.movable_joint_indices:
            info = pb.getJointInfo(self.robot, joint, physicsClientId=self._client_id)
            low, high = float(info[8]), float(info[9])
            if high <= low:
                low, high = -np.pi, np.pi
            lower.append(low)
            upper.append(high)
            ranges.append(high - low)
            rest.append(
                pb.getJointState(
                    self.robot, joint, physicsClientId=self._client_id
                )[0]
            )
        solution = pb.calculateInverseKinematics(
            bodyUniqueId=self.robot,
            endEffectorLinkIndex=self.tcp_link_index,
            targetPosition=pose[0],
            targetOrientation=pose[1],
            lowerLimits=lower,
            upperLimits=upper,
            jointRanges=ranges,
            restPoses=rest,
            maxNumIterations=300,
            residualThreshold=1e-5,
            physicsClientId=self._client_id,
        )
        return np.asarray(solution[:7], dtype=np.float64)

    def move_ee_pose(
        self,
        pose: Tuple[Sequence[float], Sequence[float]],
        speed: float = 0.02,
    ) -> bool:
        target = self.solve_ik(pose)
        if not self.move_joints(target, speed=speed):
            return False
        state = pb.getLinkState(
            self.robot,
            self.tcp_link_index,
            computeForwardKinematics=True,
            physicsClientId=self._client_id,
        )
        position_error = float(np.linalg.norm(np.asarray(state[4]) - pose[0]))
        return position_error < 0.012

    def check_grasp_reachability(
        self,
        pose: Sequence[float],
        pregrasp_offset: float = 0.12,
        waypoint_spacing: float = 0.03,
        position_tolerance: float = 0.012,
    ) -> Tuple[bool, str, float]:
        """Check a grasp path with IK without changing the live scene state."""
        action = np.asarray(pose, dtype=np.float64)
        if action.shape != (7,) or not np.all(np.isfinite(action)):
            return False, "invalid-pose", float("inf")

        target = action[:3]
        quaternion = action[3:]
        approach = Rotation.from_quat(quaternion).as_matrix()[:, 2]
        pregrasp = target - approach * pregrasp_offset
        distance = float(np.linalg.norm(target - pregrasp))
        steps = max(1, int(np.ceil(distance / waypoint_spacing)))
        waypoints = [
            (
                "pregrasp" if index == 0 else f"approach-{index}/{steps}",
                pregrasp + alpha * (target - pregrasp),
            )
            for index, alpha in enumerate(np.linspace(0.0, 1.0, steps + 1))
        ]

        state_id = pb.saveState(physicsClientId=self._client_id)
        maximum_error = 0.0
        try:
            for name, waypoint in waypoints:
                joints = self.solve_ik((waypoint, quaternion))
                if joints.shape != (7,) or not np.all(np.isfinite(joints)):
                    return False, name, float("inf")
                for joint, angle in zip(self.arm_joint_indices, joints):
                    pb.resetJointState(
                        self.robot,
                        joint,
                        float(angle),
                        physicsClientId=self._client_id,
                    )
                actual = np.asarray(
                    pb.getLinkState(
                        self.robot,
                        self.tcp_link_index,
                        computeForwardKinematics=True,
                        physicsClientId=self._client_id,
                    )[4],
                    dtype=np.float64,
                )
                error = float(np.linalg.norm(actual - waypoint))
                maximum_error = max(maximum_error, error)
                if error > position_tolerance:
                    return False, name, error
            return True, "reachable", maximum_error
        finally:
            pb.restoreState(stateId=state_id, physicsClientId=self._client_id)
            pb.removeState(stateUniqueId=state_id, physicsClientId=self._client_id)

    def straight_move(
        self,
        pose0: Sequence[float],
        pose1: Sequence[float],
        rotation: Sequence[float],
        speed: float = 0.02,
        **_: Any,
    ) -> bool:
        start = np.asarray(pose0, dtype=np.float64)
        finish = np.asarray(pose1, dtype=np.float64)
        distance = float(np.linalg.norm(finish - start))
        if distance < 1e-6:
            return self.move_ee_pose((finish, rotation), speed=speed)
        steps = max(1, int(np.ceil(distance / 0.01)))
        for alpha in np.linspace(0.0, 1.0, steps + 1)[1:]:
            target = start + alpha * (finish - start)
            if not self.move_ee_pose((target, rotation), speed=speed):
                return False
        return True

    def _move_gripper(self, target: float, timeout: float = 2.0) -> float:
        start = time.time()
        while time.time() - start < self._wall_timeout(timeout):
            pb.setJointMotorControlArray(
                bodyUniqueId=self.robot,
                jointIndices=self.gripper_joint_indices,
                controlMode=pb.POSITION_CONTROL,
                targetPositions=np.full(len(self.gripper_joint_indices), target),
                positionGains=np.full(len(self.gripper_joint_indices), 0.3),
                forces=np.full(len(self.gripper_joint_indices), 80.0),
                physicsClientId=self._client_id,
            )
            self._step_simulation()
            current = pb.getJointState(
                self.robot, self.drive_joint_index, physicsClientId=self._client_id
            )[0]
            if abs(current - target) < 0.01:
                break
        self._step_simulation(30)
        return float(
            pb.getJointState(
                self.robot, self.drive_joint_index, physicsClientId=self._client_id
            )[0]
        )

    def open_gripper(self, is_slow: bool = False) -> None:
        del is_slow
        if self.grasp_constraint is not None:
            try:
                pb.removeConstraint(
                    self.grasp_constraint, physicsClientId=self._client_id
                )
            except pb.error:
                pass
            self.grasp_constraint = None
        self._move_gripper(0.0)

    def close_gripper(self, is_slow: bool = True) -> None:
        del is_slow
        self._move_gripper(0.85)

    @property
    def is_gripper_closed(self) -> bool:
        angle = pb.getJointState(
            self.robot, self.drive_joint_index, physicsClientId=self._client_id
        )[0]
        return float(angle) < 0.82

    def _contact_object(self) -> Optional[int]:
        best: Optional[int] = None
        best_count = 0
        for object_id in self.obj_ids["rigid"]:
            contacts = pb.getContactPoints(
                bodyA=self.robot,
                bodyB=object_id,
                physicsClientId=self._client_id,
            )
            count = sum(int(contact[3]) in self.contact_link_indices for contact in contacts)
            if count > best_count:
                best = object_id
                best_count = count
        return best

    def _bilateral_contact_object(self) -> Optional[int]:
        """Return an object simultaneously enclosed by the two finger sides."""
        for object_id in self.obj_ids["rigid"]:
            contact_links = {
                int(contact[3])
                for contact in pb.getContactPoints(
                    bodyA=self.robot,
                    bodyB=object_id,
                    physicsClientId=self._client_id,
                )
            }
            if (
                contact_links & self.left_contact_link_indices
                and contact_links & self.right_contact_link_indices
            ):
                return object_id
        return None

    def _stabilize_grasp(self, object_id: int) -> None:
        """Stabilize a verified bilateral grasp against mesh-contact jitter."""
        tcp_position, tcp_orientation = pb.getLinkState(
            self.robot,
            self.tcp_link_index,
            computeForwardKinematics=True,
            physicsClientId=self._client_id,
        )[4:6]
        object_position, object_orientation = pb.getBasePositionAndOrientation(
            object_id, physicsClientId=self._client_id
        )
        inverse_position, inverse_orientation = pb.invertTransform(
            tcp_position, tcp_orientation
        )
        relative_position, relative_orientation = pb.multiplyTransforms(
            inverse_position,
            inverse_orientation,
            object_position,
            object_orientation,
        )
        self.grasp_constraint = pb.createConstraint(
            parentBodyUniqueId=self.robot,
            parentLinkIndex=self.tcp_link_index,
            childBodyUniqueId=object_id,
            childLinkIndex=-1,
            jointType=pb.JOINT_FIXED,
            jointAxis=(0, 0, 0),
            parentFramePosition=relative_position,
            childFramePosition=(0, 0, 0),
            parentFrameOrientation=relative_orientation,
            childFrameOrientation=(0, 0, 0, 1),
            physicsClientId=self._client_id,
        )
        pb.changeConstraint(
            self.grasp_constraint,
            maxForce=250.0,
            physicsClientId=self._client_id,
        )

    def _close_until_bilateral_contact(self) -> Optional[int]:
        """Close gradually and stop as soon as both finger sides make contact."""
        current = float(
            pb.getJointState(
                self.robot, self.drive_joint_index, physicsClientId=self._client_id
            )[0]
        )
        for target in np.arange(max(0.0, current) + 0.005, 0.855, 0.005):
            target = min(float(target), 0.85)
            pb.setJointMotorControlArray(
                bodyUniqueId=self.robot,
                jointIndices=self.gripper_joint_indices,
                controlMode=pb.POSITION_CONTROL,
                targetPositions=np.full(len(self.gripper_joint_indices), target),
                positionGains=np.full(len(self.gripper_joint_indices), 0.2),
                forces=np.full(len(self.gripper_joint_indices), 80.0),
                physicsClientId=self._client_id,
            )
            self._step_simulation(3)
            object_id = self._bilateral_contact_object()
            if object_id is not None:
                self._stabilize_grasp(object_id)
                return object_id
        return None

    def grasp(
        self, pose: Sequence[float], speed: float = 0.02
    ) -> Tuple[bool, Optional[int], Optional[float]]:
        action = np.asarray(pose, dtype=np.float64)
        if action.shape != (7,):
            return False, None, None
        target = action[:3]
        quaternion = action[3:]
        rotation = Rotation.from_quat(quaternion).as_matrix()
        approach = rotation[:, 2]
        pregrasp = target - approach * 0.12
        initial_heights = {
            object_id: pb.getBasePositionAndOrientation(
                object_id, physicsClientId=self._client_id
            )[0][2]
            for object_id in self.obj_ids["rigid"]
        }

        self.open_gripper()
        stage = "home"
        success = self.go_home()
        if success:
            stage = "pregrasp"
            success = self.move_ee_pose((pregrasp, quaternion), speed=0.025)
        if success:
            stage = "approach"
            success = self.straight_move(
                pregrasp, target, quaternion, speed=max(speed, 0.015)
            )
        grasped: Optional[int] = None
        if success:
            stage = "close"
            grasped = self._close_until_bilateral_contact()
            success = grasped is not None
        if success:
            stage = "retreat"
            success = self.straight_move(
                target, pregrasp, quaternion, speed=max(speed, 0.015)
            )
        if success and grasped is not None:
            stage = "lift-check"
            height = pb.getBasePositionAndOrientation(
                grasped, physicsClientId=self._client_id
            )[0][2]
            success = height > initial_heights[grasped] + 0.03

        position_distance: Optional[float] = None
        if grasped is not None and self.target_obj_ids:
            object_position = np.asarray(
                pb.getBasePositionAndOrientation(
                    grasped, physicsClientId=self._client_id
                )[0]
            )
            position_distance = min(
                float(
                    np.linalg.norm(
                        object_position
                        - np.asarray(
                            pb.getBasePositionAndOrientation(
                                target_id, physicsClientId=self._client_id
                            )[0]
                        )
                    )
                )
                for target_id in self.target_obj_ids
            )
        if not success:
            tcp_position = pb.getLinkState(
                self.robot,
                self.tcp_link_index,
                computeForwardKinematics=True,
                physicsClientId=self._client_id,
            )[4]
            drive_angle = pb.getJointState(
                self.robot,
                self.drive_joint_index,
                physicsClientId=self._client_id,
            )[0]
            print(
                "xArm7 grasp failure: "
                f"stage={stage}, tcp={np.round(tcp_position, 4).tolist()}, "
                f"drive_joint={drive_angle:.4f}, object={grasped}"
            )
            self.open_gripper()
            self.go_home()
        print(f"xArm7 grasp at {action}, success={success}, object={grasped}")
        return bool(success), grasped, position_distance
