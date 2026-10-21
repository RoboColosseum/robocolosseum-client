"""SO-101 base-frame EEF deltas backed by measured FK and calibrated IK."""
from __future__ import annotations

import logging
import math
from pathlib import Path
from xml.etree import ElementTree

import numpy as np

log = logging.getLogger(__name__)
_JOINT_LIMIT_EPS_DEG = 1e-4


class SO101EEFController:
    def __init__(self, robot, urdf_path):
        path = Path(urdf_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"SO-101 EEF URDF is missing: {path}")
        root = ElementTree.parse(path).getroot()
        from .so101 import JOINTS
        if set(JOINTS) - {joint.get('name') for joint in root.findall('joint')}:
            raise ValueError('SO-101 EEF URDF does not contain the configured arm joints')
        if 'gripper_frame_link' not in {link.get('name') for link in root.findall('link')}:
            raise ValueError('SO-101 EEF URDF lacks gripper_frame_link')
        for mesh in root.findall('.//mesh'):
            filename = mesh.get('filename')
            if filename and not (path.parent / filename).is_file():
                raise FileNotFoundError(f"SO-101 EEF mesh is missing: {filename}")
        from lerobot.model.kinematics import RobotKinematics
        from lerobot.utils.rotation import Rotation
        self.rotation = Rotation
        self.kinematics = RobotKinematics(
            str(path), target_frame_name='gripper_frame_link', joint_names=list(JOINTS))
        for name, low, high in zip(JOINTS, robot.low, robot.high):
            self.kinematics.robot.set_joint_limits(name, math.radians(low), math.radians(high))
        self.robot = robot
        self.expected_pose = None
        self.expected_gripper = None
        self.measured = None
        self._warned_ik_limit = False

    def begin_chunk(self):
        self.expected_pose = None
        self.expected_gripper = None
        self._warned_ik_limit = False

    def observe(self, positions):
        measured = np.asarray(positions, dtype=np.float64)
        pose = np.asarray(self.kinematics.forward_kinematics(measured[:5]), dtype=np.float64)
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            raise ValueError('SO-101 FK returned an invalid pose')
        rotation = self.rotation.from_matrix(pose[:3, :3])
        xyzw = rotation.as_quat()
        self.measured = measured.copy()
        return {
            'eef_state': np.asarray([*pose[:3, 3], *rotation.as_rotvec(), measured[5]], dtype=np.float32),
            'eef_quat_wxyz': np.asarray([xyzw[3], *xyzw[:3]], dtype=np.float32),
        }

    def execute(self, action):
        delta = np.asarray(action, dtype=np.float64)
        if self.measured is None:
            raise ValueError('SO-101 EEF action requires a measured observation')
        if delta.shape != (7,) or not np.isfinite(delta).all():
            raise ValueError('SO-101 EEF action must contain seven finite deltas')
        current = self.measured
        pose = np.asarray(self.kinematics.forward_kinematics(current[:5]), dtype=np.float64)
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            raise ValueError('SO-101 FK returned an invalid pose')
        desired = pose.copy() if self.expected_pose is None else self.expected_pose.copy()
        desired[:3, 3] += delta[:3]
        desired[:3, :3] = self.rotation.from_rotvec(delta[3:6]).as_matrix() @ desired[:3, :3]
        gripper_target = (current[5] if self.expected_gripper is None else self.expected_gripper) + delta[6]
        # Action chunks arrive as float32. Repeated additions can put an exact
        # endpoint target (0 or 100) a few millionths outside the range.
        if not -1e-4 <= gripper_target <= 100 + 1e-4:
            raise ValueError('SO-101 EEF gripper target exceeds [0, 100]')
        gripper_target = float(np.clip(gripper_target, 0, 100))
        gripper = gripper_target
        if np.any(delta[:6] != 0):
            try:
                joints = np.asarray(self.kinematics.inverse_kinematics(
                    current[:5], desired, position_weight=1.0, orientation_weight=.01), dtype=np.float64)
            except Exception as exc:
                raise ValueError('SO-101 inverse kinematics failed') from exc
        else:
            joints = current[:5].copy()
        if joints.shape != (5,) or not np.isfinite(joints).all():
            raise ValueError('SO-101 IK returned invalid joint values')
        substantially_outside = ((joints < self.robot.low - _JOINT_LIMIT_EPS_DEG)
                                 | (joints > self.robot.high + _JOINT_LIMIT_EPS_DEG))
        if np.any(substantially_outside):
            from .so101 import JOINTS
            detail = ', '.join(
                f'{JOINTS[i]}={joints[i]:.6f} outside '
                f'[{self.robot.low[i]:.6f}, {self.robot.high[i]:.6f}]'
                for i in np.flatnonzero(substantially_outside))
            # A rejected IK target must not advance the chunk's expected pose.
            # Send a measured-position hold so the recording contains the command
            # actually accepted by the driver for this evaluation step.
            if (np.any(current[:5] < self.robot.low - _JOINT_LIMIT_EPS_DEG)
                    or np.any(current[:5] > self.robot.high + _JOINT_LIMIT_EPS_DEG)
                    or not -_JOINT_LIMIT_EPS_DEG <= current[5] <= 100 + _JOINT_LIMIT_EPS_DEG):
                raise ValueError('SO-101 measured position exceeds calibrated limits; cannot hold safely')
            target = np.r_[np.clip(current[:5], self.robot.low, self.robot.high),
                           np.clip(current[5], 0, 100)]
            desired = pose
            gripper_target = float(target[5])
            if not self._warned_ik_limit:
                log.warning('SO-101 IK target exceeds calibrated limits; holding measured pose: %s', detail)
                self._warned_ik_limit = True
        else:
            # Placo may return a value a few floating-point steps beyond a limit.
            target = np.r_[np.clip(joints, self.robot.low, self.robot.high), gripper]
        sent = self.robot._send(target)
        from .so101 import MOTORS
        if not isinstance(sent, dict) or any(f'{motor}.pos' not in sent for motor in MOTORS):
            raise ValueError('SO-101 driver did not report the accepted joint command')
        accepted = np.asarray([sent[f'{motor}.pos'] for motor in MOTORS], dtype=np.float64)
        if (accepted.shape != (6,) or not np.isfinite(accepted).all()
                or np.any(accepted[:5] < self.robot.low) or np.any(accepted[:5] > self.robot.high)
                or not 0 <= accepted[5] <= 100):
            raise ValueError('SO-101 driver returned an invalid accepted joint command')
        self.expected_pose = desired
        self.expected_gripper = gripper_target
        return accepted.astype(np.float32)
