"""Mixed local trial records the joint command actually accepted for EEF control."""
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow.parquet as pq

from colosseum_client import colosseum_pb2 as pb
from colosseum_client import evaluation, local_policy
from colosseum_client.lerobot_export import export_trial
from colosseum_client.robot_config import RobotClientConfig
from colosseum_client.robot_interface import RobotObservation
from colosseum_client.tensors import tensor_from_numpy


def test_eef_trial_records_accepted_joint_command(tmp_path, monkeypatch):
    monkeypatch.setattr('colosseum_client.recording.shutil.which', lambda _: '/usr/bin/ffmpeg')
    accepted = np.array([1, 2, 3, 4, 5, 40], dtype=np.float32)
    proposed = np.array([.001, 0, 0, 0, 0, 0, 5], dtype=np.float32)

    class Robot:
        action_space_name = 'joint_position'
        action_dim = 7

        def __init__(self, config, *, action_space, control_mode):
            assert action_space == 'joint_position'
            assert control_mode == 'eef_delta_pose'
            self.executed = []

        def configure_model(self, model):
            assert model['action_dim'] == 6

        def get_observation(self):
            return RobotObservation(
                images={'head_image': np.zeros((2, 2, 3), dtype=np.uint8)},
                joints=np.zeros(5, dtype=np.float32), gripper=np.array([40], dtype=np.float32),
                cartesian_position=np.empty(0, dtype=np.float32),
                extra_state={'eef_state': np.array([0, 0, 0, 0, 0, 0, 40], dtype=np.float32),
                             'eef_quat_wxyz': np.array([1, 0, 0, 0], dtype=np.float32)})

        def execute(self, action):
            self.executed.append(np.asarray(action).copy())
            return accepted.copy()

        def close(self):
            pass

    class Client:
        native_control_mode = 'eef_delta_pose'
        model = {'model_type': 'llm', 'action_space': 'joint_position',
                 'action_dim': 6, 'control_hz': 10}

        def __init__(self, *args, **kwargs):
            pass

        async def connect(self, **kwargs):
            return SimpleNamespace(action_spaces=['joint_position'])

        async def infer(self, observation, **kwargs):
            assert list(observation.state['eef_state'].shape) == [7]
            assert list(observation.state['eef_quat_wxyz'].shape) == [4]
            return pb.ActionPlan(start_step=0, valid_until_step=0, control_hz=10,
                                 actions=tensor_from_numpy(proposed[None]))

        async def before_action(self, step):
            assert step == 0

        async def finish(self):
            pass

        async def close(self):
            pass

    monkeypatch.setattr(local_policy, 'LocalPolicyClient', Client)
    config = RobotClientConfig(url='ws://router.local', token='test', cameras={'head_image': 'dummy'},
                               policy_server_url='ws://127.0.0.1:8000', robot_type='so101',
                               llm_control_mode='eef_delta_pose', test=True, control_hz=30)
    assignment = {'inference_mode': 'local', 'task': {
        'instruction': 'move', 'cameras': ['head_image'], 'max_steps': 1}}
    asyncio.run(evaluation.run_trial(config, assignment, {'id': 'run'}, tmp_path,
                                     SimpleNamespace(), robot_factory=Robot))

    frames = [json.loads(line) for line in (tmp_path / 'frames.jsonl').read_text().splitlines()]
    assert frames[0]['action'] == accepted.tolist()
    assert frames[1]['action'] is None
    eef_log = json.loads((tmp_path / 'eef-actions.jsonl').read_text())
    assert eef_log['eef_delta'] == proposed.tolist()
    assert eef_log['accepted_joint_command'] == accepted.tolist()
    context = json.loads((tmp_path / 'recording-context.json').read_text())
    assert context['native_control_mode'] == 'eef_delta_pose'
    assert context['action_recording'] == 'accepted_joint_commands_after_eef_ik'
    assert context['fps'] == 10

    def encode(command, **kwargs):
        Path(command[-1]).write_bytes(b'video placeholder')

    monkeypatch.setattr('colosseum_client.lerobot_export.subprocess.run', encode)
    dataset = export_trial(tmp_path, ['head_image'], fps=context['fps'], robot_type='so101',
                           task='move', metadata=context)
    table = pq.read_table(dataset / 'data/chunk-000/file-000.parquet').to_pydict()
    assert table['action'] == [accepted.tolist()]
    exported = json.loads((dataset / 'meta/colosseum.json').read_text())
    assert exported['action_semantics'] == 'driver-accepted joint command after EEF IK'
