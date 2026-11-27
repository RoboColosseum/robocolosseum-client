"""Exercise real client constructors before any network/device connection."""
from types import SimpleNamespace

import pytest

from colosseum_client import evaluation
from colosseum_client.client import ColosseumClient
from colosseum_client.local_policy import LocalPolicyClient
from colosseum_client.robot_config import RobotClientConfig


@pytest.mark.parametrize('mode', ['local', 'remote'])
@pytest.mark.parametrize('state_hz', [0, 100])
async def test_recording_frequency_does_not_change_real_client_protocol(mode, state_hz, tmp_path, monkeypatch):
    class BeforeNetwork(RuntimeError):
        pass

    connected = []
    async def connect(client, **kwargs):
        connected.append(client.robot_spec.control_hz)
        assert client.robot_spec.control_hz == 30
        raise BeforeNetwork('stop before network and hardware')

    monkeypatch.setattr(LocalPolicyClient if mode == 'local' else ColosseumClient, 'connect', connect)
    created = []
    def factory(config):
        assert mode == 'remote'  # Local mode must connect before robot construction.
        created.append(True)
        return SimpleNamespace(joint_count=12, has_gripper=True, action_space_name='joint_position',
                               action_dim=14, close=lambda: None)

    config = RobotClientConfig(url='ws://offline.invalid', token='test', test=True,
        cameras={}, robot_type='yam', control_hz=30, roboeval_state_hz=state_hz,
        policy_server_url='ws://localhost:8000', recording=False)
    assignment = dict(id='offline-assignment', inference_mode=mode,
                      task=dict(instruction='offline', max_steps=1, cameras=[]))
    with pytest.raises(BeforeNetwork):
        await evaluation.run_trial(config, assignment, dict(id='offline-run'), tmp_path,
                                   SimpleNamespace(), robot_factory=factory)
    assert connected == [30]
    assert bool(created) == (mode == 'remote')
