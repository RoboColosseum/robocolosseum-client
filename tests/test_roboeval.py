import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import numpy as np
import pyarrow.parquet as pq
import pytest

from colosseum_client import colosseum_pb2 as pb
from colosseum_client.roboeval import RoboEvalRecorder, RAW_SOURCE
from colosseum_client.roboeval_setup import Setup
from colosseum_client.roboeval_camera import ContinuousCamera
from colosseum_client.roboeval_export import source_extension, validate_uploaded_extension
from colosseum_client.tensors import tensor_from_numpy


def setup_file(path, sides=('left', 'right')):
    path.mkdir(parents=True, exist_ok=True)
    (path / 'model.xml').write_text('<mujoco><worldbody/></mujoco>')
    (path / 'robot.urdf').write_text('<robot name="fake"/>')
    document = dict(calibrated=True, calibration_version='fake-test-only',
        model_path='model.xml', urdf_path='robot.urdf',
        arms={side: dict(base_to_world=np.eye(4).tolist(), camera_to_eef=np.eye(4).tolist(),
            camera_serial=side, eef_site='grasp_site',
            gripper=dict(readings=[0, 1], widths_m=[.01, .09])) for side in reversed(sides)})
    filename = path / 'setup.json'
    filename.write_text(json.dumps(document))
    return filename


class FakeKinematics:
    def __init__(self, setup):
        self.setup = setup

    def pose(self, side, q):
        return [q[0], q[1], 1 if side == 'left' else 2], [1, 0, 0, 0]


class FakeCamera:
    timeout_ms = 100
    def __init__(self):
        self.closed = threading.Event()
        self.index = 0

    def read_frame(self):
        if self.closed.wait(1/30):
            raise RuntimeError('camera stopped')
        self.index += 1
        return dict(pixels=np.full((24, 32, 3), self.index % 256, dtype=np.uint8),
            source_frame_number=self.index, source_timestamp_ms=self.index * 1000/30,
            source_timestamp_domain='fake-test-clock', received_monotonic_s=time.monotonic())

    def close(self):
        self.closed.set()


class FakeRobot:
    def __init__(self, sides=('left', 'right'), stale=False):
        self.roboeval_arm_ids = sides
        self.stale = stale
        self.cameras = {}
        self.feedback_sequence = 0

    def roboeval_camera_metadata(self):
        return {side: dict(serial=side, intrinsics=dict(fx=500)) for side in self.roboeval_arm_ids}

    def roboeval_limits(self):
        return {'test_fixture': True}

    def start_roboeval(self, recorder):
        self.recorder = recorder
        self.cameras = {side: ContinuousCamera(FakeCamera(), side, recorder)
                        for side in self.roboeval_arm_ids}

    def stop_roboeval(self):
        for camera in self.cameras.values():
            camera.detach()

    def get_roboeval_feedback(self):
        # This fake creates a fresh sweep per call; do not infer its sequence
        # from a coarse Windows clock bucket. The SDK snapshot test separately
        # verifies repeated reads of a real sweep keep their original sequence.
        self.feedback_sequence += 1
        return {side: dict(sequence=1 if self.stale else self.feedback_sequence,
            received_monotonic_s=time.monotonic(), running=True,
            position=np.arange(6)*.01, velocity=np.arange(6)*.02,
            torque_nm=np.arange(6)*.03, gripper_reading=1.05, gripper_effort_nm=-.42)
            for side in self.roboeval_arm_ids}

    def close(self):
        for camera in self.cameras.values():
            camera.close()


def original_plan(sides=('left', 'right')):
    raw = np.arange(4 * len(sides)*7, dtype=np.float64).reshape(4, len(sides)*7) / 10
    return pb.ActionPlan(plan_id=12, start_step=0, raw_actions=tensor_from_numpy(raw),
                         raw_action_source=RAW_SOURCE, raw_action_space='joint_position'), raw


def capture(path, setup, sides=('left', 'right'), identity=None, **options):
    recorder = RoboEvalRecorder(path, setup, hz=30, identity=identity or dict(run_id='fake'),
                               kinematics_factory=FakeKinematics, **options)
    robot = FakeRobot(sides)
    recorder.start(robot)
    try:
        plan, raw = original_plan(sides)
        recorder.plan(plan, action_space='joint_position', executable_length=2)
        recorder.select(plan.plan_id, 0, 0)
        for side in sides:
            recorder.sent(side, np.r_[np.zeros(6), .5])
        # Simulate a blocking inference: no policy observations/actions during this period.
        time.sleep(.55)
        recorder.close(complete=True)
    finally:
        try:
            recorder.close()
        finally:
            robot.close()
    return recorder, raw


@pytest.mark.parametrize('sides', [('right',), ('left', 'right')])
def test_independent_feedback_video_raw_chunks_and_shared_arm_schema(tmp_path, sides):
    setup = setup_file(tmp_path / 'calibration', sides)
    recorder, raw = capture(tmp_path / 'run', setup, sides)
    states = pq.read_table(recorder.meta / 'states.parquet').to_pylist()
    frames = pq.read_table(recorder.meta / 'frames.parquet').to_pylist()
    actions = pq.read_table(recorder.meta / 'actions.parquet').to_pylist()
    assert {r['arm_id'] for r in states} == set(sides)
    assert len(states) >= 12 * len(sides)  # Continued through the simulated inference.
    assert len(frames) >= 8 * len(sides)
    for row in states:
        assert row['gripper_reading_normalized_unclipped'] == 1.05
        assert row['gripper_effort_nm'] == -.42
        assert row['gripper_width_m'] == pytest.approx(.094)  # Preserve feedback overshoot.
        assert row['joint_velocity_rad_s'][2] == pytest.approx(.04)
        assert row['joint_torque_nm'][3] == pytest.approx(.09)
        assert row['eef_orientation_world_wxyz'] == [1, 0, 0, 0]
    for i, side in enumerate(sides):
        original = [r['values'] for r in actions if r['arm_id'] == side and r['kind'] == 'raw_policy']
        np.testing.assert_array_equal(original, raw[:, 7*i:7*(i+1)])
        sent = [r for r in actions if r['arm_id'] == side and r['kind'] == 'sent_command']
        assert len(sent) == 1 and sent[0]['gripper_command_width_m'] == pytest.approx(.05)
    assert json.loads((recorder.meta / 'status.json').read_text())['complete'] is True
    assert source_extension(tmp_path / 'run')['signature']


def test_missing_raw_output_rejected_before_any_command(tmp_path):
    recorder = RoboEvalRecorder(tmp_path / 'run', setup_file(tmp_path / 'cal'), hz=30,
        identity={}, kinematics_factory=FakeKinematics)
    with pytest.raises(ValueError, match='before execution'):
        recorder.plan(pb.ActionPlan(), action_space='joint_position', executable_length=1)
    recorder.close()
    assert json.loads((recorder.meta / 'status.json').read_text())['complete'] is False


def test_repeated_feedback_is_failure_instead_of_fresh_samples(tmp_path):
    recorder = RoboEvalRecorder(tmp_path / 'run', setup_file(tmp_path / 'cal'), hz=30,
        identity={}, kinematics_factory=FakeKinematics)
    robot = FakeRobot(stale=True)
    recorder.start(robot)
    try:
        assert recorder.stop.wait(1)
        with pytest.raises(RuntimeError, match='incomplete'):
            recorder.close(complete=True)
        assert recorder.error is not None and 'repeated' in str(recorder.error)
        assert not (recorder.root / 'manifest.json').exists()
    finally:
        robot.close()


def test_queue_saturation_does_not_block_producer_or_claim_complete(tmp_path):
    recorder = RoboEvalRecorder(tmp_path / 'run', setup_file(tmp_path / 'cal'), hz=30,
        identity={}, kinematics_factory=FakeKinematics, queue_size=1)
    started = time.monotonic()
    recorder.frame('left', {})
    recorder.frame('left', {})
    assert time.monotonic() - started < .1
    with pytest.raises(RuntimeError, match='incomplete'):
        recorder.close(complete=True)
    assert 'queue full' in str(recorder.error)


def test_setup_requires_true_geometry_and_camera_identity(tmp_path):
    path = setup_file(tmp_path)
    document = json.loads(path.read_text())
    document['calibrated'] = False
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match='verified calibration'):
        Setup(path)
    document['calibrated'] = True
    path.write_text(json.dumps(document))
    setup = Setup(path)
    with pytest.raises(ValueError, match='serial differs'):
        setup.validate_cameras(dict(left_image='wrong', right_image='right'))


def test_legacy_export_and_recursive_upload_include_extension_and_reject_tamper(tmp_path):
    from colosseum_client.recording import TrialRecorder
    from colosseum_client.robot_interface import RobotObservation
    from colosseum_client.lerobot_export import export_trial
    from colosseum_client.evaluation import EvalAPI
    import base64
    import hashlib
    run = tmp_path / 'run'
    capture(run, setup_file(tmp_path / 'cal'))
    legacy = TrialRecorder(run, ['head_image'])
    obs = RobotObservation({'head_image': np.zeros((24, 32, 3), np.uint8)},
                            np.zeros(12), np.array([.5, .5]), np.empty(0))
    legacy.add(obs, np.zeros(14))
    legacy.close()
    options = dict(fps=30, robot_type='yam', task='fake', metadata={})
    output = export_trial(run, ['head_image'], **options)
    main = pq.read_table(output / 'data/chunk-000/file-000.parquet')
    assert main.num_rows == 1 and 'joint_velocity_rad_s' not in main.column_names
    assert len(list((output / 'data').glob('*/*.parquet'))) == 1
    assert {p.name for p in (output / 'meta/roboeval').glob('*.parquet')} == {
        'states.parquet', 'actions.parquet', 'frames.parquet'}
    validate_uploaded_extension(output)
    api = EvalAPI.__new__(EvalAPI)
    manifests = []
    def request(method, route, body=None):
        if route.endswith('/dataset'):
            manifests.append(body)
            for item in body['files']:
                data = (output / item['path']).read_bytes()
                assert item['checksum_sha256'] == base64.b64encode(hashlib.sha256(data).digest()).decode()
        return {'complete': True}
    api.request = request
    api.upload_dataset('fake', output)
    assert any(f['path'] == 'meta/roboeval/states.parquet' for f in manifests[0]['files'])
    assert export_trial(run, ['head_image'], **options) == output  # Images already cleaned.
    (output / 'meta/roboeval/states.parquet').write_bytes(b'changed')
    with pytest.raises(ValueError, match='modified'):
        api.upload_dataset('fake', output)
    with pytest.raises(ValueError, match='modified'):
        export_trial(run, ['head_image'], **options)


def test_required_missing_and_incomplete_sources_cannot_export(tmp_path):
    (tmp_path / 'recording-context.json').write_text('{"roboeval_required":true}')
    with pytest.raises(ValueError, match='missing'):
        source_extension(tmp_path)
    (tmp_path / 'roboeval').mkdir()
    with pytest.raises(ValueError, match='incomplete'):
        source_extension(tmp_path)


def test_encoder_failure_stops_recording_and_preserves_partial_tables(tmp_path):
    class FailedEncoder:
        def __init__(self, *args): pass
        def add(self, pixels): raise OSError('simulated encoder failure')
        def close(self): pass
        def abort(self): pass
    recorder = RoboEvalRecorder(tmp_path / 'run', setup_file(tmp_path / 'cal'), hz=30,
        identity={}, kinematics_factory=FakeKinematics, video_factory=FailedEncoder)
    robot = FakeRobot()
    recorder.start(robot)
    try:
        assert recorder.stop.wait(2)
        with pytest.raises(RuntimeError, match='incomplete'):
            recorder.close(complete=True)
        status = json.loads((recorder.meta / 'status.json').read_text())
        assert status['complete'] is False and 'encoder failure' in status['error']
        assert not (recorder.root / 'manifest.json').exists()
        pq.read_table(recorder.meta / 'states.parquet')  # Partial numeric data remains readable.
    finally:
        robot.close()


def test_video_segment_indices_follow_source_frames(tmp_path, monkeypatch):
    from colosseum_client import roboeval
    monkeypatch.setattr(roboeval, 'VIDEO_SEGMENT_FRAMES', 3)
    class Encoder:
        def __init__(self, path, shape, fps):
            self.frames, self.closed = 0, False
        def add(self, pixels): self.frames += 1
        def close(self): self.closed = True
        def abort(self): pass
    recorder = RoboEvalRecorder(tmp_path / 'run', setup_file(tmp_path / 'cal'), hz=30,
        identity={}, kinematics_factory=FakeKinematics, video_factory=Encoder)
    rows = [recorder._write_video('left', dict(pixels=np.zeros((24, 32, 3), np.uint8),
        source_frame_number=i, received_monotonic_s=100+i/30, source_timestamp_ms=i*1000/30,
        source_timestamp_domain='test')) for i in range(7)]
    assert [r['video_frame_index'] for r in rows] == [0, 1, 2, 0, 1, 2, 0]
    assert rows[0]['video_path'] != rows[3]['video_path'] != rows[6]['video_path']
    assert recorder.video[('left', 0)].closed and recorder.video[('left', 1)].closed
    recorder.close()


@pytest.mark.parametrize('missing_raw', [False, True])
async def test_trial_lifecycle_prepares_before_robot_and_closes_before_home(tmp_path, monkeypatch, missing_raw):
    import asyncio
    import sys
    from colosseum_client import evaluation, local_policy, roboeval
    from colosseum_client.robot_config import RobotClientConfig
    from colosseum_client.robot_interface import RobotObservation
    events = []
    class Recorder(RoboEvalRecorder):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs, kinematics_factory=FakeKinematics)
            events.append('prepared')
    monkeypatch.setattr(roboeval, 'RoboEvalRecorder', Recorder)
    monkeypatch.setitem(sys.modules, 'i2rt.motor_drivers.dm_driver',
        SimpleNamespace(DMChainCanInterface=SimpleNamespace(get_feedback_snapshot=lambda: None)))
    setup = setup_file(tmp_path / 'cal')
    config = RobotClientConfig(url='ws://offline.invalid', token='test', control_hz=30,
        robot_type='yam', policy_server_url='ws://localhost:8000',
        cameras=dict(left_image='left', right_image='right'), roboeval_setup=str(setup))
    class Robot(FakeRobot):
        action_space_name, action_dim, joint_count, has_gripper, closed = 'joint_position', 14, 12, True, False
        def get_observation(self):
            return RobotObservation({f'{side}_image': camera.read() for side, camera in self.cameras.items()},
                                     np.zeros(12), np.ones(2)*.5, np.empty(0))
        def execute(self, action):
            events.append('execute')
            for i, side in enumerate(self.roboeval_arm_ids):
                self.recorder.sent(side, action[7*i:7*(i+1)])
        async def finish_trial(self, robot_call):
            events.append('home')
            status = json.loads((tmp_path / 'run/roboeval/meta/roboeval/status.json').read_text())
            assert status['complete'] is not missing_raw
        async def hold_for_recovery(self, robot_call):
            raise AssertionError('Fake return must not fail')
        def close(self):
            events.append('robot_closed')
            self.closed = True
            super().close()
    def factory(*args, **kwargs):
        assert events == ['prepared', 'connected']
        events.append('robot_constructed')
        return Robot()
    class Client:
        model = dict(action_space='joint_position', url='fake', revision='test')
        def __init__(self, *args, **kwargs): pass
        async def connect(self, **kwargs):
            events.append('connected')
            return SimpleNamespace(action_spaces=['joint_position'])
        async def infer(self, observation, **kwargs):
            await asyncio.sleep(.4)
            plan, _ = original_plan()
            plan.start_step = observation.control_step
            plan.valid_until_step = plan.start_step
            plan.actions.CopyFrom(tensor_from_numpy(np.zeros((1, 14), np.float32)))
            if missing_raw: plan.ClearField('raw_actions')
            return plan
        async def before_action(self, step): pass
        async def finish(self): events.append('finished')
        async def close(self): events.append('client_closed')
    monkeypatch.setattr(local_policy, 'LocalPolicyClient', Client)
    assignment = dict(id='fake-assignment', inference_mode='local',
        task=dict(instruction='offline test', max_steps=1, cameras=['left_image', 'right_image']))
    call = evaluation.run_trial(config, assignment, dict(id='fake-run'), tmp_path / 'run',
        SimpleNamespace(), robot_factory=factory, confirm_live=lambda *args: True)
    await call
    assert events.index('execute') < events.index('home') < events.index('finished')
    assert (tmp_path / 'run/roboeval-error.json').exists() is missing_raw
    assert 'robot_closed' in events and 'client_closed' in events


@pytest.fixture
def extended_export(tmp_path):
    from colosseum_client.recording import TrialRecorder
    from colosseum_client.robot_interface import RobotObservation
    from colosseum_client.lerobot_export import export_trial
    run = tmp_path / 'fake-assignment/fake'
    identity = dict(assignment_id='fake-assignment', run_id='fake', side='A')
    capture(run, setup_file(tmp_path / 'cal'), identity=identity)
    recorder = TrialRecorder(run, ['head_image'])
    recorder.add(RobotObservation({'head_image': np.zeros((24, 32, 3), np.uint8)},
        np.zeros(12), np.array([.5, .5]), np.empty(0)), np.zeros(14))
    recorder.close()
    output = export_trial(run, ['head_image'], fps=30, robot_type='yam',
                          task='fake', metadata=identity)
    return run, output


@pytest.mark.parametrize('change', ['missing', 'added', 'encoder_log', 'identity'])
def test_committed_inventory_is_exact_for_every_uploaded_file(extended_export, change):
    run, output = extended_export
    validate_uploaded_extension(output)
    if change == 'missing':
        (output / 'videos/roboeval/left/chunk-000/file-000.mp4').unlink()
    elif change == 'added':
        (output / 'meta/roboeval/extra.json').write_text('{}')
    elif change == 'encoder_log':
        (output / 'meta/roboeval/uncommitted.encoder.log').write_text('extra')
    else:
        marker = output / 'meta/colosseum.json'
        document = json.loads(marker.read_text())
        document['run_id'] = 'another-run'
        marker.write_text(json.dumps(document))
    with pytest.raises(ValueError, match='inventory|identity'):
        validate_uploaded_extension(output)


def test_source_identity_is_verified_before_reusing_export(extended_export):
    from colosseum_client.lerobot_export import export_trial
    run, output = extended_export
    with pytest.raises(ValueError, match='identity'):
        export_trial(run, ['head_image'], fps=30, robot_type='yam', task='fake',
                     metadata=dict(assignment_id='another-assignment', run_id='fake', side='A'))
