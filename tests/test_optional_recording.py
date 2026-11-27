"""Fault injection for optional data; all hardware/network interfaces are fakes."""
import ast
import asyncio
from dataclasses import replace
import json
from pathlib import Path
import re
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from colosseum_client import evaluation as e
from colosseum_client.recording import OptionalRecording
from colosseum_client.roboeval_camera import ContinuousCamera
from colosseum_client.roboeval_export import source_extension
from test_roboeval import FakeCamera
from test_roboeval_shared import assignment, result
from test_yam import config, hardware, make_robot, yam


def broken(*args, **kwargs):
    raise OSError('injected recording failure')


def test_sent_recording_error_keeps_both_commands_and_real_control_errors_propagate(tmp_path, hardware):
    robot = make_robot(config())
    observer = OptionalRecording(tmp_path)
    observer.recorder = SimpleNamespace(sent=broken, fail=lambda e: None, close=lambda **kw: None)
    robot._roboeval = observer
    action = np.r_[robot.arms[0].position, robot.arms[1].position]
    robot._send_action(action)
    assert [len(arm.commands) for arm in robot.arms] == [1, 1]
    assert not robot.closed and isinstance(observer.error, OSError)
    observer.close(complete=True)
    assert json.loads((tmp_path / 'roboeval-error.json').read_text())['incomplete']
    with pytest.raises(ValueError, match='incomplete'):
        source_extension(tmp_path)
    robot.arms[0].command_joint_pos = broken
    with pytest.raises(OSError):
        robot._send_action(action)
    assert robot.closed and len(robot.arms[1].commands) == 1


def test_factory_metadata_failure_leaves_original_camera_and_arm_initialization(tmp_path, hardware, monkeypatch):
    monkeypatch.setattr(yam._RealSenseCamera, 'cache_factory_calibration', broken, raising=False)
    robot = make_robot(replace(config(), roboeval_setup='configured.json'))
    assert len(robot.arms) == 2 and robot.get_observation().images
    with pytest.raises(RuntimeError, match='metadata incomplete'):
        robot.roboeval_camera_metadata()
    robot.close()


def test_frame_sink_failure_does_not_poison_live_camera_and_source_error_still_raises(tmp_path):
    observer = OptionalRecording(tmp_path)
    observer.recorder = SimpleNamespace(frame=broken, fail=lambda e: None, close=lambda **kw: None)
    source = FakeCamera()
    camera = ContinuousCamera(source, 'left', observer)
    try:
        first = camera.read()
        deadline = time.monotonic() + 2
        while source.index < 3 and time.monotonic() < deadline:
            time.sleep(.01)
        second = camera.read()
        assert first[0, 0, 0] < second[0, 0, 0]
        assert camera.error is None and isinstance(observer.error, OSError)
        source.close()
        camera.thread.join(2)
        with pytest.raises(RuntimeError, match='camera failed'):
            camera.read()
    finally:
        camera.close()


@pytest.mark.parametrize('fault', ['init', 'start', 'plan', 'select', 'close', 'timing', 'event'])
def test_extra_faults_finish_trial_and_preserve_required_marker(tmp_path, monkeypatch, fault):
    from colosseum_client import roboeval, roboeval_shared
    calls = []
    class Extra:
        def __init__(self, *a, **kw):
            if fault == 'init': broken()
            self.setup = SimpleNamespace(validate_cameras=lambda c: None)
        def fail(self, error): pass
        def start(self, robot, **kw):
            if fault == 'start': broken()
        def plan(self, *a, **kw):
            if fault == 'plan': broken()
        def select(self, *a, **kw):
            if fault == 'select': broken()
        def close(self, **kw):
            if fault == 'close': broken()
    class Robot:
        joint_count, has_gripper, action_dim, action_space_name = 12, True, 14, 'joint_position'
        def __init__(self, cfg): pass
        def get_observation(self): return object()
        def execute(self, action): calls.append('execute')
        def roboeval_observation_timing(self):
            if fault == 'timing': broken()
            return {}
        async def finish_trial(self, robot_call): calls.append('home')
        def close(self): calls.append('closed')
    class Client:
        def __init__(self, *a, **kw): pass
        async def connect(self, **kw): return SimpleNamespace(action_spaces=['joint_position'])
        async def infer(self, *a, **kw): return SimpleNamespace(plan_id=1, request_sequence=1, start_step=0)
        async def close(self): pass
    class OriginalRecorder:
        started = 0.
        def __init__(self, *a): pass
        def add(self, *a, **kw): pass
        def close(self): pass
    monkeypatch.setattr(roboeval, 'RoboEvalRecorder', Extra)
    monkeypatch.setitem(sys.modules, 'i2rt.motor_drivers.dm_driver', SimpleNamespace(
        DMChainCanInterface=SimpleNamespace(get_feedback_snapshot=lambda: None)))
    monkeypatch.setattr(e, 'ColosseumClient', Client)
    monkeypatch.setattr(e, 'TrialRecorder', OriginalRecorder)
    monkeypatch.setattr(e, 'protobuf_observation', lambda *a, **kw: object())
    monkeypatch.setattr(e, 'action_chunk', lambda *a, **kw: np.zeros((1, 14)))
    if fault == 'event': monkeypatch.setattr(roboeval_shared, 'record_event', broken)
    cfg = replace(config(), roboeval_setup='configured.json', evaluation_dir=str(tmp_path))
    a = dict(id='ev', task=dict(instruction='cup', cameras=[], max_steps=1))
    api = SimpleNamespace(request=lambda *a: calls.append('finished'))
    asyncio.run(e.run_trial(cfg, a, dict(id='r'), tmp_path, api, robot_factory=Robot))
    assert calls == ['execute', 'home', 'finished', 'closed']
    context = json.loads((tmp_path / 'recording-context.json').read_text())
    assert context['roboeval_required'] and context['roboeval_shared_required']
    marker = 'shared-events-error.json' if fault == 'event' else 'roboeval-error.json'
    assert json.loads((tmp_path / marker).read_text())['incomplete']


@pytest.mark.parametrize('feedback', ['', 'B worked better'])
def test_export_failure_submits_once_preserves_inputs_and_continues_normal_prompt(tmp_path, monkeypatch, feedback):
    a = assignment()
    a.update(track='open', state='pending', recording=True, test=False,
             task=dict(instruction='cup', cameras=[], max_steps=1))
    a['runs'].reverse()
    for run in a['runs']: run['state'] = 'assigned'
    submissions, prompts, exports = [], [], []
    class API:
        def __init__(self, cfg): pass
        def request(self, method, path, body=None):
            if path == '/reset': return {}
            if path.endswith('/result'):
                submissions.append(body)
                return {}
            return a
        def upload_dataset(self, *args): pytest.fail('Incomplete exports must not upload')
        def close(self): pass
    async def run(cfg, assigned, trial, path, api, **kw):
        path.mkdir(parents=True, exist_ok=True)
        (path / 'frames.jsonl').write_text('preserved raw input\n')
        trial['state'] = 'finished'
    def export(cfg, assigned, trial, folder, result):
        assert len(submissions) == 1
        exports.append(trial['id'])
        broken()
    monkeypatch.setattr(e, 'EvalAPI', API)
    monkeypatch.setattr(e, 'run_trial', run)
    monkeypatch.setattr(e, 'export_expanded_run', export)
    answers = iter(['', '60', '2', '', '100', 'B', feedback, 'n'])
    def answer(prompt):
        prompts.append(prompt)
        return next(answers)
    monkeypatch.setattr('builtins.input', answer)
    cfg = replace(config(), roboeval_setup='configured.json', evaluation_dir=str(tmp_path),
                  instruction='cup', scene='desk', skip_upload=False)
    e.run_evaluation(cfg, track='open')
    saved = json.loads((tmp_path / a['id'] / 'manifest.json').read_text())
    assert saved['submitted'] and len(submissions) == 1
    assert submissions[0]['feedback'] == feedback
    assert exports == [r['id'] for r in a['runs']]
    assert len(prompts) == 8 and prompts[-1].startswith('Continue')
    for r in a['runs']:
        assert not saved['dataset_exports'][r['id']]['complete']
        assert (tmp_path / a['id'] / r['id'] / 'frames.jsonl').read_text() == 'preserved raw input\n'




@pytest.mark.parametrize('previous_failure', [False, True])
def test_existing_backfill_reexports_when_no_dataset_exists(tmp_path, monkeypatch, previous_failure):
    from colosseum_client import dataset_upload as upload
    a = assignment()
    a['track'] = 'open'
    folder = tmp_path / a['id']
    folder.mkdir()
    statuses = {r['id']: {'complete': False} for r in a['runs']} if previous_failure else {}
    (folder / 'manifest.json').write_text(json.dumps(dict(assignment=a, result=result(), submitted=True,
                                                       dataset_exports=statuses)))
    for r in a['runs']:
        path = folder / r['id']
        path.mkdir()
        (path / 'recording-context.json').write_text('{"roboeval_required":true}')
    events = []
    class API:
        def __init__(self, cfg): pass
        def request(self, *args): return a
        def upload_dataset(self, run_id, path):
            assert path.is_dir()
            events.append(('upload', run_id))
        def close(self): pass
    def export(cfg, assigned, run, folder, original):
        assert original == result()
        (folder / run['id'] / 'lerobot').mkdir()
        events.append(('export', run['id']))
        return folder / run['id'] / 'lerobot'
    monkeypatch.setattr(upload, 'EvalAPI', API)
    monkeypatch.setattr(upload, 'export_expanded_run', export)
    monkeypatch.setattr(upload.RobotClientConfig, 'from_yaml', lambda p: SimpleNamespace(evaluation_dir=str(tmp_path)))
    monkeypatch.setattr(sys, 'argv', ['colosseum-upload-dataset', 'offline.yaml'])
    upload.main()
    assert events == [(verb, r['id']) for r in a['runs'] for verb in ('export', 'upload')]


def test_keyboard_interrupt_is_not_swallowed(tmp_path):
    observer = OptionalRecording(tmp_path)
    def interrupt(): raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt): observer.call(interrupt)


@pytest.mark.parametrize('fps,accepted', [(15., True), (14., False)])
def test_camera_minimum_accepts_exactly_fifteen_fps(fps, accepted):
    from colosseum_client.roboeval import RoboEvalRecorder
    recorder = RoboEvalRecorder.__new__(RoboEvalRecorder)
    recorder.started, recorder.hz = 0., 30
    recorder.continuous = False
    recorder.setup = SimpleNamespace(arms=['left', 'right'])
    recorder.counts = dict(states=122, actions=4)
    recorder.action_kinds = {s: {'raw_policy', 'sent_command'} for s in recorder.setup.arms}
    recorder.state_stats = {s: dict(samples=61, first=0., last=2., max_gap_s=1/30)
                            for s in recorder.setup.arms}
    recorder.required_camera_keys = ['left', 'right', 'head_image']
    recorder.frame_stats = {s: dict(frames=int(2*fps)+1, first=0., last=2., max_gap_s=1/fps)
                            for s in recorder.required_camera_keys}
    if accepted:
        recorder._validate_capture(2.)
        assert all(r['actual_fps'] == 15 for r in recorder.frame_stats.values())
    else:
        with pytest.raises(RuntimeError, match='15 fps'):
            recorder._validate_capture(2.)


def test_each_successful_hardware_send_has_exactly_one_record(tmp_path,hardware):
    robot=make_robot(config());observer=OptionalRecording(tmp_path);records=[]
    observer.recorder=SimpleNamespace(sent=lambda side,value:records.append((side,value.copy())),fail=lambda e:None)
    robot._roboeval=observer
    action=np.r_[robot.arms[0].position,robot.arms[1].position]
    robot._send_action(action)
    assert [side for side,value in records]==['left','right']
    assert [len(arm.commands) for arm in robot.arms]==[1,1]
    for i,(side,value) in enumerate(records):np.testing.assert_array_equal(value,action[7*i:7*i+7])
    robot.close()
