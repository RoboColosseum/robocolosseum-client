import importlib.util
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import pyarrow.parquet as pq
import pytest

from colosseum_client.roboeval_feedback import parse_feedback, build_comparison, outcome_for_run
from colosseum_client.roboeval_shared import record_event, shared_payload
from colosseum_client.roboeval_export import validate_uploaded_extension
from colosseum_client.roboeval_calibration import factory_extrinsics
from colosseum_client.roboeval_setup import Setup
from test_roboeval import setup_file, FakeRobot, FakeCamera, FakeKinematics, original_plan

FEEDBACK = ('Task=Place the orange in the dish. Policy A: Task success=no; '
    'Completed stages=approach -> grasp the orange -> lift; '
    'Main issue=dropped the orange during transfer before placing it in the dish; '
    'Drop=yes; Damage=no; Spill=na. Policy B: Task success=yes; '
    'Completed stages=approach -> grasp the orange -> lift -> transfer -> place in the dish -> release; '
    'Main issue=none; Drop=no; Damage=no; Spill=na. Final preference=B; '
    'Comparison reason=B completed the task, while A dropped the orange before placement.')


def assignment():
    # Deliberately reverse list order; A/B is never the left/right arm or list index.
    return {'id': 'ev_shared', 'runs': [
        {'id': 'policy_second', 'side': 'B', 'model_id': 'model-success'},
        {'id': 'policy_first', 'side': 'A', 'model_id': 'model-failed'}]}


def result(feedback=FEEDBACK):
    return {'feedback': feedback, 'preference': 'B', 'outcomes': {
        'A': {'success': False, 'partial_success': .6},
        'B': {'success': True, 'partial_success': 1.0}}}


def test_fixed_english_example_roundtrip_and_run_identity():
    comp = build_comparison(assignment(), result())
    assert comp['raw_feedback'] == FEEDBACK
    assert comp['policies']['A']['drop'] == 'yes'
    assert comp['policies']['B']['spill'] == 'not_applicable'
    assert comp['runs']['A']['run_id'] == 'policy_first'
    assert comp['runs']['B']['model'] == {'model_id': 'model-success'}
    for side, rid in [('A', 'policy_first'), ('B', 'policy_second')]:
        own = outcome_for_run(comp, rid)
        assert own['self_side'] == side
        assert own['episode_labels']['success'] is (side == 'B')
        assert own['comparison']['runs'] == comp['runs']
    with pytest.raises(ValueError, match='does not belong'):
        outcome_for_run(comp, 'unrelated')


def test_no_grasp_is_failed_task_without_drop():
    raw = FEEDBACK.replace('approach -> grasp the orange -> lift;', 'approach;', 1)
    raw = raw.replace('dropped the orange during transfer before placing it in the dish',
                      'failed to grasp the orange').replace('Drop=yes;', 'Drop=no;', 1)
    a = build_comparison(assignment(), result(raw))['policies']['A']
    assert a['completed_stages'] == ['approach']
    assert a['success'] is False and a['drop'] == 'no' and a['damage'] == 'no'


@pytest.mark.parametrize('raw', [FEEDBACK, FEEDBACK.lower(),
    FEEDBACK.replace('Task success=', 'Task success =  ').replace('Spill=na', 'Spill=NA')])
def test_allowed_case_whitespace_preserves_original(raw):
    parsed = parse_feedback(raw)
    assert parsed['raw_feedback'] == raw
    assert parsed['preference'] == 'B'


def test_unknown_is_not_false_or_not_applicable():
    raw = FEEDBACK.replace('Task success=no', 'Task success=unknown', 1)
    raw = raw.replace('Completed stages=approach -> grasp the orange -> lift;',
                      'Completed stages=unknown;', 1).replace('Drop=yes', 'Drop=unknown', 1)
    a = build_comparison(assignment(), result(raw))['policies']['A']
    assert a['success'] is None and a['drop'] == 'unknown'
    assert a['spill'] == 'not_applicable' and a['completed_stages_status'] == 'unknown'


@pytest.mark.parametrize('label', ['Task', 'Task success', 'Completed stages', 'Main issue',
                                  'Drop', 'Damage', 'Spill', 'Final preference', 'Comparison reason'])
def test_missing_or_renamed_labels_fail(label):
    with pytest.raises(ValueError):
        parse_feedback(FEEDBACK.replace(label + '=', 'Different=', 1))


@pytest.mark.parametrize('raw', [
    FEEDBACK + '\n', FEEDBACK.replace('Drop=yes', 'Drop=maybe'),
    FEEDBACK.replace('Task success=no', 'Task success=na'),
    FEEDBACK.replace('Drop=yes;', 'Drop=yes; Drop=no;'),
    FEEDBACK.replace('Policy B:', 'Policy A:'),
    FEEDBACK.replace('Main issue=none', 'Main issue=Policy B: none'),
    FEEDBACK.replace('approach -> grasp the orange -> lift;', 'approach -> -> lift;', 1),
    FEEDBACK.replace('Task=Place', 'Task=\u4efb\u52a1'),
])
def test_malformed_feedback_never_confirms_labels(raw):
    with pytest.raises(ValueError):
        parse_feedback(raw)


@pytest.mark.parametrize('change', ['preference', 'success', 'missing_side', 'duplicate_id', 'unknown_side'])
def test_conflicts_and_ambiguous_identity_fail(change):
    a, r = assignment(), result()
    if change == 'preference': r['preference'] = 'A'
    if change == 'success': r['feedback'] = FEEDBACK.replace('Task success=no', 'Task success=yes', 1)
    if change == 'missing_side': a['runs'] = a['runs'][:1]
    if change == 'duplicate_id': a['runs'][1]['id'] = a['runs'][0]['id']
    if change == 'unknown_side': a['runs'][1]['side'] = 'left'
    if change in ('preference', 'success'):
        comparison = build_comparison(a, r)
        assert comparison['raw_feedback'] == r['feedback']
        assert comparison['preference'] == r['preference']
        assert comparison['policies']['A']['drop'] == 'unknown'
        assert comparison['policies']['A']['success'] is False
    else:
        with pytest.raises(ValueError):
            build_comparison(a, r)


def test_shared_event_order_timing_and_foreign_run_rejection(tmp_path):
    own = outcome_for_run(build_comparison(assignment(), result()), 'policy_first')
    record_event(tmp_path, 'ev_shared', 'policy_first', 'episode_start', stamp=10.)
    record_event(tmp_path, 'ev_shared', 'policy_first', 'inference', stamp=11.,
        observation_monotonic_s=10.9, action_ready_monotonic_s=12., plan_id=7, status='ready')
    record_event(tmp_path, 'ev_shared', 'policy_first', 'inference', stamp=12.1,
        observation_monotonic_s=12., action_ready_monotonic_s=None, status='cancelled')
    record_event(tmp_path, 'ev_shared', 'policy_first', 'episode_end', stamp=13.)
    record_event(tmp_path, 'ev_shared', 'policy_first', 'reset_end', stamp=14.)
    p = shared_payload(tmp_path, own)
    assert p['clock']['event_count'] == 5 and p['rows'][1]['plan_id'] == 7
    assert p['rows'][2]['status'] == 'cancelled' and p['rows'][-1]['kind'] == 'reset_end'
    record_event(tmp_path, 'ev_other', 'policy_first', 'emergency_stop', stamp=15.)
    with pytest.raises(ValueError, match='another run'):
        shared_payload(tmp_path, own)


@pytest.mark.parametrize('problem', ['partial', 'duplicate', 'bad_time', 'missing_end'])
def test_incomplete_or_invalid_event_log_cannot_export(tmp_path, problem):
    own = outcome_for_run(build_comparison(assignment(), result()), 'policy_first')
    record_event(tmp_path, 'ev_shared', 'policy_first', 'episode_start', stamp=10.)
    if problem != 'missing_end':
        record_event(tmp_path, 'ev_shared', 'policy_first', 'episode_end', stamp=11.)
    log = tmp_path / 'shared-events.jsonl'
    if problem == 'partial': log.write_bytes(log.read_bytes() + b'{')
    if problem == 'duplicate': log.write_bytes(log.read_bytes() + log.read_bytes().splitlines(keepends=True)[0])
    if problem == 'bad_time': log.write_text(log.read_text().replace('10.0', '-10.0'))
    with pytest.raises(ValueError):
        shared_payload(tmp_path, own)


class ThreeCameraRobot(FakeRobot):
    def roboeval_camera_metadata(self):
        return {name: {'serial': name, 'intrinsics': {'fx': 500}, 'factory_extrinsics': []}
                for name in ('head_image', 'left_image', 'right_image')}

    def start_roboeval(self, recorder):
        super().start_roboeval(recorder)
        from colosseum_client.roboeval_camera import ContinuousCamera
        self.cameras['head_image'] = ContinuousCamera(FakeCamera(), 'head_image', recorder)


def make_recording(path, calibration, rid):
    from colosseum_client.roboeval import RoboEvalRecorder
    from colosseum_client.recording import TrialRecorder
    from colosseum_client.robot_interface import RobotObservation
    recorder = RoboEvalRecorder(path, calibration, hz=30,
        identity={'assignment_id': 'ev_shared', 'run_id': rid}, kinematics_factory=FakeKinematics)
    robot = ThreeCameraRobot()
    record_event(path, 'ev_shared', rid, 'episode_start')
    recorder.start(robot)
    cameras = ['head_image', 'left_image', 'right_image']
    legacy = TrialRecorder(path, cameras)
    try:
        observed = time.monotonic()
        request = time.monotonic()
        # No policy observation while inference blocks; all cameras keep recording.
        time.sleep(.38)
        plan, _ = original_plan()
        recorder.plan(plan, action_space='joint_position', executable_length=2)
        record_event(path, 'ev_shared', rid, 'inference', stamp=request,
            observation_monotonic_s=observed, action_ready_monotonic_s=time.monotonic(),
            call_id=rid + ':0', plan_id=plan.plan_id, status='ready')
        for step in range(2):
            recorder.select(plan.plan_id, step, step)
            for side in robot.roboeval_arm_ids: recorder.sent(side, np.r_[np.zeros(6), .5])
            obs = RobotObservation({c: np.zeros((24, 32, 3), np.uint8) for c in cameras},
                                   np.zeros(12), np.array([.5, .5]), np.empty(0))
            legacy.add(obs, np.zeros(14))
            time.sleep(.04)
        legacy.close()
        record_event(path, 'ev_shared', rid, 'episode_end')
        recorder.close(complete=True)
        # Reset is still recorded after motor/camera capture has ended.
        record_event(path, 'ev_shared', rid, 'reset_start', reset_kind='robot_return')
        record_event(path, 'ev_shared', rid, 'reset_end', reset_kind='robot_return')
    finally:
        legacy.close()
        recorder.close()
        robot.close()
    (path / 'recording-context.json').write_text(json.dumps({'roboeval_required': True,
        'roboeval_shared_required': True, 'robot_type': 'yam', 'fps': 30, 'test': True,
        'action_space': 'joint_position', 'execution_enabled': True}))
    return cameras


def test_actual_two_dataset_exports_labels_top_video_reload_retry_and_integrity(tmp_path):
    from colosseum_client.lerobot_export import export_trial
    calibration = setup_file(tmp_path / 'cal')
    comp = build_comparison(assignment(), result())
    for side, rid in [('B', 'policy_second'), ('A', 'policy_first')]:
        run = tmp_path / rid
        cameras = make_recording(run, calibration, rid)
        with pytest.raises(ValueError, match='Confirmed comparison'):
            export_trial(run, cameras, fps=30, robot_type='yam', task='orange', metadata={})
        own = outcome_for_run(comp, rid)
        options = dict(fps=30, robot_type='yam', task='orange', metadata={
            'assignment_id': 'ev_shared', 'run_id': rid, 'roboeval_outcome': own})
        output = export_trial(run, cameras, **options)
        marker = json.loads((output / 'meta/colosseum.json').read_text())
        stored = marker['roboeval_layout']['documents']['outcome']
        assert stored == own and stored['comparison']['raw_feedback'] == FEEDBACK
        ep = pq.read_table(output / 'meta/episodes/chunk-000/file-000.parquet').to_pylist()[0]
        assert ep['roboeval/self_run_id'] == rid and ep['roboeval/self_side'] == side
        assert ep['roboeval/success'] is (side == 'B')
        from colosseum_client.roboeval_compact import load_stream
        frames = load_stream(output, 'frames').to_pylist()
        top = [row for row in frames if row['camera_id'] == 'head_image']
        assert len(top) >= 8 and all(row['arm_id'] is None for row in top)
        assert all((output / row['video_path']).is_file() for row in top)
        shared = load_stream(output, 'shared').to_pylist()
        assert shared[-1]['kind'] == 'reset_end' and shared[1]['kind'] == 'inference'
        marker = json.loads((output / 'meta/colosseum.json').read_text())
        assert len([p for p in output.rglob('*') if p.is_file()]) == 9
        assert 'meta/episodes/chunk-000/file-000.parquet' in marker['files']
        assert not (output / 'meta/roboeval').exists()
        validate_uploaded_extension(output)
        assert export_trial(run, cameras, **options) == output  # Raw PNGs were cleaned.
        changed = json.loads(json.dumps(options))
        changed['metadata']['roboeval_outcome']['comparison']['comparison_reason'] = 'changed'
        with pytest.raises(ValueError, match='differs'):
            export_trial(run, cameras, **changed)
        (output / 'meta/episodes/chunk-000/file-000.parquet').write_bytes(b'changed')
        with pytest.raises(ValueError, match='modified'):
            validate_uploaded_extension(output)


def test_camera_installation_matrices_are_optional_but_arm_world_is_not(tmp_path):
    path = setup_file(tmp_path)
    doc = json.loads(path.read_text())
    doc['arms']['left']['camera_to_eef'] = None
    doc['fixed_cameras'] = {'head_image': {'serial': 'top', 'camera_to_world': None}}
    path.write_text(json.dumps(doc))
    Setup(path).validate_cameras({'left_image': 'left', 'right_image': 'right', 'head_image': 'top'})
    doc['arms']['left']['base_to_world'] = None
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match='base_to_world'):
        Setup(path)


def test_factory_calibration_groups_stream_transforms_within_one_camera():
    rs = SimpleNamespace(stream=SimpleNamespace(depth='depth', infrared='ir'),
                         format=SimpleNamespace(z16='z16', y8='y8'))
    color = SimpleNamespace(stream_type=lambda: 'color', stream_index=lambda: 0)
    class Profile:
        def __init__(self, stream, index, fmt): self.stream, self.index, self.fmt = stream, index, fmt
        def is_video_stream_profile(self): return True
        def as_video_stream_profile(self): return self
        def width(self): return 640
        def height(self): return 480
        def fps(self): return 30
        def stream_type(self): return self.stream
        def stream_index(self): return self.index
        def format(self): return self.fmt
        def get_extrinsics_to(self, target):
            assert target is color
            return SimpleNamespace(rotation=np.eye(3).ravel().tolist(), translation=[.01, 0, 0])
    profiles = [Profile('depth', 0, 'z16'), Profile('ir', 1, 'y8'), Profile('ir', 2, 'y8')]
    device = SimpleNamespace(query_sensors=lambda: [SimpleNamespace(get_stream_profiles=lambda: profiles)])
    transforms = factory_extrinsics(device, color, rs)
    assert len(transforms) == 3 and all(t['to_stream'] == 'color' for t in transforms)
    assert all(t['reference'] == 'within this device; not robot/world' for t in transforms)


def test_three_camera_metadata_records_contain_their_own_factory_transforms():
    from colosseum_client.robots.yam import YAMRobot
    robot = YAMRobot.__new__(YAMRobot)
    robot._camera_serials = {'head_image': 'top', 'left_image': 'left', 'right_image': 'right'}
    transforms = [{'from_stream': source, 'to_stream': 'color'} for source in ('depth', 'ir1', 'ir2')]
    robot.cameras = {name: SimpleNamespace(calibration=lambda: {'fx': 500},
        factory_extrinsics=lambda: transforms) for name in robot._camera_serials}
    records = robot.roboeval_camera_metadata()
    assert len(records) == 3
    assert records['head_image']['arm_id'] is None
    assert records['left_image']['arm_id'] == 'left'
    assert all(len(record['factory_extrinsics']) == 3 for record in records.values())




def test_client_defers_both_exports_until_feedback_and_resumes_without_replaying(tmp_path, monkeypatch):
    from colosseum_client import evaluation as e
    from colosseum_client.robot_config import RobotClientConfig
    a = assignment()
    # Use normal A then B execution, but mapping is still by each run's side.
    a.update(track='open', state='pending', recording=True, test=False,
        task={'instruction': 'orange', 'cameras': ['head_image', 'left_image', 'right_image'],
              'max_steps': 2})
    a['runs'].reverse()
    for run in a['runs']: run['state'] = 'assigned'
    calibration = setup_file(tmp_path / 'cal')
    executions, submissions, uploads = [], [], []
    class API:
        def __init__(self, cfg): pass
        def close(self): pass
        def request(self, method, path, body=None):
            if path == '/reset': return {}
            if path.endswith('/result'):
                # Optional export must not delay or block original publication.
                for run in a['runs']:
                    output = tmp_path / 'ev_shared' / run['id'] / 'lerobot'
                    assert not output.exists()
                submissions.append(body)
                if len(submissions) == 1: raise RuntimeError('fake network result failure')
                return {}
            return a
        def upload_dataset(self, rid, output):
            assert len(submissions) == 2
            validate_uploaded_extension(output)
            stored = json.loads((output / 'meta/colosseum.json').read_text())['roboeval_layout']['documents']['outcome']
            assert stored['comparison']['raw_feedback'] == FEEDBACK
            uploads.append(rid)
    async def run(cfg, assigned, trial, path, api, **kwargs):
        if trial['side'] == 'B':
            first = path.parent / 'policy_first'
            assert not (first / 'lerobot').exists()
            assert (first / 'head_image/000000.png').exists()
        executions.append(trial['id'])
        make_recording(path, calibration, trial['id'])
        trial['state'] = 'finished'
    monkeypatch.setattr(e, 'EvalAPI', API)
    monkeypatch.setattr(e, 'run_trial', run)
    cfg = RobotClientConfig(url='ws://localhost', token='fake', robot_type='yam', control_hz=30,
        cameras={'head_image': 'top', 'left_image': 'left', 'right_image': 'right'},
        instruction='orange', scene='desk', test=False, evaluation_dir=str(tmp_path),
        roboeval_setup=str(calibration), recording=True, skip_upload=False)
    answers = iter(['', '60', '2', '', '100', 'B', FEEDBACK])
    monkeypatch.setattr('builtins.input', lambda prompt: next(answers))
    with pytest.raises(RuntimeError, match='fake network'):
        e.run_evaluation(cfg, track='open')
    assert not uploads
    answers = iter(['n'])
    monkeypatch.setattr('builtins.input', lambda prompt: next(answers))
    e.run_evaluation(cfg, resume='ev_shared')
    assert executions == ['policy_first', 'policy_second']
    assert uploads == executions and submissions[0] == submissions[1]
