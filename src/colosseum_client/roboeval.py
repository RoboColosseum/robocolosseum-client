"""Independent YAM recording. All constructors here are safe before robot startup."""
from __future__ import annotations

import hashlib
import json
from importlib import metadata
from pathlib import Path
import queue
import shutil
import subprocess
import threading
import time
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .roboeval_setup import Setup, Kinematics, array, setup_snapshot
from .tensors import tensor_to_numpy

RAW_SOURCE = 'model_output_before_action_safety_clipping'
VIDEO_SEGMENT_FRAMES = 3600  # Two minutes at 30 fps; bounded below receiver's 512 MiB/file limit.
VECTOR = pa.list_(pa.float64())
STATE_SCHEMA = pa.schema([
    ('arm_id', pa.string()), ('sample_index', pa.int64()),
    ('sample_monotonic_s', pa.float64()), ('feedback_sequence', pa.int64()),
    ('feedback_received_monotonic_s', pa.float64()),
    ('joint_position_rad', VECTOR), ('joint_velocity_rad_s', VECTOR),
    ('joint_torque_nm', VECTOR), ('eef_position_world_m', VECTOR),
    ('eef_orientation_world_wxyz', VECTOR),
    ('gripper_effort_nm', pa.float64()),
    ('gripper_reading_normalized_unclipped', pa.float64()), ('gripper_width_m', pa.float64())])
ACTION_SCHEMA = pa.schema([
    ('kind', pa.string()), ('arm_id', pa.string()), ('plan_id', pa.uint64()),
    ('chunk_index', pa.int64()), ('control_step', pa.int64()),
    ('command_monotonic_s', pa.float64()), ('recorded_monotonic_s', pa.float64()), ('action_space', pa.string()),
    ('values', VECTOR), ('gripper_command_width_m', pa.float64()),
    ('executable_prefix_length', pa.int64()), ('source_block_id', pa.string()),
    ('evidence_json', pa.string())])
FRAME_SCHEMA = pa.schema([
    ('arm_id', pa.string()), ('camera_id', pa.string()), ('frame_index', pa.int64()), ('source_frame_number', pa.int64()),
    ('source_timestamp_ms', pa.float64()), ('source_timestamp_domain', pa.string()),
    ('received_monotonic_s', pa.float64()), ('video_path', pa.string()),
    ('video_frame_index', pa.int64()),
    ('video_timestamp_s', pa.float64())])


def _json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


class _Video:
    def __init__(self, path, shape, fps):
        height, width, channels = shape
        if channels != 3 or height % 2 or width % 2:
            raise ValueError('RoboEval RGB video requires even width/height')
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path, self.shape, self.frames = path, shape, 0
        self.closed = False
        self.log = (path.with_suffix('.encoder.log')).open('wb')
        try:
            self.process = subprocess.Popen(['ffmpeg', '-hide_banner', '-loglevel', 'error',
                '-nostdin', '-n', '-f', 'rawvideo', '-pixel_format', 'rgb24',
                '-video_size', f'{width}x{height}', '-framerate', str(fps), '-i', 'pipe:0',
                '-an', '-c:v', 'libx264', '-preset', 'ultrafast', '-threads', '1',
                '-crf', '23', '-maxrate', '8M', '-bufsize', '16M',
                '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(path)],
                stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self.log, bufsize=0)
        except Exception:
            self.log.close()
            raise

    def add(self, pixels):
        if pixels.shape != self.shape or self.process.poll() is not None:
            raise RuntimeError('Wrist video encoder failed or image shape changed')
        remaining = memoryview(pixels.tobytes())
        while remaining:
            written = self.process.stdin.write(remaining)
            if not written:
                raise RuntimeError('Wrist video encoder pipe closed')
            remaining = remaining[written:]
        self.frames += 1

    def abort(self):
        if self.process.poll() is None:
            self.process.kill()

    def close(self):
        if self.closed:
            return
        try:
            self.process.stdin.close()
            if self.process.wait(timeout=30) != 0:
                raise RuntimeError('Wrist video encoding failed; inspect encoder log')
            stream = json.loads(subprocess.check_output(['ffprobe', '-v', 'error',
                '-select_streams', 'v:0', '-show_entries', 'stream=nb_frames',
                '-of', 'json', str(self.path)], timeout=30))['streams'][0]
            if int(stream['nb_frames']) != self.frames:
                raise RuntimeError('Encoded wrist frame count mismatch')
        finally:
            self.abort()
            self.process.wait(timeout=5)
            self.log.close()
            self.closed = True


class RoboEvalRecorder:
    """Queue-only producers; one worker computes FK and writes tables/video.

    A bounded queue fails visibly rather than dropping samples or blocking the
    control loop. Receipt time is host time for a bus sweep/source frame; no
    hardware exposure-clock synchronization is claimed. Shared inference/operator
    events and confirmed outcomes are finalized separately after both trials.
    """
    def __init__(self, path, setup_path, *, hz, identity, queue_size=128,
                 kinematics_factory=Kinematics, video_factory=_Video, continuous=False):
        if shutil.which('ffmpeg') is None or shutil.which('ffprobe') is None:
            raise RuntimeError('RoboEval requires existing ffmpeg and ffprobe')
        self.setup = Setup(setup_path)
        self.kinematics = kinematics_factory(self.setup)
        self.root = Path(path) / 'roboeval'
        self.root.mkdir(parents=True, exist_ok=False)
        self.meta = self.root / 'meta/roboeval'
        identity = dict(identity, software=dict(python=sys.version,
            packages={name: _version(name) for name in ('colosseum-client', 'i2rt', 'mujoco', 'pyrealsense2')},
            client_source_sha256={name: sha256(Path(__file__).parent / name)
                for name in ('roboeval.py', 'roboeval_setup.py', 'robots/yam.py', 'evaluation.py', 'colosseum_pb2.py')}))
        setup_snapshot(self.setup, self.meta, cameras={}, identity=identity, limits={})
        self.hz, self.video_factory = hz, video_factory
        self.continuous = continuous
        self.missed_sampling_ticks = 0
        self.duplicate_feedback_reads = {}
        self.quality_issues = {}
        self.feedback_read_stats = {}
        self.queue = queue.Queue(maxsize=queue_size)
        self.stop = threading.Event()
        self.error = None
        self.closed = False
        self.worker = self.sampler = None
        self.robot = None
        self.context = (None, None, None)
        self._context_lock = threading.Lock()
        self.counts = {name: 0 for name in ('states', 'actions', 'frames')}
        self.video, self.frame_stats = {}, {}
        self.state_stats = {}
        self.action_kinds = {side: set() for side in self.setup.arms}
        self.required_camera_keys = list(self.setup.arms)
        self._status(False)

    def _status(self, complete, **extra):
        _json(self.meta / 'status.json', dict(schema_version=1, complete=complete,
            error=None if self.error is None else str(self.error),
            selected_fields=self.setup.document.get('selected_fields',
                [*(f'A{i}' for i in range(1, 10)), 'B1', 'B2', 'B3', 'B5', 'B6',
                 'C1', 'C2', 'C3', 'C4', 'C5', 'C8']),
            counts=self.counts, target_state_hz=self.hz,
            frame_stats=self.frame_stats, state_stats=self.state_stats,
            continuous=self.continuous, missed_sampling_ticks=self.missed_sampling_ticks,
            duplicate_feedback_reads=self.duplicate_feedback_reads,
            feedback_read_stats=self.feedback_read_stats,
            completion_scope='raw_capture_integrity' if self.continuous else 'capture_quality',
            feedback_storage=getattr(self, 'feedback_storage', 'latest_snapshot'),
            quality_issues=self.quality_issues, **extra))

    def _feedback_issue(self, side, stamp, age, sequence, previous, *, details=None):
        # A quality violation invalidates certification, not the other streams.
        # Keep bounded evidence, skip this invalid state and resume on fresh data.
        issues = self.quality_issues.setdefault(side, dict(count=0, first_monotonic_s=stamp,
            last_monotonic_s=stamp, max_age_s=age, sequence=sequence, previous=previous))
        issues.update(count=issues['count'] + 1, last_monotonic_s=stamp,
                      max_age_s=max(issues['max_age_s'], age), sequence=sequence, previous=previous)
        if details is not None:
            events = issues.setdefault('events', [])
            if len(events) < 32:
                events.append(dict(sample_monotonic_s=stamp, age_s=age,
                                   sequence=sequence, previous=previous, **details))
            issues['events_omitted'] = issues['count'] - len(events)
        if issues['count'] == 1:
            print(f'RoboEval {side} feedback quality gap: age={age:.6f}s; '
                  'cameras continue; training-interval source age will be checked after capture.', flush=True)

    def fail(self, error):
        if self.error is None:
            self.error = error
        self.stop.set()

    def check(self):
        if self.error is not None:
            raise RuntimeError('RoboEval recording failed; data is incomplete') from self.error

    def _put(self, kind, payload):
        if self.closed or (self.stop.is_set() and (self.error is not None or kind != 'state_batch')):
            return
        try:
            self.queue.put_nowait((kind, payload))
        except queue.Full:
            self.fail(RuntimeError('RoboEval writer queue full; no samples silently discarded'))

    def start(self, robot, *, observer=None):
        if self.worker is not None:
            raise RuntimeError('RoboEval recording already started')
        if set(robot.roboeval_arm_ids) != set(self.setup.arms):
            raise ValueError('Robot arm IDs differ from verified setup')
        self.robot = robot
        # Metadata access is passive. Device/model constructors must already have finished.
        snapshot = json.loads((self.meta / 'setup.json').read_text(encoding='utf-8'))
        snapshot.update(cameras=robot.roboeval_camera_metadata(),
                        runtime_limits=robot.roboeval_limits())
        if 'head_image' in snapshot['cameras']:
            self.required_camera_keys.append('head_image')
        policy_file = self.root.parent / 'policy.json'
        snapshot['identity']['policy'] = json.loads(policy_file.read_text()) if policy_file.exists() else None
        _json(self.meta / 'setup.json', snapshot)
        self.started = time.monotonic()
        self.feedback_storage = 'latest_snapshot'
        if self.continuous and callable(getattr(robot, 'start_roboeval_feedback_history', None)):
            robot.start_roboeval_feedback_history()
            self.feedback_storage = 'all_received_sweeps'
        self.worker = threading.Thread(target=self._write_loop, name='roboeval-writer', daemon=True)
        self.worker.start()
        robot.start_roboeval(self if observer is None else observer)
        self.sampler = threading.Thread(target=self._sample_loop, name='roboeval-feedback', daemon=True)
        self.sampler.start()

    def _sample_loop(self):
        if getattr(self, 'feedback_storage', None) == 'all_received_sweeps':
            return self._history_loop()
        previous, received, index, deadline = {}, {}, 0, time.monotonic()
        self.duplicate_feedback_reads = getattr(self, 'duplicate_feedback_reads', {})
        self.quality_issues = getattr(self, 'quality_issues', {})
        self.feedback_read_stats = getattr(self, 'feedback_read_stats', {})
        try:
            while not self.stop.is_set():
                read_started = time.monotonic()
                feedback = self.robot.get_roboeval_feedback()
                stamp = time.monotonic()
                duration = stamp - read_started
                stats = self.feedback_read_stats
                stats.update(count=stats.get('count', 0) + 1,
                    max_duration_s=max(stats.get('max_duration_s', 0.), duration),
                    over_20ms=stats.get('over_20ms', 0) + int(duration > .02),
                    max_start_lateness_s=max(stats.get('max_start_lateness_s', 0.), read_started-deadline))
                for side in self.setup.arms:
                    state = feedback[side]
                    duplicate = state['sequence'] == previous.get(side, -1)
                    age = stamp - (received[side] if duplicate else state['received_monotonic_s'])
                    if (not state['running'] or age > (.02 if self.continuous else .1) or age < -.01
                            or state['sequence'] < previous.get(side, -1)
                            or duplicate and not self.continuous):
                        if self.continuous:
                            self._feedback_issue(side, stamp, age, state['sequence'], previous.get(side),
                                details=dict(feedback_received_monotonic_s=state['received_monotonic_s'],
                                    read_started_monotonic_s=read_started, read_completed_monotonic_s=stamp,
                                    scheduled_monotonic_s=deadline, running=bool(state['running'])))
                            continue
                        raise RuntimeError(f'{side} motor feedback is stopped, stale or repeated: '
                            f'running={state["running"]}, age_s={age:.6f}, '
                            f'sequence={state["sequence"]}, previous={previous.get(side)}')
                    if duplicate:
                        self.duplicate_feedback_reads[side] = self.duplicate_feedback_reads.get(side, 0) + 1
                        continue
                    previous[side] = state['sequence']
                    received[side] = state['received_monotonic_s']
                    self._put('states', dict(arm_id=side, sample_index=index,
                        sample_monotonic_s=stamp, feedback_sequence=state['sequence'],
                        feedback_received_monotonic_s=state['received_monotonic_s'],
                        joint_position_rad=array(state['position'], (6,), 'q').tolist(),
                        joint_velocity_rad_s=array(state['velocity'], (6,), 'dq').tolist(),
                        joint_torque_nm=array(state['torque_nm'], (6,), 'torque').tolist(),
                        gripper_effort_nm=(None if state.get('gripper_effort_nm') is None else
                            float(array([state['gripper_effort_nm']], (1,), 'gripper effort')[0])),
                        gripper_reading_normalized_unclipped=float(array(
                            [state['gripper_reading']], (1,), 'gripper reading')[0])))
                index += 1
                deadline += 1 / self.hz
                # No catch-up burst of repeated caches after scheduling delays.
                now = time.monotonic()
                if deadline < now:
                    missed = int((now - deadline) * self.hz) + 1
                    self.missed_sampling_ticks += missed
                    deadline += missed / self.hz
                self.stop.wait(max(0, deadline - time.monotonic()))
        except Exception as exc:
            self.fail(exc)

    def _drain_history(self, *, read_started, deadline):
        feedback = self.robot.get_roboeval_feedback_history()
        stamp = time.monotonic()
        previous, received = self._history_previous, self._history_received
        rows = []
        for side in self.setup.arms:
            for state in feedback[side]:
                seq, receipt = state['sequence'], state['received_monotonic_s']
                if (not state['running'] or not np.isfinite(receipt) or receipt>stamp
                        or side in previous and (seq!=previous[side]+1 or receipt<=received[side])):
                    raise RuntimeError(f'{side} stopped, missing, reordered or invalid feedback history')
                rows.append(dict(arm_id=side, sample_index=self._history_index, sample_monotonic_s=stamp,
                    feedback_sequence=seq, feedback_received_monotonic_s=receipt,
                    joint_position_rad=array(state['position'], (6,), 'q').tolist(),
                    joint_velocity_rad_s=array(state['velocity'], (6,), 'dq').tolist(),
                    joint_torque_nm=array(state['torque_nm'], (6,), 'torque').tolist(),
                        gripper_effort_nm=(None if state.get('gripper_effort_nm') is None else
                            float(array([state['gripper_effort_nm']], (1,), 'gripper effort')[0])),
                    gripper_reading_normalized_unclipped=float(array([state['gripper_reading']], (1,), 'gripper')[0])))
                self._history_index += 1
                previous[side], received[side] = seq, receipt
            if side in received and stamp-received[side]>.02:
                self._feedback_issue(side, stamp, stamp-received[side], previous[side], previous[side],
                    details=dict(feedback_received_monotonic_s=received[side],
                        read_started_monotonic_s=read_started, read_completed_monotonic_s=stamp,
                        scheduled_monotonic_s=deadline, running=True))
        if rows:
            self._put('state_batch', rows)

    def _history_loop(self):
        self._history_previous, self._history_received, self._history_index = {}, {}, 0
        deadline = time.monotonic()
        try:
            while not self.stop.is_set():
                read_started = time.monotonic()
                self._drain_history(read_started=read_started, deadline=deadline)
                stamp = time.monotonic()
                duration = stamp-read_started
                stats = self.feedback_read_stats
                stats.update(count=stats.get('count', 0)+1,
                    max_duration_s=max(stats.get('max_duration_s', 0.), duration),
                    over_20ms=stats.get('over_20ms', 0)+int(duration>.02),
                    max_start_lateness_s=max(stats.get('max_start_lateness_s', 0.), read_started-deadline))
                deadline += 1/self.hz
                now = time.monotonic()
                if deadline<now:
                    missed = int((now-deadline)*self.hz)+1
                    self.missed_sampling_ticks += missed
                    deadline += missed/self.hz
                self.stop.wait(max(0, deadline-time.monotonic()))
        except Exception as exc:
            self.fail(exc)

    def plan(self, plan, *, action_space, executable_length):
        self.check()
        if plan.action_evidence_json:
            return self._evidence_plan(plan, action_space=action_space, executable_length=executable_length)
        if (not plan.HasField('raw_actions') or plan.raw_action_source != RAW_SOURCE
                or plan.raw_action_space != action_space):
            raise ValueError('RoboEval requires the original model action chunk before execution')
        raw = tensor_to_numpy(plan.raw_actions)
        if (raw.ndim != 2 or raw.shape[1] != 7 * len(self.setup.arms)
                or not executable_length <= len(raw) <= 4096 or not np.isfinite(raw).all()):
            raise ValueError('Invalid original policy action chunk')
        rows = []
        for index, values in enumerate(raw):
            for arm_index, side in enumerate(self.setup.arms):
                rows.append(dict(kind='raw_policy', arm_id=side, plan_id=plan.plan_id,
                    chunk_index=index, control_step=plan.start_step + index,
                    command_monotonic_s=None, action_space=action_space,
                    recorded_monotonic_s=time.monotonic(),
                    values=values[7*arm_index:7*(arm_index+1)].tolist(),
                    gripper_command_width_m=None, executable_prefix_length=executable_length))
        self._put('action_batch', rows)
        self.check()

    def _evidence_plan(self, plan, *, action_space, executable_length):
        evidence = json.loads(plan.action_evidence_json)
        if evidence.get('version') != 1 or evidence.get('source') not in {'inspect_agent', 'dreamlabs'}:
            raise ValueError('Unknown raw action evidence schema')
        mapping = evidence['execution']
        kind = mapping.get('kind')
        if kind not in {'hold', 'unused_terminal', 'prefix', 'linear_interpolation_after_gripper_clip'}:
            raise ValueError('Missing raw-to-execution mapping')
        known = dict(getattr(self, '_source_blocks', {}))
        rows = []
        for block in evidence['blocks']:
            raw = np.asarray(block['values'], dtype=np.float64)
            if raw.size == 0 and kind == 'hold':
                known[block['id']] = 0
                continue
            if (raw.ndim != 2 or raw.shape[1] != 7 * len(self.setup.arms)
                    or not 1 <= len(raw) <= 4096 or not np.isfinite(raw).all()):
                raise ValueError('Invalid original action evidence block')
            if block['id'] in known:
                raise ValueError('Duplicate original action evidence block')
            known[block['id']] = len(raw)
            block_metadata = json.dumps({k:v for k,v in block.items() if k != 'values'})
            for index, values in enumerate(raw):
                for arm_index, side in enumerate(self.setup.arms):
                    rows.append(dict(kind='raw_policy', arm_id=side,
                        plan_id=plan.plan_id, source_block_id=block['id'], chunk_index=index,
                        control_step=None, action_space=action_space,
                        recorded_monotonic_s=time.monotonic(),
                        values=values[7*arm_index:7*(arm_index+1)].tolist(),
                        evidence_json=block_metadata))
        if kind not in {'hold', 'unused_terminal'} and mapping.get('source_block') not in known:
            raise ValueError('Execution refers to an unrecorded source block')
        length_key = 'rows' if kind == 'prefix' else 'lower_rows'
        if kind not in {'hold', 'unused_terminal'} and len(mapping[length_key]) != executable_length:
            raise ValueError('Execution mapping length mismatch')
        if kind == 'prefix':
            indices = mapping['rows']
            if indices != list(range(executable_length)) or executable_length > known[mapping['source_block']]:
                raise ValueError('Invalid executable prefix mapping')
        if kind == 'linear_interpolation_after_gripper_clip':
            lower, upper, weights = mapping['lower_rows'], mapping['upper_rows'], mapping['weights']
            size = known[mapping['source_block']]
            if (len(upper) != executable_length or len(weights) != executable_length
                    or any(type(a) is not int or type(b) is not int or not 0 <= a <= b < size
                           or b-a > 1 or not isinstance(w, (int,float)) or not np.isfinite(w) or not 0 <= w <= 1
                           for a,b,w in zip(lower,upper,weights))):
                raise ValueError('Invalid interpolation source mapping')
        # Separate association row: a hold or interpolated target is not a raw model action.
        rows.append(dict(kind='execution_mapping', plan_id=plan.plan_id,
            control_step=plan.start_step, recorded_monotonic_s=time.monotonic(),
            evidence_json=json.dumps(evidence | {'blocks': []})))
        # One model reply can contain hundreds of raw rows despite executing only
        # a short prefix. Validate and enqueue it atomically, rather than consuming
        # the state/video queue one slot per raw row. Disk writes remain bounded
        # Arrow batches; no control command is delayed waiting for queue capacity.
        self._put('action_batch', rows)
        self.check()
        self._source_blocks = known

    def select(self, plan_id, chunk_index, control_step):
        self.check()
        with self._context_lock:
            self.context = (plan_id, chunk_index, control_step)

    def sent(self, side, values):
        # Called just after a successful SDK command. Never throw into safety handling.
        try:
            values = array(values, (7,), 'sent command')
            with self._context_lock:
                plan_id, index, step = self.context
            self._put('actions', dict(kind='sent_command', arm_id=side, plan_id=plan_id,
                chunk_index=index, control_step=step, command_monotonic_s=time.monotonic(),
                recorded_monotonic_s=time.monotonic(),
                action_space='joint_position', values=values.tolist(),
                gripper_command_width_m=self.setup.width(side, values[6]),
                executable_prefix_length=None))
        except Exception as exc:
            self.fail(exc)

    def frame(self, side, frame):
        self._put('frames', (side, frame))

    def _write_loop(self):
        schemas = dict(states=STATE_SCHEMA, actions=ACTION_SCHEMA, frames=FRAME_SCHEMA)
        writers, buffers = {}, {name: [] for name in schemas}
        try:
            for name, schema in schemas.items():
                writers[name] = pq.ParquetWriter(self.meta / f'{name}.parquet', schema)
            while True:
                item = self.queue.get()
                try:
                    if item is None:
                        break
                    name, payload = item
                    rows = payload if name in ('action_batch', 'state_batch') else [payload]
                    name = {'action_batch': 'actions', 'state_batch': 'states'}.get(name, name)
                    for row in rows:
                        self._write_row(name, row, writers, buffers, schemas)
                finally:
                    self.queue.task_done()
        except Exception as exc:
            self.fail(exc)
        finally:
            for name, writer in writers.items():
                try:
                    if buffers[name]:
                        writer.write_table(pa.Table.from_pylist(buffers[name], schema=schemas[name]))
                    writer.close()
                except Exception as exc:
                    self.fail(exc)

    def _write_row(self, name, row, writers, buffers, schemas):
        if name == 'states':
            side = row['arm_id']
            position, orientation = self.kinematics.pose(side, row['joint_position_rad'])
            position = array(position, (3,), 'world position').tolist()
            orientation = array(orientation, (4,), 'world quaternion')
            if not np.isclose(np.linalg.norm(orientation), 1, atol=1e-6):
                raise ValueError('World quaternion must be normalized')
            row.update(eef_position_world_m=position,
                eef_orientation_world_wxyz=orientation.tolist(),
                gripper_width_m=self.setup.width(side, row['gripper_reading_normalized_unclipped']))
            stamp = row['feedback_received_monotonic_s'] if self.continuous else row['sample_monotonic_s']
            stats = self.state_stats.setdefault(side,
                dict(samples=0, first=stamp, last=stamp, max_gap_s=0.))
            stats.update(samples=stats['samples']+1, last=stamp,
                max_gap_s=max(stats['max_gap_s'], stamp-stats['last']))
        elif name == 'frames':
            row = self._write_video(*row)
        elif name == 'actions' and row.get('arm_id') is not None:
            self.action_kinds[row['arm_id']].add(row['kind'])
        buffers[name].append(row)
        self.counts[name] += 1
        if len(buffers[name]) >= 128:
            writers[name].write_table(pa.Table.from_pylist(buffers[name], schema=schemas[name]))
            buffers[name].clear()

    def _write_video(self, side, frame):
        array([frame['received_monotonic_s'], frame['source_timestamp_ms']], (2,), 'frame timestamps')
        pixels = np.asarray(frame['pixels'])
        if self.continuous and pixels.shape != (480, 640, 3):
            raise ValueError('Continuous source frames must be native 640x480 RGB')
        if pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[2] != 3:
            raise ValueError('Wrist frame must be RGB uint8')
        if side not in self.frame_stats:
            self.frame_stats[side] = dict(frames=0, first=frame['received_monotonic_s'],
                last=frame['received_monotonic_s'], max_gap_s=0., last_source_frame=-1,
                last_source_timestamp_ms=None, source_timestamp_domain=frame['source_timestamp_domain'])
        stats = self.frame_stats[side]
        segment, video_index = divmod(stats['frames'], VIDEO_SEGMENT_FRAMES)
        relative = f'videos/roboeval/{side}/chunk-000/file-{segment:03d}.mp4'
        key = (side, segment)
        if key not in self.video:
            if segment:
                self.video[(side, segment-1)].close()
            self.video[key] = self.video_factory(self.root / relative, pixels.shape, 30)
        video = self.video[key]
        stamp, source_id = frame['received_monotonic_s'], frame['source_frame_number']
        if (source_id <= stats['last_source_frame']
                or stats['frames'] and (stamp <= stats['last']
                    or frame['source_timestamp_ms'] <= stats['last_source_timestamp_ms'])
                or frame['source_timestamp_domain'] != stats['source_timestamp_domain']):
            raise RuntimeError('Wrist source frame/timestamp repeated, reordered or changed clock domain')
        gap = stamp - stats['last']
        video.add(pixels)
        row = dict(arm_id=side if side in self.setup.arms else None,
            camera_id=f'{side}_image' if side in self.setup.arms else side,
            frame_index=stats['frames'], source_frame_number=source_id,
            source_timestamp_ms=frame['source_timestamp_ms'],
            source_timestamp_domain=frame['source_timestamp_domain'], received_monotonic_s=stamp,
            video_path=relative, video_frame_index=video_index, video_timestamp_s=video_index / 30)
        stats.update(frames=stats['frames'] + 1, last=stamp, last_source_frame=source_id,
                     last_source_timestamp_ms=frame['source_timestamp_ms'],
                     max_gap_s=max(stats['max_gap_s'], gap))
        return row

    def close(self, *, complete=False):
        if self.closed:
            self.check()
            return
        self.stop.set()
        stopped_at = time.monotonic()
        if self.robot is not None:
            try:
                self.robot.stop_roboeval()
            except Exception as exc:
                self.fail(exc)
        if self.sampler is not None:
            self.sampler.join(timeout=3)
            if self.sampler.is_alive():
                self.fail(RuntimeError('Feedback recorder did not stop'))
        if getattr(self, 'feedback_storage', None) == 'all_received_sweeps':
            try:
                if self.error is None and hasattr(self, '_history_previous'):
                    self._drain_history(read_started=stopped_at, deadline=stopped_at)
            except Exception as exc:
                self.fail(exc)
            try:
                self.robot.stop_roboeval_feedback_history()
            except Exception as exc:
                self.fail(exc)
        if self.worker is not None:
            try:
                self.queue.put(None, timeout=5)
            except queue.Full:
                self.fail(RuntimeError('RoboEval writer did not drain'))
                for video in list(self.video.values()):
                    video.abort()
            self.worker.join(timeout=10)
            if self.worker.is_alive():
                for video in list(self.video.values()):
                    video.abort()
                self.worker.join(timeout=3)
                if self.worker.is_alive():
                    self.fail(RuntimeError('RoboEval writer did not stop'))
        for video in list(self.video.values()):
            try:
                video.close()
            except Exception as exc:
                self.fail(exc)
        self.closed = True
        ended = stopped_at
        if complete and self.error is None:
            try:
                self._validate_capture(ended)
            except Exception as exc:
                self.fail(exc)
        self._status(complete and self.error is None,
                     duration_s=ended - self.started if self.worker is not None else 0)
        # Hash only closed immutable files. Encoder diagnostics remain local.
        if complete and self.error is None:
            files = {p.relative_to(self.root).as_posix(): sha256(p)
                     for p in sorted(self.root.rglob('*'))
                     if p.is_file() and not p.name.endswith('.encoder.log')}
            _json(self.root / 'manifest.json', dict(schema_version=1, files=files))
        self.check()

    def _validate_capture(self, ended):
        if not self.continuous and getattr(self, 'quality_issues', {}):
            raise RuntimeError('Feedback quality gaps recorded; raw streams retained, strict continuous export rejected')
        if not self.counts['states'] or not self.counts['actions']:
            raise RuntimeError('Missing arm state/action data')
        # Validate bounded online summaries. Materializing a long Parquet file
        # while the motors are live could starve the Python watchdog thread.
        for side in self.setup.arms:
            if self.action_kinds[side] != {'raw_policy', 'sent_command'}:
                raise RuntimeError(f'{side} missing original policy output or sent commands')
            state = self.state_stats.get(side)
            if self.continuous:
                if state is None or state['samples'] < 2 or state['last'] <= state['first']:
                    raise RuntimeError(f'{side} missing feedback source coverage')
                state['actual_hz'] = (state['samples']-1)/(state['last']-state['first'])
                continue  # Physical source age is certified on the action grid after robot cleanup.
            if (state is None or state['samples'] < 2 or state['last'] <= state['first']
                    or (state['samples']-1)/(state['last']-state['first']) < (99 if self.continuous else self.hz * .9)
                    or state['max_gap_s'] > (.02 if self.continuous else max(.1, 2/self.hz))
                    or state['first']-self.started > .1 or ended-state['last'] > .1):
                raise RuntimeError(f'{side} feedback sampling rate below 90% of control rate')
            state['actual_hz'] = (state['samples']-1)/(state['last']-state['first'])
        for side in self.required_camera_keys:
            stats = self.frame_stats.get(side)
            if self.continuous:
                if stats is None or stats['frames'] < 2 or stats['last'] <= stats['first']:
                    raise RuntimeError(f'{side} missing camera source coverage')
                stats['actual_fps'] = (stats['frames']-1)/(stats['last']-stats['first'])
                continue
            if (stats is None or stats['frames'] < 2 or stats['last'] <= stats['first']
                    or (stats['frames']-1)/(stats['last']-stats['first']) < 15
                    or stats['max_gap_s'] > (.1 if self.continuous else .2) or stats['first'] - self.started > .2
                    or ended - stats['last'] > .2):
                raise RuntimeError(f'{side} wrist capture failed the 15 fps/coverage check')
            stats['actual_fps'] = (stats['frames']-1)/(stats['last']-stats['first'])
