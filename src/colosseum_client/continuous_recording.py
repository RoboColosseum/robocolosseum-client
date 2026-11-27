"""Causal 100 Hz export from immutable observations, targets and source videos.

No robot calls, interpolation, image upscaling or synthetic feedback. A repeated
source index identifies reuse on the export grid, not another sensor sample.
"""
from bisect import bisect_right
from pathlib import Path
import json
import math
import subprocess
import tempfile

import numpy as np
import pyarrow.parquet as pq

FPS = 100
STATE_MAX_AGE = .020
FRAME_MAX_AGE = .100


def _ordered(rows, key):
    stamps = [r[key] for r in rows]
    if not stamps or not all(isinstance(t, (int, float)) and math.isfinite(t) for t in stamps):
        raise ValueError(f'Missing/nonfinite {key}')
    if any(b <= a for a, b in zip(stamps, stamps[1:])):
        raise ValueError(f'Repeated or reordered {key}')
    return stamps


def _latest(rows, times, stamp, limit):
    index = bisect_right(times, stamp) - 1
    if index < 0 or stamp - times[index] > limit + 1e-9:
        raise ValueError(f'No causal source within {limit}s at {stamp}')
    return rows[index]


def align(states, frames, targets, *, end, cameras, version=3, grid_hz=FPS):
    """Pure alignment used by the exporter and corruption verifier."""
    if grid_hz not in (30, 100):
        raise ValueError('Unsupported alignment grid')
    action_times = _ordered(targets, 'action_executed_monotonic_s')
    start = action_times[0]
    if not isinstance(end, (int, float)) or not math.isfinite(end) or end <= start or end < action_times[-1]:
        raise ValueError('Missing/invalid successful trial boundary')
    by_arm = {side: [r for r in states if r['arm_id'] == side] for side in ('left', 'right')}
    if version not in (1, 2, 3):
        raise ValueError('Unsupported continuous alignment version')
    state_times = {side: _ordered(rows, 'sample_monotonic_s' if version == 1
                                else 'feedback_received_monotonic_s') for side, rows in by_arm.items()}
    for side, rows in by_arm.items():
        _ordered(rows, 'feedback_sequence')
        _ordered(rows, 'feedback_received_monotonic_s')
        copies = [r['sample_monotonic_s'] for r in rows]
        if not all(math.isfinite(t) for t in copies) or any(b < a for a, b in zip(copies, copies[1:])):
            raise ValueError('Invalid/reordered feedback copy timestamps')
        if any(r['feedback_received_monotonic_s'] > r['sample_monotonic_s'] for r in rows):
            raise ValueError('Feedback cannot be received after sampling')
    by_camera = {c: [r for r in frames if r['camera_id'] == c] for c in cameras}
    frame_times = {c: _ordered(rows, 'received_monotonic_s') for c, rows in by_camera.items()}
    for rows in by_camera.values():
        _ordered(rows, 'source_frame_number')
    count = math.ceil((end - start) * grid_hz - 1e-8)
    rows, selected_frames, columns = [], {c: [] for c in cameras}, {}

    def column(key, value):
        columns.setdefault(key, []).append(value)

    for index in range(count):
        stamp = start + index / grid_hz
        target = _latest(targets, action_times, stamp, math.inf)
        action = np.asarray(target['action'], dtype=np.float32)
        if action.shape != (14,) or not np.isfinite(action).all():
            raise ValueError('Continuous policy target must be finite 14-D')
        sampled = {s: _latest(by_arm[s], state_times[s], stamp,
                             STATE_MAX_AGE if version < 3 else math.inf) for s in by_arm}
        for side, source in sampled.items():
            if version < 3 and stamp - source['feedback_received_monotonic_s'] > STATE_MAX_AGE + 1e-9:
                raise ValueError(f'{side} feedback receipt too old at grid {index}: '
                                 f'{(stamp-source["feedback_received_monotonic_s"])*1000:.6f} ms > 20 ms')
            for field in ('sample_index', 'feedback_sequence', 'sample_monotonic_s', 'feedback_received_monotonic_s'):
                column(f'observation.source_{side}_{field}', source[field])
        joints = sampled['left']['joint_position_rad'] + sampled['right']['joint_position_rad']
        # Match Robot._read_positions: the training state clips gripper feedback;
        # the extension retains the original unclipped measurement and metres.
        grippers = [float(np.clip(sampled[s]['gripper_reading_normalized_unclipped'], 0, 1))
                    for s in ('left', 'right')]
        rows.append(dict(frame=index, seconds=index/grid_hz, captured_monotonic_s=stamp,
                         joints=joints, gripper=grippers, cartesian_position=[], action=action.tolist()))
        effort = [sampled[side].get('gripper_effort_nm') for side in ('left', 'right')]
        if any(value is not None for value in effort):
            if any(value is None for value in effort) or not np.isfinite(effort).all():
                raise ValueError('Missing or nonfinite gripper effort in aligned feedback')
            rows[-1]['gripper_effort_nm'] = effort
        column('action_source_frame_index', target['frame'])
        column('action_source_executed_monotonic_s', target['action_executed_monotonic_s'])
        for camera in cameras:
            source = _latest(by_camera[camera], frame_times[camera], stamp, FRAME_MAX_AGE)
            selected_frames[camera].append(source)
            for field in ('frame_index', 'source_frame_number', 'received_monotonic_s'):
                column(f'observation.source_{camera}_{field}', source[field])
    columns = {key: np.asarray(values, dtype=np.float64 if key.endswith('_s') else np.int64)
               for key, values in columns.items()}
    return rows, columns, selected_frames


def _source_resolution(root, frames):
    from .roboeval_compact import probe
    root = Path(root).resolve()
    sizes = set()
    for name in sorted({row['video_path'] for row in frames}):
        source = (root / name).resolve()
        if root not in source.parents:
            raise ValueError('Unsafe source video path')
        stream = next(s for s in probe(source)['streams'] if s['codec_type'] == 'video')
        sizes.add((stream['width'], stream['height']))
    if len(sizes) != 1 or not sizes <= {(640, 480), (1280, 720)}:
        raise ValueError('Continuous source dimensions differ or are unsupported')
    return next(iter(sizes))


def command_coverage(targets, columns, actions):
    """Require paired command evidence for every target used by the grid."""
    by_step = {}
    for action in actions:
        if action['kind'] == 'sent_command':
            by_step.setdefault(action['control_step'], {}).setdefault(action['arm_id'], []).append(action['command_monotonic_s'])
    verified = set()
    for target in targets:
        sent = by_step.get(target['frame'], {})
        if all(any(t is not None and math.isfinite(t) and t <= target['action_executed_monotonic_s'] for t in sent.get(side, [])) for side in ('left', 'right')):
            verified.add(target['frame'])
    used = {int(i) for i in columns['action_source_frame_index']}
    if not used <= verified:
        raise ValueError('Training target lacks paired command evidence: ' + str(sorted(used - verified)))
    return dict(all_referenced_targets_verified=True, unreferenced_targets_without_paired_commands=[t['frame'] for t in targets if t['frame'] not in verified])


def build_continuous(path, targets, recording, cameras, *, training_fps=100):
    if training_fps not in (30, 100):
        raise ValueError("Unsupported training FPS")
    if recording.get('roboeval_state_hz') != FPS or not recording.get('execution_enabled'):
        raise ValueError('Continuous export requires real executed targets and 100 Hz recording')
    if set(cameras) != {'left_image', 'right_image', 'head_image'}:
        raise ValueError('Continuous export requires all three cameras')
    from .roboeval_export import source_extension
    extension = source_extension(path)
    if extension is None:
        raise ValueError('Continuous source is missing')
    root = Path(path) / 'roboeval/meta/roboeval'
    status = json.loads((root / 'status.json').read_text())
    if not status.get('complete') or not status.get('continuous') or status.get('target_state_hz') != FPS:
        raise ValueError('Capture is not a completed continuous recording')
    states = pq.read_table(root / 'states.parquet').to_pylist()
    frames = pq.read_table(root / 'frames.parquet').to_pylist()
    from .continuity_quality import assess_timing, require_timing
    require_timing(assess_timing(states, frames, targets,
        end=recording.get('continuous_end_monotonic_s'), cameras=cameras))
    # Evidence is embedded so an uploader can recheck alignment, not just hashes.
    metadata = dict(version=4 if training_fps == 30 else 3, state_hz=FPS, source_video_fps=30, training_video_fps=training_fps,
                    source_resolution=list(_source_resolution(Path(path) / "roboeval", frames)), state_max_age_s=None,
                    state_age_policy='record_only', state_age_reference_s=STATE_MAX_AGE,
                    frame_max_age_s=FRAME_MAX_AGE, start_monotonic_s=targets[0].get('action_executed_monotonic_s'),
                    end_monotonic_s=recording.get('continuous_end_monotonic_s'),
                    policy_targets=[{k: t.get(k) for k in ('frame', 'action', 'action_executed_monotonic_s')}
                                    for t in targets],
                    alignment='physical feedback receipt <= grid; recorder copy time retained separately; no interpolation',
                    action_semantics='selected policy target after successful execution call; not final SDK command',
                    video_semantics=f'{training_fps} FPS training view reuses 30 FPS source frames; explicit source indices',
                    initial_wait='preserved in source streams; excluded from training rows')
    rows, columns, selected = align(states, frames, metadata['policy_targets'],
                                    end=metadata['end_monotonic_s'], cameras=cameras, grid_hz=training_fps)
    if training_fps == 30:
        metadata.update(training_hz=30, raw_alignment_hz=100)
        metadata['command_coverage'] = command_coverage(metadata['policy_targets'], columns, pq.read_table(root / 'actions.parquet').to_pylist())
    return dict(rows=rows, columns=columns, frames=selected, metadata=metadata)


def _read_frame(stream, size):
    data = bytearray()
    while len(data) < size:
        block = stream.read(size - len(data))
        if not block:
            raise ValueError('Source video ended before referenced frame')
        data.extend(block)
    return bytes(data)


def _stop_process(proc):
    """Close pipes before SIGTERM: ffmpeg may otherwise block flushing stdout."""
    for stream in (proc.stdout, proc.stdin):
        if stream is not None and not stream.closed:
            try:
                stream.close()
            except (BrokenPipeError, OSError):
                pass
    if proc.poll() is None:
        proc.terminate()
    try:
        proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=3)


def encode_aligned_video(run, rows, output, fps):
    """Stream reused decoded frames to ffmpeg; no 100 FPS PNG staging."""
    root = (Path(run) / 'roboeval').resolve()
    total, squares = np.zeros(3), np.zeros(3)
    minimum, maximum = np.ones(3), np.zeros(3)
    width, height = _source_resolution(root, rows)
    pixels_per_frame = width * height
    decoder = encoder = None
    current_path, current_index, pixels = None, -1, None
    frame_stats = None
    with tempfile.TemporaryFile() as errors:
        try:
            encoder = subprocess.Popen(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
                '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{width}x{height}', '-r', str(fps), '-i', 'pipe:0',
                '-an', '-c:v', 'libx264', '-preset', 'fast', '-pix_fmt', 'yuv420p',
                '-movflags', '+faststart', str(output)], stdin=subprocess.PIPE, stderr=errors)
            for row in rows:
                source = (root / row['video_path']).resolve()
                if root not in source.parents:
                    raise ValueError('Unsafe source video path')
                if source != current_path:
                    if decoder is not None:
                        _stop_process(decoder)
                    from .roboeval_compact import probe
                    profile = next(s for s in probe(source)['streams'] if s['codec_type'] == 'video')
                    if (profile['width'], profile['height']) != (width, height):
                        raise ValueError('Source video dimensions differ')
                    decoder = subprocess.Popen(['ffmpeg', '-hide_banner', '-loglevel', 'error',
                        '-i', str(source), '-map', '0:v:0', '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1'],
                        stdout=subprocess.PIPE, stderr=errors)
                    current_path, current_index = source, -1
                requested = row['video_frame_index']
                if requested < current_index:
                    raise ValueError('Source video indices moved backwards')
                changed = False
                while current_index < requested:
                    pixels = _read_frame(decoder.stdout, pixels_per_frame * 3)
                    current_index += 1; changed = True
                if changed:
                    array = np.frombuffer(pixels, np.uint8).reshape(-1, 3).astype(np.float64) / 255
                    frame_stats = (array.sum(axis=0), np.square(array).sum(axis=0),
                                   array.min(axis=0), array.max(axis=0))
                total += frame_stats[0]; squares += frame_stats[1]
                minimum = np.minimum(minimum, frame_stats[2]); maximum = np.maximum(maximum, frame_stats[3])
                encoder.stdin.write(pixels)
            encoder.stdin.close()
            if encoder.wait(timeout=120) != 0:
                errors.seek(0)
                raise RuntimeError(errors.read().decode(errors='replace'))
        finally:
            for proc in (decoder, encoder):
                if proc is not None:
                    _stop_process(proc)
    mean = total / (len(rows)*pixels_per_frame)
    std = np.sqrt(np.maximum(squares/(len(rows)*pixels_per_frame) - mean*mean, 0))
    return {**{k: v.reshape(3, 1, 1).tolist() for k, v in
               [('min', minimum), ('max', maximum), ('mean', mean), ('std', std)]}, 'count': [len(rows)]}


def validate_continuous(root, marker, *, states, frames):
    meta = marker['continuous_recording']
    fps = 30 if meta.get('version') == 4 else 100
    if meta.get('version') == 4 and (meta.get('training_hz'), meta.get('raw_alignment_hz')) != (30,100):
        raise ValueError('Invalid dual-rate metadata')
    if meta.get('version') not in (1, 2, 3, 4) or (meta.get('state_hz'), meta.get('training_video_fps'), meta.get('source_video_fps')) != (100,fps,30):
        raise ValueError('Unsupported continuous recording metadata')
    resolution = meta.get('source_resolution')
    allowed = [[1280, 720]] if meta['version'] < 3 else [[1280, 720], [640, 480]]
    if resolution not in allowed:
        raise ValueError('Unsupported continuous source resolution')
    width, height = resolution
    if (meta.get('state_max_age_s'), meta.get('frame_max_age_s')) != (None if meta['version']>=3 else .02,.1):
        raise ValueError('Continuous recording requirements differ')
    if meta['version']>=3 and (meta.get('state_age_policy'),meta.get('state_age_reference_s'))!=('record_only',.02):
        raise ValueError('Continuous state age policy differs')
    if meta.get('start_monotonic_s') != meta['policy_targets'][0].get('action_executed_monotonic_s'):
        raise ValueError('Continuous start boundary differs')
    if meta['version'] >= 2:
        from .continuity_quality import assess_timing, require_timing
        require_timing(assess_timing(states, frames, meta['policy_targets'],
            end=meta['end_monotonic_s'], cameras=marker['cameras'],version=min(meta['version'],3)))
    rows, columns, _ = align(states, frames, meta['policy_targets'], end=meta['end_monotonic_s'], cameras=marker['cameras'], version=min(meta['version'],3), grid_hz=fps)
    data = pq.read_table(Path(root) / 'data/chunk-000/file-000.parquet').to_pydict()
    expected = dict(columns, timestamp=np.arange(len(rows), dtype=np.float32)/fps,
                    action=np.asarray([r['action'] for r in rows], np.float32),
                    **{'observation.state': np.asarray([r['joints'][:6]+r['gripper'][:1]+r['joints'][6:]+r['gripper'][1:] for r in rows], np.float32),
                       'observation.capture_timestamp': np.asarray([r['seconds'] for r in rows]),
                       'observation.capture_monotonic_s': np.asarray([r['captured_monotonic_s'] for r in rows])})
    if marker.get('training_auxiliary_columns') == 'omitted':
        expected = {k: v for k, v in expected.items()
                    if k in ('timestamp', 'action', 'observation.state')}
        if any(k.startswith(('observation.source_', 'observation.capture_', 'action_source_')) for k in data):
            raise ValueError('Unexpected auxiliary column in clean training table')
    efforts = [row.get('gripper_effort_nm') for row in rows]
    if any(value is not None for value in efforts):
        if any(value is None for value in efforts):
            raise ValueError('Incomplete gripper effort recording')
        expected['observation.gripper_effort_nm'] = np.asarray(efforts, np.float32)
    elif 'observation.gripper_effort_nm' in data:
        raise ValueError('Gripper effort has no original source evidence')
    if marker.get('delivery_profile'):
        from .training_delivery import physical_vectors
        action_rows = pq.read_table(Path(root) / 'meta/roboeval/actions.parquet').to_pylist()
        expected.update(physical_vectors(rows, states, action_rows))
    for key, value in expected.items():
        if key not in data or not np.array_equal(np.asarray(data[key]), value):
            raise ValueError(f'Continuous alignment differs: {key}')
    info = json.loads((Path(root) / 'meta/info.json').read_text())
    if info['fps'] != fps or info['total_frames'] != len(rows):
        raise ValueError('Continuous main table rate/length differs')
    from .roboeval_compact import probe
    for camera in marker['cameras']:
        feature = info['features'][f'observation.images.{camera}']
        if feature['shape'] != [height, width, 3] or (feature['info']['video.width'], feature['info']['video.height']) != (width, height):
            raise ValueError('Continuous feature dimensions differ')
        video=Path(root)/f'videos/observation.images.{camera}/chunk-000/file-000.mp4'
        stream=next(s for s in probe(video)['streams'] if s['codec_type']=='video')
        if (stream['width'],stream['height'],stream['r_frame_rate'],int(stream['nb_frames'])) != (width,height,f'{fps}/1',len(rows)):
            raise ValueError('Continuous training video dimensions/rate/count differ')

    if marker.get('roboeval_layout'):
        from .roboeval_compact import load_stream
        source_rows = load_stream(root, 'frames').to_pylist()
    else:
        source_rows = frames
    checked = {}
    for row in source_rows:
        name, index = row['video_path'], row.get('video_stream_index', 0)
        if name not in checked:
            checked[name] = probe(Path(root) / name)['streams']
        stream = next(s for s in checked[name] if s['index'] == index)
        if (stream['width'], stream['height']) != (width, height):
            raise ValueError('Continuous source dimensions differ')

    if meta['version'] == 4:
        if marker.get('roboeval_layout'):
            actions = load_stream(root, 'actions', original=True).to_pylist()
        else:
            actions = pq.read_table(Path(root) / 'meta/roboeval/actions.parquet').to_pylist()
        if command_coverage(meta['policy_targets'], columns, actions) != meta.get('command_coverage'):
            raise ValueError('Command coverage metadata differs')
