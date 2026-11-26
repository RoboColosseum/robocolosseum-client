"""Versioned timing assessment; retain observations without manufacturing feedback."""
from bisect import bisect_right
import json
import math
from pathlib import Path

from .continuous_recording import _ordered, FPS, STATE_MAX_AGE, FRAME_MAX_AGE


def assess_timing(states, frames, targets, *, end, cameras, version=3):
    if version not in (2,3):
        raise ValueError('Unsupported timing assessment version')
    action_times = _ordered(targets, 'action_executed_monotonic_s')
    start = action_times[0]
    if not isinstance(end, (int, float)) or not math.isfinite(end) or end <= start or end < action_times[-1]:
        raise ValueError('Invalid successful execution boundary')
    count = math.ceil((end-start)*FPS-1e-8)
    streams = {}
    for side in ('left', 'right'):
        rows = [r for r in states if r['arm_id'] == side]
        _ordered(rows, 'feedback_sequence')
        copies = [r['sample_monotonic_s'] for r in rows]
        if (not all(math.isfinite(t) for t in copies)
                or any(b < a for a, b in zip(copies, copies[1:]))
                or any(r['feedback_received_monotonic_s'] > r['sample_monotonic_s'] for r in rows)):
            raise ValueError('Invalid feedback receipt/copy relation')
        streams[side] = (rows, 'feedback_received_monotonic_s', STATE_MAX_AGE, 99.)
    for camera in cameras:
        rows = [r for r in frames if r['camera_id'] == camera]
        _ordered(rows, 'source_frame_number')
        streams[camera] = (rows, 'received_monotonic_s', FRAME_MAX_AGE, 15.)
    report = dict(version=version, scope='successful_action_interval', start_monotonic_s=start,
                  end_monotonic_s=end, grid_rows=count, complete=True, streams={})
    for name, (rows, key, limit, minimum_hz) in streams.items():
        times = _ordered(rows, key)
        window = [t for t in times if start <= t < end]
        hz = (len(window)-1)/(window[-1]-window[0]) if len(window)>1 else 0.
        summary = dict(samples=len(times), interval_samples=len(window), actual_source_hz=hz, minimum_source_hz=minimum_hz,
                       max_age_s=0., limit_s=limit, violations=0, missing_sources=0,
                       age_is_blocking=version<3 or name in cameras,
                       examples=[], complete=hz+1e-9>=minimum_hz)
        for i in range(count):
            stamp = start+i/FPS
            source = bisect_right(times, stamp)-1
            age = None if source < 0 else stamp-times[source]
            if age is not None:
                summary['max_age_s'] = max(summary['max_age_s'], age)
            if age is None or age > limit+1e-9:
                summary['violations'] += 1
                if age is None:
                    summary['missing_sources'] += 1
                if age is None or summary['age_is_blocking']:
                    summary['complete'] = False
                if len(summary['examples']) < 32:
                    summary['examples'].append(dict(grid_index=i, grid_monotonic_s=stamp,
                        source_index=source, age_s=age))
        report['streams'][name] = summary
        report['complete'] &= summary['complete']
    return report


def require_timing(report):
    problems = []
    for name, value in report['streams'].items():
        if value['actual_source_hz']+1e-9 < value['minimum_source_hz']:
            problems.append(f"{name}: source rate {value['actual_source_hz']:.3f} Hz < {value['minimum_source_hz']:g} Hz")
        if value.get('missing_sources'):
            problems.append(f"{name}: {value['missing_sources']} grid points have no causal source")
        if value['violations'] and value.get('age_is_blocking',True):
            problems.append(f"{name}: {value['violations']} grid sources exceed {value['limit_s']*1000:g} ms or are missing; max {value['max_age_s']*1000:.6f} ms")
    if problems:
        raise ValueError('; '.join(problems))


def assess_run(path):
    """Run only after robot cleanup. A report is not an exported package receipt."""
    import pyarrow.parquet as pq
    from .roboeval_export import source_extension
    from .continuous_recording import align
    path = Path(path)
    report = dict(version=3, complete=False, scope='successful_action_interval')
    try:
        extension = source_extension(path)  # All immutable source hashes required.
        if extension is None:
            raise ValueError('Missing raw capture')
        root = path/'roboeval/meta/roboeval'
        context = json.loads((path/'recording-context.json').read_text())
        status = json.loads((root/'status.json').read_text())
        targets = [r for line in (path/'frames.jsonl').read_text().splitlines()
                   if (r := json.loads(line)).get('action') is not None]
        states = pq.read_table(root/'states.parquet').to_pylist()
        frames = pq.read_table(root/'frames.parquet').to_pylist()
        if len(states) != status['counts']['states'] or len(frames) != status['counts']['frames']:
            raise ValueError('Raw stream counts differ')
        cameras = ['head_image', 'left_image', 'right_image']
        report = assess_timing(states, frames, targets, end=context['continuous_end_monotonic_s'], cameras=cameras)
        report['raw_source_sha256'] = extension['signature']
        report['capture_quality_events'] = status.get('quality_issues', {})
        if report['complete']:
            align(states, frames, targets, end=context['continuous_end_monotonic_s'], cameras=cameras)
        require_timing(report)
    except Exception as exc:
        report.update(complete=False, error=f'{type(exc).__name__}: {exc}')
    (path/'continuity-quality.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    return report
